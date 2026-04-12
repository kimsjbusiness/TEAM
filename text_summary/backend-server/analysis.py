import pandas as pd
import argparse
import logging
import sys
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Analyze Summary Scores")
    parser.add_argument("--input", default="history.csv", help="Path to input CSV")
    parser.add_argument("--ref-col", default="ollama", help="Reference column (Ground Truth). Default is 'ollama'.")
    parser.add_argument("--chunk-size", type=int, default=100, help="Rows per chunk for processing")
    args = parser.parse_args()
    input_path = args.input

    bert_scorer = None
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        logging.error("Missing required library: rouge-score. Please install: pip install rouge-score")
        sys.exit(1)

    try:
        import torch
        from bert_score import BERTScorer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {device} for BERTScore")
        bert_scorer = BERTScorer(lang="ko", rescale_with_baseline=False, device=device)
    except ImportError as e:
        logging.warning(f"bert-score or torch library not found ({e}). Skipping BERTScore calculation.")
    except Exception as e:
        logging.warning(f"Failed to initialize BERTScorer (likely torch issue): {e}. Skipping BERTScore.")

    try:
        # Check columns first by reading just the header
        df_head = pd.read_csv(input_path, nrows=0)
        available_cols = df_head.columns.tolist()
        
        logging.info(f"Available columns: {available_cols}")

        if args.ref_col not in available_cols:
             logging.warning(f"Reference column '{args.ref_col}' not found in CSV.")
             # Try to find a suitable reference, or fallback to 'context' if present
             if 'context' in available_cols:
                 logging.warning(f"Falling back to 'context' as reference.")
                 args.ref_col = 'context'
             else:
                 logging.error("No valid reference column found (neither specified nor 'context'). Exiting.")
                 sys.exit(1)
        
        # Determine candidate columns: all except metadata and reference
        exclude_cols = {'id', 'timestamp', 'title', 'context', args.ref_col, 'Unnamed: 0'}
        # Also exclude potential index columns or empty ones
        cand_cols = [c for c in available_cols if c not in exclude_cols]
        
        logging.info(f"Reference Column: '{args.ref_col}'")
        logging.info(f"Candidate Columns to Evaluate: {cand_cols}")
        
        if not cand_cols:
            logging.error("No candidate columns found to evaluate.")
            sys.exit(1)

        # Initialize scorers
        rouge_evaluator = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Global metrics accumulators
        # Structure: {candidate: {'rouge1': sum, 'rouge2': sum, 'rougeL': sum, 'bert_P': sum, 'bert_R': sum, 'bert_F1': sum, 'count': 0}}
        metrics = {c: {'r1': 0.0, 'r2': 0.0, 'rl': 0.0, 'b_p': 0.0, 'b_r': 0.0, 'b_f1': 0.0, 'count': 0} for c in cand_cols}
        
        total_rows = 0
        
        # Iterate via chunks to handle large files
        chunk_iterator = pd.read_csv(input_path, chunksize=args.chunk_size)
        
        logging.info("Starting analysis...")
        
        for chunk in chunk_iterator:
            # Fill NaNs with empty string
            chunk = chunk.fillna("").astype(str)
            
            # Extract reference texts
            refs = chunk[args.ref_col].tolist()
            
            # Process each candidate column
            for cand in cand_cols:
                preds = chunk[cand].tolist()
                
                # Pair filtering: ignore empty refs or empty preds
                valid_pairs = []
                valid_indices = []
                
                for i, (p, r) in enumerate(zip(preds, refs)):
                    if p.strip() and r.strip():
                        valid_pairs.append((p, r))
                        valid_indices.append(i)
                
                if not valid_pairs:
                    continue
                    
                vp_preds, vp_refs = zip(*valid_pairs)
                
                # --- ROUGE Calculation ---
                batch_r1, batch_r2, batch_rl = 0, 0, 0
                for p, r in valid_pairs:
                    s = rouge_evaluator.score(r, p) # target, prediction
                    batch_r1 += s['rouge1'].fmeasure
                    batch_r2 += s['rouge2'].fmeasure
                    batch_rl += s['rougeL'].fmeasure
                
                metrics[cand]['r1'] += batch_r1
                metrics[cand]['r2'] += batch_r2
                metrics[cand]['rl'] += batch_rl
                
                # --- BERTScore Calculation ---
                if bert_scorer:
                    try:
                        P, R, F1 = bert_scorer.score(list(vp_preds), list(vp_refs))
                        # Sum up the scores for this batch
                        metrics[cand]['b_p'] += P.sum().item()
                        metrics[cand]['b_r'] += R.sum().item()
                        metrics[cand]['b_f1'] += F1.sum().item()
                    except Exception as e:
                        logging.warning(f"BERTScore failed for a batch in {cand}: {e}")
                
                metrics[cand]['count'] += len(valid_pairs)
            
            total_rows += len(chunk)
            print(f"Processed {total_rows} rows...", end='\r')

        print("\n" + "="*80)
        print(f"ANALYSIS REPORT | Reference: {args.ref_col} | Rows: {total_rows}")
        print("="*80)
        print(f"{'Model':<15} | {'ROUGE-1':<8} | {'ROUGE-2':<8} | {'ROUGE-L':<8} | {'BERT-P':<8} | {'BERT-R':<8} | {'BERT-F1':<8}")
        print("-" * 80)
        
        for cand, m in metrics.items():
            cnt = m['count']
            if cnt == 0:
                print(f"{cand:<15} | {'N/A':<8} | {'N/A':<8} | {'N/A':<8} | {'N/A':<8} | {'N/A':<8} | {'N/A':<8}")
            else:
                bp = m['b_p']/cnt if bert_scorer else 0.0
                br = m['b_r']/cnt if bert_scorer else 0.0
                bf1 = m['b_f1']/cnt if bert_scorer else 0.0
                
                bp_str = f"{bp:.4f}" if bert_scorer else "N/A"
                br_str = f"{br:.4f}" if bert_scorer else "N/A"
                bf1_str = f"{bf1:.4f}" if bert_scorer else "N/A"

                print(f"{cand:<15} | {m['r1']/cnt:.4f}   | {m['r2']/cnt:.4f}   | {m['rl']/cnt:.4f}   | {bp_str:<8} | {br_str:<8} | {bf1_str:<8}")
        
        print("="*80)
        logging.info("Analysis complete.")

    except Exception as e:
        logging.error(f"Analysis process failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
