import pandas as pd
import os
import sys

try:
    from rouge_score import rouge_scorer
except ImportError:
    print("rouge-score library not found.")
    sys.exit(1)

try:
    import torch
    from bert_score import BERTScorer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bert_scorer = BERTScorer(lang="ko", rescale_with_baseline=False, device=device)
except ImportError:
    bert_scorer = None
except Exception as e:
    bert_scorer = None

def main():
    history_file = os.path.join(os.path.dirname(__file__), 'history.csv')
    if not os.path.exists(history_file):
        print(f"Error: {history_file} not found.")
        return

    df = pd.read_csv(history_file, encoding='utf-8-sig').fillna("")
    if df.empty:
        print("No data in history.csv to evaluate.")
        return

    rouge_evaluator = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    summary_keys = ["tf_idf", "text_rank", "lsa", "lex_rank", "mmr"]
    
    # Initialize rankings
    metrics = ['rouge1', 'rouge2', 'rougeL']
    if bert_scorer:
        metrics.extend(['bert_p', 'bert_r', 'bert_f1'])

    ranks = {m: {k: {'1st': 0, '2nd': 0} for k in summary_keys} for m in metrics}

    print(f"Analyzing rankings for {len(df)} total records...")

    valid_records = 0
    for index, row in df.iterrows():
        reference = row.get("ollama", "")
        if not reference.strip(): continue

        row_scores = {k: {m: 0.0 for m in metrics} for k in summary_keys}
        
        # Calculate scores for all candidate summaries
        for key in summary_keys:
            candidate = row.get(key, "")
            if not candidate.strip(): continue

            r_scores = rouge_evaluator.score(reference, candidate)
            row_scores[key]['rouge1'] = r_scores['rouge1'].fmeasure
            row_scores[key]['rouge2'] = r_scores['rouge2'].fmeasure
            row_scores[key]['rougeL'] = r_scores['rougeL'].fmeasure

            if bert_scorer:
                P, R, F1 = bert_scorer.score([candidate], [reference])
                row_scores[key]['bert_p'] = P.item()
                row_scores[key]['bert_r'] = R.item()
                row_scores[key]['bert_f1'] = F1.item()

        # Update rankings for this row independently for each metric
        for m in metrics:
            sorted_candidates = sorted(summary_keys, key=lambda k: row_scores[k][m], reverse=True)
            if not sorted_candidates:
                continue

            # Group keys by their score to handle ties properly
            score_to_keys = {}
            for k in summary_keys:
                score = row_scores[k][m]
                if score not in score_to_keys:
                    score_to_keys[score] = []
                score_to_keys[score].append(k)
                
            # Sort the unique scores descending
            unique_scores = sorted(list(score_to_keys.keys()), reverse=True)
            
            # 1st place keys are always those with the highest score
            first_place_score = unique_scores[0]
            first_place_keys = score_to_keys[first_place_score]
            
            for k in first_place_keys:
                ranks[m][k]['1st'] += 1
                
            # If there was a tie for 1st place with 2 or more candidates, 
            # they take up the 1st AND 2nd rank slots, so no one gets 2nd place.
            # Only if exactly ONE candidate got 1st place do we award 2nd place
            # to the candidate(s) with the next highest score.
            if len(first_place_keys) == 1 and len(unique_scores) > 1:
                second_place_score = unique_scores[1]
                second_place_keys = score_to_keys[second_place_score]
                for k in second_place_keys:
                    ranks[m][k]['2nd'] += 1
        
        valid_records += 1
        print(f"Processed {valid_records}/{len(df)} records...", end='\r')

    print("\n\n" + "="*80)
    print("RANKING COUNT (1st / 2nd place)")
    print("="*80)
    print(f"{'Method':<12} | {'ROUGE-1 (1st/2nd)':<18} | {'ROUGE-2 (1st/2nd)':<18} | {'ROUGE-L (1st/2nd)':<18}")
    if bert_scorer:
        print(f"{'':<12} | {'BERT-P (1st/2nd)':<18} | {'BERT-R (1st/2nd)':<18} | {'BERT-F1 (1st/2nd)':<18}")
    print("-" * 80)

    for k in summary_keys:
        r1_str = f"{ranks['rouge1'][k]['1st']}/{ranks['rouge1'][k]['2nd']}"
        r2_str = f"{ranks['rouge2'][k]['1st']}/{ranks['rouge2'][k]['2nd']}"
        rl_str = f"{ranks['rougeL'][k]['1st']}/{ranks['rougeL'][k]['2nd']}"
        print(f"{k:<12} | {r1_str:<18} | {r2_str:<18} | {rl_str:<18}")

        if bert_scorer:
            bp_str = f"{ranks['bert_p'][k]['1st']}/{ranks['bert_p'][k]['2nd']}"
            br_str = f"{ranks['bert_r'][k]['1st']}/{ranks['bert_r'][k]['2nd']}"
            bf1_str = f"{ranks['bert_f1'][k]['1st']}/{ranks['bert_f1'][k]['2nd']}"
            print(f"{'':<12} | {bp_str:<18} | {br_str:<18} | {bf1_str:<18}")
        print("-" * 80)

    print("\n" + "="*80)

if __name__ == "__main__":
    main()
