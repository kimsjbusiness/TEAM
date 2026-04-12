from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pandas as pd
import os
import csv
from datetime import datetime
from summarizer import Summarizer
from scraper import scrape_article
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Scorers
rouge_evaluator = None
try:
    from rouge_score import rouge_scorer
    rouge_evaluator = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    logger.info("ROUGE Scorer initialized.")
except ImportError:
    logger.error("rouge-score library not found. Please install it with 'pip install rouge-score'.")
except Exception as e:
    logger.error(f"Error initializing ROUGE Scorer: {e}")


bert_scorer = None
try:
    import torch
    from bert_score import BERTScorer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bert_scorer = BERTScorer(lang="ko", rescale_with_baseline=False, device=device)
    logger.info(f"BERT Scorer initialized on {device}.")
except ImportError:
    logger.error("bert-score library not found. Please install it with 'pip install bert-score'.")
except Exception as e:
    logger.warning(f"BERT Scorer initialization failed: {e}")

app = FastAPI()

# Mount static files
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
if not os.path.exists("templates"):
    os.makedirs("templates")
templates = Jinja2Templates(directory="templates")

# Initialize Summarizer
summarizer = Summarizer()

# CSV File Path
HISTORY_FILE = "history.csv"

# Ensure CSV exists with headers
if not os.path.exists(HISTORY_FILE):
    df = pd.DataFrame(columns=["id", "timestamp", "title", "context", "tf_idf", "text_rank", "lsa", "lex_rank", "mmr", "ollama"])
    df.to_csv(HISTORY_FILE, index=False, encoding="utf-8-sig")

class SummarizeRequest(BaseModel):
    title: str
    text: str

class SaveRequest(BaseModel):
    title: str
    context: str
    tf_idf: str
    text_rank: str
    lsa: str
    lex_rank: str
    mmr: str
    ollama: str
    # Scores are calculated on demand, not saved in CSV for simplicity (or can be saved if needed)

class ScrapeRequest(BaseModel):
    url: str

def calculate_scores(reference, candidate):
    scores = {}
    
    # ROUGE
    if rouge_evaluator and reference and candidate:
        try:
            r_scores = rouge_evaluator.score(reference, candidate)
            scores['rouge'] = {
                'r1': r_scores['rouge1'].fmeasure,
                'r2': r_scores['rouge2'].fmeasure,
                'rl': r_scores['rougeL'].fmeasure
            }
        except Exception as e:
            logger.error(f"ROUGE calculation error: {e}")
            scores['rouge'] = None

    # BERT
    if bert_scorer and reference and candidate:
        try:
            P, R, F1 = bert_scorer.score([candidate], [reference])
            scores['bert'] = {
                'f1': F1.item(),
                'p': P.item(),
                'r': R.item()
            }
        except Exception as e:
            logger.error(f"BERT calculation error: {e}")
            scores['bert'] = None
            
    return scores

@app.post("/scrape")
async def scrape_url(request: ScrapeRequest):
    result = scrape_article(request.url)
    if "error" in result:
        return JSONResponse(content={"error": result["error"]}, status_code=500)
    return JSONResponse(content=result)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/summarize")
async def summarize(request: SummarizeRequest):
    text = request.text
    
    # Generate summaries
    results = {
        "tf_idf": summarizer.tfidf_summary(text),
        "text_rank": summarizer.textrank_summary(text),
        "lsa": summarizer.lsa_summary(text),
        "lex_rank": summarizer.lexrank_summary(text),
        "mmr": summarizer.mmr_summary(text),
        "ollama": summarizer.ollama_summary(text)
    }
    
    # Calculate scores against Ollama (Reference)
    scores = {}
    reference = results.get("ollama", "")
    
    if reference:
        for key, summary in results.items():
            if key == "ollama": continue
            scores[key] = calculate_scores(reference, summary)
            
    results["scores"] = scores
    
    return JSONResponse(content=results)

@app.post("/save")
async def save_summary(request: SaveRequest):
    new_data = {
        "id": datetime.now().strftime("%Y%m%d%H%M%S"),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "title": request.title,
        "context": request.context,
        "tf_idf": request.tf_idf,
        "text_rank": request.text_rank,
        "lsa": request.lsa,
        "lex_rank": request.lex_rank,
        "mmr": request.mmr,
        "ollama": request.ollama
    }
    
    df = pd.DataFrame([new_data])
    
    # Append to CSV
    if not os.path.exists(HISTORY_FILE):
        df.to_csv(HISTORY_FILE, index=False, encoding="utf-8-sig")
    else:
        df.to_csv(HISTORY_FILE, mode='a', header=False, index=False, encoding="utf-8-sig")
        
    return JSONResponse(content={"message": "Saved successfully", "id": new_data["id"]})

@app.get("/history", response_class=HTMLResponse)
async def read_history(request: Request):
    if os.path.exists(HISTORY_FILE):
        try:
            df = pd.read_csv(HISTORY_FILE, encoding="utf-8-sig")
            # Fill NaN
            df = df.fillna("")
            history_list = df.to_dict(orient="records")
            # Sort by timestamp descending
            history_list.reverse()
        except pd.errors.EmptyDataError:
             history_list = []
    else:
        history_list = []
        
    return templates.TemplateResponse("history.html", {"request": request, "history": history_list})

@app.get("/history/{item_id}", response_class=JSONResponse)
async def get_history_item(item_id: str):
    if os.path.exists(HISTORY_FILE):
        try:
            df = pd.read_csv(HISTORY_FILE, encoding="utf-8-sig")
            df = df.fillna("")
            # Ensure id is treated as string
            df['id'] = df['id'].astype(str)
            item_list = df[df['id'] == item_id].to_dict(orient="records")
            
            if item_list:
                item = item_list[0]
                
                # Calculate scores on the fly
                scores = {}
                reference = item.get("ollama", "")
                
                # Available summarization keys in CSV
                summary_keys = ["tf_idf", "text_rank", "lsa", "lex_rank", "mmr"]
                
                if reference:
                    for key in summary_keys:
                        if key in item:
                            scores[key] = calculate_scores(reference, item[key])
                
                item["scores"] = scores
                return JSONResponse(content=item)
                
        except Exception as e:
            logger.error(f"Error reading history item: {e}")
            return JSONResponse(content={"error": "Error processing item"}, status_code=500)
            
    return JSONResponse(content={"error": "Item not found"}, status_code=404)

@app.delete("/history/{item_id}")
async def delete_history_item(item_id: str):
    if os.path.exists(HISTORY_FILE):
        df = pd.read_csv(HISTORY_FILE, encoding="utf-8-sig")
        df['id'] = df['id'].astype(str)
        
        # Check if item exists
        if item_id not in df['id'].values:
            return JSONResponse(content={"error": "Item not found"}, status_code=404)
            
        # Remove item
        df = df[df['id'] != item_id]
        df.to_csv(HISTORY_FILE, index=False, encoding="utf-8-sig")
        
        return JSONResponse(content={"message": "Deleted successfully"})
    return JSONResponse(content={"error": "History file not found"}, status_code=404)

if __name__ == "__main__":
    import uvicorn
    # 0.0.0.0으로 해야 외부에서 접속 가능
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
