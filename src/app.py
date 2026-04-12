from __future__ import annotations

import asyncio
import json
import logging
import uuid
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import uvicorn
import yaml
from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag_chat import build_vectorstore, _load_config as load_rag_config
from video_analyzer import analyze_video
import ollama

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Video Transcriber API")

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve data directory for video preview and assets
data_dir = Path("data")
data_dir.mkdir(parents=True, exist_ok=True)
app.mount("/data", StaticFiles(directory="data"), name="data")

CONFIG_PATH = Path("src/config.yaml")

# Global state for progress tracking
analysis_jobs: Dict[str, Dict] = {}

class AnalyzeRequest(BaseModel):
    video_path: str
    frame_interval: Optional[float] = None
    max_frames: Optional[int] = None

class ChatRequest(BaseModel):
    query: str
    video_name: str

@app.get("/api/config")
async def get_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

@app.get("/api/videos")
async def list_videos():
    video_dir = Path("data/videos")
    if not video_dir.exists():
        video_dir.mkdir(parents=True, exist_ok=True)
    
    videos = []
    for ext in [".mp4", ".mkv", ".avi", ".mov"]:
        videos.extend(list(video_dir.glob(f"*{ext}")))
    
    return [{"name": v.name, "path": str(v.absolute())} for v in videos]

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file to the data/videos directory."""
    video_dir = Path("data/videos")
    video_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = video_dir / file.filename
    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        # Return relative path for frontend preview
        rel_path = f"data/videos/{file.filename}"
        return {"filename": file.filename, "path": str(file_path.absolute()), "url": rel_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

def run_analysis_task(job_id: str, video_path: str, interval: Optional[float], max_f: Optional[int]):
    try:
        def progress_cb(percent: float, status: str):
            analysis_jobs[job_id]["progress"] = percent
            analysis_jobs[job_id]["status_msg"] = status
            logger.info(f"Job {job_id}: {percent}% - {status}")

        analysis_path = analyze_video(
            video_path=Path(video_path),
            config_path=CONFIG_PATH,
            frame_interval_seconds_override=interval,
            max_frames_override=max_f,
            progress_callback=progress_cb
        )
        
        # Load summary from the generated JSON
        with analysis_path.open("r", encoding="utf-8") as f:
            result_data = json.load(f)
            
        analysis_jobs[job_id]["status"] = "completed"
        analysis_jobs[job_id]["analysis_file"] = str(analysis_path)
        analysis_jobs[job_id]["summary"] = result_data.get("video_summary", "")
        
        progress_cb(95.0, "Building vector store/index...")
        build_vectorstore(analysis_path, CONFIG_PATH)
        progress_cb(100.0, "Done.")
        
    except Exception as e:
        analysis_jobs[job_id]["status"] = "failed"
        analysis_jobs[job_id]["error"] = str(e)
        logger.error(f"Job {job_id} failed: {e}")

@app.post("/api/analyze")
async def start_analysis(req: AnalyzeRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    analysis_jobs[job_id] = {
        "id": job_id,
        "status": "processing",
        "progress": 0.0,
        "status_msg": "Queued",
        "video_path": req.video_path,
        "video_name": Path(req.video_path).stem,
        "summary": ""
    }
    
    background_tasks.add_task(
        run_analysis_task, 
        job_id, 
        req.video_path, 
        req.frame_interval, 
        req.max_frames
    )
    
    return {"job_id": job_id}

@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in analysis_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return analysis_jobs[job_id]

@app.post("/api/chat")
async def chat(req: ChatRequest):
    cfg = load_rag_config(CONFIG_PATH)
    ollama_cfg = cfg["ollama"]
    paths_cfg = cfg["paths"]
    
    import chromadb
    from sentence_transformers import SentenceTransformer
    
    vdb_path = Path(paths_cfg["vectorstore_dir"])
    if not vdb_path.exists():
        raise HTTPException(status_code=404, detail="Vector store not found. Analyze a video first.")
        
    vdb = chromadb.PersistentClient(path=str(vdb_path))
    collection = vdb.get_or_create_collection("video_chunks")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    llm = ollama.Client(host=ollama_cfg["host"])
    
    # Check if we have a summary in memory for this video
    job_summary = ""
    for job in analysis_jobs.values():
        if job["video_name"] == req.video_name:
            job_summary = job.get("summary", "")
            break
            
    q_emb = embedder.encode([req.query]).tolist()[0]
    results = collection.query(query_embeddings=[q_emb], n_results=5)
    docs = results.get("documents", [[]])[0]
    
    context = "\n\n---\n\n".join(docs) if docs else "No relevant context found."
    
    # Inject summary as primary context for general questions
    if any(q in req.query.lower() for q in ["what", "summarize", "about", "overall", "gist"]):
        context = f"GLOBAL VIDEO SUMMARY:\n{job_summary}\n\nDETAILED CHUNKS:\n{context}"

    prompt = (
        "You are a helpful assistant answering questions about a video content.\n"
        "Reference the 'GLOBAL VIDEO SUMMARY' if it helps answer high-level questions.\n"
        "Use 'DETAILED CHUNKS' for specific facts.\n"
        "Provide a clear, detailed, and human-friendly response.\n"
        "DO NOT mention watermarks like 'clideo.com'.\n"
        "DO NOT just list timestamps unless asked.\n\n"
        f"Context:\n{context}\n\nQuestion:\n{req.query}"
    )
    
    try:
        resp = llm.chat(
            model=ollama_cfg["text_model"],
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.4},
        )
        return {"answer": resp["message"]["content"].strip(), "context": docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
