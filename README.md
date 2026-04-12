# Video Intelligence & RAG Chat

A high-performance, local-first platform to analyze videos, generate rich summaries, and interact with your footage using a Retrieval-Augmented Generation (RAG) chatbot.

## Core Features
- **Parallel AI Analysis**: Uses multi-threaded frame captioning with Ollama local vision models.
- **Privacy First**: Everything runs locally on your machine—no cloud APIs or data leaks.
- **Premium Web UI**: Side-by-side layout with a Video Player, AI Summary, and Persistent Chat.
- **RAG Powered**: Query specific details about your video; the AI "remembers" the visual context.
- **Fast Uploads**: Drag-and-drop file ingestion with a modern glassmorphism design.

## Recommended Models (Optimized for RTX 5070)
- **Vision (Fast & Good)**: `moondream` (extremely fast, ~1.6GB)
- **Vision (High Quality)**: `minicpm-v:8b` (GPU accelerated)
- **Text/Chat**: `qwen2.5:7b-instruct`
- **Embeddings**: `all-MiniLM-L6-v2` (Local)

## 1) Installation

```bash
# Clone and setup environment
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Initialize Frontend
cd frontend
npm install
cd ..
```

## 2) Pull Models
Ensure Ollama is running, then pull the optimized models:
```bash
ollama pull moondream
ollama pull minicpm-v:8b
ollama pull qwen2.5:7b-instruct
```

## 3) Run the Application

**Start the Backend API:**
```bash
python src/app.py
```

**Start the React Frontend:**
```bash
cd frontend
npm run dev
```
Navigate to `http://localhost:5173`.

## 4) Configuration (`src/config.yaml`)
Tweak performance settings to match your hardware:
- `max_workers`: Increase to 8+ for parallel vision processing.
- `frame_interval_seconds`: Lower (e.g., 4s) for more detail, higher (e.g., 10s) for speed.
- `vision_model`: Switch between `moondream` (speed) and `minicpm-v:8b` (intelligence).

## Project Structure
- `src/`: Core Python logic (Analysis, RAG, API).
- `frontend/`: React + Vite application.
- `data/`: Local storage for uploaded videos, frame captures, and vector databases.
