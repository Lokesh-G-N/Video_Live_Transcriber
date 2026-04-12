from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import chromadb
import ollama
import yaml
from sentence_transformers import SentenceTransformer


def _load_config(config_path: Path) -> Dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _chunk_frame_captions(frame_captions: List[Dict], chunk_size: int = 12) -> List[Tuple[str, str]]:
    docs: List[Tuple[str, str]] = []
    for i in range(0, len(frame_captions), chunk_size):
        part = frame_captions[i : i + chunk_size]
        start_t = part[0]["timestamp_sec"]
        end_t = part[-1]["timestamp_sec"]
        lines = [f"[{x['timestamp_sec']:.2f}s] {x['caption']}" for x in part]
        text = "\n".join(lines)
        doc_id = f"chunk_{i//chunk_size:04d}"
        docs.append((doc_id, f"Time range {start_t:.2f}s - {end_t:.2f}s\n{text}"))
    return docs


def build_vectorstore(analysis_json_path: Path, config_path: Path) -> None:
    cfg = _load_config(config_path)
    paths_cfg = cfg["paths"]

    data = json.loads(analysis_json_path.read_text(encoding="utf-8"))
    frame_captions = data["frame_captions"]
    summary = data["video_summary"]

    persist_dir = Path(paths_cfg["vectorstore_dir"])
    persist_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection("video_chunks")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    docs = _chunk_frame_captions(frame_captions)
    ids = [d[0] for d in docs]
    texts = [d[1] for d in docs]
    embeddings = embedder.encode(texts, show_progress_bar=True).tolist()

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=[{"source": str(analysis_json_path)} for _ in docs],
    )

    summary_id = f"{analysis_json_path.stem}_global_summary"
    sum_emb = embedder.encode([summary]).tolist()
    collection.upsert(
        ids=[summary_id],
        documents=[summary],
        embeddings=sum_emb,
        metadatas=[{"source": str(analysis_json_path), "type": "global_summary"}],
    )


def chat_loop(config_path: Path) -> None:
    cfg = _load_config(config_path)
    ollama_cfg = cfg["ollama"]
    paths_cfg = cfg["paths"]

    vdb = chromadb.PersistentClient(path=str(Path(paths_cfg["vectorstore_dir"])))
    collection = vdb.get_or_create_collection("video_chunks")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    llm = ollama.Client(host=ollama_cfg["host"])

    print("RAG chat ready. Type 'exit' to stop.")
    while True:
        q = input("\nYou: ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break

        q_emb = embedder.encode([q]).tolist()[0]
        results = collection.query(query_embeddings=[q_emb], n_results=5)
        docs = results.get("documents", [[]])[0]

        context = "\n\n---\n\n".join(docs) if docs else "No relevant context found."
        prompt = (
            "You are a helpful assistant answering questions about a video.\n"
            "Use ONLY the provided context. If uncertain, say so.\n\n"
            f"Context:\n{context}\n\nQuestion:\n{q}"
        )

        resp = llm.chat(
            model=ollama_cfg["text_model"],
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.2},
        )
        print(f"\nAssistant: {resp['message']['content'].strip()}")
