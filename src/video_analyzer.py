from __future__ import annotations

import base64
import json
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional

import ollama
import yaml
from PIL import Image
from tqdm import tqdm

from frame_sampler import sample_frames


def _load_config(config_path: Path) -> Dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _prepare_temp_image(image_path: Path, max_dim: int, quality: int) -> Path:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        w, h = img.size
        scale = min(1.0, float(max_dim) / float(max(w, h)))
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp_path = Path(tmp.name)
        tmp.close()
        img.save(tmp_path, format="JPEG", quality=quality, optimize=True)
        return tmp_path


def _image_to_b64(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def _vision_caption_with_retry(
    client: ollama.Client,
    model: str,
    image_path: Path,
    timestamp_sec: float,
    max_retries: int = 3,
) -> str:
    prompt = (
        "You are describing one frame from a video.\n"
        "Focus on important actions, objects, scene context, and any visible text.\n"
        "Keep it factual and concise (2-4 lines).\n"
        f"Frame timestamp: {timestamp_sec:.2f} seconds."
    )

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        temp_path: Path | None = None
        try:
            if attempt == 0:
                send_path = image_path
            else:
                max_dim = max(384, 960 - attempt * 224)
                quality = max(45, 85 - attempt * 15)
                temp_path = _prepare_temp_image(image_path, max_dim=max_dim, quality=quality)
                send_path = temp_path

            response = client.chat(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [_image_to_b64(send_path)],
                    }
                ],
                options={"temperature": 0.2},
            )
            return response["message"]["content"].strip()
        except Exception as err:  # noqa: BLE001
            last_error = err
            time.sleep(min(2 + attempt, 5))
        finally:
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass

    raise RuntimeError(
        f"Vision captioning failed after retries for frame at {timestamp_sec:.2f}s: {last_error}"
    )


def _summarize_captions(client: ollama.Client, model: str, captions: List[Dict]) -> str:
    # Remove timestamps for the summary to prevent "framewise" reporting
    raw_captions = [c['caption'] for c in captions]
    joined = " ".join(raw_captions)

    prompt = (
        "You are an executive content strategist. Below is a sequence of observations from a video. "
        "Your task is to synthesize these observations into a single, high-level executive summary paragraph.\n\n"
        "OBSERVATIONS:\n"
        f"{joined}\n\n"
        "FINAL INSTRUCTIONS (MANDATORY):\n"
        "- Write exactly ONE cohesive and professional paragraph summarizing the ENTIRE video.\n"
        "- Do NOT list individual frames, images, or specific timestamps.\n"
        "- NEVER start with 'The image shows' or 'In this scene'.\n"
        "- Synthesize the core product, theme, and purpose (e.g., if it's an ad for Genspark, say so).\n"
        "- NO bullet points, NO lists, NO colons. Keep it under 8 lines.\n\n"
        "EXECUTIVE SUMMARY:"
    )

    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1},
    )
    return response["message"]["content"].strip()


def analyze_video(
    video_path: Path,
    config_path: Path,
    frame_interval_seconds_override: Optional[float] = None,
    max_frames_override: Optional[int] = None,
    resize_width_override: Optional[int] = None,
    vision_delay_seconds_override: Optional[float] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Path:
    cfg = _load_config(config_path)
    paths_cfg = cfg["paths"]
    video_cfg = cfg["video"]
    ollama_cfg = cfg["ollama"]

    frames_dir = Path(paths_cfg["frames_dir"]) / video_path.stem
    captions_dir = Path(paths_cfg["captions_dir"])
    captions_dir.mkdir(parents=True, exist_ok=True)

    client = ollama.Client(host=ollama_cfg["host"])

    if progress_callback:
        progress_callback(0.0, "Sampling frames...")

    frame_infos = sample_frames(
        video_path=video_path,
        output_dir=frames_dir,
        frame_interval_seconds=(
            frame_interval_seconds_override
            if frame_interval_seconds_override is not None
            else video_cfg["frame_interval_seconds"]
        ),
        max_frames=max_frames_override if max_frames_override is not None else video_cfg["max_frames"],
        resize_width=resize_width_override if resize_width_override is not None else video_cfg["resize_width"],
    )
    if not frame_infos:
        raise RuntimeError("No frames were sampled from the video.")

    num_frames = len(frame_infos)
    all_captions: List[Dict] = [None] * num_frames  # Keep order
    failed_frames: List[Dict] = []
    retry_count = int(ollama_cfg.get("vision_retries", 3))
    max_workers = int(ollama_cfg.get("max_workers", 5))

    if progress_callback:
        progress_callback(10.0, f"Starting captioning with {max_workers} threads...")

    def _process_frame(fi_idx: int, frame_info):
        try:
            caption = _vision_caption_with_retry(
                client=client,
                model=ollama_cfg["vision_model"],
                image_path=frame_info.frame_path,
                timestamp_sec=frame_info.timestamp_sec,
                max_retries=retry_count,
            )
            return (fi_idx, {
                "frame_index": frame_info.index,
                "timestamp_sec": frame_info.timestamp_sec,
                "frame_path": str(frame_info.frame_path),
                "caption": caption,
            }, None)
        except Exception as err:
            return (fi_idx, None, {
                "frame_index": frame_info.index,
                "timestamp_sec": frame_info.timestamp_sec,
                "frame_path": str(frame_info.frame_path),
                "error": str(err),
            })

    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_frame, i, fi): i for i, fi in enumerate(frame_infos)}
        for future in tqdm(as_completed(futures), total=num_frames, desc="Captioning frames"):
            idx, caption_data, error_data = future.result()
            if caption_data:
                all_captions[idx] = caption_data
            else:
                failed_frames.append(error_data)
            
            completed += 1
            if progress_callback:
                # Progress from 10% to 90%
                p = 10.0 + (completed / num_frames) * 80.0
                progress_callback(p, f"Processed {completed}/{num_frames} frames...")

    # Filter out None from failed frames
    all_captions = [c for c in all_captions if c is not None]

    if not all_captions:
        out_path = captions_dir / f"{video_path.stem}_analysis_failed.json"
        payload = {
            "video_path": str(video_path),
            "frames_count": 0,
            "failed_frames_count": len(failed_frames),
            "failed_frames": failed_frames,
            "frame_captions": [],
            "video_summary": "",
        }
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        raise RuntimeError(
            f"No frame captions were generated. Failure report saved at: {out_path}"
        )

    if progress_callback:
        progress_callback(90.0, "Generating final summary...")

    video_summary = _summarize_captions(client, ollama_cfg["text_model"], all_captions)

    if progress_callback:
        progress_callback(100.0, "Analysis complete.")

    out_path = captions_dir / f"{video_path.stem}_analysis.json"
    payload = {
        "video_path": str(video_path),
        "frames_count": len(all_captions),
        "failed_frames_count": len(failed_frames),
        "failed_frames": failed_frames,
        "frame_captions": all_captions,
        "video_summary": video_summary,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def test_vision_smoke(config_path: Path, image_path: Optional[Path] = None) -> None:
    """One request to Ollama vision model — use to confirm runner works before long analyze runs."""
    cfg = _load_config(config_path)
    ollama_cfg = cfg["ollama"]
    model = ollama_cfg["vision_model"]
    client = ollama.Client(host=ollama_cfg["host"])

    created_temp = False
    if image_path is not None:
        img_path = image_path
    else:
        created_temp = True
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp_path = Path(tmp.name)
        tmp.close()
        Image.new("RGB", (256, 256), color=(240, 240, 240)).save(tmp_path, format="JPEG", quality=85)
        img_path = tmp_path

    try:
        response = client.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "Describe this image in one short sentence.",
                    "images": [_image_to_b64(img_path)],
                }
            ],
            options={"temperature": 0.2},
        )
        text = response["message"]["content"].strip()
        print(f"OK — vision model '{model}' responded:\n{text}")
    finally:
        if created_temp and img_path.exists():
            try:
                img_path.unlink()
            except OSError:
                pass
