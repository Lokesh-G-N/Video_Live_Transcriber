from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2


@dataclass
class FrameInfo:
    index: int
    timestamp_sec: float
    frame_path: Path


def sample_frames(
    video_path: Path,
    output_dir: Path,
    frame_interval_seconds: float = 3.0,
    max_frames: int = 250,
    resize_width: int = 960,
) -> List[FrameInfo]:
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise RuntimeError("Video FPS not detected")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval_frames = max(1, int(frame_interval_seconds * fps))

    frame_infos: List[FrameInfo] = []
    frame_idx = 0
    sampled_count = 0

    while True:
        if sampled_count >= max_frames:
            break

        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval_frames == 0:
            h, w = frame.shape[:2]
            if w > resize_width:
                scale = resize_width / float(w)
                frame = cv2.resize(frame, (resize_width, int(h * scale)))

            frame_file = output_dir / f"frame_{sampled_count:04d}.jpg"
            cv2.imwrite(str(frame_file), frame)
            timestamp = frame_idx / fps
            frame_infos.append(
                FrameInfo(index=sampled_count, timestamp_sec=timestamp, frame_path=frame_file)
            )
            sampled_count += 1

        frame_idx += 1
        if total_frames > 0 and frame_idx >= total_frames:
            break

    cap.release()
    return frame_infos
