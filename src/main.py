from __future__ import annotations

import argparse
from pathlib import Path

from rag_chat import build_vectorstore, chat_loop
from video_analyzer import analyze_video, test_vision_smoke


def main() -> None:
    parser = argparse.ArgumentParser(description="Local Video Analyzer + RAG chatbot")
    sub = parser.add_subparsers(dest="cmd", required=True)

    analyze_cmd = sub.add_parser("analyze", help="Sample frames and summarize video")
    analyze_cmd.add_argument("--video", required=True, type=str, help="Path to video file")
    analyze_cmd.add_argument("--config", default="src/config.yaml", type=str, help="Config file path")
    analyze_cmd.add_argument("--frame-interval", type=float, default=None, help="Seconds between sampled frames")
    analyze_cmd.add_argument("--max-frames", type=int, default=None, help="Maximum number of sampled frames")
    analyze_cmd.add_argument("--resize-width", type=int, default=None, help="Resize frame width before vision model")
    analyze_cmd.add_argument(
        "--vision-delay",
        type=float,
        default=None,
        help="Extra delay in seconds between vision requests",
    )

    index_cmd = sub.add_parser("index", help="Build vector store from analysis json")
    index_cmd.add_argument("--analysis", required=True, type=str, help="Path to *_analysis.json file")
    index_cmd.add_argument("--config", default="src/config.yaml", type=str, help="Config file path")

    chat_cmd = sub.add_parser("chat", help="Run RAG chatbot")
    chat_cmd.add_argument("--config", default="src/config.yaml", type=str, help="Config file path")

    test_cmd = sub.add_parser("test-vision", help="Single Ollama vision request (diagnose runner crashes)")
    test_cmd.add_argument("--config", default="src/config.yaml", type=str, help="Config file path")
    test_cmd.add_argument("--image", type=str, default=None, help="Optional JPEG/PNG path; default is a tiny synthetic image")

    args = parser.parse_args()

    if args.cmd == "analyze":
        out = analyze_video(
            Path(args.video),
            Path(args.config),
            frame_interval_seconds_override=args.frame_interval,
            max_frames_override=args.max_frames,
            resize_width_override=args.resize_width,
            vision_delay_seconds_override=args.vision_delay,
        )
        print(f"Analysis saved to: {out}")
    elif args.cmd == "index":
        build_vectorstore(Path(args.analysis), Path(args.config))
        print("Vector store updated.")
    elif args.cmd == "chat":
        chat_loop(Path(args.config))
    elif args.cmd == "test-vision":
        test_vision_smoke(Path(args.config), Path(args.image) if args.image else None)


if __name__ == "__main__":
    main()
