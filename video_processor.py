import os
import io
import cv2
import torch
import base64
import requests
import re
from PIL import Image
from dotenv import load_dotenv
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load environment variables
load_dotenv()

# Gemini Vision API
GEMINI_API_KEY = "Your API KEY"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

# FLAN-T5 Summarizer (optional - not used if you only want combined captions)
summarizer_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
summarizer_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")

# === Clean caption: remove unwanted special characters and newlines, keep sentences complete ===
def clean_caption(text):
    text = text.replace("Here is a description of the image:", "")
    text = text.replace("**Description:**", "").replace("**", "")
    # Remove \, /, \n, ', ", (, )
    text = re.sub(r"[\\/\n\'\"\(\)]", "", text)
    # Replace multiple spaces with single space
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# === Gemini captioning ===
def caption_frame_with_gemini(image_np):
    image_pil = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
    buffered = io.BytesIO()
    image_pil.save(buffered, format="JPEG")
    image_bytes = buffered.getvalue()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{
            "parts": [
                {"inline_data": {"mime_type": "image/jpeg", "data": image_b64}},
                {"text": "Describe the image using 1–2 short but complete sentences. Do not leave any sentence incomplete."}
            ]
        }]
    }

    try:
        response = requests.post(GEMINI_URL, headers=headers, json=data)
        response.raise_for_status()
        raw_text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        return clean_caption(raw_text)
    except Exception as e:
        print(f"[Gemini Error] {e}")
        return None

# === Scene detection ===
def detect_scenes(video_path):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()
    video_manager.release()
    return scene_list

# === Keyframe extraction ===
def extract_keyframes(video_path, scene_list, max_frames=20):
    cap = cv2.VideoCapture(video_path)
    frames = []
    step = max(1, len(scene_list) // max_frames)
    for i, (start, _) in enumerate(scene_list[::step]):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start.get_frames())
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        if len(frames) >= max_frames:
            break
    cap.release()
    return frames

# === Optional summarization ===
def summarize_with_flan(captions: list[str], max_tokens=50) -> str:
    if not captions:
        return "No captions to summarize."
    prompt = "Summarize the following image captions into one short sentence:\n" + "\n".join(captions)
    inputs = summarizer_tokenizer(prompt, return_tensors="pt", truncation=True)
    summary_ids = summarizer_model.generate(inputs["input_ids"], max_length=max_tokens, num_beams=4, early_stopping=True)
    output = summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return output.strip()

# === Main process function ===
def process_video(video_path):
    scenes = detect_scenes(video_path)
    frames = extract_keyframes(video_path, scenes, max_frames=20)

    total_duration = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_POS_MSEC) / 1000
    segment_duration = 3  # seconds per caption segment

    caption_chunks = []
    num_segments = len(frames)

    for i, frame in enumerate(frames):
        caption = caption_frame_with_gemini(frame)
        if caption:
            start_time = i * segment_duration
            end_time = start_time + segment_duration
            caption_chunks.append({
                "start": start_time,
                "end": end_time,
                "caption": caption
            })

    return caption_chunks
