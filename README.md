# 🎬 Video Transcribe: AI-Powered Captioning App

This Flask-based web application allows users to upload a video and automatically generates **live captions** using AI. Captions are displayed **as the video plays**, with typewriter-style animation. Scene detection and keyframe extraction are used to ensure accurate visual summaries.

---

## 📁 Project Structure
```
Video_Transcribe/
│
├── static/ # Static assets (optional)
├── templates/
│ └── index.html # Main HTML interface
├── uploads/ # Stores uploaded videos
├── venv/ # Virtual environment (auto-generated)
├── app.py # Flask app backend
├── video_processor.py # Core logic: scene detection, keyframe extraction, captioning
├── requirements.txt # Python dependencies
├── .env # Environment variables (for Gemini API key)
└── README.md # You're reading this!
```
---

## ⚙️ Features

- Upload any `.mp4` video
- Automatically extracts ~20 keyframes using scene detection
- Captions each frame using **Google Gemini Vision API**
- Live captions display in sync with video playback
- Typewriter effect for smooth caption rendering
- Jump to captions by clicking text or scrubbing through the video
- Resets captions on replay

---

## 🌐 Requirements

- Python 3.9+
- Google Gemini API Key
- Internet connection (for Gemini API)
- (Optional) GPU acceleration for faster processing

---

## 📦 Installation

1. **Clone the repository**:

```
git clone https://github.com/yourusername/Video_Transcribe.git
cd Video_Transcribe
```

Create a virtual environment:
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install dependencies:
```
pip install -r requirements.txt
```

🔐 Google Gemini API Setup
Go to Google AI Studio and generate your Gemini API key.
Create a .env file in the root directory:
```
GEMINI_API_KEY=your_google_gemini_api_key_here
This key will be used in video_processor.py to send requests to Gemini’s vision model.
```

🚀 Run the App
```
python app.py
```

Then open your browser and go to:
http://127.0.0.1:5000
Upload a video and watch the AI-generated captions live.

🧠 How It Works
Uses PySceneDetect to detect scenes in the video.
Extracts 20 frames (or less) spaced across these scenes.
Sends each frame to Google Gemini Vision API.
Cleans up the AI response using regex.
Creates caption segments (every 3 seconds).
Streams them live as you watch the video, with smooth typewriter animation.

✏️ To Customize
Change caption timing: segment_duration in video_processor.py
Change max keyframes: max_frames in extract_keyframes
Modify prompt for Gemini inside caption_frame_with_gemini()

🛠 Dependencies
Flask
requests
opencv-python
python-dotenv
transformers
torch
scenedetect
Pillow
