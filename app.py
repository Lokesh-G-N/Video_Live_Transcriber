import time

from flask import Flask, render_template, request
from video_processor import process_video
import os

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Video Summary App</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        h1 { color: #333; }
        .container { max-width: 700px; margin: auto; }
        textarea { width: 100%; height: 250px; margin-top: 20px; padding: 10px; font-size: 16px; }
        .summary-title { margin-top: 40px; font-weight: bold; font-size: 20px; }

        /* Loader Style */
        #loader {
            border: 6px solid #f3f3f3;
            border-top: 6px solid #555;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
            display: none;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
    <script>
        function showLoader() {
            document.getElementById("loader").style.display = "block";
        }
    </script>
</head>
<body>
    <div class="container">
        <h1>Upload a Video to Generate Summary</h1>
        <form method="POST" enctype="multipart/form-data" onsubmit="showLoader()">
            <input type="file" name="video" accept="video/*" required>
            <button type="submit">Upload & Summarize</button>
        </form>

        <div id="loader"></div>

        {% if summary %}
        <div class="summary-title">Summary:</div>
        <textarea readonly>{{ summary }}</textarea>
        {% endif %}
    </div>
</body>
</html>
"""

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        video_file = request.files["video"]
        if video_file:
            os.makedirs("static/uploads", exist_ok=True)
            video_path = os.path.join("static/uploads", video_file.filename)
            video_file.save(video_path)

            caption_chunks = process_video(video_path)
            return render_template("index.html", video_url=video_path, chunks=caption_chunks)
    return render_template("index.html", video_url=None, chunks=[])



if __name__ == '__main__':
    app.run(debug=True)
