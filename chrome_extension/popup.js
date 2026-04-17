document.addEventListener('DOMContentLoaded', async () => {
  const transcribeBtn = document.getElementById('transcribe-btn');
  const videoUrlEl = document.getElementById('video-url');
  const statusMsg = document.getElementById('status-msg');
  const btnText = document.getElementById('btn-text');

  // Get current active tab
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

  if (tab && tab.url && (tab.url.includes('youtube.com/watch') || tab.url.includes('youtu.be/'))) {
    videoUrlEl.textContent = tab.url;
    transcribeBtn.disabled = false;
    statusMsg.textContent = "YouTube video detected!";
  } else {
    videoUrlEl.textContent = "No supported video detected.";
    transcribeBtn.disabled = true;
    statusMsg.textContent = "Navigate to a YouTube video to transcribe it.";
  }

  transcribeBtn.addEventListener('click', async () => {
    transcribeBtn.classList.add('loading');
    transcribeBtn.disabled = true;
    statusMsg.textContent = "Sending to Video Transcriber...";

    try {
      const response = await fetch('http://localhost:8000/api/youtube', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ url: tab.url }),
      });

      if (response.ok) {
        const data = await response.json();
        statusMsg.style.color = "#10b981"; // Green
        statusMsg.textContent = "Job started! Check the web app to track progress.";
        chrome.tabs.create({ url: 'http://localhost:3000' }); // Assuming frontend is here
      } else {
        const error = await response.text();
        throw new Error(error);
      }
    } catch (err) {
      statusMsg.style.color = "#ef4444"; // Red
      statusMsg.textContent = "Error: Make sure your local server is running.";
      console.error(err);
    } finally {
      transcribeBtn.classList.remove('loading');
    }
  });
});
