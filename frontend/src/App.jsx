import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { 
  Video, 
  Send, 
  Settings, 
  Play, 
  CheckCircle, 
  AlertCircle, 
  Loader2,
  MessageSquare,
  Layout,
  Clock,
  Upload,
  RefreshCcw,
  FileVideo,
  X,
  FileText
} from 'lucide-react';

const API_BASE = 'http://localhost:8000/api';
const STATIC_BASE = 'http://localhost:8000';

function App() {
  const [videoPath, setVideoPath] = useState('');
  const [videoUrl, setVideoUrl] = useState('');
  const [videoName, setVideoName] = useState('');
  const [jobId, setJobId] = useState(null);
  const [status, setStatus] = useState(null);
  const [chatQuery, setChatQuery] = useState('');
  const [messages, setMessages] = useState([]);
  const [isChatLoading, setIsChatLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [ytUrl, setYtUrl] = useState('');
  const [isYtLoading, setIsYtLoading] = useState(false);
  
  const statusInterval = useRef(null);
  const chatEndRef = useRef(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    if (jobId && (!status || (status.status !== 'completed' && status.status !== 'failed'))) {
      statusInterval.current = setInterval(async () => {
        try {
          const resp = await axios.get(`${API_BASE}/status/${jobId}`);
          setStatus(resp.data);
          if (resp.data.status === 'completed' || resp.data.status === 'failed') {
            clearInterval(statusInterval.current);
          }
        } catch (err) {
          console.error("Polling error", err);
        }
      }, 2000);
    }
    return () => clearInterval(statusInterval.current);
  }, [jobId, status]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleReset = () => {
    setVideoPath('');
    setVideoUrl('');
    setVideoName('');
    setJobId(null);
    setStatus(null);
    setMessages([]);
    setChatQuery('');
    if (statusInterval.current) clearInterval(statusInterval.current);
  };

  const handleFileUpload = async (file) => {
    if (!file) return;
    const formData = new FormData();
    formData.append('file', file);
    
    setIsUploading(true);
    try {
      const resp = await axios.post(`${API_BASE}/upload`, formData);
      setVideoPath(resp.data.path);
      setVideoName(resp.data.filename);
      setVideoUrl(`${STATIC_BASE}/${resp.data.url}`);
    } catch (err) {
      alert("Upload failed: " + err.message);
    } finally {
      setIsUploading(false);
    }
  };

  const onDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('video/')) {
      handleFileUpload(file);
    } else {
      alert("Please drop a valid video file.");
    }
  };

  const startAnalysis = async () => {
    if (!videoPath) return;
    try {
      setJobId(null);
      setStatus(null);
      const resp = await axios.post(`${API_BASE}/analyze`, {
        video_path: videoPath
      });
      setJobId(resp.data.job_id);
    } catch (err) {
      alert("Failed to start analysis: " + err.message);
    }
  };

  const handleYoutubeSubmit = async (e) => {
    e.preventDefault();
    if (!ytUrl.trim()) return;
    
    setIsYtLoading(true);
    try {
      const resp = await axios.post(`${API_BASE}/youtube`, {
        url: ytUrl
      });
      setJobId(resp.data.job_id);
      setYtUrl('');
      // We don't have the video path yet, the backend will update it.
      setVideoName("YouTube Video");
      setStatus({ status: 'processing', status_msg: 'Queuing YouTube Download...', progress: 0 });
    } catch (err) {
      alert("Failed to start YouTube analysis: " + err.message);
    } finally {
      setIsYtLoading(false);
    }
  };

  const handleChat = async (e) => {
    e.preventDefault();
    if (!chatQuery.trim() || isChatLoading) return;

    const userMsg = { role: 'user', content: chatQuery };
    setMessages(prev => [...prev, userMsg]);
    setChatQuery('');
    setIsChatLoading(true);

    try {
      const resp = await axios.post(`${API_BASE}/chat`, {
        query: chatQuery,
        video_name: videoName || 'current'
      });
      setMessages(prev => [...prev, { role: 'assistant', content: resp.data.answer }]);
    } catch (err) {
      setMessages(prev => [...prev, { role: 'assistant', content: "Error: " + err.message }]);
    } finally {
      setIsChatLoading(false);
    }
  };

  return (
    <div className="container fade-in">
      <header>
        <h1 className="logo">Video Transcriber</h1>
        <p className="subtitle">High-Quality Video Analysis & RAG Chat</p>
      </header>

      <div className="grid">
        {/* Left Column: Video & Summary */}
        <section className="column-left">
          <div className="glass card">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Video size={20} className="text-primary" />
                <h3 style={{ margin: 0 }}>Video Player</h3>
              </div>
              {(videoPath || status) && (
                <button className="btn btn-danger" onClick={handleReset} style={{ padding: '0.4rem 0.8rem', fontSize: '0.75rem' }}>
                  <RefreshCcw size={12} />
                  Reset
                </button>
              )}
            </div>

            {videoUrl ? (
              <div className="video-player fade-in">
                <video src={videoUrl} controls style={{ width: '100%', borderRadius: '0.75rem', border: '1px solid var(--glass-border)', boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.3)' }} />
              </div>
            ) : (
              <>
                <div 
                className={`drop-zone ${isDragging ? 'active' : ''}`}
                onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={onDrop}
                onClick={() => fileInputRef.current.click()}
                style={{ height: '300px' }}
              >
                <input 
                  type="file" 
                  ref={fileInputRef} 
                  style={{ display: 'none' }} 
                  onChange={(e) => handleFileUpload(e.target.files[0])}
                  accept="video/*"
                />
                {isUploading ? <Loader2 size={40} className="spinner drop-zone-icon" /> : <Upload size={40} className="drop-zone-icon" />}
                <p style={{ fontWeight: 600, marginTop: '0.5rem' }}>Drag Video or Click to Browse</p>
              </div>
              
              <div style={{ marginTop: '1.5rem', borderTop: '1px solid var(--glass-border)', paddingTop: '1.5rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
                  <Play size={18} className="text-primary" />
                  <h4 style={{ margin: 0 }}>... or Paste YouTube Link</h4>
                </div>
                <form onSubmit={handleYoutubeSubmit} style={{ display: 'flex', gap: '0.5rem' }}>
                  <input 
                    type="text" 
                    placeholder="https://www.youtube.com/watch?v=..." 
                    value={ytUrl}
                    onChange={(e) => setYtUrl(e.target.value)}
                    className="glass-input"
                    style={{ flex: 1, padding: '0.75rem', borderRadius: '0.5rem', border: '1px solid var(--glass-border)', background: 'rgba(255,255,255,0.05)', color: 'white' }}
                  />
                  <button type="submit" className="btn btn-primary" disabled={isYtLoading || !ytUrl.trim()}>
                    {isYtLoading ? <Loader2 size={18} className="spinner" /> : <Send size={18} />}
                    Process
                  </button>
                </form>
              </div>
            </>
            )}

            {videoPath && !status && !isUploading && (
              <div style={{ marginTop: '1.5rem' }}>
                <button className="btn btn-primary" style={{ width: '100%', padding: '1rem' }} onClick={startAnalysis}>
                  <Play size={18} />
                  Start AI Analysis
                </button>
              </div>
            )}

            {status && (
              <div className="analysis-status fade-in" style={{ marginTop: '1.5rem', padding: '1rem', background: 'rgba(255,255,255,0.02)', borderRadius: '0.75rem', border: '1px solid var(--glass-border)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                  <span className="status-text" style={{ fontWeight: 600, color: 'var(--primary)' }}>{status.status_msg}</span>
                  <span className="status-text">{Math.round(status.progress)}%</span>
                </div>
                <div className="progress-container">
                  <div className="progress-bar" style={{ width: `${status.progress}%` }}></div>
                </div>
              </div>
            )}
          </div>

          <div className="glass card fade-in" style={{ marginTop: '1.5rem', flex: 1 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.2rem' }}>
              <FileText size={20} className="text-primary" />
              <h3 style={{ margin: 0 }}>Video Summary</h3>
            </div>
            
            <div className="summary-content markdown-body">
              {status?.summary ? (
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {status.summary}
                </ReactMarkdown>
              ) : status?.status === 'processing' ? (
                <div style={{ textAlign: 'center', padding: '3rem', color: 'var(--text-dim)' }}>
                  <Loader2 size={32} className="spinner" style={{ marginBottom: '1rem', opacity: 0.5 }} />
                  <p>Analyzing video content and generating summary...</p>
                </div>
              ) : (
                <div style={{ textAlign: 'center', padding: '3rem', color: 'var(--text-dim)' }}>
                  <Clock size={32} style={{ marginBottom: '1rem', opacity: 0.3 }} />
                  <p>Summary will appear here after analysis.</p>
                </div>
              )}
            </div>
          </div>
        </section>

        {/* Right Column: Chat Interface */}
        <section className="column-right">
          <div className="glass card chat-card" style={{ display: 'flex', flexDirection: 'column' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.5rem' }}>
              <MessageSquare size={20} className="text-primary" />
              <h3 style={{ margin: 0 }}>AI Assistant</h3>
            </div>

            <div className="chat-container">
              <div className="messages" style={{ flex: 1 }}>
                {messages.length === 0 && (
                  <div style={{ textAlign: 'center', color: 'var(--text-dim)', marginTop: '30%', padding: '2rem' }}>
                    <Layout size={48} style={{ opacity: 0.05, marginBottom: '1rem' }} />
                    <p>Analysis complete! You can now ask anything about what happened in the video.</p>
                  </div>
                )}
                {messages.map((m, i) => (
                  <div key={i} className={`message ${m.role} fade-in`}>
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content}</ReactMarkdown>
                  </div>
                ))}
                {isChatLoading && (
                  <div className="message assistant" style={{ background: 'transparent' }}>
                    <Loader2 size={24} className="spinner text-dim" />
                  </div>
                )}
                <div ref={chatEndRef} />
              </div>
              <form className="chat-input" onSubmit={handleChat}>
                <input 
                  type="text" 
                  placeholder="Ask a question..." 
                  value={chatQuery}
                  onChange={(e) => setChatQuery(e.target.value)}
                  disabled={status?.status !== 'completed' && messages.length === 0}
                />
                <button type="submit" className="btn btn-primary" disabled={isChatLoading || (status?.status !== 'completed' && messages.length === 0)}>
                  <Send size={18} />
                </button>
              </form>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}

export default App;
