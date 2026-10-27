import React, { useState, useEffect, useRef } from 'react';
import { ArrowLeft, Sparkles, Download, RefreshCw, Trash2, Video as VideoIcon, Film, Maximize2, X } from 'lucide-react';

const VIDEO_PRESETS = [
  "College students walking on a sunny university campus.",
  "Futuristic cyberpunk city street with neon lights reflecting on wet asphalt.",
  "A cinematic drone shot flying over green mountains at sunrise.",
  "A majestic waterfall cascading into a tropical turquoise lagoon.",
  "An astronaut exploring a crystalline alien cavern in deep space.",
  "A serene Japanese zen garden with crimson maple foliage."
];

export function Exp10VideoGeneration({ onBack }) {
  const [prompt, setPrompt] = useState(VIDEO_PRESETS[0]);
  const [numFrames, setNumFrames] = useState(24);
  const [fps, setFps] = useState(8);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [status, setStatus] = useState(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const videoRef = useRef(null);
  const fullscreenVideoRef = useRef(null);

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') {
        setIsFullscreen(false);
      }
    };
    if (isFullscreen) {
      window.addEventListener('keydown', handleKeyDown);
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      document.body.style.overflow = '';
    };
  }, [isFullscreen]);

  const handleNativeFullscreen = () => {
    const target = fullscreenVideoRef.current || videoRef.current;
    if (target) {
      if (target.requestFullscreen) {
        target.requestFullscreen();
      } else if (target.webkitRequestFullscreen) {
        target.webkitRequestFullscreen();
      }
    }
  };

  useEffect(() => {
    fetch('/api/exp10/status')
      .then((res) => res.json())
      .then((d) => setStatus(d))
      .catch((e) => console.error(e));
  }, []);

  const handleGenerate = async () => {
    if (!prompt.trim()) {
      setError('Please provide a descriptive video prompt.');
      return;
    }
    setLoading(true);
    setError(null);

    try {
      const res = await fetch('/api/exp10/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: prompt.trim(),
          num_frames: numFrames,
          fps: fps,
          model_id: 'cerspense/zeroscope_v2_576w'
        })
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'Video generation failed');
      }
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setPrompt('');
    setResult(null);
    setError(null);
  };

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      
      {/* Navigation & Header */}
      <div>
        <button
          onClick={onBack}
          className="inline-flex items-center gap-2 text-xs font-semibold text-zinc-600 hover:text-zinc-950 mb-4 transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Lab Overview
        </button>

        <div className="border-b border-zinc-200 pb-5">
          <span className="text-xs font-mono text-zinc-500 uppercase tracking-wider font-semibold">
            Experiment 10
          </span>
          <h1 className="text-2xl font-bold tracking-tight text-zinc-950 mt-1">
            Generation of Videos from Text Prompts using a Pretrained Open-Source Video Generation Foundation Model
          </h1>
          <p className="text-sm text-zinc-500 mt-1">
            Synthesize continuous temporal frames and export genuine H.264 MP4 video sequences conditioned on natural language prompts.
          </p>
        </div>
      </div>

      {/* Main Interactive Studio Form */}
      <div className="border border-zinc-200 rounded-xl p-6 bg-white space-y-6">
        
        {/* Preset Prompts Chips */}
        <div>
          <span className="text-xs font-semibold text-zinc-500 uppercase tracking-wider">Example Prompts:</span>
          <div className="flex flex-wrap gap-2 mt-2">
            {VIDEO_PRESETS.map((p, i) => (
              <button
                key={i}
                type="button"
                onClick={() => setPrompt(p)}
                className="text-left text-xs bg-zinc-50 border border-zinc-200 hover:border-zinc-400 hover:bg-zinc-100 px-3 py-1.5 rounded-lg text-zinc-700 transition-colors"
              >
                {p}
              </button>
            ))}
          </div>
        </div>

        {/* Large Prompt Input Box */}
        <div>
          <label className="block text-xs font-semibold text-zinc-800 uppercase tracking-wider mb-1.5">
            Video Scene Prompt
          </label>
          <textarea
            rows={3}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="A cinematic drone shot flying over green mountains at sunrise."
            className="w-full text-sm p-3 border border-zinc-300 rounded-lg focus:outline-none focus:ring-1 focus:ring-zinc-900 leading-relaxed text-zinc-900"
          />
        </div>

        {/* Frame & FPS Parameters */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div>
            <label className="block text-xs font-semibold text-zinc-700 uppercase tracking-wider mb-1.5">
              Temporal Frame Count
            </label>
            <div className="grid grid-cols-3 gap-2">
              {[16, 24, 32].map((f) => (
                <button
                  key={f}
                  type="button"
                  onClick={() => setNumFrames(f)}
                  className={`py-2 text-xs font-mono font-medium rounded-lg border transition-colors ${
                    numFrames === f
                      ? 'bg-zinc-950 text-white border-zinc-950'
                      : 'bg-white text-zinc-700 border-zinc-200 hover:bg-zinc-50'
                  }`}
                >
                  {f} Frames
                </button>
              ))}
            </div>
          </div>

          <div>
            <label className="block text-xs font-semibold text-zinc-700 uppercase tracking-wider mb-1.5">
              Playback Frame Rate
            </label>
            <div className="grid grid-cols-3 gap-2">
              {[8, 12, 16].map((rate) => (
                <button
                  key={rate}
                  type="button"
                  onClick={() => setFps(rate)}
                  className={`py-2 text-xs font-mono font-medium rounded-lg border transition-colors ${
                    fps === rate
                      ? 'bg-zinc-950 text-white border-zinc-950'
                      : 'bg-white text-zinc-700 border-zinc-200 hover:bg-zinc-50'
                  }`}
                >
                  {rate} FPS
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center gap-3 pt-2">
          <button
            onClick={handleGenerate}
            disabled={loading}
            className="flex-1 py-3 rounded-lg bg-zinc-950 hover:bg-zinc-800 text-white font-semibold text-xs transition-colors flex items-center justify-center gap-2 disabled:opacity-50"
          >
            {loading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Film className="w-4 h-4" />}
            {loading ? 'Synthesizing Spatio-Temporal Video...' : 'Generate Video'}
          </button>

          <button
            onClick={handleClear}
            className="px-4 py-3 rounded-lg border border-zinc-200 hover:bg-zinc-100 text-zinc-700 text-xs font-semibold transition-colors flex items-center gap-1.5"
          >
            <Trash2 className="w-4 h-4" />
            Clear Prompt
          </button>
        </div>

      </div>

      {error && (
        <div className="p-4 rounded-xl bg-red-50 border border-red-200 text-red-800 text-xs">
          {error}
        </div>
      )}

      {/* Generated Video Output Section */}
      {result && (
        <div className="border border-zinc-200 rounded-xl p-6 bg-zinc-50/50 space-y-4 animate-in fade-in duration-200">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-zinc-200 pb-3">
            <div className="flex items-center gap-2 text-zinc-950 font-semibold text-sm">
              <VideoIcon className="w-4 h-4 text-emerald-600" />
              <span>Generated Video Preview ({result.duration_seconds}s • {result.dimensions})</span>
            </div>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => setIsFullscreen(true)}
                className="px-3 py-1.5 rounded-lg border border-zinc-200 bg-white hover:bg-zinc-100 text-zinc-800 text-xs font-semibold flex items-center gap-1.5 transition-colors shadow-sm"
                title="Open Fullscreen Preview"
              >
                <Maximize2 className="w-3.5 h-3.5 text-zinc-700" />
                Fullscreen Preview
              </button>
              <a
                href={result.video_url}
                download={result.filename}
                className="px-3 py-1.5 rounded-lg bg-zinc-950 text-white hover:bg-zinc-800 text-xs font-semibold flex items-center gap-1.5 transition-colors shadow-sm"
              >
                <Download className="w-3.5 h-3.5" />
                Download MP4
              </a>
            </div>
          </div>

          <div className="p-4 bg-white rounded-lg border border-zinc-200 flex flex-col items-center justify-center space-y-4">
            <div className="w-full max-w-2xl aspect-[16/9] bg-zinc-950 rounded-xl overflow-hidden relative group flex items-center justify-center shadow-md">
              <video
                key={result.video_url}
                ref={videoRef}
                controls
                autoPlay
                loop
                src={result.video_url}
                className="w-full h-full object-contain"
              />
              <button
                type="button"
                onClick={() => setIsFullscreen(true)}
                className="absolute top-3 right-3 opacity-0 group-hover:opacity-100 transition-opacity px-2.5 py-1.5 bg-black/80 hover:bg-black text-white text-xs font-medium rounded-lg backdrop-blur-sm flex items-center gap-1.5 z-10 border border-white/20 shadow-lg"
                title="Open Fullscreen Preview"
              >
                <Maximize2 className="w-3.5 h-3.5" />
                Fullscreen
              </button>
            </div>

            <div className="text-center space-y-1">
              <p className="text-xs text-zinc-800 font-medium italic">"{result.prompt}"</p>
              <p className="text-[11px] text-zinc-500 font-mono">
                {result.model} • {result.num_frames} frames @ {result.fps} FPS ({result.dimensions} H.264 MP4)
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Fullscreen Video Preview Modal */}
      {isFullscreen && result && (
        <div className="fixed inset-0 z-50 bg-black/95 backdrop-blur-md flex flex-col justify-between p-4 sm:p-6 animate-in fade-in duration-200">
          {/* Header */}
          <div className="flex items-center justify-between border-b border-zinc-800 pb-3">
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2 text-white font-semibold text-sm">
                <VideoIcon className="w-4 h-4 text-emerald-400" />
                <span>Fullscreen Video Preview</span>
              </div>
              <span className="hidden sm:inline-block text-xs font-mono px-2 py-0.5 rounded bg-zinc-800 text-zinc-300">
                {result.duration_seconds}s • {result.dimensions} • {result.fps} FPS
              </span>
            </div>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={handleNativeFullscreen}
                className="px-3 py-1.5 rounded-lg border border-zinc-700 hover:bg-zinc-800 text-zinc-200 text-xs font-medium flex items-center gap-1.5 transition-colors"
                title="Enter Native Device Fullscreen"
              >
                <Maximize2 className="w-3.5 h-3.5" />
                <span className="hidden sm:inline">Native Fullscreen</span>
              </button>
              <a
                href={result.video_url}
                download={result.filename}
                className="px-3 py-1.5 rounded-lg bg-zinc-100 text-zinc-900 hover:bg-white text-xs font-semibold flex items-center gap-1.5 transition-colors"
              >
                <Download className="w-3.5 h-3.5" />
                Download MP4
              </a>
              <button
                type="button"
                onClick={() => setIsFullscreen(false)}
                className="p-1.5 rounded-lg text-zinc-400 hover:text-white hover:bg-zinc-800 transition-colors ml-2"
                title="Close fullscreen (Esc)"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
          </div>

          {/* Centered Large Video */}
          <div className="flex-1 flex items-center justify-center py-4 px-2 overflow-hidden">
            <div className="w-full max-w-5xl max-h-[76vh] aspect-video bg-zinc-950 rounded-2xl overflow-hidden flex items-center justify-center shadow-2xl border border-zinc-800 relative">
              <video
                key={`fs-${result.video_url}`}
                ref={fullscreenVideoRef}
                controls
                autoPlay
                loop
                src={result.video_url}
                className="w-full h-full object-contain"
              />
            </div>
          </div>

          {/* Footer */}
          <div className="border-t border-zinc-800 pt-3 flex flex-col sm:flex-row items-center justify-between text-xs text-zinc-400 gap-2">
            <p className="italic text-zinc-200 font-medium truncate max-w-2xl text-center sm:text-left">
              "{result.prompt}"
            </p>
            <div className="flex items-center gap-4 text-[11px] font-mono text-zinc-500">
              <span>Press <kbd className="px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-300 font-sans">Esc</kbd> or click close to exit</span>
            </div>
          </div>
        </div>
      )}

    </div>
  );
}
