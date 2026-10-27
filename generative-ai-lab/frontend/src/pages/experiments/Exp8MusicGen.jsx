import React, { useState, useEffect, useRef } from 'react';
import { ArrowLeft, Sparkles, Download, RefreshCw, Trash2, Music, Volume2 } from 'lucide-react';

const MUSIC_PROMPTS = [
  "Peaceful instrumental music with piano and soft strings.",
  "Upbeat electronic dance groove with energetic synth chords.",
  "Cinematic ambient soundtrack with atmospheric pads and subtle piano.",
  "Calm acoustic guitar arpeggios in a warm reverberant room."
];

export function Exp8MusicGen({ onBack }) {
  const [prompt, setPrompt] = useState(MUSIC_PROMPTS[0]);
  const [duration, setDuration] = useState(5);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const audioRef = useRef(null);

  const handleGenerate = async () => {
    if (!prompt.trim()) {
      setError('Please provide a descriptive music prompt.');
      return;
    }
    setLoading(true);
    setError(null);

    try {
      const res = await fetch('/api/exp8/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: prompt.trim(),
          duration_seconds: duration,
          model_id: 'facebook/musicgen-small'
        })
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'Music generation failed');
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
      
      {/* Header */}
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
            Experiment 8
          </span>
          <h1 className="text-2xl font-bold tracking-tight text-zinc-950 mt-1">
            Generation of Novel Music Compositions using a Pretrained Open-Source Foundation Model (MusicGen)
          </h1>
          <p className="text-sm text-zinc-500 mt-1">
            Condition foundation autoregressive audio models on natural language to synthesize novel instrumental compositions.
          </p>
        </div>
      </div>

      {/* Main Interactive Form */}
      <div className="border border-zinc-200 rounded-xl p-6 bg-white space-y-6">
        
        {/* Presets */}
        <div>
          <span className="text-xs font-semibold text-zinc-500 uppercase tracking-wider">Example Audio Prompts:</span>
          <div className="flex flex-wrap gap-2 mt-2">
            {MUSIC_PROMPTS.map((p, i) => (
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

        {/* Prompt Input */}
        <div>
          <label className="block text-xs font-semibold text-zinc-800 uppercase tracking-wider mb-1.5">
            Music Description Prompt
          </label>
          <textarea
            rows={3}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="e.g. Peaceful instrumental music with piano and soft strings."
            className="w-full text-sm p-3 border border-zinc-300 rounded-lg focus:outline-none focus:ring-1 focus:ring-zinc-900 leading-relaxed text-zinc-900"
          />
        </div>

        {/* Duration Selector */}
        <div>
          <label className="block text-xs font-semibold text-zinc-800 uppercase tracking-wider mb-2">
            Composition Duration
          </label>
          <div className="grid grid-cols-3 gap-3">
            {[5, 10, 15].map((sec) => (
              <button
                key={sec}
                type="button"
                onClick={() => setDuration(sec)}
                className={`py-2 text-xs font-semibold rounded-lg border transition-colors ${
                  duration === sec
                    ? 'bg-zinc-950 text-white border-zinc-950'
                    : 'bg-white text-zinc-700 border-zinc-200 hover:bg-zinc-50'
                }`}
              >
                {sec} Seconds
              </button>
            ))}
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center gap-3 pt-2">
          <button
            onClick={handleGenerate}
            disabled={loading}
            className="flex-1 py-3 rounded-lg bg-zinc-950 hover:bg-zinc-800 text-white font-semibold text-xs transition-colors flex items-center justify-center gap-2 disabled:opacity-50"
          >
            {loading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Music className="w-4 h-4" />}
            {loading ? 'Synthesizing Polyphonic Composition...' : 'Generate Music'}
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

      {/* Generated Audio Card */}
      {result && (
        <div className="border border-zinc-200 rounded-xl p-6 bg-zinc-50/50 space-y-5 animate-in fade-in duration-200">
          <div className="flex items-center justify-between border-b border-zinc-200 pb-3">
            <div className="flex items-center gap-2 text-zinc-950 font-semibold text-sm">
              <Volume2 className="w-4 h-4 text-emerald-600" />
              Generated Audio Track ({result.duration}s)
            </div>
            <a
              href={result.audio_url}
              download={result.filename}
              className="px-3 py-1.5 rounded-lg bg-zinc-950 text-white hover:bg-zinc-800 text-xs font-semibold flex items-center gap-1.5 transition-colors"
            >
              <Download className="w-3.5 h-3.5" />
              Download WAV File
            </a>
          </div>

          <div className="p-6 bg-white rounded-lg border border-zinc-200 flex flex-col items-center justify-center space-y-4">
            {/* Visual audio waveform simulation */}
            <div className="w-full flex items-center justify-center gap-1 h-14 px-4 py-2 bg-zinc-50 rounded-lg border border-zinc-200">
              {[40, 75, 55, 90, 60, 30, 85, 100, 65, 45, 95, 70, 50, 80, 60, 90, 40, 85, 70, 55, 90, 65, 45, 80, 60, 100, 75, 40, 90, 60].map((h, i) => (
                <div
                  key={i}
                  style={{ height: `${h}%` }}
                  className="w-1.5 bg-zinc-900 rounded-full transition-all duration-300"
                />
              ))}
            </div>

            {/* Standard HTML5 Audio Player */}
            <audio
              ref={audioRef}
              controls
              src={result.audio_url}
              className="w-full max-w-md accent-zinc-900"
            />

            <div className="text-center space-y-1">
              <p className="text-xs text-zinc-800 font-medium italic">"{result.prompt}"</p>
              <p className="text-[11px] text-zinc-500 font-mono">
                {result.provider} • 44.1kHz 16-Bit Stereo WAV
              </p>
            </div>
          </div>
        </div>
      )}

    </div>
  );
}
