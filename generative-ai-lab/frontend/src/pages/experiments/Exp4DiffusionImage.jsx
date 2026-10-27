import React, { useState, useEffect } from 'react';
import { ArrowLeft, Sparkles, Download, RefreshCw, Trash2, Image as ImageIcon, AlertCircle } from 'lucide-react';

const PROMPT_CHIPS = [
  "A photorealistic portrait of an astronaut exploring a bioluminescent crystal cavern",
  "A vintage mechanical pocket watch on antique dark oak desk, cinematic chiaroscuro lighting",
  "Japanese zen garden in autumn with crimson maple leaves, morning mist and stone lantern",
  "Futuristic cyberpunk city street at dusk with neon signage reflecting on rain-slicked asphalt"
];

export function Exp4DiffusionImage({ onBack, onOpenSettings }) {
  const [prompt, setPrompt] = useState(PROMPT_CHIPS[0]);
  const [negativePrompt, setNegativePrompt] = useState('blurry, low quality, distorted, watermark, signature');
  const [aspectRatio, setAspectRatio] = useState('1:1');
  const [modelId, setModelId] = useState('stabilityai/stable-diffusion-xl-base-1.0');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [status, setStatus] = useState(null);

  useEffect(() => {
    fetch('/api/exp4/status')
      .then((res) => res.json())
      .then((d) => setStatus(d))
      .catch((e) => console.error(e));
  }, []);

  const handleGenerate = async () => {
    if (!prompt.trim()) {
      setError('Please provide a descriptive text prompt.');
      return;
    }
    setLoading(true);
    setError(null);

    try {
      const res = await fetch('/api/exp4/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: prompt.trim(),
          negative_prompt: negativePrompt.trim(),
          aspect_ratio: aspectRatio,
          model_id: modelId
        })
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'Diffusion generation failed');
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
    setNegativePrompt('');
    setResult(null);
    setError(null);
  };

  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      
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
            Experiment 4
          </span>
          <h1 className="text-2xl font-bold tracking-tight text-zinc-950 mt-1">
            Text-to-Image Generation using a Pre-trained Diffusion Model
          </h1>
          <p className="text-sm text-zinc-500 mt-1">
            Synthesize high-fidelity photorealistic imagery conditioned on text prompts using Latent Diffusion and SDXL foundation models.
          </p>
        </div>
      </div>

      {/* Configuration Strip */}
      <div className="p-4 rounded-xl border border-zinc-200 bg-zinc-50 flex items-center justify-between gap-4">
        <div className="flex items-center gap-3 w-full sm:w-auto">
          <label className="text-xs font-semibold text-zinc-700">Diffusion Model:</label>
          <select
            value={modelId}
            onChange={(e) => setModelId(e.target.value)}
            className="text-xs border border-zinc-300 rounded-lg px-3 py-1.5 bg-white text-zinc-900 focus:outline-none focus:ring-1 focus:ring-zinc-900"
          >
            <option value="stabilityai/stable-diffusion-xl-base-1.0">Stable Diffusion XL (Base 1.0)</option>
            <option value="runwayml/stable-diffusion-v1-5">Stable Diffusion v1.5</option>
            <option value="black-forest-labs/FLUX.1-schnell">FLUX.1 Schnell</option>
          </select>
        </div>
      </div>

      {/* Prompt Form */}
      <div className="border border-zinc-200 rounded-xl p-6 bg-white space-y-5">
        
        {/* Presets */}
        <div>
          <span className="text-xs font-semibold text-zinc-500 uppercase tracking-wider">Example Prompts:</span>
          <div className="flex flex-wrap gap-2 mt-2">
            {PROMPT_CHIPS.map((chip, i) => (
              <button
                key={i}
                type="button"
                onClick={() => setPrompt(chip)}
                className="text-left text-xs bg-zinc-50 border border-zinc-200 hover:border-zinc-400 hover:bg-zinc-100 px-3 py-1.5 rounded-lg text-zinc-700 transition-colors"
              >
                {chip.slice(0, 50)}...
              </button>
            ))}
          </div>
        </div>

        {/* Large Prompt Box */}
        <div>
          <label className="block text-xs font-semibold text-zinc-800 uppercase tracking-wider mb-1.5">
            Generation Prompt
          </label>
          <textarea
            rows={3}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Describe the visual scene in detail..."
            className="w-full text-sm p-3 border border-zinc-300 rounded-lg focus:outline-none focus:ring-1 focus:ring-zinc-900 leading-relaxed text-zinc-900"
          />
        </div>

        {/* Negative Prompt & Aspect Ratio Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="md:col-span-2">
            <label className="block text-xs font-semibold text-zinc-800 uppercase tracking-wider mb-1.5">
              Negative Prompt (Avoid features)
            </label>
            <input
              type="text"
              value={negativePrompt}
              onChange={(e) => setNegativePrompt(e.target.value)}
              placeholder="e.g. blurry, low quality, distorted"
              className="w-full text-xs p-2.5 border border-zinc-300 rounded-lg focus:outline-none focus:ring-1 focus:ring-zinc-900 text-zinc-900"
            />
          </div>

          <div>
            <label className="block text-xs font-semibold text-zinc-800 uppercase tracking-wider mb-1.5">
              Aspect Ratio
            </label>
            <select
              value={aspectRatio}
              onChange={(e) => setAspectRatio(e.target.value)}
              className="w-full text-xs p-2.5 border border-zinc-300 rounded-lg bg-white text-zinc-900 focus:outline-none focus:ring-1 focus:ring-zinc-900"
            >
              <option value="1:1">1:1 Square (512x512)</option>
              <option value="16:9">16:9 Widescreen (768x432)</option>
              <option value="9:16">9:16 Portrait (432x768)</option>
              <option value="4:3">4:3 Standard (640x480)</option>
            </select>
          </div>
        </div>

        {/* Buttons */}
        <div className="flex items-center gap-3 pt-2">
          <button
            onClick={handleGenerate}
            disabled={loading}
            className="flex-1 py-3 rounded-lg bg-zinc-950 hover:bg-zinc-800 text-white font-semibold text-xs transition-colors flex items-center justify-center gap-2 disabled:opacity-50"
          >
            {loading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Sparkles className="w-4 h-4" />}
            {loading ? 'Diffusing Latent Vectors...' : 'Generate Image'}
          </button>

          <button
            onClick={handleClear}
            className="px-4 py-3 rounded-lg border border-zinc-200 hover:bg-zinc-100 text-zinc-700 text-xs font-semibold transition-colors flex items-center gap-1.5"
          >
            <Trash2 className="w-4 h-4" />
            Clear
          </button>
        </div>

      </div>

      {/* Error state */}
      {error && (
        <div className="p-4 rounded-xl bg-amber-50 border border-amber-200 text-amber-900 text-xs flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-amber-700 mt-0.5 shrink-0" />
          <div className="space-y-1">
            <div className="font-semibold">Diffusion Inference Notice</div>
            <div>{error}</div>
            <button
              onClick={onOpenSettings}
              className="font-semibold underline text-amber-950 block mt-1"
            >
              Enter your Hugging Face API Token in Settings to enable high-resolution inference
            </button>
          </div>
        </div>
      )}

      {/* Generated Image Result */}
      {result && (
        <div className="border border-zinc-200 rounded-xl p-6 bg-zinc-50/50 space-y-4 animate-in fade-in duration-200">
          <div className="flex items-center justify-between border-b border-zinc-200 pb-3">
            <div className="flex items-center gap-2 text-zinc-950 font-semibold text-sm">
              <ImageIcon className="w-4 h-4 text-zinc-800" />
              Generated Diffusion Output
            </div>
            <a
              href={result.image_url}
              download={result.filename}
              className="px-3 py-1.5 rounded-lg bg-zinc-950 text-white hover:bg-zinc-800 text-xs font-semibold flex items-center gap-1.5 transition-colors"
            >
              <Download className="w-3.5 h-3.5" />
              Download Image
            </a>
          </div>

          <div className="flex flex-col items-center justify-center p-4 bg-white rounded-lg border border-zinc-200">
            <img
              src={result.image_url}
              alt={result.prompt}
              className="max-h-[500px] w-auto rounded-lg shadow-sm object-contain"
            />
            <div className="mt-4 text-center max-w-xl">
              <p className="text-xs text-zinc-800 font-medium italic">"{result.prompt}"</p>
            </div>
          </div>
        </div>
      )}

    </div>
  );
}
