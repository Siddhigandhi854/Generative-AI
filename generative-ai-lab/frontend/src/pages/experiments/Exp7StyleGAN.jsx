import React, { useState, useEffect } from 'react';
import { ArrowLeft, Sparkles, Download, RefreshCw, Sliders, Dice5 } from 'lucide-react';

export function Exp7StyleGAN({ onBack }) {
  const [numSamples, setNumSamples] = useState(4);
  const [seed, setSeed] = useState(777);
  const [truncationPsi, setTruncationPsi] = useState(0.7);
  const [samples, setSamples] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [status, setStatus] = useState(null);

  useEffect(() => {
    fetch('/api/exp7/status')
      .then((res) => res.json())
      .then((d) => setStatus(d))
      .catch((e) => console.error(e));

    handleGenerate(4, 777, 0.7);
  }, []);

  const handleGenerate = async (customCount = null, customSeed = null, customPsi = null) => {
    setLoading(true);
    setError(null);

    const count = customCount !== null ? customCount : numSamples;
    const currentSeed = customSeed !== null ? customSeed : seed;
    const currentPsi = customPsi !== null ? customPsi : truncationPsi;

    try {
      const res = await fetch('/api/exp7/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          num_samples: count,
          seed: currentSeed !== '' ? Number(currentSeed) : null,
          truncation_psi: parseFloat(currentPsi)
        })
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'StyleGAN synthesis failed');
      }

      const data = await res.json();
      setSamples(data.samples || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleRandomize = () => {
    const nextSeed = Math.floor(Math.random() * 500000);
    setSeed(nextSeed);
    handleGenerate(numSamples, nextSeed, truncationPsi);
  };

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      
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
            Experiment 7
          </span>
          <h1 className="text-2xl font-bold tracking-tight text-zinc-950 mt-1">
            Implementation of StyleGANs for Generating High-Quality Images
          </h1>
          <p className="text-sm text-zinc-500 mt-1">
            High-fidelity face synthesis utilizing StyleGAN's disentangled latent space (W-space) and Adaptive Instance Normalization (AdaIN).
          </p>
        </div>
      </div>

      {/* Control Panel */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        
        {/* Basic options */}
        <div className="border border-zinc-200 rounded-xl p-5 bg-white space-y-4">
          <h3 className="text-xs font-semibold text-zinc-800 uppercase tracking-wider">
            Generation Controls
          </h3>

          <div>
            <label className="block text-xs text-zinc-600 mb-1">Number of Portraits</label>
            <div className="grid grid-cols-4 gap-2">
              {[1, 2, 4, 8].map((cnt) => (
                <button
                  key={cnt}
                  type="button"
                  onClick={() => setNumSamples(cnt)}
                  className={`py-1.5 text-xs font-mono rounded-lg border transition-colors ${
                    numSamples === cnt
                      ? 'bg-zinc-950 text-white border-zinc-950'
                      : 'bg-white text-zinc-700 border-zinc-200 hover:bg-zinc-50'
                  }`}
                >
                  {cnt}
                </button>
              ))}
            </div>
          </div>

          <div>
            <label className="block text-xs text-zinc-600 mb-1">Random Seed (Z Vector)</label>
            <div className="flex gap-2">
              <input
                type="number"
                value={seed}
                onChange={(e) => setSeed(e.target.value)}
                className="w-full text-xs font-mono border border-zinc-300 rounded-lg px-3 py-1.5 focus:outline-none focus:ring-1 focus:ring-zinc-900"
              />
              <button
                onClick={handleRandomize}
                className="px-3 py-1.5 text-xs font-medium rounded-lg bg-zinc-100 hover:bg-zinc-200 text-zinc-800 transition-colors flex items-center gap-1.5"
              >
                <Dice5 className="w-3.5 h-3.5" />
                Randomize
              </button>
            </div>
          </div>
        </div>

        {/* Truncation & Latent controls */}
        <div className="border border-zinc-200 rounded-xl p-5 bg-white space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-semibold text-zinc-800 uppercase tracking-wider">
              Truncation Trick (Psi)
            </h3>
            <span className="text-xs font-mono font-medium text-zinc-900">
              {truncationPsi}
            </span>
          </div>

          <div className="space-y-1.5">
            <input
              type="range"
              min="0.2"
              max="1.0"
              step="0.05"
              value={truncationPsi}
              onChange={(e) => setTruncationPsi(parseFloat(e.target.value))}
              className="w-full h-1.5 bg-zinc-200 rounded-lg appearance-none cursor-pointer accent-zinc-950"
            />
            <div className="flex justify-between text-[10px] text-zinc-400 font-mono">
              <span>0.2 (Average Face)</span>
              <span>0.7 (Optimal)</span>
              <span>1.0 (High Diversity)</span>
            </div>
          </div>

          <p className="text-xs text-zinc-500 leading-relaxed">
            Scales intermediate latent vector <span className="font-mono text-zinc-800">w</span> toward the empirical average <span className="font-mono text-zinc-800">w_avg</span> in W-space, balancing photorealism against variety.
          </p>
        </div>

        {/* Action Button & Architecture info */}
        <div className="border border-zinc-200 rounded-xl p-5 bg-white flex flex-col justify-between space-y-4">
          <div className="space-y-2">
            <h3 className="text-xs font-semibold text-zinc-800 uppercase tracking-wider">
              AdaIN Style Synthesis
            </h3>
            <p className="text-xs text-zinc-500 leading-relaxed">
              Injects style features across coarse (pose), medium (features), and fine (micro-texture) layers with noise injection.
            </p>
          </div>

          <button
            onClick={() => handleGenerate()}
            disabled={loading}
            className="w-full py-2.5 rounded-lg bg-zinc-950 hover:bg-zinc-800 text-white font-semibold text-xs transition-colors flex items-center justify-center gap-2 disabled:opacity-50"
          >
            {loading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Sparkles className="w-4 h-4" />}
            Synthesize Portraits
          </button>
        </div>

      </div>

      {error && (
        <div className="p-4 rounded-xl bg-red-50 border border-red-200 text-red-800 text-xs">
          {error}
        </div>
      )}

      {/* Gallery */}
      <div className="border border-zinc-200 rounded-xl p-6 bg-zinc-50/50 space-y-4">
        <div className="flex items-center justify-between border-b border-zinc-200 pb-3">
          <h3 className="text-sm font-semibold text-zinc-950">
            Synthesized StyleGAN Portraits ({samples.length} Outputs)
          </h3>
          <span className="text-xs font-mono text-zinc-500">128x128 RGB • Upscaled to 256x256</span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-5 pt-2">
          {samples.map((s, idx) => (
            <div
              key={idx}
              className="group flex flex-col items-center p-3 rounded-xl bg-white border border-zinc-200 hover:border-zinc-400 hover:shadow-sm transition-all"
            >
              <img
                src={s.image_url}
                alt={`StyleGAN Portrait ${idx}`}
                className="w-full aspect-square rounded-lg object-cover bg-zinc-100 shadow-inner"
              />
              <div className="w-full flex items-center justify-between mt-3 text-xs">
                <span className="font-mono text-[11px] text-zinc-500">Seed: {s.seed}</span>
                <span className="font-mono text-[11px] text-zinc-500">&psi;: {s.truncation_psi}</span>
                <a
                  href={s.image_url}
                  download={s.filename}
                  className="font-semibold text-zinc-900 hover:text-zinc-600 flex items-center gap-1 transition-colors"
                >
                  <Download className="w-3.5 h-3.5" />
                  Save
                </a>
              </div>
            </div>
          ))}
        </div>
      </div>

    </div>
  );
}
