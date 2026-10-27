import React, { useState, useEffect } from 'react';
import { ArrowLeft, Sparkles, Download, RefreshCw, Sliders } from 'lucide-react';

export function Exp5VariationalAutoencoder({ onBack }) {
  const [numSamples, setNumSamples] = useState(4);
  const [seed, setSeed] = useState(42);
  const [z1, setZ1] = useState(0.0);
  const [z2, setZ2] = useState(0.0);
  const [samples, setSamples] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [status, setStatus] = useState(null);

  useEffect(() => {
    fetch('/api/exp5/status')
      .then((res) => res.json())
      .then((d) => setStatus(d))
      .catch((e) => console.error(e));

    // Auto-generate initial batch
    handleGenerate(4, 42);
  }, []);

  const handleGenerate = async (customCount = null, customSeed = null) => {
    setLoading(true);
    setError(null);

    const count = customCount !== null ? customCount : numSamples;
    const currentSeed = customSeed !== null ? customSeed : seed;

    try {
      const res = await fetch('/api/exp5/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          num_samples: count,
          seed: currentSeed !== '' ? Number(currentSeed) : null,
          z_coords: [parseFloat(z1), parseFloat(z2)]
        })
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'VAE generation failed');
      }

      const data = await res.json();
      setSamples(data.samples || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleRandomizeSeed = () => {
    const nextSeed = Math.floor(Math.random() * 100000);
    setSeed(nextSeed);
    handleGenerate(numSamples, nextSeed);
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
            Experiment 5
          </span>
          <h1 className="text-2xl font-bold tracking-tight text-zinc-950 mt-1">
            Variational Autoencoder (VAE) for New Image Generation
          </h1>
          <p className="text-sm text-zinc-500 mt-1">
            Sample continuous Gaussian latent distributions z ~ N(0, I) and reconstruct synthetic visual manifolds through a deep VAE decoder.
          </p>
        </div>
      </div>

      {/* Control Strip */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        
        {/* Sampling Parameters */}
        <div className="border border-zinc-200 rounded-xl p-5 bg-white space-y-4">
          <h3 className="text-xs font-semibold text-zinc-800 uppercase tracking-wider">
            Batch Generation Settings
          </h3>

          <div>
            <label className="block text-xs text-zinc-600 mb-1">Number of Samples</label>
            <div className="grid grid-cols-4 gap-2">
              {[1, 4, 8, 16].map((count) => (
                <button
                  key={count}
                  type="button"
                  onClick={() => setNumSamples(count)}
                  className={`py-1.5 text-xs font-mono rounded-lg border transition-colors ${
                    numSamples === count
                      ? 'bg-zinc-950 text-white border-zinc-950'
                      : 'bg-white text-zinc-700 border-zinc-200 hover:bg-zinc-50'
                  }`}
                >
                  {count}
                </button>
              ))}
            </div>
          </div>

          <div>
            <label className="block text-xs text-zinc-600 mb-1">Random Seed</label>
            <div className="flex gap-2">
              <input
                type="number"
                value={seed}
                onChange={(e) => setSeed(e.target.value)}
                className="w-full text-xs font-mono border border-zinc-300 rounded-lg px-2.5 py-1.5 focus:outline-none focus:ring-1 focus:ring-zinc-900"
              />
              <button
                onClick={handleRandomizeSeed}
                className="px-3 py-1.5 text-xs font-semibold rounded-lg bg-zinc-100 hover:bg-zinc-200 text-zinc-800 transition-colors whitespace-nowrap"
              >
                Randomize
              </button>
            </div>
          </div>

          <button
            onClick={() => handleGenerate()}
            disabled={loading}
            className="w-full py-2.5 rounded-lg bg-zinc-950 hover:bg-zinc-800 text-white font-semibold text-xs transition-colors flex items-center justify-center gap-2 disabled:opacity-50"
          >
            {loading ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Sparkles className="w-3.5 h-3.5" />}
            Generate New Images
          </button>
        </div>

        {/* Latent Space Manifold Sliders */}
        <div className="md:col-span-2 border border-zinc-200 rounded-xl p-5 bg-white space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-semibold text-zinc-800 uppercase tracking-wider flex items-center gap-1.5">
              <Sliders className="w-3.5 h-3.5 text-zinc-700" />
              Latent Manifold Traversal (z1, z2)
            </h3>
            <span className="text-xs font-mono text-zinc-500">Continuous Interpolation</span>
          </div>
          <p className="text-xs text-zinc-500">
            Directly manipulate latent coordinates z1 and z2 to traverse the learned continuous manifold and observe real-time structural transformation.
          </p>

          <div className="space-y-3 pt-2">
            <div>
              <div className="flex justify-between text-xs font-mono text-zinc-700 mb-1">
                <span>Latent Coordinate z₁</span>
                <span>{z1}</span>
              </div>
              <input
                type="range"
                min="-2.5"
                max="2.5"
                step="0.1"
                value={z1}
                onChange={(e) => {
                  setZ1(e.target.value);
                  handleGenerate();
                }}
                className="w-full accent-zinc-950"
              />
            </div>

            <div>
              <div className="flex justify-between text-xs font-mono text-zinc-700 mb-1">
                <span>Latent Coordinate z₂</span>
                <span>{z2}</span>
              </div>
              <input
                type="range"
                min="-2.5"
                max="2.5"
                step="0.1"
                value={z2}
                onChange={(e) => {
                  setZ2(e.target.value);
                  handleGenerate();
                }}
                className="w-full accent-zinc-950"
              />
            </div>
          </div>
        </div>

      </div>

      {error && (
        <div className="p-4 rounded-xl bg-red-50 border border-red-200 text-red-800 text-xs">
          {error}
        </div>
      )}

      {/* Generated Images Grid */}
      <div className="border border-zinc-200 rounded-xl p-6 bg-zinc-50/50 space-y-4">
        <div className="flex items-center justify-between border-b border-zinc-200 pb-3">
          <h3 className="text-sm font-semibold text-zinc-950">
            Synthesized VAE Latent Grid ({samples.length} Samples)
          </h3>
          <span className="text-xs font-mono text-zinc-500">Resolution: 140x140 (Upscaled from 28x28)</span>
        </div>

        {samples.length === 0 && !loading && (
          <div className="p-8 text-center text-xs text-zinc-500">
            No samples generated yet. Click "Generate New Images" to sample from latent space.
          </div>
        )}

        <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-8 gap-4 pt-2">
          {samples.map((s, idx) => (
            <div
              key={idx}
              className="group flex flex-col items-center p-3 rounded-lg bg-white border border-zinc-200 hover:border-zinc-400 hover:shadow-sm transition-all"
            >
              <img
                src={s.image_url}
                alt={`VAE Latent Sample ${idx}`}
                className="w-24 h-24 rounded object-contain bg-zinc-950 p-1"
              />
              <div className="mt-2 text-[10px] font-mono text-zinc-500">
                z: [{s.latent_z[0]}, {s.latent_z[1]}]
              </div>
              <a
                href={s.image_url}
                download={s.filename}
                className="mt-2 text-[11px] font-semibold text-zinc-900 hover:text-zinc-600 flex items-center gap-1 opacity-80 group-hover:opacity-100 transition-opacity"
              >
                <Download className="w-3 h-3" /> Save
              </a>
            </div>
          ))}
        </div>
      </div>

    </div>
  );
}
