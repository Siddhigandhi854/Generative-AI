import React, { useState, useEffect } from 'react';
import { ArrowLeft, Sparkles, Download, RefreshCw, Dice5 } from 'lucide-react';

export function Exp6DCGANFaces({ onBack }) {
  const [numSamples, setNumSamples] = useState(4);
  const [seed, setSeed] = useState(1337);
  const [samples, setSamples] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [status, setStatus] = useState(null);

  useEffect(() => {
    fetch('/api/exp6/status')
      .then((res) => res.json())
      .then((d) => setStatus(d))
      .catch((e) => console.error(e));

    // Initial batch load
    handleGenerate(4, 1337);
  }, []);

  const handleGenerate = async (customCount = null, customSeed = null) => {
    setLoading(true);
    setError(null);

    const count = customCount !== null ? customCount : numSamples;
    const currentSeed = customSeed !== null ? customSeed : seed;

    try {
      const res = await fetch('/api/exp6/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          num_samples: count,
          seed: currentSeed !== '' ? Number(currentSeed) : null,
          latent_dim: 100
        })
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'DCGAN generation failed');
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
            Experiment 6
          </span>
          <h1 className="text-2xl font-bold tracking-tight text-zinc-950 mt-1">
            Implementation of Generative Adversarial Networks (DCGAN)
          </h1>
          <p className="text-sm text-zinc-500 mt-1">
            Generate photorealistic synthetic faces from 100-dimensional Gaussian noise vectors using deep convolutional transposed generator networks.
          </p>
        </div>
      </div>

      {/* Control Bar */}
      <div className="p-6 rounded-xl border border-zinc-200 bg-white grid grid-cols-1 sm:grid-cols-3 gap-6 items-end">
        
        <div>
          <label className="block text-xs font-semibold text-zinc-700 uppercase tracking-wider mb-1.5">
            Number of Face Samples
          </label>
          <div className="grid grid-cols-4 gap-2">
            {[1, 4, 8, 12].map((cnt) => (
              <button
                key={cnt}
                type="button"
                onClick={() => setNumSamples(cnt)}
                className={`py-2 text-xs font-mono font-medium rounded-lg border transition-colors ${
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
          <label className="block text-xs font-semibold text-zinc-700 uppercase tracking-wider mb-1.5">
            Random Seed (Latent Vector Z)
          </label>
          <div className="flex gap-2">
            <input
              type="number"
              value={seed}
              onChange={(e) => setSeed(e.target.value)}
              className="w-full text-xs font-mono border border-zinc-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-1 focus:ring-zinc-900"
            />
            <button
              onClick={handleRandomize}
              className="px-3 py-2 text-xs font-semibold rounded-lg bg-zinc-100 hover:bg-zinc-200 text-zinc-800 transition-colors flex items-center gap-1.5 whitespace-nowrap"
            >
              <Dice5 className="w-3.5 h-3.5" />
              Randomize
            </button>
          </div>
        </div>

        <div>
          <button
            onClick={() => handleGenerate()}
            disabled={loading}
            className="w-full py-2.5 rounded-lg bg-zinc-950 hover:bg-zinc-800 text-white font-semibold text-xs transition-colors flex items-center justify-center gap-2 disabled:opacity-50"
          >
            {loading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Sparkles className="w-4 h-4" />}
            Generate Faces
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
            Synthesized Face Gallery ({samples.length} Outputs)
          </h3>
          <span className="text-xs font-mono text-zinc-500">64x64 RGB Tensor • Upscaled to 256x256</span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-5 pt-2">
          {samples.map((s, idx) => (
            <div
              key={idx}
              className="group flex flex-col items-center p-3 rounded-xl bg-white border border-zinc-200 hover:border-zinc-400 hover:shadow-sm transition-all"
            >
              <img
                src={s.image_url}
                alt={`DCGAN Face ${idx}`}
                className="w-full aspect-square rounded-lg object-cover bg-zinc-100"
              />
              <div className="w-full flex items-center justify-between mt-3 text-xs">
                <span className="font-mono text-[11px] text-zinc-500">Seed: {s.seed}</span>
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
