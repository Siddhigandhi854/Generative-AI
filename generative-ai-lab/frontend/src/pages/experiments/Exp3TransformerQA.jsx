import React, { useState, useEffect } from 'react';
import { ArrowLeft, Sparkles, RefreshCw, Trash2, CheckCircle2, AlertCircle, Copy, Check, SlidersHorizontal, BookOpen } from 'lucide-react';

const PRESETS = [
  {
    id: 'transformers',
    label: 'Transformer Architecture',
    question: 'What does the Transformer architecture rely entirely on to compute representations?',
    context: `The Transformer model, introduced in the seminal 2017 paper "Attention Is All You Need" by Ashish Vaswani and colleagues at Google Brain, revolutionized artificial intelligence. Unlike recurrence-based architectures such as LSTMs and GRUs, the Transformer relies entirely on self-attention mechanisms to compute representations of its input and output without using sequence-aligned RNNs. This parallel self-attention architecture enables massive scalability during pre-training on large language corpora. Modern generative models like GPT-4, Llama 3, and BERT all derive their core architecture from this foundation.`
  },
  {
    id: 'diffusion',
    label: 'Diffusion Models',
    question: 'What network predicts and subtracts Gaussian noise in Latent Diffusion Models?',
    context: `Latent Diffusion Models (LDMs) achieve high-resolution text-to-image synthesis by performing diffusion denoising in a low-dimensional latent space compressed by a Variational Autoencoder (VAE). A time-conditional UNet backbone predicts and subtracts Gaussian noise over discrete reverse diffusion steps, guided by cross-attention embeddings from pre-trained text encoders like CLIP or T5. Decoding the final denoised latent vector through the VAE decoder reconstructs the synthesized RGB image.`
  },
  {
    id: 'rlhf',
    label: 'LLM Alignment (RLHF)',
    question: 'What optimization algorithm is used in RLHF along with a KL-divergence penalty?',
    context: `Reinforcement Learning from Human Feedback (RLHF) and Direct Preference Optimization (DPO) are key alignment methodologies for foundation language models. RLHF trains a reward model on human pairwise preferences, then fine-tunes the policy network using Proximal Policy Optimization (PPO) with a KL-divergence penalty to prevent policy collapse and preserve linguistic fluency while aligning responses with safety guidelines.`
  }
];

export function Exp3TransformerQA({ onBack, onOpenSettings }) {
  const [question, setQuestion] = useState(PRESETS[0].question);
  const [context, setContext] = useState(PRESETS[0].context);
  const [modelName, setModelName] = useState('deepset/roberta-base-squad2');
  const [provider, setProvider] = useState('huggingface');
  const [temperature, setTemperature] = useState(0.3);
  const [maxTokens, setMaxTokens] = useState(120);
  const [showGenParams, setShowGenParams] = useState(false);
  const [activeTab, setActiveTab] = useState('edit'); // 'edit' | 'highlight'
  const [copied, setCopied] = useState(false);
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState(null);
  const [error, setError] = useState(null);

  const handleGenerate = async () => {
    if (!question.trim()) {
      setError('Please enter a question.');
      return;
    }
    setLoading(true);
    setError(null);
    setResponse(null);
    setCopied(false);

    try {
      const res = await fetch('/api/exp3/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: question.trim(),
          context: context.trim(),
          provider: provider,
          model_name: modelName,
          temperature: parseFloat(temperature),
          max_tokens: parseInt(maxTokens)
        })
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'Failed to generate answer.');
      }
      setResponse(data);
      if (data.start_char !== null && data.start_char !== undefined && data.start_char !== -1) {
        setActiveTab('highlight');
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setQuestion('');
    setContext('');
    setResponse(null);
    setError(null);
    setActiveTab('edit');
  };

  const handleCopy = () => {
    if (response?.answer) {
      navigator.clipboard.writeText(response.answer);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  // Render context with highlighted answer span
  const renderHighlightedContext = () => {
    if (!response || response.start_char === null || response.start_char === undefined || response.start_char === -1) {
      return <span>{context}</span>;
    }
    const s = response.start_char;
    const e = response.end_char || (s + response.answer.length);
    const before = context.slice(0, s);
    const match = context.slice(s, e);
    const after = context.slice(e);

    return (
      <span className="leading-relaxed">
        {before}
        <mark className="bg-amber-100 text-amber-950 px-1 py-0.5 rounded font-medium border border-amber-300">
          {match}
        </mark>
        {after}
      </span>
    );
  };

  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      
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
            Experiment 3
          </span>
          <h1 className="text-2xl font-bold tracking-tight text-zinc-950 mt-1">
            Text Generation using a Pre-trained Transformer Model for a Question–Answer System
          </h1>
          <p className="text-sm text-zinc-500 mt-1">
            Query pre-trained Transformer language models with context grounding, factual span extraction, and generative question answering.
          </p>
        </div>
      </div>

      {/* Model Selection */}
      <div className="p-4 rounded-xl border border-zinc-200 bg-zinc-50 flex flex-col sm:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-3 w-full sm:w-auto">
          <label className="text-xs font-semibold text-zinc-700 whitespace-nowrap">Model Architecture:</label>
          <select
            value={modelName}
            onChange={(e) => {
              const val = e.target.value;
              setModelName(val);
              if (val.includes('gemini')) setProvider('gemini');
              else if (val.includes('gpt')) setProvider('openai');
              else setProvider('huggingface');
            }}
            className="text-xs border border-zinc-300 rounded-lg px-3 py-1.5 bg-white text-zinc-900 focus:outline-none focus:ring-1 focus:ring-zinc-900"
          >
            <option value="deepset/roberta-base-squad2">RoBERTa Base (Extractive SQuAD2 Transformer)</option>
            <option value="google/flan-t5-base">Flan-T5 Base (Seq2Seq Generative Transformer)</option>
            <option value="gemini-1.5-flash">Gemini 1.5 Flash (Google AI)</option>
            <option value="gpt-4o-mini">GPT-4o Mini (OpenAI)</option>
          </select>
        </div>

        <button
          onClick={() => setShowGenParams(!showGenParams)}
          className="inline-flex items-center gap-1.5 text-xs text-zinc-600 hover:text-zinc-950 px-2.5 py-1.5 rounded-lg border border-zinc-200 bg-white hover:bg-zinc-100 transition-colors"
        >
          <SlidersHorizontal className="w-3.5 h-3.5" />
          <span>{showGenParams ? 'Hide Parameters' : 'Generation Parameters'}</span>
        </button>
      </div>

      {/* Generation Parameters Panel */}
      {showGenParams && (
        <div className="p-4 rounded-xl border border-zinc-200 bg-white grid grid-cols-1 sm:grid-cols-2 gap-6 animate-in fade-in duration-150">
          <div>
            <div className="flex items-center justify-between text-xs font-medium text-zinc-700 mb-1.5">
              <span>Sampling Temperature: <strong className="font-mono text-zinc-950">{temperature}</strong></span>
              <span className="text-[11px] text-zinc-500">{temperature < 0.4 ? 'Greedy / Factual' : 'Stochastic / Creative'}</span>
            </div>
            <input
              type="range"
              min="0.0"
              max="1.0"
              step="0.05"
              value={temperature}
              onChange={(e) => setTemperature(parseFloat(e.target.value))}
              className="w-full h-1.5 bg-zinc-200 rounded-lg appearance-none cursor-pointer accent-zinc-900"
            />
          </div>

          <div>
            <div className="flex items-center justify-between text-xs font-medium text-zinc-700 mb-1.5">
              <span>Max Generation Tokens: <strong className="font-mono text-zinc-950">{maxTokens}</strong></span>
              <span className="text-[11px] text-zinc-500">Output Length Cap</span>
            </div>
            <input
              type="range"
              min="32"
              max="256"
              step="16"
              value={maxTokens}
              onChange={(e) => setMaxTokens(parseInt(e.target.value))}
              className="w-full h-1.5 bg-zinc-200 rounded-lg appearance-none cursor-pointer accent-zinc-900"
            />
          </div>
        </div>
      )}

      {/* Main Interactive Form */}
      <div className="border border-zinc-200 rounded-xl p-6 bg-white space-y-6">
        
        {/* Preset Selector Chips */}
        <div>
          <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
            <span className="text-xs font-semibold text-zinc-800 uppercase tracking-wider">
              Domain Context Presets:
            </span>
            <div className="flex flex-wrap gap-1.5">
              {PRESETS.map((p) => (
                <button
                  key={p.id}
                  onClick={() => {
                    setContext(p.context);
                    setQuestion(p.question);
                    setResponse(null);
                    setError(null);
                    setActiveTab('edit');
                  }}
                  className="text-xs px-2.5 py-1 rounded-md border border-zinc-200 bg-zinc-50 hover:bg-zinc-100 text-zinc-700 transition-colors font-medium"
                >
                  {p.label}
                </button>
              ))}
            </div>
          </div>

          {/* Context Tab Controls if Answer is Present */}
          {response && response.start_char !== null && response.start_char !== undefined && response.start_char !== -1 && (
            <div className="flex items-center gap-2 mb-2">
              <button
                onClick={() => setActiveTab('edit')}
                className={`text-xs px-3 py-1 rounded-md font-medium transition-colors ${
                  activeTab === 'edit'
                    ? 'bg-zinc-950 text-white'
                    : 'bg-zinc-100 text-zinc-600 hover:text-zinc-950'
                }`}
              >
                Edit Context
              </button>
              <button
                onClick={() => setActiveTab('highlight')}
                className={`text-xs px-3 py-1 rounded-md font-medium transition-colors flex items-center gap-1.5 ${
                  activeTab === 'highlight'
                    ? 'bg-zinc-950 text-white'
                    : 'bg-zinc-100 text-zinc-600 hover:text-zinc-950'
                }`}
              >
                <BookOpen className="w-3.5 h-3.5" />
                Grounded Span View
              </button>
            </div>
          )}

          {activeTab === 'edit' ? (
            <textarea
              rows={5}
              value={context}
              onChange={(e) => setContext(e.target.value)}
              placeholder="Paste reference paragraph or document context here..."
              className="w-full text-xs font-mono p-3 border border-zinc-300 rounded-lg focus:outline-none focus:ring-1 focus:ring-zinc-900 leading-relaxed text-zinc-900"
            />
          ) : (
            <div className="w-full text-xs font-mono p-3 border border-zinc-300 rounded-lg bg-zinc-50/50 leading-relaxed text-zinc-900 min-h-[120px] whitespace-pre-wrap">
              {renderHighlightedContext()}
            </div>
          )}
        </div>

        {/* Question Input */}
        <div>
          <label className="block text-xs font-semibold text-zinc-800 uppercase tracking-wider mb-1.5">
            Your Question
          </label>
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Enter your question regarding the context or general topic..."
            className="w-full text-sm p-3 border border-zinc-300 rounded-lg focus:outline-none focus:ring-1 focus:ring-zinc-900 text-zinc-900"
          />
        </div>

        {/* Action Controls */}
        <div className="flex items-center gap-3 pt-2">
          <button
            onClick={handleGenerate}
            disabled={loading}
            className="flex-1 py-3 rounded-lg bg-zinc-950 hover:bg-zinc-800 text-white font-semibold text-xs transition-colors flex items-center justify-center gap-2 disabled:opacity-50"
          >
            {loading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Sparkles className="w-4 h-4" />}
            {loading ? 'Synthesizing Answer...' : 'Generate Answer'}
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
            <div className="font-semibold">Notice</div>
            <div>{error}</div>
          </div>
        </div>
      )}

      {/* Response Area */}
      {response && (
        <div className="border border-zinc-200 rounded-xl p-6 bg-zinc-50/50 space-y-4 animate-in fade-in duration-200">
          <div className="flex items-center justify-between border-b border-zinc-200 pb-3">
            <div className="flex items-center gap-2 text-zinc-950 font-semibold text-sm">
              <CheckCircle2 className="w-4 h-4 text-emerald-600" />
              Generated Response
            </div>
            <button
              onClick={handleCopy}
              className="inline-flex items-center gap-1.5 text-xs text-zinc-600 hover:text-zinc-950 px-2.5 py-1 rounded-md border border-zinc-200 bg-white hover:bg-zinc-100 transition-colors"
            >
              {copied ? <Check className="w-3.5 h-3.5 text-emerald-600" /> : <Copy className="w-3.5 h-3.5" />}
              <span>{copied ? 'Copied' : 'Copy Answer'}</span>
            </button>
          </div>

          <div className="p-4 bg-white rounded-lg border border-zinc-200 text-sm text-zinc-900 font-sans leading-relaxed whitespace-pre-wrap">
            {response.answer}
          </div>
        </div>
      )}

    </div>
  );
}
