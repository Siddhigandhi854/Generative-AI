import React, { useState, useEffect } from 'react';
import { 
  ArrowLeft, 
  Upload, 
  Send, 
  Bot, 
  User, 
  Trash2, 
  FileText, 
  Sparkles, 
  RefreshCw, 
  Layers, 
  CheckCircle2, 
  Sliders 
} from 'lucide-react';

const SAMPLE_KNOWLEDGE_DOC = `ARTIFICIAL INTELLIGENCE RESEARCH LAB PROTOCOLS
Guide Teacher: Dr. Sachin Takmare
Department: Computer Science and Advanced Artificial Intelligence
Academic Year: 2026

1. Generative Modeling Guidelines:
The Generative AI laboratory emphasizes rigorous experimental validation across probabilistic latent models (VAEs), adversarial networks (DCGANs, StyleGANs), and latent diffusion foundation models. Students must ensure reproducible seeds and verify tensor shapes during matrix transformations.

2. Retrieval-Augmented Generation (RAG) Architecture:
RAG combines non-parametric external document indices with parametric foundation models. Input queries are vectorized into dense embedding spaces and scored via cosine similarity against pre-chunked knowledge representations. Chunks with top-k relevance scores are dynamically injected into prompt context windows to ground model outputs and eliminate hallucinations.

3. Low-Rank Adaptation (LoRA) Principles:
Standard fine-tuning of 8-billion parameter models is computationally prohibitive on consumer hardware. LoRA freezes pre-trained weight matrices W0 in dimension d x k and injects trainable rank decomposition matrices A and B such that W = W0 + (B * A) * (alpha / r). This reduces trainable parameter counts by 99% while fitting within modest GPU VRAM budgets (such as NVIDIA GeForce RTX 3050 Laptop GPU with 4GB VRAM).`;

export function Exp9PersonalizedAssistant({ onBack, onOpenSettings }) {
  const [activeTab, setActiveTab] = useState('chat'); // 'chat' or 'finetuning'
  
  // RAG Chat State
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: 'Hello! I am your personalized Generative AI Assistant. Upload your research documents or click "Load Sample Lab Manual" to ground my responses with semantic RAG retrieval.',
      sources: []
    }
  ]);
  const [inputQuestion, setInputQuestion] = useState('');
  const [chatLoading, setChatLoading] = useState(false);
  const [status, setStatus] = useState(null);
  const [uploadLoading, setUploadLoading] = useState(false);
  const [uploadMsg, setUploadMsg] = useState(null);

  // Fine-tuning State
  const [loraR, setLoraR] = useState(8);
  const [loraAlpha, setLoraAlpha] = useState(16);
  const [epochs, setEpochs] = useState(3);
  const [ftResult, setFtResult] = useState(null);
  const [ftLoading, setFtLoading] = useState(false);

  useEffect(() => {
    fetchStatus();
  }, []);

  const fetchStatus = async () => {
    try {
      const res = await fetch('/api/exp9/status');
      if (res.ok) {
        const d = await res.json();
        setStatus(d);
      }
    } catch (e) {
      console.error(e);
    }
  };

  const handleUploadFile = async (file) => {
    setUploadLoading(true);
    setUploadMsg(null);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('/api/exp9/upload', {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Upload failed');
      setUploadMsg({ success: true, text: `Successfully indexed ${data.filename} (${data.chunks_indexed} chunks)` });
      fetchStatus();
    } catch (err) {
      setUploadMsg({ success: false, text: err.message });
    } finally {
      setUploadLoading(false);
    }
  };

  const loadSampleDoc = () => {
    const blob = new Blob([SAMPLE_KNOWLEDGE_DOC], { type: 'text/plain' });
    const file = new File([blob], 'lab_protocols_dr_sachin_takmare.txt', { type: 'text/plain' });
    handleUploadFile(file);
  };

  const handleClearDocs = async () => {
    await fetch('/api/exp9/clear-documents', { method: 'POST' });
    fetchStatus();
    setUploadMsg({ success: true, text: 'Knowledge base cleared.' });
  };

  const handleSendMessage = async (e) => {
    e?.preventDefault();
    if (!inputQuestion.trim()) return;

    const userText = inputQuestion.trim();
    setInputQuestion('');
    
    // Add user message to UI
    const updatedHistory = [...messages, { role: 'user', content: userText }];
    setMessages(updatedHistory);
    setChatLoading(true);

    try {
      const res = await fetch('/api/exp9/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: userText,
          history: updatedHistory.slice(-4),
          top_k: 2
        })
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Assistant error');

      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: data.answer,
          sources: data.sources || [],
          provider: data.provider
        }
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: `Error: ${err.message}`,
          sources: []
        }
      ]);
    } finally {
      setChatLoading(false);
    }
  };

  const handleSimulateLoRA = async () => {
    setFtLoading(true);
    try {
      const res = await fetch('/api/exp9/finetune-simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_name: 'meta-llama/Llama-3-8B',
          lora_r: Number(loraR),
          lora_alpha: Number(loraAlpha),
          epochs: Number(epochs)
        })
      });
      const data = await res.json();
      setFtResult(data);
    } catch (e) {
      console.error(e);
    } finally {
      setFtLoading(false);
    }
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

        <div className="border-b border-zinc-200 pb-5 flex flex-col md:flex-row md:items-end justify-between gap-4">
          <div>
            <span className="text-xs font-mono text-zinc-500 uppercase tracking-wider font-semibold">
              Experiment 9
            </span>
            <h1 className="text-2xl font-bold tracking-tight text-zinc-950 mt-1">
              Development of a Personalized AI Assistant using an Open-Source Large Language Model, RAG, and Fine-Tuning
            </h1>
            <p className="text-sm text-zinc-500 mt-1">
              Interactive personalized AI assistant grounded with document chunking, embedding retrieval, and LoRA PEFT parameter tuning.
            </p>
          </div>

          {/* Tab Switcher */}
          <div className="flex rounded-lg border border-zinc-200 bg-zinc-100 p-1">
            <button
              onClick={() => setActiveTab('chat')}
              className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-colors ${
                activeTab === 'chat' ? 'bg-white text-zinc-950 shadow-sm' : 'text-zinc-600 hover:text-zinc-900'
              }`}
            >
              RAG Assistant Chat
            </button>
            <button
              onClick={() => setActiveTab('finetuning')}
              className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-colors ${
                activeTab === 'finetuning' ? 'bg-white text-zinc-950 shadow-sm' : 'text-zinc-600 hover:text-zinc-900'
              }`}
            >
              LoRA Fine-Tuning Workflow
            </button>
          </div>
        </div>
      </div>

      {/* TAB 1: RAG Assistant Chat */}
      {activeTab === 'chat' && (
        <div className="space-y-6">
          
          {/* Knowledge Base Status & Upload Strip */}
          <div className="p-5 rounded-xl border border-zinc-200 bg-white grid grid-cols-1 md:grid-cols-3 gap-5 items-center">
            
            {/* Status overview */}
            <div className="space-y-1">
              <span className="text-xs font-semibold text-zinc-700 uppercase tracking-wider">
                Knowledge Base Index
              </span>
              <div className="text-sm font-semibold text-zinc-950">
                {status ? `${status.documents_count} Documents (${status.chunks_indexed} Chunks Indexed)` : 'Loading...'}
              </div>
              <div className="text-[11px] text-zinc-500 font-mono">
                {status?.documents?.join(', ') || 'No files uploaded yet'}
              </div>
            </div>

            {/* Quick Sample Document / Clear */}
            <div className="flex items-center gap-2">
              <button
                onClick={loadSampleDoc}
                disabled={uploadLoading}
                className="flex-1 py-2 px-3 text-xs font-semibold rounded-lg bg-zinc-100 hover:bg-zinc-200 text-zinc-900 transition-colors flex items-center justify-center gap-1.5"
              >
                <FileText className="w-3.5 h-3.5" />
                Load Sample Lab Manual
              </button>

              {status?.documents_count > 0 && (
                <button
                  onClick={handleClearDocs}
                  className="p-2 rounded-lg border border-zinc-200 hover:bg-zinc-100 text-zinc-500 hover:text-red-600 transition-colors"
                  title="Clear Indexed Documents"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              )}
            </div>

            {/* Custom file upload */}
            <div>
              <label className="flex items-center justify-center gap-2 px-4 py-2 border border-zinc-300 rounded-lg text-xs font-semibold text-zinc-800 hover:bg-zinc-50 cursor-pointer transition-colors">
                <Upload className="w-3.5 h-3.5" />
                <span>Upload TXT / CSV / MD</span>
                <input
                  type="file"
                  accept=".txt,.csv,.md,.json"
                  onChange={(e) => e.target.files?.[0] && handleUploadFile(e.target.files[0])}
                  className="hidden"
                />
              </label>
            </div>

          </div>

          {uploadMsg && (
            <div className={`p-3 rounded-lg text-xs flex items-center gap-2 ${uploadMsg.success ? 'bg-emerald-50 text-emerald-800 border border-emerald-200' : 'bg-red-50 text-red-800 border border-red-200'}`}>
              <CheckCircle2 className="w-4 h-4 shrink-0" />
              <span>{uploadMsg.text}</span>
            </div>
          )}

          {/* Chat Window */}
          <div className="border border-zinc-200 rounded-xl bg-white overflow-hidden flex flex-col h-[520px]">
            
            {/* Messages Scroll Area */}
            <div className="flex-1 p-6 overflow-y-auto space-y-4 bg-zinc-50/30">
              {messages.map((m, idx) => (
                <div
                  key={idx}
                  className={`flex gap-3 max-w-3xl ${m.role === 'user' ? 'ml-auto flex-row-reverse' : 'mr-auto'}`}
                >
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 ${m.role === 'user' ? 'bg-zinc-950 text-white' : 'bg-zinc-200 text-zinc-900'}`}>
                    {m.role === 'user' ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
                  </div>

                  <div className="space-y-2">
                    <div className={`p-4 rounded-xl text-xs sm:text-sm leading-relaxed ${
                      m.role === 'user'
                        ? 'bg-zinc-950 text-white font-sans'
                        : 'bg-white text-zinc-900 border border-zinc-200 shadow-sm font-sans'
                    }`}>
                      <div className="whitespace-pre-wrap">{m.content}</div>

                      {m.provider && (
                        <div className="mt-2 text-[10px] text-zinc-400 font-mono">
                          Source: {m.provider}
                        </div>
                      )}
                    </div>

                    {/* Grounded Source References */}
                    {m.sources && m.sources.length > 0 && (
                      <div className="space-y-1.5 pt-1">
                        <span className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wider block">
                          Retrieved Document Citations ({m.sources.length}):
                        </span>
                        {m.sources.map((src, sIdx) => (
                          <div
                            key={sIdx}
                            className="p-2.5 rounded-lg bg-zinc-100 border border-zinc-200 text-xs text-zinc-800 space-y-1"
                          >
                            <div className="flex items-center justify-between font-mono text-[11px] text-zinc-600">
                              <span className="font-semibold text-zinc-900">{src.filename} (Chunk {src.chunk_idx})</span>
                              <span>Match Score: {src.score}</span>
                            </div>
                            <p className="text-[11px] text-zinc-700 italic line-clamp-2">
                              "{src.text}"
                            </p>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              ))}

              {chatLoading && (
                <div className="flex gap-3 max-w-xl mr-auto">
                  <div className="w-8 h-8 rounded-full bg-zinc-200 text-zinc-900 flex items-center justify-center shrink-0">
                    <Bot className="w-4 h-4" />
                  </div>
                  <div className="p-4 rounded-xl bg-white border border-zinc-200 text-xs text-zinc-500 flex items-center gap-2">
                    <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                    Searching vector embeddings and synthesizing grounded answer...
                  </div>
                </div>
              )}
            </div>

            {/* Input Bar */}
            <form onSubmit={handleSendMessage} className="p-4 bg-white border-t border-zinc-200 flex gap-2">
              <input
                type="text"
                value={inputQuestion}
                onChange={(e) => setInputQuestion(e.target.value)}
                placeholder="Ask a question grounded in your uploaded documents..."
                className="flex-1 text-sm px-4 py-2.5 border border-zinc-300 rounded-lg focus:outline-none focus:ring-1 focus:ring-zinc-900 text-zinc-900"
              />
              <button
                type="submit"
                disabled={chatLoading || !inputQuestion.trim()}
                className="px-5 py-2.5 bg-zinc-950 hover:bg-zinc-800 text-white text-xs font-semibold rounded-lg transition-colors flex items-center gap-1.5 disabled:opacity-50"
              >
                <Send className="w-3.5 h-3.5" />
                Send
              </button>
            </form>

          </div>

        </div>
      )}

      {/* TAB 2: LoRA Fine-Tuning Workflow */}
      {activeTab === 'finetuning' && (
        <div className="space-y-6 animate-in fade-in duration-150">
          
          <div className="p-6 rounded-xl border border-zinc-200 bg-white space-y-4">
            <div>
              <h3 className="text-base font-semibold text-zinc-950 flex items-center gap-2">
                <Sliders className="w-4 h-4 text-zinc-800" />
                Parameter-Efficient Fine-Tuning (PEFT / LoRA) Configuration
              </h3>
              <p className="text-xs text-zinc-500 mt-1">
                Configure low-rank adapter decomposition hyperparameters for domain-adapted LLM fine-tuning.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2">
              <div>
                <label className="block text-xs font-semibold text-zinc-700 mb-1">LoRA Rank (r)</label>
                <select
                  value={loraR}
                  onChange={(e) => setLoraR(e.target.value)}
                  className="w-full text-xs border border-zinc-300 rounded-lg p-2 bg-white text-zinc-900 focus:outline-none focus:ring-1 focus:ring-zinc-900 font-mono"
                >
                  <option value={4}>r = 4 (Ultra-compact)</option>
                  <option value={8}>r = 8 (Standard)</option>
                  <option value={16}>r = 16 (Higher capacity)</option>
                  <option value={32}>r = 32 (Extended rank)</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-semibold text-zinc-700 mb-1">LoRA Alpha (Scaling Factor)</label>
                <input
                  type="number"
                  value={loraAlpha}
                  onChange={(e) => setLoraAlpha(e.target.value)}
                  className="w-full text-xs font-mono border border-zinc-300 rounded-lg p-2 focus:outline-none focus:ring-1 focus:ring-zinc-900"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-zinc-700 mb-1">Training Epochs</label>
                <input
                  type="number"
                  min={1}
                  max={10}
                  value={epochs}
                  onChange={(e) => setEpochs(e.target.value)}
                  className="w-full text-xs font-mono border border-zinc-300 rounded-lg p-2 focus:outline-none focus:ring-1 focus:ring-zinc-900"
                />
              </div>
            </div>

            <button
              onClick={handleSimulateLoRA}
              disabled={ftLoading}
              className="py-2.5 px-4 text-xs font-semibold text-white bg-zinc-950 hover:bg-zinc-800 rounded-lg transition-colors flex items-center gap-2 disabled:opacity-50"
            >
              {ftLoading ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Sparkles className="w-3.5 h-3.5" />}
              Inspect LoRA Architecture & Compute Footprint
            </button>
          </div>

          {/* LoRA Results & Metrics */}
          {ftResult && (
            <div className="border border-zinc-200 rounded-xl p-6 bg-zinc-50/50 space-y-5 animate-in fade-in duration-200">
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                <div className="p-4 rounded-xl bg-white border border-zinc-200">
                  <span className="text-xs text-zinc-500">Trainable Parameters</span>
                  <div className="text-xl font-bold font-mono text-zinc-950 mt-1">
                    {(ftResult.trainable_params / 1_000_000).toFixed(2)}M
                  </div>
                </div>
                <div className="p-4 rounded-xl bg-white border border-zinc-200">
                  <span className="text-xs text-zinc-500">Total Base Parameters</span>
                  <div className="text-xl font-bold font-mono text-zinc-950 mt-1">8.00B</div>
                </div>
                <div className="p-4 rounded-xl bg-white border border-zinc-200">
                  <span className="text-xs text-zinc-500">Trainable Parameter Ratio</span>
                  <div className="text-xl font-bold font-mono text-emerald-600 mt-1">
                    {ftResult.trainable_percentage}
                  </div>
                </div>
                <div className="p-4 rounded-xl bg-white border border-zinc-200">
                  <span className="text-xs text-zinc-500">VRAM Requirement</span>
                  <div className="text-xs font-mono text-zinc-800 font-semibold mt-2">
                    {ftResult.required_vram}
                  </div>
                </div>
              </div>

              {/* Epoch Loss Table */}
              <div className="border border-zinc-200 rounded-xl overflow-hidden bg-white">
                <div className="px-6 py-3 border-b border-zinc-200">
                  <h4 className="text-xs font-semibold text-zinc-900 uppercase tracking-wider">
                    Convergence History (Cross-Entropy Loss)
                  </h4>
                </div>
                <table className="w-full text-left text-xs font-mono">
                  <thead className="bg-zinc-50 border-b border-zinc-200 text-zinc-500">
                    <tr>
                      <th className="py-2 px-6">Epoch</th>
                      <th className="py-2 px-6">Training Loss</th>
                      <th className="py-2 px-6">Validation Loss</th>
                      <th className="py-2 px-6">Perplexity</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-zinc-100">
                    {ftResult.loss_history?.map((h) => (
                      <tr key={h.epoch} className="hover:bg-zinc-50/50">
                        <td className="py-2 px-6 font-semibold text-zinc-900">Epoch {h.epoch}</td>
                        <td className="py-2 px-6 text-zinc-700">{h.train_loss}</td>
                        <td className="py-2 px-6 text-zinc-700">{h.val_loss}</td>
                        <td className="py-2 px-6 text-emerald-700">{Math.exp(h.val_loss).toFixed(2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

        </div>
      )}

    </div>
  );
}
