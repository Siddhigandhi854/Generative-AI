import React, { useState, useEffect } from 'react';
import { X, Key, Cpu, HardDrive, CheckCircle2, AlertCircle, RefreshCw } from 'lucide-react';

export function SettingsModal({ isOpen, onClose, onSave }) {
  const [diagnostics, setDiagnostics] = useState(null);
  const [hfToken, setHfToken] = useState('');
  const [geminiKey, setGeminiKey] = useState('');
  const [openaiKey, setOpenaiKey] = useState('');
  const [saving, setSaving] = useState(false);
  const [saveStatus, setSaveStatus] = useState(null);

  useEffect(() => {
    if (isOpen) {
      fetchDiagnostics();
    }
  }, [isOpen]);

  const fetchDiagnostics = async () => {
    try {
      const res = await fetch('/api/diagnostics');
      if (res.ok) {
        const data = await res.json();
        setDiagnostics(data);
      }
    } catch (err) {
      console.error("Failed to fetch diagnostics:", err);
    }
  };

  const handleSave = async (e) => {
    e.preventDefault();
    setSaving(true);
    setSaveStatus(null);
    try {
      const res = await fetch('/api/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          hf_token: hfToken,
          gemini_key: geminiKey,
          openai_key: openaiKey
        })
      });
      if (res.ok) {
        setSaveStatus({ success: true, message: 'Settings saved successfully!' });
        fetchDiagnostics();
        if (onSave) onSave();
        setTimeout(() => setSaveStatus(null), 3000);
      } else {
        setSaveStatus({ success: false, message: 'Failed to update settings.' });
      }
    } catch (err) {
      setSaveStatus({ success: false, message: 'Network error saving settings.' });
    } finally {
      setSaving(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-zinc-950/40 backdrop-blur-sm animate-in fade-in duration-150">
      <div className="w-full max-w-xl bg-white rounded-xl shadow-xl border border-zinc-200 overflow-hidden">
        
        {/* Modal Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-zinc-200">
          <div className="flex items-center gap-2">
            <Key className="w-5 h-5 text-zinc-900" />
            <h2 className="text-base font-semibold text-zinc-950">System Diagnostics & API Settings</h2>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-md text-zinc-400 hover:text-zinc-600 hover:bg-zinc-100 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-6 space-y-6 max-h-[80vh] overflow-y-auto">
          
          {/* Diagnostics Card */}
          <div className="bg-zinc-50 rounded-lg p-4 border border-zinc-200 space-y-3">
            <div className="flex items-center justify-between text-xs font-semibold text-zinc-500 uppercase tracking-wider">
              <span>Hardware & Environment</span>
              <button 
                onClick={fetchDiagnostics} 
                className="hover:text-zinc-900 flex items-center gap-1"
                title="Refresh diagnostics"
              >
                <RefreshCw className="w-3 h-3" /> Refresh
              </button>
            </div>
            
            <div className="grid grid-cols-2 gap-3 text-sm">
              <div className="flex items-start gap-2">
                <Cpu className="w-4 h-4 text-zinc-600 mt-0.5" />
                <div>
                  <div className="text-xs text-zinc-500">Accelerator</div>
                  <div className="font-medium text-zinc-900 text-xs">
                    {diagnostics?.gpu_accelerator || 'NVIDIA RTX 3050 Laptop GPU'}
                  </div>
                </div>
              </div>

              <div className="flex items-start gap-2">
                <HardDrive className="w-4 h-4 text-zinc-600 mt-0.5" />
                <div>
                  <div className="text-xs text-zinc-500">Outputs Generated</div>
                  <div className="font-medium text-zinc-900 text-xs">
                    {diagnostics ? `${diagnostics.active_outputs_count} files (${diagnostics.outputs_size_mb} MB)` : 'Scanning...'}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* API Keys Configuration Form */}
          <form onSubmit={handleSave} className="space-y-4">
            <div>
              <label className="block text-xs font-semibold text-zinc-700 uppercase tracking-wider mb-1">
                Hugging Face Access Token (Optional)
              </label>
              <input
                type="password"
                placeholder={diagnostics?.keys_status?.hf_configured ? '•••••••••••••••• (Configured)' : 'hf_xxxxxxxxxxxxxxxxxxxxxx'}
                value={hfToken}
                onChange={(e) => setHfToken(e.target.value)}
                className="w-full px-3 py-2 text-sm border border-zinc-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-zinc-900/10 focus:border-zinc-900 font-mono"
              />
              <p className="mt-1 text-xs text-zinc-500">
                Enables remote inference for Stable Diffusion (Exp 4), MusicGen (Exp 8), and Transformer models.
              </p>
            </div>

            <div>
              <label className="block text-xs font-semibold text-zinc-700 uppercase tracking-wider mb-1">
                Google Gemini API Key (Optional)
              </label>
              <input
                type="password"
                placeholder={diagnostics?.keys_status?.gemini_configured ? '•••••••••••••••• (Configured)' : 'AIzaSy...'}
                value={geminiKey}
                onChange={(e) => setGeminiKey(e.target.value)}
                className="w-full px-3 py-2 text-sm border border-zinc-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-zinc-900/10 focus:border-zinc-900 font-mono"
              />
              <p className="mt-1 text-xs text-zinc-500">
                Powers Gemini 1.5 Flash in Q&A (Exp 3) and Personalized RAG Assistant (Exp 9).
              </p>
            </div>

            <div>
              <label className="block text-xs font-semibold text-zinc-700 uppercase tracking-wider mb-1">
                OpenAI API Key (Optional)
              </label>
              <input
                type="password"
                placeholder={diagnostics?.keys_status?.openai_configured ? '•••••••••••••••• (Configured)' : 'sk-proj-...'}
                value={openaiKey}
                onChange={(e) => setOpenaiKey(e.target.value)}
                className="w-full px-3 py-2 text-sm border border-zinc-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-zinc-900/10 focus:border-zinc-900 font-mono"
              />
              <p className="mt-1 text-xs text-zinc-500">
                Alternative LLM engine for GPT-4o-mini inference.
              </p>
            </div>

            {saveStatus && (
              <div className={`p-3 rounded-lg text-xs flex items-center gap-2 ${saveStatus.success ? 'bg-emerald-50 text-emerald-800 border border-emerald-200' : 'bg-red-50 text-red-800 border border-red-200'}`}>
                {saveStatus.success ? <CheckCircle2 className="w-4 h-4" /> : <AlertCircle className="w-4 h-4" />}
                <span>{saveStatus.message}</span>
              </div>
            )}

            <div className="pt-2 flex items-center justify-end gap-3 border-t border-zinc-200">
              <button
                type="button"
                onClick={onClose}
                className="px-4 py-2 text-xs font-medium text-zinc-700 hover:bg-zinc-100 rounded-lg transition-colors"
              >
                Close
              </button>
              <button
                type="submit"
                disabled={saving}
                className="px-4 py-2 text-xs font-medium text-white bg-zinc-950 hover:bg-zinc-800 rounded-lg transition-colors disabled:opacity-50"
              >
                {saving ? 'Saving...' : 'Save Configuration'}
              </button>
            </div>
          </form>

        </div>

      </div>
    </div>
  );
}
