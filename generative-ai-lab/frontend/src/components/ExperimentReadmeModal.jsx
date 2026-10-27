import React, { useEffect } from 'react';
import { X, BookOpen, Cpu, Layers, CheckCircle2, Sliders, FileDown, ArrowRight, Code } from 'lucide-react';
import { EXPERIMENT_DOCS } from '../data/experimentDocs';

export function ExperimentReadmeModal({ expId, onClose, onLaunch }) {
  const doc = EXPERIMENT_DOCS[expId];

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    document.body.style.overflow = 'hidden';
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      document.body.style.overflow = '';
    };
  }, [onClose]);

  if (!doc) return null;

  return (
    <div className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4 sm:p-6 animate-in fade-in duration-150">
      <div 
        onClick={(e) => e.stopPropagation()}
        className="bg-white rounded-2xl max-w-2xl w-full border border-zinc-200 shadow-2xl max-h-[90vh] flex flex-col overflow-hidden animate-in zoom-in-95 duration-150"
      >
        {/* Header */}
        <div className="p-5 sm:p-6 border-b border-zinc-200 flex items-start justify-between gap-4 bg-zinc-50/50">
          <div>
            <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-md bg-zinc-900 text-white text-[11px] font-mono font-medium mb-2">
              <BookOpen className="w-3 h-3" />
              README • EXPERIMENT {doc.id}
            </div>
            <h2 className="text-lg sm:text-xl font-bold tracking-tight text-zinc-950 leading-snug">
              {doc.title}
            </h2>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="p-1.5 rounded-lg text-zinc-400 hover:text-zinc-950 hover:bg-zinc-100 transition-colors shrink-0"
            title="Close (Esc)"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Scrollable Content Body */}
        <div className="p-5 sm:p-6 space-y-6 overflow-y-auto text-sm text-zinc-700">
          
          {/* Foundation Model / Architecture Box */}
          <div className="p-3.5 rounded-xl border border-zinc-200 bg-zinc-50/70 space-y-2">
            <span className="text-[11px] font-mono uppercase tracking-wider text-zinc-500 font-semibold flex items-center gap-1.5">
              <Cpu className="w-3.5 h-3.5 text-zinc-900" />
              Model & Architecture
            </span>
            <p className="text-sm font-semibold text-zinc-950 font-mono">
              {doc.model}
            </p>
          </div>

          {/* Tech Stack Chips */}
          <div className="space-y-2">
            <span className="text-xs font-semibold text-zinc-900 uppercase tracking-wider flex items-center gap-1.5">
              <Layers className="w-3.5 h-3.5 text-zinc-700" />
              Key Libraries & Technologies
            </span>
            <div className="flex flex-wrap gap-2">
              {doc.techStack.map((tech, i) => (
                <span
                  key={i}
                  className="px-2.5 py-1 rounded-md text-xs font-medium bg-zinc-100 text-zinc-800 border border-zinc-200"
                >
                  {tech}
                </span>
              ))}
            </div>
          </div>

          {/* Core Concept & How It Works */}
          <div className="space-y-2">
            <span className="text-xs font-semibold text-zinc-900 uppercase tracking-wider">
              Core Concept & Overview
            </span>
            <p className="text-xs sm:text-sm text-zinc-600 leading-relaxed bg-white border border-zinc-200 p-3.5 rounded-xl">
              {doc.concept}
            </p>
          </div>

          {/* Execution Pipeline Steps */}
          <div className="space-y-2.5">
            <span className="text-xs font-semibold text-zinc-900 uppercase tracking-wider flex items-center gap-1.5">
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
              Pipeline Execution Workflow
            </span>
            <ul className="space-y-2 bg-zinc-50/50 border border-zinc-200 p-4 rounded-xl">
              {doc.pipeline.map((step, idx) => (
                <li key={idx} className="flex items-start gap-2.5 text-xs text-zinc-700 leading-relaxed">
                  <span className="w-4 h-4 rounded-full bg-zinc-900 text-white text-[10px] font-mono font-bold flex items-center justify-center shrink-0 mt-0.5">
                    {idx + 1}
                  </span>
                  <span>{step}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Parameters & Interactive Controls */}
          <div className="space-y-2.5">
            <span className="text-xs font-semibold text-zinc-900 uppercase tracking-wider flex items-center gap-1.5">
              <Sliders className="w-3.5 h-3.5 text-zinc-700" />
              Interactive Controls & Parameters
            </span>
            <div className="grid grid-cols-1 gap-2">
              {doc.parameters.map((param, idx) => (
                <div key={idx} className="p-3 rounded-lg border border-zinc-200 bg-white space-y-0.5">
                  <p className="text-xs font-semibold text-zinc-950 font-mono">{param.name}</p>
                  <p className="text-xs text-zinc-500">{param.desc}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Outputs & File Exports */}
          <div className="space-y-2">
            <span className="text-xs font-semibold text-zinc-900 uppercase tracking-wider flex items-center gap-1.5">
              <FileDown className="w-3.5 h-3.5 text-zinc-700" />
              Generated Outputs & File Exports
            </span>
            <p className="text-xs text-zinc-700 bg-zinc-50 border border-zinc-200 p-3 rounded-lg leading-relaxed">
              {doc.outputs}
            </p>
          </div>

          {/* API Endpoints */}
          <div className="space-y-1.5">
            <span className="text-xs font-semibold text-zinc-900 uppercase tracking-wider flex items-center gap-1.5">
              <Code className="w-3.5 h-3.5 text-zinc-700" />
              Backend API Route
            </span>
            <p className="text-[11px] font-mono text-zinc-600 bg-zinc-100 border border-zinc-200 p-2.5 rounded-lg break-all">
              {doc.endpoint}
            </p>
          </div>

        </div>

        {/* Footer Actions */}
        <div className="p-4 sm:p-5 border-t border-zinc-200 bg-zinc-50 flex items-center justify-between gap-3">
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-2 rounded-lg border border-zinc-200 bg-white hover:bg-zinc-100 text-zinc-700 text-xs font-semibold transition-colors"
          >
            Close
          </button>

          <button
            type="button"
            onClick={() => onLaunch(doc.id)}
            className="px-4 py-2 rounded-lg bg-zinc-950 hover:bg-zinc-800 text-white text-xs font-semibold transition-colors flex items-center gap-1.5 shadow-sm"
          >
            Open Interactive Experiment
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
    </div>
  );
}
