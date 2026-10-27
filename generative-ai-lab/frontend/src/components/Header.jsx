import React from 'react';
import { Settings, Cpu, ShieldCheck } from 'lucide-react';

export function Header({ backendStatus, onOpenSettings, onNavigateHome }) {
  return (
    <header className="sticky top-0 z-40 w-full border-b border-zinc-200 bg-white/95 backdrop-blur-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        
        {/* Brand / Logo */}
        <div 
          onClick={onNavigateHome}
          className="flex items-center gap-3 cursor-pointer group transition-opacity hover:opacity-90"
        >
          <div className="w-9 h-9 rounded-lg bg-zinc-950 flex items-center justify-center text-white font-mono text-sm font-semibold shadow-sm">
            AI
          </div>
          <div>
            <div className="font-semibold text-zinc-950 text-base leading-tight tracking-tight">
              Generative AI Lab
            </div>
            <div className="text-xs text-zinc-500 font-medium">
              Interactive Experimentation Platform
            </div>
          </div>
        </div>

        {/* Right Section: Guide Teacher attribution */}
        <div className="flex items-center">
          <div className="flex flex-col text-right">
            <span className="text-[11px] text-zinc-400 uppercase tracking-wider font-semibold">
              Guide Teacher
            </span>
            <span className="text-sm font-semibold text-zinc-900 tracking-tight">
              Dr. Sachin Takmare
            </span>
          </div>
        </div>
      </div>
    </header>
  );
}
