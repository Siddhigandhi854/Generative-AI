import React, { useState } from 'react';
import { EXPERIMENTS } from '../data/experiments';
import { ExperimentReadmeModal } from '../components/ExperimentReadmeModal';
import { 
  Database, 
  Cpu, 
  Sparkles, 
  Image as ImageIcon, 
  Network, 
  Boxes, 
  Layers, 
  Music, 
  Bot, 
  Video, 
  ArrowRight,
  BookOpen
} from 'lucide-react';

const ICON_MAP = {
  Database: Database,
  Cpu: Cpu,
  Sparkles: Sparkles,
  Image: ImageIcon,
  Network: Network,
  Boxes: Boxes,
  Layers: Layers,
  Music: Music,
  Bot: Bot,
  Video: Video,
};

export function HomePage({ onSelectExperiment }) {
  const [activeReadmeExpId, setActiveReadmeExpId] = useState(null);

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-8">
      
      {/* Hero Heading & Subtitle */}
      <div className="max-w-3xl space-y-2">
        <h1 className="text-3xl sm:text-4xl font-bold tracking-tight text-zinc-950 font-sans">
          Generative AI Lab
        </h1>
        <p className="text-base sm:text-lg text-zinc-600 font-normal italic">
          Explore and run interactive Generative AI experiments.
        </p>
      </div>

      {/* 10 Experiment Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-2 gap-5 pt-2">
        {EXPERIMENTS.map((exp) => {
          const IconComponent = ICON_MAP[exp.iconName] || Sparkles;

          return (
            <div
              key={exp.id}
              onClick={() => onSelectExperiment(exp.id)}
              className="group relative flex flex-col justify-between p-6 bg-white rounded-xl border border-zinc-200 hover:border-zinc-400 hover:shadow-md transition-all duration-200 cursor-pointer"
            >
              <div className="space-y-4">
                {/* Header with icon and tags */}
                <div className="flex items-center justify-between">
                  <div className="w-10 h-10 rounded-lg bg-zinc-100 border border-zinc-200 flex items-center justify-center text-zinc-900 group-hover:bg-zinc-950 group-hover:text-white transition-colors duration-200">
                    <IconComponent className="w-5 h-5" />
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-[11px] font-mono text-zinc-500 bg-zinc-50 border border-zinc-200 px-2 py-0.5 rounded">
                      Exp {exp.id}
                    </span>
                    <span className="text-[11px] font-medium text-zinc-600 bg-zinc-100 px-2 py-0.5 rounded">
                      {exp.tag}
                    </span>
                  </div>
                </div>

                {/* Exact Title */}
                <div>
                  <h3 className="text-base font-semibold text-zinc-950 leading-snug tracking-tight group-hover:text-zinc-800 transition-colors">
                    {exp.title}
                  </h3>
                  <p className="mt-2 text-xs text-zinc-500 line-clamp-2 leading-relaxed">
                    {exp.description}
                  </p>
                </div>
              </div>

              {/* Action Buttons: README & Open Experiment */}
              <div className="mt-6 pt-4 border-t border-zinc-100 flex items-center justify-between gap-3">
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    setActiveReadmeExpId(exp.id);
                  }}
                  className="inline-flex items-center gap-1.5 text-xs font-semibold text-zinc-700 hover:text-zinc-950 bg-zinc-100 hover:bg-zinc-200 border border-zinc-200 px-2.5 py-1.5 rounded-lg transition-colors shadow-sm"
                  title={`View README & model information for Experiment ${exp.id}`}
                >
                  <BookOpen className="w-3.5 h-3.5 text-zinc-800" />
                  README
                </button>

                <button
                  type="button"
                  className="inline-flex items-center gap-1.5 text-xs font-semibold text-zinc-950 group-hover:translate-x-0.5 transition-transform"
                >
                  Open Experiment
                  <ArrowRight className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          );
        })}
      </div>

      {/* Experiment Technical Documentation Modal */}
      {activeReadmeExpId && (
        <ExperimentReadmeModal
          expId={activeReadmeExpId}
          onClose={() => setActiveReadmeExpId(null)}
          onLaunch={(id) => {
            setActiveReadmeExpId(null);
            onSelectExperiment(id);
          }}
        />
      )}

    </div>
  );
}
