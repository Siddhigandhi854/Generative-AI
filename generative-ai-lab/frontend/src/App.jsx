import React, { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { SettingsModal } from './components/SettingsModal';
import { HomePage } from './pages/HomePage';

import { Exp1DatasetAnalysis } from './pages/experiments/Exp1DatasetAnalysis';
import { Exp2TensorOperations } from './pages/experiments/Exp2TensorOperations';
import { Exp3TransformerQA } from './pages/experiments/Exp3TransformerQA';
import { Exp4DiffusionImage } from './pages/experiments/Exp4DiffusionImage';
import { Exp5VariationalAutoencoder } from './pages/experiments/Exp5VariationalAutoencoder';
import { Exp6DCGANFaces } from './pages/experiments/Exp6DCGANFaces';
import { Exp7StyleGAN } from './pages/experiments/Exp7StyleGAN';
import { Exp8MusicGen } from './pages/experiments/Exp8MusicGen';
import { Exp9PersonalizedAssistant } from './pages/experiments/Exp9PersonalizedAssistant';
import { Exp10VideoGeneration } from './pages/experiments/Exp10VideoGeneration';

export function App() {
  const [currentExpId, setCurrentExpId] = useState(null);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [backendHealth, setBackendHealth] = useState({ healthy: false });

  // Handle URL Hash-based routing for bookmarking / browser back button
  useEffect(() => {
    const handleHashChange = () => {
      const hash = window.location.hash;
      const match = hash.match(/#\/exp\/(\d+)/);
      if (match) {
        setCurrentExpId(parseInt(match[1], 10));
      } else {
        setCurrentExpId(null);
      }
    };

    handleHashChange();
    window.addEventListener('hashchange', handleHashChange);
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, []);

  // Poll backend health once on mount
  useEffect(() => {
    checkHealth();
  }, []);

  const checkHealth = async () => {
    try {
      const res = await fetch('/api/health');
      if (res.ok) {
        const data = await res.json();
        setBackendHealth({ healthy: true, ...data });
      } else {
        setBackendHealth({ healthy: false });
      }
    } catch {
      setBackendHealth({ healthy: false });
    }
  };

  const navigateToExperiment = (id) => {
    setCurrentExpId(id);
    window.location.hash = `#/exp/${id}`;
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const navigateToHome = () => {
    setCurrentExpId(null);
    window.location.hash = '#/';
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <div className="min-h-screen bg-white text-zinc-900 flex flex-col font-sans">
      
      {/* Top Header */}
      <Header
        backendStatus={backendHealth}
        onOpenSettings={() => setIsSettingsOpen(true)}
        onNavigateHome={navigateToHome}
      />

      {/* Main Content Router */}
      <main className="flex-1 pb-16">
        {currentExpId === null && (
          <HomePage onSelectExperiment={navigateToExperiment} />
        )}

        {currentExpId === 1 && (
          <Exp1DatasetAnalysis onBack={navigateToHome} />
        )}

        {currentExpId === 2 && (
          <Exp2TensorOperations onBack={navigateToHome} />
        )}

        {currentExpId === 3 && (
          <Exp3TransformerQA
            onBack={navigateToHome}
            onOpenSettings={() => setIsSettingsOpen(true)}
          />
        )}

        {currentExpId === 4 && (
          <Exp4DiffusionImage
            onBack={navigateToHome}
            onOpenSettings={() => setIsSettingsOpen(true)}
          />
        )}

        {currentExpId === 5 && (
          <Exp5VariationalAutoencoder onBack={navigateToHome} />
        )}

        {currentExpId === 6 && (
          <Exp6DCGANFaces onBack={navigateToHome} />
        )}

        {currentExpId === 7 && (
          <Exp7StyleGAN onBack={navigateToHome} />
        )}

        {currentExpId === 8 && (
          <Exp8MusicGen onBack={navigateToHome} />
        )}

        {currentExpId === 9 && (
          <Exp9PersonalizedAssistant
            onBack={navigateToHome}
            onOpenSettings={() => setIsSettingsOpen(true)}
          />
        )}

        {currentExpId === 10 && (
          <Exp10VideoGeneration onBack={navigateToHome} />
        )}
      </main>

      {/* Minimalist Footer */}
      <footer className="border-t border-zinc-200 py-6 bg-zinc-50/50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between text-xs text-zinc-500 gap-3">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-zinc-900">Generative AI Lab</span>
          </div>
          <div>
            Guide Teacher: <strong className="text-zinc-900 font-semibold">Dr. Sachin Takmare</strong>
          </div>
        </div>
      </footer>

      {/* Global Settings Modal */}
      <SettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        onSave={() => checkHealth()}
      />

    </div>
  );
}

export default App;
