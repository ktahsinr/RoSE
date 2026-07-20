import React, { useEffect, useState } from 'react';
import { api } from './api.js';
import DemoPage from './DemoPage.jsx';
import Dashboard from './Dashboard.jsx';

export default function App() {
  const [tab, setTab] = useState('demo');
  const [health, setHealth] = useState(null);

  const refreshHealth = () => api.health().then(setHealth).catch(() => setHealth(null));
  useEffect(() => { refreshHealth(); }, []);

  const ready = health?.ollama?.ok && health?.ollama?.hasChat && health?.ollama?.hasEmbed;

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="logo">🌹</span>
          <div>
            <h1>RoSE</h1>
            <p className="tagline">Reasoning with Orchestrated Streaming Experiences · EMNLP 2024</p>
          </div>
        </div>
        <nav className="tabs">
          <button className={tab === 'demo' ? 'active' : ''} onClick={() => setTab('demo')}>
            Interactive Demo
          </button>
          <button className={tab === 'dash' ? 'active' : ''} onClick={() => setTab('dash')}>
            Benchmark Dashboard
          </button>
        </nav>
        <StatusPill health={health} ready={ready} onRefresh={refreshHealth} />
      </header>

      {!ready && <SetupBanner health={health} />}

      <main>
        {tab === 'demo' ? (
          <DemoPage onPoolChange={refreshHealth} />
        ) : (
          <Dashboard />
        )}
      </main>

      <footer>
        Local implementation · Ollama ({health?.ollama?.chatModel || '…'}) ·{' '}
        {health?.datasets
          ? Object.entries(health.datasets).map(([k, v]) => `${k} (${v})`).join(' · ')
          : ''}
      </footer>
    </div>
  );
}

function StatusPill({ health, ready, onRefresh }) {
  let label = 'checking…', cls = 'warn';
  if (health && !health.ollama?.ok) { label = 'Ollama offline'; cls = 'bad'; }
  else if (health && !ready) { label = 'models missing'; cls = 'warn'; }
  else if (ready) {
    const total = Object.values(health.pools || {}).reduce((a, b) => a + b, 0);
    label = `pools: ${total}`; cls = 'good';
  }
  return (
    <button className={`pill ${cls}`} onClick={onRefresh} title="click to refresh">
      ● {label}
    </button>
  );
}

function SetupBanner({ health }) {
  if (!health) return null;
  if (!health.ollama?.ok)
    return (
      <div className="banner bad">
        Ollama isn't reachable. Start it with <code>brew services start ollama</code>.
      </div>
    );
  const missing = [];
  if (!health.ollama.hasChat) missing.push(health.ollama.chatModel);
  if (!health.ollama.hasEmbed) missing.push(health.ollama.embedModel);
  if (missing.length)
    return (
      <div className="banner warn">
        Still downloading / missing models: {missing.map((m) => <code key={m}>{m}</code>)}. Pull with{' '}
        <code>ollama pull {missing[0]}</code>.
      </div>
    );
  return null;
}
