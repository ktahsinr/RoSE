import React, { useEffect, useState } from 'react';
import { api } from './api.js';

export default function DemoPage({ onPoolChange }) {
  const [datasets, setDatasets] = useState([]);
  const [dataset, setDataset] = useState('CommonsenseQA');
  const [questions, setQuestions] = useState([]);
  const [selectedId, setSelectedId] = useState('');
  const [numPaths, setNumPaths] = useState(10);
  const [compare, setCompare] = useState(true);
  const [busy, setBusy] = useState(false);
  const [busyLabel, setBusyLabel] = useState('');
  const [result, setResult] = useState(null);
  const [poolSize, setPoolSize] = useState(0);
  const [error, setError] = useState(null);

  // Load the dataset registry once.
  useEffect(() => { api.datasets().then(setDatasets); }, []);

  // (Re)load questions + pool whenever the selected dataset changes.
  useEffect(() => {
    setResult(null);
    api.dataset(dataset, 40).then((qs) => {
      setQuestions(qs);
      setSelectedId(qs[0]?.id || '');
    });
    api.pool(dataset).then((p) => setPoolSize(p.size));
  }, [dataset]);

  const meta = datasets.find((d) => d.key === dataset);
  const selected = questions.find((q) => q.id === selectedId);
  const isMC = selected?.type === 'multiple_choice';

  async function run() {
    setBusy(true); setBusyLabel('Orchestrating experiences & sampling reasoning paths…');
    setError(null); setResult(null);
    try {
      const r = await api.answer({ id: selectedId, numPaths, compareBaselines: compare });
      if (r.error) throw new Error(r.error);
      setResult(r);
      const p = await api.pool(dataset); setPoolSize(p.size); onPoolChange?.();
    } catch (e) { setError(e.message); }
    finally { setBusy(false); }
  }

  async function warmup() {
    setBusy(true); setBusyLabel('Warming the experience pool…'); setError(null);
    try {
      await api.warmup(dataset, 5, 5);
      const p = await api.pool(dataset); setPoolSize(p.size); onPoolChange?.();
    } catch (e) { setError(e.message); }
    finally { setBusy(false); }
  }

  async function resetPool() {
    await api.resetPool(dataset);
    const p = await api.pool(dataset); setPoolSize(p.size); onPoolChange?.();
  }

  return (
    <div className="page grid2">
      <section className="panel">
        <h2>1 · Choose a benchmark</h2>
        <div className="dspicker">
          {datasets.map((d) => (
            <button
              key={d.key}
              className={`dschip ${d.key === dataset ? 'active' : ''}`}
              onClick={() => setDataset(d.key)}
            >
              <b>{d.label}</b>
              <span>{d.kind} · {d.count}</span>
            </button>
          ))}
        </div>

        <h2>2 · Pick a question</h2>
        <select value={selectedId} onChange={(e) => setSelectedId(e.target.value)}>
          {questions.map((q) => (
            <option key={q.id} value={q.id}>{q.question.slice(0, 70)}</option>
          ))}
        </select>

        {selected && (
          <div className="qcard">
            <p className="qtext">{selected.question}</p>
            {isMC ? (
              <ul className="choices">
                {selected.choices.map((c) => (
                  <li key={c.label} className={c.label === selected.answer ? 'gold' : ''}>
                    <b>{c.label}</b> {c.text}
                    {c.label === selected.answer && <span className="tag">gold</span>}
                  </li>
                ))}
              </ul>
            ) : (
              <div className="numgold">
                gold answer: <b>{selected.answer}</b>
              </div>
            )}
          </div>
        )}

        <h2>3 · Settings</h2>
        <label className="row">
          Reasoning paths (self-consistency): <b>{numPaths}</b>
          <input type="range" min="1" max="20" value={numPaths}
            onChange={(e) => setNumPaths(+e.target.value)} />
        </label>
        <label className="row checkbox">
          <input type="checkbox" checked={compare} onChange={(e) => setCompare(e.target.checked)} />
          Compare against Zero-Shot-CoT & Auto-CoT
        </label>

        <div className="poolbar">
          <span>{dataset} pool: <b>{poolSize}</b></span>
          <button className="ghost" onClick={warmup} disabled={busy}>Warm up (+5)</button>
          <button className="ghost" onClick={resetPool} disabled={busy}>Reset</button>
        </div>

        <button className="primary big" onClick={run} disabled={busy || !selectedId}>
          {busy ? 'Running…' : '▶ Run RoSE'}
        </button>
        <p className="hint">
          The pool starts empty and is separate per benchmark. RoSE self-improves as you answer more
          questions — warm it up or run several to see orchestration kick in.
        </p>
        {busy && <div className="spinner">{busyLabel}</div>}
        {error && <div className="banner bad">{error}</div>}
      </section>

      <section className="panel">
        <h2>Result {meta && <span className="muted small">· {meta.label} ({meta.type === 'numeric' ? 'numeric' : 'multiple choice'})</span>}</h2>
        {!result && <p className="muted">Run a question to see RoSE's orchestrated reasoning.</p>}
        {result && <ResultView result={result} />}
      </section>
    </div>
  );
}

function verdict(correct) {
  if (correct == null) return <span className="tag">no gold</span>;
  return correct
    ? <span className="tag ok">✓ correct</span>
    : <span className="tag no">✗ wrong</span>;
}

function pred(item, p) {
  const mc = item.type === 'multiple_choice';
  return mc ? `(${p ?? '—'})` : (p ?? '—');
}

// Build one row per option. For multiple-choice we list EVERY choice (A–E) in
// order, including options that received zero votes, annotated with their text.
// For numeric there is no fixed option set, so we show the values that got votes,
// highest first.
function voteRows(item, rose) {
  const dist = rose.voteDistribution || {};
  if (item.type === 'multiple_choice' && item.choices) {
    return item.choices.map((c) => ({
      label: c.label,
      text: c.text,
      n: dist[c.label] || 0,
      win: c.label === rose.predicted,
    }));
  }
  return Object.entries(dist)
    .sort((a, b) => b[1] - a[1])
    .map(([label, n]) => ({ label, text: null, n, win: label === rose.predicted }));
}

function ResultView({ result }) {
  const { item, rose, zeroShot, autoCoT } = result;
  return (
    <div className="result">
      <div className="methodcard highlight">
        <div className="mhead">
          <h3>RoSE <span className="predbadge">→ {pred(item, rose.predicted)}</span></h3>
          {verdict(rose.correct)}
        </div>
        <div className="metrics">
          <Metric label="uncertainty" value={rose.uncertainty?.toFixed(3)} />
          <Metric label="complexity" value={rose.complexity} />
          <Metric label="paths" value={rose.numPaths} />
          <Metric label="pool used" value={rose.poolSizeBefore} />
        </div>

        <details open>
          <summary>Reasoning path</summary>
          <p className="reasoning">{rose.reasoning}</p>
        </details>

        <details open>
          <summary>Vote distribution — all {rose.numPaths} paths across every option</summary>
          <div className="votes">
            {voteRows(item, rose).map(({ label, text, n, win }) => (
              <div key={label} className={`voterow ${win ? 'winrow' : ''}`}>
                <span className="votelabel">{label}</span>
                <div className="votebar"><div style={{ width: `${(n / rose.numPaths) * 100}%` }} /></div>
                <span className="votecount">{n}</span>
                {text && <span className="votetext">{text}</span>}
              </div>
            ))}
            {rose.unparsed > 0 && (
              <div className="voterow unparsed">
                <span className="votelabel">?</span>
                <div className="votebar"><div style={{ width: `${(rose.unparsed / rose.numPaths) * 100}%` }} /></div>
                <span className="votecount">{rose.unparsed}</span>
                <span className="votetext">unparsed</span>
              </div>
            )}
          </div>
        </details>

        <details>
          <summary>Orchestrated experiences ({rose.demonstrations?.length || 0} buckets)</summary>
          {rose.demonstrations?.length ? (
            <table className="demos">
              <thead><tr><th>bucket</th><th>picked question</th><th>sim</th><th>unc</th><th>cplx</th></tr></thead>
              <tbody>
                {rose.demonstrations.map((b) => (
                  <tr key={b.index}>
                    <td>{b.index}</td>
                    <td className="qcell">{b.picked.question.slice(0, 55)}…</td>
                    <td>{b.picked.similarity.toFixed(2)}</td>
                    <td>{b.picked.uncertainty.toFixed(2)}</td>
                    <td>{b.picked.complexity}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : <p className="muted">Pool was empty — RoSE fell back to zero-shot for this question.</p>}
        </details>
      </div>

      {zeroShot && (
        <div className="methodcard">
          <div className="mhead"><h3>Zero-Shot-CoT <span className="predbadge">→ {pred(item, zeroShot.predicted)}</span></h3>{verdict(zeroShot.correct)}</div>
          <details><summary>Reasoning</summary><p className="reasoning">{zeroShot.reasoning}</p></details>
        </div>
      )}
      {autoCoT && (
        <div className="methodcard">
          <div className="mhead"><h3>Auto-CoT <span className="predbadge">→ {pred(item, autoCoT.predicted)}</span></h3>{verdict(autoCoT.correct)}</div>
          <p className="muted small">{autoCoT.demonstrations.length} clustered demos</p>
          <details><summary>Reasoning</summary><p className="reasoning">{autoCoT.reasoning}</p></details>
        </div>
      )}
    </div>
  );
}

function Metric({ label, value }) {
  return <div className="metric"><span className="mval">{value ?? '—'}</span><span className="mlabel">{label}</span></div>;
}
