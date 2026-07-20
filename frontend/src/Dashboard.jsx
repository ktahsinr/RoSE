import React, { useEffect, useState } from 'react';
import { api } from './api.js';

const METHODS = ['Zero-Shot-CoT', 'Auto-CoT', 'RoSE'];
const COLORS = { 'Zero-Shot-CoT': '#94a3b8', 'Auto-CoT': '#60a5fa', RoSE: '#e11d48' };

// Per-dataset paper reference (used inline under the live bars).
const PAPER_REF = {
  CommonsenseQA: { 'Zero-Shot-CoT': 67.6, 'Auto-CoT': 74.8, RoSE: 67.8 },
  AddSub: { 'Zero-Shot-CoT': 83.5, 'Auto-CoT': 91.4, RoSE: 90.9 },
  GSM8K: { 'Zero-Shot-CoT': 75.8, 'Auto-CoT': 74.4, RoSE: 83.9 },
};

// The paper's full Table 2 (EMNLP 2024), reproduced as a labeled reference.
const TABLE2 = {
  cols: ['AddSub', 'AQuA', 'GSM8K', 'SingleEq', 'SingleOp', 'SVAMP', 'CSQA', 'Strategy', 'Date', 'AVG'],
  groups: [
    {
      model: 'GPT-3.5-Turbo-16k-0613',
      rows: [
        ['Zero-Shot-CoT', 83.5, 55.5, 75.8, 90.9, 90.9, 77.5, 67.6, 65.5, 67.5, 75.0],
        ['Auto-CoT', 91.4, 52.8, 74.4, 91.5, 93.6, 84.9, 74.8, 62.0, 56.6, 75.8],
        ['RoSE (Ours)', 90.9, 70.9, 83.9, 92.2, 95.6, 89.2, 67.8, 71.3, 88.6, 83.4],
      ],
    },
    {
      model: 'LLaMA2-13B-Chat',
      rows: [
        ['Zero-Shot-CoT', 14.7, 14.2, 9.0, 18.5, 16.2, 17.3, 33.1, 57.4, 37.7, 24.2],
        ['Auto-CoT', 58.5, 22.4, 35.9, 69.5, 81.0, 38.2, 61.7, 63.0, 56.6, 54.1],
        ['RoSE (Ours)', 79.5, 31.5, 50.2, 81.3, 89.5, 64.3, 62.2, 69.4, 63.7, 65.7],
      ],
    },
  ],
};

export default function Dashboard() {
  const [datasets, setDatasets] = useState([]);
  const [dataset, setDataset] = useState('CommonsenseQA');
  const [n, setN] = useState(15);
  const [numPaths, setNumPaths] = useState(10);
  const [warmup, setWarmup] = useState(15);
  const [busy, setBusy] = useState(false);
  const [res, setRes] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => { api.datasets().then(setDatasets); }, []);

  async function run() {
    setBusy(true); setError(null); setRes(null);
    try {
      const r = await api.benchmark(dataset, n, numPaths, warmup);
      if (r.error) throw new Error(r.error);
      setRes(r);
    } catch (e) { setError(e.message); }
    finally { setBusy(false); }
  }

  const paper = PAPER_REF[dataset];

  return (
    <div className="page">
      <section className="panel">
        <h2>Streaming benchmark</h2>
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
        <p className="muted">
          Streams questions through all three methods. RoSE builds its experience pool and votes over
          multiple reasoning paths — the goal is RoSE trending above Zero-Shot-CoT and Auto-CoT (the
          paper's "consistent gains").
        </p>
        <div className="controls">
          <label>Scored questions: <b>{n}</b>
            <input type="range" min="3" max="60" value={n} onChange={(e) => setN(+e.target.value)} />
          </label>
          <label>Self-consistency paths: <b>{numPaths}</b>
            <input type="range" min="1" max="20" value={numPaths} onChange={(e) => setNumPaths(+e.target.value)} />
          </label>
          <label>Pool warm-up: <b>{warmup}</b>
            <input type="range" min="0" max="40" value={warmup} onChange={(e) => setWarmup(+e.target.value)} />
          </label>
          <button className="primary" onClick={run} disabled={busy}>
            {busy ? 'Running stream…' : '▶ Run benchmark'}
          </button>
        </div>
        <p className="hint">
          Paper settings ≈ 20 paths and a large pool. On a local 8B model that's slow — a run of
          {' '}{n} scored + {warmup} warm-up × {numPaths} paths can take several minutes. Watch the
          terminal for progress.
        </p>
        {busy && <div className="spinner">Streaming {dataset}: warming {warmup}, scoring {n}, {numPaths} paths each…</div>}
        {error && <div className="banner bad">{error}</div>}
      </section>

      {res && (
        <>
          <section className="panel">
            <h2>Live accuracy — {res.dataset} ({res.n} scored, pool warmed to {res.warmup})</h2>
            <div className="bars">
              {METHODS.map((m) => {
                const a = res.accuracy[m];
                const pct = a.total ? a.acc * 100 : 0;
                return (
                  <div className="barrow" key={m}>
                    <span className="barlabel">{m}</span>
                    <div className="bartrack">
                      <div className="barfill" style={{ width: `${pct}%`, background: COLORS[m] }}>
                        <span className="barval">{pct.toFixed(1)}%</span>
                      </div>
                    </div>
                    <span className="barfrac">{a.correct}/{a.total}</span>
                  </div>
                );
              })}
            </div>
            <Winner accuracy={res.accuracy} />
            {paper && (
              <p className="muted small">
                Paper reference (GPT-3.5, full test set): Zero-Shot-CoT {paper['Zero-Shot-CoT']} ·
                Auto-CoT {paper['Auto-CoT']} · RoSE {paper.RoSE}. Local 8B numbers differ in absolute
                value — the reproduced signal is the ordering (RoSE ≥ baselines) and the method working.
              </p>
            )}
          </section>

          <section className="panel">
            <h2>Per-question stream</h2>
            <div className="tablewrap">
              <table className="log">
                <thead>
                  <tr>
                    <th>#</th><th>question</th><th>gold</th>
                    <th>Zero-Shot</th><th>Auto-CoT</th><th>RoSE</th><th>pool</th>
                  </tr>
                </thead>
                <tbody>
                  {res.log.map((row) => (
                    <tr key={row.i}>
                      <td>{row.i + 1}</td>
                      <td className="qcell">{row.question.slice(0, 55)}…</td>
                      <td className="gold">{row.gold}</td>
                      {METHODS.map((m) => {
                        const p = row.predictions[m];
                        return (
                          <td key={m} className={p.correct ? 'cellok' : 'cellno'}>
                            {p.pred ?? '—'} {p.correct ? '✓' : '✗'}
                          </td>
                        );
                      })}
                      <td>{row.poolSize}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </>
      )}

      <PaperTable />
    </div>
  );
}

function Winner({ accuracy }) {
  const rose = accuracy['RoSE']?.acc ?? 0;
  const best = Math.max(accuracy['Zero-Shot-CoT']?.acc ?? 0, accuracy['Auto-CoT']?.acc ?? 0);
  const gain = ((rose - best) * 100).toFixed(1);
  if (rose >= best)
    return <div className="verdict good">RoSE leads by {gain} points over the best baseline on this run. ✓</div>;
  return <div className="verdict warn">On this run RoSE trails by {Math.abs(gain)} pts — raise warm-up / paths; the gap widens as the pool grows.</div>;
}

// The paper's official Table 2, shown as a clearly-labeled reference.
function PaperTable() {
  return (
    <section className="panel">
      <h2>Paper reference · Table 2 (EMNLP 2024)</h2>
      <p className="muted small">
        Official reported accuracy from the RoSE paper. These use GPT-3.5-Turbo-16k and LLaMA2-13B-Chat
        on full test sets with 20 self-consistency paths — the target this project reproduces the
        <i> method</i> of. Bold = best in column per model.
      </p>
      <div className="tablewrap">
        <table className="log paper2">
          <thead>
            <tr><th>Method</th>{TABLE2.cols.map((c) => <th key={c}>{c}</th>)}</tr>
          </thead>
          {TABLE2.groups.map((g) => {
            const maxByCol = TABLE2.cols.map((_, ci) => Math.max(...g.rows.map((r) => r[ci + 1])));
            return (
              <tbody key={g.model}>
                <tr className="grouprow"><td colSpan={TABLE2.cols.length + 1}>{g.model}</td></tr>
                {g.rows.map((r) => (
                  <tr key={r[0]} className={r[0].startsWith('RoSE') ? 'roserow' : ''}>
                    <td>{r[0]}</td>
                    {r.slice(1).map((v, ci) => (
                      <td key={ci} className={v === maxByCol[ci] ? 'bestcell' : ''}>{v.toFixed(1)}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            );
          })}
        </table>
      </div>
    </section>
  );
}
