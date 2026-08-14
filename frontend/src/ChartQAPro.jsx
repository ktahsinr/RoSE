import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { api } from './api.js';

// The vision extension runs offline on a GPU (Colab/Kaggle) and writes result
// files. This page reads those files: every accuracy, ablation delta and judge
// verdict shown here is computed by chartqapro/summarize.py through the
// backend, so the dashboard and the CLI tables always agree.

// Emphasis palette: RoSE carries the accent, everything else is context gray.
// Validated against the panel surface for contrast and CVD separation; every
// mark is also directly labeled, so colour is never the only encoding.
const ACCENT = '#e11d48';
const CONTEXT = '#7c8598';

const METHOD_LABEL = {
  zero_shot_cot: 'Zero-shot CoT · pool cold',
  rose_few_shot: 'RoSE few-shot · orchestrated',
};

const TYPE_LABEL = { factoid: 'Factoid', mcq: 'MCQ', unknown: 'Unknown' };

const VOTE_MODE_LABEL = {
  exact: 'Exact-string majority (Eq. 1–3)',
  cluster: 'Numeric tolerance clustering',
  weighted: 'Confidence-weighted vote',
};

// The documented deviations from the published algorithm. The three marked
// offline only change how the m per-path answers are combined, so they can be
// ablated from a finished run; the rest change what the model is shown or how
// it decodes and need their own run.
const EXTENSIONS = [
  ['mcq_permute_options', 'Rotate MCQ option order across the m paths so position bias cancels', false],
  ['numeric_vote_clustering', 'Group numeric answers within tolerance, take the median of the largest group', true],
  ['type_aware_retrieval', 'Retrieve demonstrations only from the same question type', false],
  ['drop_malformed_paths', 'Exclude paths that never emitted a "Final Answer:" line', true],
  ['greedy_first_path', 'Decode path 0 greedily (T=0) instead of sampling every path', false],
  ['chart_reading_scaffold', 'Axis-and-units-first reading scaffold in the prompt', false],
  ['mcq_demo_anchor_fix', 'Store MCQ demonstrations as "(B) 45%" so the letter stays anchored to a value', false],
  ['two_stage_reasoning', 'Describe the chart first, then answer from that description', false],
  ['visual_hybrid_retrieval', 'Blend CLIP image similarity into demonstration retrieval', false],
  ['confidence_weighted_vote', 'Weight each path in the vote by decoding and well-formedness', true],
];

const pct = (v) => `${Number(v ?? 0).toFixed(1)}%`;
const num = (v, d = 2) => (v === null || v === undefined || v === '' ? '—' : Number(v).toFixed(d));

export default function ChartQAPro() {
  const [runs, setRuns] = useState(null);
  const [run, setRun] = useState('');
  const [summary, setSummary] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  const loadRuns = useCallback(async (preferred) => {
    try {
      const r = await api.cqa.runs();
      if (r.error) throw new Error(r.error);
      setRuns(r.runs);
      setRun((cur) => {
        const want = preferred || cur;
        if (want && r.runs.some((x) => x.id === want)) return want;
        return r.runs[0]?.id || '';
      });
    } catch (e) {
      setError(e.message);
      setRuns([]);
    }
  }, []);

  useEffect(() => { loadRuns(); }, [loadRuns]);

  useEffect(() => {
    if (!run) { setSummary(null); return; }
    let stale = false;
    setBusy(true); setError(null); setSummary(null);
    api.cqa.summary(run)
      .then((s) => {
        if (stale) return;
        if (s.error) throw new Error(s.error);
        setSummary(s);
      })
      .catch((e) => !stale && setError(e.message))
      .finally(() => !stale && setBusy(false));
    return () => { stale = true; };
  }, [run]);

  const meta = runs?.find((r) => r.id === run);

  return (
    <div className="page">
      <RunPicker
        runs={runs}
        run={run}
        onPick={setRun}
        onReload={loadRuns}
        busy={busy}
      />

      {error && <div className="banner bad">{error}</div>}
      {busy && <div className="spinner">Re-scoring the run and recomputing the ablation…</div>}

      {runs && runs.length === 0 && !busy && <EmptyState />}

      {summary && (
        <>
          <RunConfig summary={summary} runMeta={meta} run={run} />
          <Breakdowns summary={summary} />
          <LearningCurve curve={summary.curve} />
          <Calibration summary={summary} />
          <Ablation ablation={summary.ablation} extensions={summary.extensions} />
          {summary.judge && <JudgePanel judge={summary.judge} file={summary.verdictsFile} />}
          <Browser run={run} summary={summary} />
        </>
      )}
    </div>
  );
}

// --------------------------------------------------------------------------
// Run selection + upload
// --------------------------------------------------------------------------

function RunPicker({ runs, run, onPick, onReload, busy }) {
  const [uploading, setUploading] = useState(null);
  const [note, setNote] = useState(null);
  const fileRef = useRef(null);

  // A run is up to three files (results, meta, verdicts). Classify each by
  // shape rather than by name — a downloaded file is often renamed.
  async function classify(file) {
    const text = await file.text();
    const name = file.name.toLowerCase();
    if (name.includes('meta')) return { kind: 'meta', text };
    if (name.includes('verdict') || text.slice(0, 8192).includes('"verdict"'))
      return { kind: 'verdicts', text };
    return { kind: 'results', text };
  }

  async function onFiles(e) {
    const files = [...(e.target.files || [])];
    if (!files.length) return;
    const base = (files.find((f) => !/meta|verdict/i.test(f.name)) || files[0]).name
      .replace(/\.json$/i, '');

    setUploading(`Uploading ${files.length} file${files.length > 1 ? 's' : ''}…`);
    setNote(null);
    let landed = null;
    try {
      for (const file of files) {
        const { kind, text } = await classify(file);
        setUploading(`Uploading ${file.name} (${kind})…`);
        const r = await api.cqa.upload(base, kind, text);
        if (r.error) throw new Error(`${file.name}: ${r.error}`);
        if (kind === 'results') landed = r.run;
      }
      setNote(`Imported ${files.length} file${files.length > 1 ? 's' : ''} as "${base}".`);
      await onReload(landed);
    } catch (err) {
      setNote(err.message);
    } finally {
      setUploading(null);
      if (fileRef.current) fileRef.current.value = '';
    }
  }

  async function remove() {
    const r = runs.find((x) => x.id === run);
    if (!r?.uploaded) return;
    if (!window.confirm(`Remove the uploaded run "${r.label}" and its files from chartqapro/results/uploads?`))
      return;
    const res = await api.cqa.remove(run);
    if (res.error) setNote(res.error);
    else await onReload();
  }

  const current = runs?.find((r) => r.id === run);

  return (
    <section className="panel">
      <h2>ChartQAPro run</h2>
      <p className="muted small">
        RoSE extended from text to vision: Qwen2.5-VL reading chart images, with the streaming
        experience pool, entropy uncertainty and complexity selection unchanged. Inference runs
        offline on a GPU — this reads the result files it produces.
      </p>

      <div className="runbar">
        <select value={run} onChange={(e) => onPick(e.target.value)} disabled={!runs?.length}>
          {!runs?.length && <option>no runs found</option>}
          {runs?.map((r) => (
            <option key={r.id} value={r.id}>
              {r.label}
              {r.uploaded ? ' · uploaded' : ''}
              {r.partial ? ' · checkpoint' : ''}
              {r.nResults ? ` · ${r.nResults} rows` : ''}
            </option>
          ))}
        </select>
        <button className="ghost" onClick={() => onReload()} disabled={busy}>↻ Rescan</button>
        <button className="ghost" onClick={() => fileRef.current?.click()} disabled={!!uploading}>
          ⬆ Import run
        </button>
        {current?.uploaded && (
          <button className="ghost danger" onClick={remove}>✕ Remove</button>
        )}
        <input
          ref={fileRef}
          type="file"
          accept=".json,application/json"
          multiple
          hidden
          onChange={onFiles}
        />
      </div>

      <p className="hint">
        Import accepts the run's <code>results.json</code>, <code>meta.json</code> and the judge's{' '}
        <code>verdicts.json</code> together — or drop them into <code>chartqapro/results/</code> and
        hit Rescan.
      </p>
      {uploading && <div className="spinner">{uploading}</div>}
      {note && <div className="banner warn">{note}</div>}
    </section>
  );
}

function EmptyState() {
  return (
    <section className="panel">
      <h2>No runs yet</h2>
      <ol className="steps">
        <li>Run <code>Notebook/RoSE_ChartQAPro_Notebook.ipynb</code> on a GPU (Colab / Kaggle T4).</li>
        <li>It writes <code>results/rose_factoid_mcq_results.json</code> and <code>results/meta.json</code>.</li>
        <li>Optionally re-score with the semantic judge:{' '}
          <code>python rescore_judge.py results/…json --judge ollama --out results/verdicts.json</code></li>
        <li>Import those files above, or copy them into <code>chartqapro/results/</code>.</li>
      </ol>
      <p className="muted small">
        Everything on this page is recomputed from the raw per-path generations logged in the
        results file, so a finished run can be re-scored and ablated without touching a GPU.
      </p>
    </section>
  );
}

// --------------------------------------------------------------------------
// Provenance + headline number
// --------------------------------------------------------------------------

function RunConfig({ summary, runMeta, run }) {
  const m = summary.meta;
  const exts = summary.extensions;
  const faithful = m?.paper_faithful;

  return (
    <section className="panel">
      <h2>Configuration & headline</h2>

      <div className="heroRow">
        <div className="hero">
          <span className="heroval">{pct(summary.overall.rescored.acc)}</span>
          <span className="herolabel">
            overall accuracy · {summary.overall.rescored.correct}/{summary.overall.rescored.total} re-scored
          </span>
        </div>
        <div className="metrics">
          <Metric label="questions" value={summary.n_rows} />
          <Metric label="errored" value={summary.n_errors} />
          <Metric label="paths / q" value={m?.m_paths ?? '—'} />
          <Metric label="k demos" value={m?.k_demonstrations ?? '—'} />
          <Metric label="λ" value={m?.lambda ?? '—'} />
          <Metric label="tolerance" value={m ? `${(m.numeric_tolerance * 100).toFixed(0)}%` : '—'} />
        </div>
      </div>

      {summary.overall.drift !== 0 && (
        <div className="banner warn">
          Re-scoring moved this run by {summary.overall.drift > 0 ? '+' : ''}{summary.overall.drift}{' '}
          questions ({pct(summary.overall.run.acc)} as written → {pct(summary.overall.rescored.acc)} now).
          The file keeps whatever <code>is_correct</code> was true at run time; the page shows what{' '}
          <code>scoring.py</code> says today.
        </div>
      )}

      {m ? (
        <div className="chips">
          <span className="chip"><b>model</b> {m.model}</span>
          <span className="chip"><b>embedder</b> {m.embed_model}</span>
          <span className="chip"><b>T</b> {m.temperature}</span>
          <span className="chip"><b>max tokens</b> {m.max_new_tokens}</span>
          {m.clip_model && <span className="chip"><b>clip</b> {m.clip_model}</span>}
          {m.visual_alpha != null && <span className="chip"><b>visual α</b> {m.visual_alpha}</span>}
          <span className="chip"><b>types</b> {(m.target_types || []).join(' + ')}</span>
          {m.finished_at && <span className="chip"><b>finished</b> {m.finished_at}</span>}
        </div>
      ) : (
        <div className="banner warn">
          No <code>meta.json</code> beside this run — the configuration that produced it is
          unrecorded, so treat the comparisons below with care.
        </div>
      )}

      <h2>Extensions beyond the paper</h2>
      {faithful ? (
        <p className="muted small">
          <span className="tag ok">PAPER-FAITHFUL</span> This run disabled every extension — RoSE
          exactly as published (with the implementation bugs fixed, which are corrections rather
          than extensions).
        </p>
      ) : (
        <p className="muted small">
          Each deviation is separately switchable, so a faithful baseline and the improved system can
          be reported side by side. <b>offline</b> marks the{' '}
          {EXTENSIONS.filter(([, , o]) => o).length} that can be ablated from this finished run; the
          others change the prompt, retrieval or decoding and need their own run.
        </p>
      )}
      <div className="extgrid">
        {EXTENSIONS.map(([key, desc, offline]) => {
          const on = exts ? Boolean(exts[key]) : null;
          return (
            <div key={key} className={`extrow ${on ? 'on' : on === null ? 'unknown' : 'off'}`}>
              <span className="extstate">{on === null ? '?' : on ? 'ON' : 'OFF'}</span>
              <div>
                <b>{key}</b>
                {offline && <span className="tag">offline-ablatable</span>}
                <p className="muted small">{desc}</p>
              </div>
            </div>
          );
        })}
      </div>

      <div className="runbar">
        <a className="ghost btnlink" href={api.cqa.csvUrl(run, false)} download>⤓ results CSV</a>
        <a className="ghost btnlink" href={api.cqa.csvUrl(run, true)} download>⤓ per-path CSV</a>
        {runMeta && (
          <span className="muted small">
            {(runMeta.bytes / 1e6).toFixed(1)} MB · {runMeta.id}
          </span>
        )}
      </div>
    </section>
  );
}

function Metric({ label, value }) {
  return (
    <div className="metric">
      <span className="mval">{value ?? '—'}</span>
      <span className="mlabel">{label}</span>
    </div>
  );
}

// --------------------------------------------------------------------------
// Accuracy breakdowns
// --------------------------------------------------------------------------

function Bars({ rows, labelOf, colorOf }) {
  return (
    <div className="bars compact">
      {rows.map((r) => {
        // A short bar cannot hold its own label — a 0% bar would clip it to
        // nothing. Below the threshold the value sits outside the fill.
        const inside = r.acc >= 12;
        return (
          <div className="barrow" key={r.key}>
            <span className="barlabel">{labelOf ? labelOf(r.key) : r.key}</span>
            <div className="bartrack">
              <div
                className="barfill"
                style={{ width: `${r.acc}%`, background: colorOf ? colorOf(r.key) : ACCENT }}
              >
                {inside && <span className="barval">{pct(r.acc)}</span>}
              </div>
              {!inside && <span className="barval outside">{pct(r.acc)}</span>}
            </div>
            <span className="barfrac">{r.correct}/{r.total}</span>
          </div>
        );
      })}
    </div>
  );
}

function Breakdowns({ summary }) {
  const methods = summary.by_method.filter((m) => m.key !== 'error');
  const zero = methods.find((m) => m.key === 'zero_shot_cot');
  const rose = methods.find((m) => m.key === 'rose_few_shot');
  const c = summary.confusion;

  return (
    <section className="panel">
      <h2>Accuracy by question type</h2>
      <Bars rows={summary.by_type} labelOf={(k) => TYPE_LABEL[k] || k} />

      <h2>Cold start vs orchestrated</h2>
      <Bars
        rows={methods}
        labelOf={(k) => METHOD_LABEL[k] || k}
        colorOf={(k) => (k === 'rose_few_shot' ? ACCENT : CONTEXT)}
      />
      {zero && rose ? (
        <p className="muted small">
          The first questions of a stream are answered zero-shot because the pool has fewer than{' '}
          <i>k</i> experiences; everything after is orchestrated. Those {zero.total} cold-start
          questions are far too few to be a fair baseline — for a real Zero-shot vs RoSE comparison,
          run the pipeline again with <code>PAPER_FAITHFUL</code> or the extensions toggled and
          compare the two runs.
        </p>
      ) : (
        <p className="muted small">This run stayed in a single phase, so there is nothing to compare here.</p>
      )}

      {summary.by_vote_mode?.length > 1 && (
        <>
          <h2>How the vote was decided</h2>
          <Bars
            rows={summary.by_vote_mode}
            labelOf={(k) => VOTE_MODE_LABEL[k] || k}
            colorOf={(k) => (k === 'exact' ? CONTEXT : ACCENT)}
          />
          <p className="muted small">
            Rows are split by the aggregation each answer actually went through. This is a
            breakdown, not a comparison — a question routed to clustering is usually a numeric one,
            so the gap mixes question difficulty with the extension's effect. The ablation below is
            the controlled version, re-running every configuration over the same rows.
          </p>
        </>
      )}

      <h2>Answerable vs unanswerable</h2>
      <Bars rows={summary.by_answerable} labelOf={(k) => (k === 'unanswerable' ? 'Unanswerable (gold)' : 'Answerable (gold)')} />
      <div className="metrics">
        <Metric label="said unanswerable, wasn't" value={c.answerable_said_unanswerable} />
        <Metric label="gave a value, was unanswerable" value={c.unanswerable_said_value} />
      </div>
      <p className="muted small">
        A large slice of ChartQAPro gold answers are literally "Unanswerable", and those two counts
        are how the abstention behaviour fails in each direction — a model that abstains too eagerly
        buys unanswerable accuracy with answerable accuracy.
      </p>
    </section>
  );
}

// --------------------------------------------------------------------------
// Learning curve — the streaming-experience claim, as one chart
// --------------------------------------------------------------------------

function LearningCurve({ curve }) {
  const pts = curve?.points || [];
  const [hover, setHover] = useState(null);
  const svgRef = useRef(null);

  const W = 940, H = 280;
  const PAD = { t: 18, r: 92, b: 36, l: 46 };

  const geom = useMemo(() => {
    if (pts.length < 2) return null;
    const last = pts[pts.length - 1].i || 1;
    const x = (i) => PAD.l + (i / last) * (W - PAD.l - PAD.r);
    const y = (v) => PAD.t + (1 - v / 100) * (H - PAD.t - PAD.b);
    const line = (key) =>
      pts.map((p, k) => `${k ? 'L' : 'M'}${x(p.i).toFixed(1)},${y(p[key]).toFixed(1)}`).join(' ');
    return { last, x, y, roll: line('roll_acc'), cum: line('cum_acc') };
  }, [pts]);

  if (!geom) return null;

  function onMove(e) {
    const box = svgRef.current.getBoundingClientRect();
    const px = ((e.clientX - box.left) / box.width) * W;
    let best = pts[0], bestD = Infinity;
    for (const p of pts) {
      const d = Math.abs(geom.x(p.i) - px);
      if (d < bestD) { bestD = d; best = p; }
    }
    setHover(best);
  }

  const final = pts[pts.length - 1];
  const switchAt = curve.phase_switch;

  return (
    <section className="panel">
      <h2>Accuracy as the experience pool grows</h2>
      <p className="muted small">
        Questions in the order they were streamed. The rolling line is a {curve.window}-question
        window; the cumulative line is the run's running mean. RoSE's claim is that the rolling line
        rises as the pool fills — a flat line means the orchestrated demonstrations are not helping.
      </p>

      <div className="chart">
        <svg
          ref={svgRef}
          viewBox={`0 0 ${W} ${H}`}
          className="curve"
          role="img"
          aria-label={`Rolling and cumulative accuracy over ${pts.length} streamed questions`}
          onMouseMove={onMove}
          onMouseLeave={() => setHover(null)}
        >
          {[0, 25, 50, 75, 100].map((v) => (
            <g key={v}>
              <line x1={PAD.l} x2={W - PAD.r} y1={geom.y(v)} y2={geom.y(v)} className="gridline" />
              <text x={PAD.l - 8} y={geom.y(v) + 4} className="axistext" textAnchor="end">{v}%</text>
            </g>
          ))}

          {[0, 0.25, 0.5, 0.75, 1].map((f) => {
            const i = Math.round(f * geom.last);
            return (
              <text key={f} x={geom.x(i)} y={H - 12} className="axistext" textAnchor="middle">{i}</text>
            );
          })}

          {switchAt != null && switchAt > 0 && (
            <g>
              <line
                x1={geom.x(switchAt)} x2={geom.x(switchAt)}
                y1={PAD.t} y2={H - PAD.b}
                className="markerline"
              />
              <text x={geom.x(switchAt) + 6} y={PAD.t + 12} className="axistext">
                pool ≥ k · orchestration starts
              </text>
            </g>
          )}

          <path d={geom.cum} fill="none" stroke={CONTEXT} strokeWidth="2" />
          <path d={geom.roll} fill="none" stroke={ACCENT} strokeWidth="2" />

          <text x={W - PAD.r + 8} y={geom.y(final.roll_acc) + 4} className="endlabel" fill={ACCENT}>
            rolling {pct(final.roll_acc)}
          </text>
          <text x={W - PAD.r + 8} y={geom.y(final.cum_acc) + 4} className="endlabel" fill={CONTEXT}>
            mean {pct(final.cum_acc)}
          </text>

          {hover && (
            <g>
              <line
                x1={geom.x(hover.i)} x2={geom.x(hover.i)}
                y1={PAD.t} y2={H - PAD.b}
                className="crosshair"
              />
              <circle cx={geom.x(hover.i)} cy={geom.y(hover.cum_acc)} r="5"
                      fill={CONTEXT} stroke="var(--panel)" strokeWidth="2" />
              <circle cx={geom.x(hover.i)} cy={geom.y(hover.roll_acc)} r="5"
                      fill={ACCENT} stroke="var(--panel)" strokeWidth="2" />
            </g>
          )}
        </svg>

        {hover && (
          <div
            className="tooltip"
            style={{
              left: `${(geom.x(hover.i) / W) * 100}%`,
              transform: geom.x(hover.i) > W * 0.6 ? 'translate(-105%, 0)' : 'translate(8px, 0)',
            }}
          >
            <b>question {hover.i + 1}</b>
            <span>rolling <b>{pct(hover.roll_acc)}</b></span>
            <span>mean <b>{pct(hover.cum_acc)}</b></span>
            <span>pool {hover.pool ?? '—'}</span>
            <span className={hover.correct ? 'ok' : 'no'}>
              this one {hover.correct ? 'correct' : 'wrong'}
            </span>
            <span className="muted">{METHOD_LABEL[hover.method] || hover.method}</span>
          </div>
        )}
      </div>

      <div className="legend">
        <span><i style={{ background: ACCENT }} /> rolling accuracy ({curve.window}-question window)</span>
        <span><i style={{ background: CONTEXT }} /> cumulative mean</span>
      </div>
    </section>
  );
}

// --------------------------------------------------------------------------
// Calibration
// --------------------------------------------------------------------------

function Calibration({ summary }) {
  const bins = summary.uncertainty_bins || [];
  const s = summary.metrics || {};
  if (!bins.length) return null;

  const monotone = bins.every((b, i) => i === 0 || b.acc <= bins[i - 1].acc + 1e-9);

  return (
    <section className="panel">
      <h2>Does the uncertainty signal mean anything?</h2>
      <p className="muted small">
        Uncertainty is the Shannon entropy over the m sampled answers (Eq. 1–3) — the same quantity
        RoSE uses to decide which experiences are worth reusing. If it is calibrated, accuracy should
        fall as it rises.
      </p>
      <Bars
        rows={bins.map((b, i) => ({ ...b, key: `${b.lo}–${b.hi}`, i }))}
        labelOf={(k) => `H ${k}`}
      />
      <div className={`verdict ${monotone ? 'good' : 'warn'}`}>
        {monotone
          ? 'Accuracy falls monotonically as entropy rises — the signal is calibrated on this run. ✓'
          : 'Accuracy does not fall monotonically with entropy on this run — with few paths per question the entropy estimate is coarse.'}
      </div>
      <div className="metrics">
        {['uncertainty', 'complexity', 'agreement'].map((k) =>
          s[k] ? (
            <div className="metric wide" key={k}>
              <span className="mval">{num(s[k].mean, 3)}</span>
              <span className="mlabel">mean {k}</span>
              <span className="muted small">median {num(s[k].median, 3)} · {num(s[k].min, 2)}–{num(s[k].max, 2)}</span>
            </div>
          ) : null
        )}
      </div>
    </section>
  );
}

// --------------------------------------------------------------------------
// Offline aggregation ablation
// --------------------------------------------------------------------------

function Ablation({ ablation, extensions }) {
  const [group, setGroup] = useState('all');
  if (!ablation?.length) {
    return (
      <section className="panel">
        <h2>Aggregation ablation</h2>
        <p className="muted">
          This run has no per-path logging, so the vote cannot be recomputed under other
          configurations. Re-run with the current pipeline — it records every path's raw generation.
        </p>
      </section>
    );
  }

  const g = ablation.find((x) => x.label === group) || ablation[0];
  const best = Math.max(...g.configs.map((c) => c.acc));

  return (
    <section className="panel">
      <h2>Aggregation ablation · recomputed offline</h2>
      <p className="muted small">
        The {EXTENSIONS.filter(([, , o]) => o).length} aggregation extensions only change how the m
        per-path answers are combined, so they can be ablated from this finished run for free — no
        GPU, no re-inference. The vote is recomputed from the logged paths under all{' '}
        {g.configs.length} configurations and re-scored.
      </p>

      <div className="dspicker">
        {ablation.map((x) => (
          <button
            key={x.label}
            className={`dschip ${x.label === group ? 'active' : ''}`}
            onClick={() => setGroup(x.label)}
          >
            <b>{TYPE_LABEL[x.label] || x.label}</b>
            <span>{x.n} rows</span>
          </button>
        ))}
      </div>

      <div className="tablewrap">
        <table className="log">
          <thead>
            <tr><th>configuration</th><th>accuracy</th><th>correct</th><th>Δ vs paper</th></tr>
          </thead>
          <tbody>
            {g.configs.map((c, i) => (
              <tr key={c.name} className={c.acc === best ? 'bestrow' : ''}>
                <td>{c.name}</td>
                <td>{pct(c.acc)}</td>
                <td className="muted">{c.correct}/{c.total}</td>
                <td className={i === 0 ? 'muted' : c.delta > 0 ? 'cellok' : c.delta < 0 ? 'cellno' : 'muted'}>
                  {i === 0 ? 'baseline' : `${c.delta > 0 ? '+' : ''}${c.delta.toFixed(1)} pp`}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {g.skipped > 0 && (
        <div className="banner warn">
          {g.skipped} rows had no per-path log; their stored prediction was reused unchanged in every
          configuration, so the deltas understate the real effect.
        </div>
      )}

      <p className="hint">
        Not ablatable here:{' '}
        {EXTENSIONS.filter(([, , offline]) => !offline).map(([k]) => k).join(', ')} — each changes
        what the model is shown or how it decodes, so each needs its own run.
        {extensions && (
          <> This file was produced with{' '}
            <b>{Object.entries(extensions).filter(([, v]) => v).map(([k]) => k).join(', ') || 'no extensions'}</b>{' '}
            active, which is baked into the generations above.</>
        )}
      </p>
    </section>
  );
}

// --------------------------------------------------------------------------
// Semantic judge
// --------------------------------------------------------------------------

function JudgePanel({ judge, file }) {
  const delta = judge.judge_acc.acc - judge.rule_acc.acc;
  return (
    <section className="panel">
      <h2>Semantic judge · {file}</h2>
      <p className="muted small">
        String matching under-counts answers that are right but written differently ("107,995" vs
        107995). The judge re-scores for meaning. It saw {judge.judged} of the run's questions
        ({judge.coverage}%), so these numbers describe that subset, not the whole run.
      </p>

      <Bars
        rows={[
          { key: 'rule', ...judge.rule_acc },
          { key: 'judge', ...judge.judge_acc },
        ]}
        labelOf={(k) => (k === 'rule' ? 'scoring.py rules' : 'semantic judge')}
        colorOf={(k) => (k === 'judge' ? ACCENT : CONTEXT)}
      />
      <div className={`verdict ${delta >= 0 ? 'good' : 'warn'}`}>
        The judge scores this subset {delta >= 0 ? '+' : ''}{delta.toFixed(1)} points against the
        rules, and agrees with them on {judge.agreed} of {judge.judged} questions.
      </div>

      <div className="grid2 tight">
        <CategoryTable title={`Recovered by the judge (${judge.recovered.reduce((a, b) => a + b.n, 0)})`}
                       rows={judge.recovered} empty="Nothing recovered — the rules caught everything the judge accepts." />
        <CategoryTable title={`Judge rejects what the rules accepted (${judge.still_wrong.reduce((a, b) => a + b.n, 0)})`}
                       rows={judge.still_wrong} empty="Nothing rejected — no false credit from the rules." />
      </div>
    </section>
  );
}

function CategoryTable({ title, rows, empty }) {
  return (
    <div>
      <h3 className="subhead">{title}</h3>
      {rows.length ? (
        <table className="log">
          <thead><tr><th>category</th><th>n</th></tr></thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.category}><td>{r.category}</td><td>{r.n}</td></tr>
            ))}
          </tbody>
        </table>
      ) : <p className="muted small">{empty}</p>}
    </div>
  );
}

// --------------------------------------------------------------------------
// Per-question browser
// --------------------------------------------------------------------------

const PAGE = 25;

function Browser({ run, summary }) {
  const [type, setType] = useState('all');
  const [only, setOnly] = useState('');
  const [q, setQ] = useState('');
  const [page, setPage] = useState(0);
  const [data, setData] = useState(null);
  const [busy, setBusy] = useState(false);
  const [openId, setOpenId] = useState(null);

  useEffect(() => { setPage(0); }, [type, only, q, run]);

  useEffect(() => {
    let stale = false;
    setBusy(true);
    const t = setTimeout(() => {
      api.cqa.rows({ run, type, only, q, offset: page * PAGE, limit: PAGE })
        .then((d) => !stale && setData(d.error ? { total: 0, rows: [], error: d.error } : d))
        .finally(() => !stale && setBusy(false));
    }, q ? 250 : 0);   // debounce only the free-text search
    return () => { stale = true; clearTimeout(t); };
  }, [run, type, only, q, page]);

  const types = ['all', ...summary.by_type.map((t) => t.key)];
  const total = data?.total ?? 0;
  const pages = Math.max(1, Math.ceil(total / PAGE));

  return (
    <section className="panel">
      <h2>Every question</h2>
      <div className="filters">
        <label>type
          <select value={type} onChange={(e) => setType(e.target.value)}>
            {types.map((t) => <option key={t} value={t}>{TYPE_LABEL[t] || t}</option>)}
          </select>
        </label>
        <label>show
          <select value={only} onChange={(e) => setOnly(e.target.value)}>
            <option value="">all</option>
            <option value="wrong">wrong only</option>
            <option value="correct">correct only</option>
            <option value="disagree">scorers disagree</option>
            <option value="errors">errored</option>
          </select>
        </label>
        <label className="grow">search
          <input type="text" value={q} placeholder="question text or id"
                 onChange={(e) => setQ(e.target.value)} />
        </label>
        <span className="muted small">{total} matching</span>
      </div>

      {data?.error && <div className="banner bad">{data.error}</div>}

      <div className="tablewrap">
        <table className="log rowtable">
          <thead>
            <tr>
              <th>id</th><th>type</th><th>question</th><th>gold</th><th>prediction</th>
              <th>rule</th><th>judge</th><th>u</th><th>c</th><th>agree</th><th>pool</th>
            </tr>
          </thead>
          <tbody>
            {(data?.rows || []).map((r) => (
              <React.Fragment key={r.id}>
                <tr
                  className={`clickable ${openId === r.id ? 'openrow' : ''}`}
                  onClick={() => setOpenId(openId === r.id ? null : r.id)}
                >
                  <td className="muted">{r.id}</td>
                  <td>{r.question_type}</td>
                  <td className="qcell">{String(r.question).slice(0, 60)}…</td>
                  <td className="gold">{String(r.ground_truth).slice(0, 24)}</td>
                  <td className={r.is_correct ? 'cellok' : 'cellno'}>
                    {String(r.prediction).slice(0, 24) || '—'}
                  </td>
                  <td className={r.is_correct ? 'cellok' : 'cellno'}>{r.is_correct ? '✓' : '✗'}</td>
                  <td className={r.judge_correct == null ? 'muted' : r.judge_correct ? 'cellok' : 'cellno'}>
                    {r.judge_correct == null ? '—' : r.judge_correct ? '✓' : '✗'}
                  </td>
                  <td>{num(r.uncertainty, 2)}</td>
                  <td>{num(r.complexity, 1)}</td>
                  <td>{num(r.agreement, 2)}</td>
                  <td className="muted">{r.pool_size ?? '—'}</td>
                </tr>
                {openId === r.id && (
                  <tr className="detailrow">
                    <td colSpan={11}><RowDetail run={run} id={r.id} row={r} /></td>
                  </tr>
                )}
              </React.Fragment>
            ))}
            {!busy && !data?.rows?.length && (
              <tr><td colSpan={11} className="muted">No questions match these filters.</td></tr>
            )}
          </tbody>
        </table>
      </div>

      <div className="pager">
        <button className="ghost" disabled={page === 0 || busy} onClick={() => setPage((p) => p - 1)}>← prev</button>
        <span className="muted small">page {page + 1} of {pages}</span>
        <button className="ghost" disabled={page + 1 >= pages || busy} onClick={() => setPage((p) => p + 1)}>next →</button>
        {busy && <span className="muted small">loading…</span>}
      </div>
      <p className="hint">Click a question to see every sampled reasoning path and its raw generation.</p>
    </section>
  );
}

function RowDetail({ run, id, row }) {
  const [full, setFull] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let stale = false;
    setFull(null); setError(null);
    api.cqa.row(run, id)
      .then((d) => !stale && (d.error ? setError(d.error) : setFull(d)))
      .catch((e) => !stale && setError(e.message));
    return () => { stale = true; };
  }, [run, id]);

  if (error) return <div className="banner bad">{error}</div>;
  if (!full) return <p className="muted small">loading paths…</p>;

  return (
    <div className="detail">
      <p className="qtext">{full.question}</p>

      {full.options_shown?.length > 0 && (
        <ul className="choices">
          {full.options_shown.map((o, i) => (
            <li key={i}><b>{String.fromCharCode(65 + i)}</b> {o}</li>
          ))}
        </ul>
      )}

      <div className="metrics">
        <Metric label="gold" value={String(full.ground_truth)} />
        <Metric label="prediction" value={String(full.prediction) || '—'} />
        <Metric label="normalized" value={String(full.prediction_norm) || '—'} />
        <Metric label="method" value={full.method} />
        <Metric label="demos used" value={full.n_demos} />
        <Metric label="pool" value={full.pool_size} />
        {full.vote_mode && <Metric label="vote mode" value={full.vote_mode} />}
        {full.paths_voting != null && (
          <Metric label="paths voting" value={`${full.paths_voting}/${full.paths.length}`} />
        )}
        {full.n_groups != null && <Metric label="answer groups" value={full.n_groups} />}
      </div>

      {full.chart_description && (
        <details>
          <summary>Stage 1 — the model's description of the chart (two_stage_reasoning)</summary>
          <p className="reasoning">{full.chart_description}</p>
        </details>
      )}

      {row.judge_category && (
        <p className="muted small">
          judge: <span className={row.judge_correct ? 'tag ok' : 'tag no'}>{row.judge_category}</span>{' '}
          {row.judge_reason}
        </p>
      )}

      <h3 className="subhead">{full.paths.length} reasoning paths</h3>
      <div className="paths">
        {full.paths.map((p) => (
          <details key={p.i}>
            <summary>
              path {p.i} · {p.decoding} · answer <b>{p.extracted || '—'}</b>
              {!p.well_formed && <span className="tag no">no Final Answer line</span>}
            </summary>
            <p className="reasoning">{p.raw}</p>
          </details>
        ))}
      </div>

      {full.best_rationale && (
        <details>
          <summary>Stored as the pool experience (argmax CountSteps, Eq. 5)</summary>
          <p className="reasoning">{full.best_rationale}</p>
        </details>
      )}

      {full.option_perms && (
        <p className="muted small">
          option order per path: {full.option_perms.map((p) => `[${p.join(' ')}]`).join(' ')} — MCQ
          options are rotated across paths so position bias cancels in the vote.
        </p>
      )}
    </div>
  );
}
