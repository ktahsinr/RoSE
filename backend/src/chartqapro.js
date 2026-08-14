// ---------------------------------------------------------------------------
// ChartQAPro bridge
// ---------------------------------------------------------------------------
// The vision extension runs on a GPU elsewhere (Colab/Kaggle) and drops JSON
// results files into chartqapro/results/. This module serves those files to
// the frontend.
//
// It deliberately does NOT reimplement scoring in JS. Every number it returns
// comes from shelling out to chartqapro/summarize.py, which imports the same
// scoring.py the run itself used — so the dashboard, the CSV export and the
// terminal tables can never disagree about the same file. The cost is a Python
// process per uncached request; results are cached on (path, mtime).
// ---------------------------------------------------------------------------

import { spawn } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(__dirname, '..', '..');
const CQA_DIR = path.join(ROOT, 'chartqapro');
const RESULTS_DIR = path.join(CQA_DIR, 'results');
const UPLOAD_DIR = path.join(RESULTS_DIR, 'uploads');

const SUMMARIZE = path.join(CQA_DIR, 'summarize.py');
const EXPORT_CSV = path.join(CQA_DIR, 'export_csv.py');

// ----------------------------- python ------------------------------------

// scoring.py/summarize.py are stdlib-only, so any Python 3 works. Prefer the
// project venv if one was created, then whatever is on PATH.
function findPython() {
  if (process.env.ROSE_PYTHON) return process.env.ROSE_PYTHON;
  const venv = path.join(ROOT, '.venv', 'bin', 'python3');
  if (fs.existsSync(venv)) return venv;
  return 'python3';
}
const PYTHON = findPython();

function runPython(args, { json = true, timeout = 120_000 } = {}) {
  return new Promise((resolve, reject) => {
    const proc = spawn(PYTHON, args, { cwd: CQA_DIR });
    let out = '', err = '';
    const timer = setTimeout(() => {
      proc.kill('SIGKILL');
      reject(new Error(`python timed out after ${timeout / 1000}s`));
    }, timeout);

    proc.stdout.on('data', (d) => { out += d; });
    proc.stderr.on('data', (d) => { err += d; });
    proc.on('error', (e) =>
      reject(new Error(
        e.code === 'ENOENT'
          ? `Python not found (tried "${PYTHON}"). Set ROSE_PYTHON to a python3 binary.`
          : e.message
      ))
    );
    proc.on('close', (code) => {
      clearTimeout(timer);
      if (code !== 0) return reject(new Error(err.trim() || `python exited ${code}`));
      if (!json) return resolve(out);
      try {
        resolve(JSON.parse(out));
      } catch {
        reject(new Error(`python returned non-JSON: ${out.slice(0, 300)}`));
      }
    });
  });
}

// ------------------------------ caching -----------------------------------

// Summaries re-score every row and run the ablation four times, so they are
// worth caching — but only for the exact bytes they were computed from.
const cache = new Map();
const MAX_CACHE = 24;

function stamp(...files) {
  return files
    .filter(Boolean)
    .map((f) => {
      try { return `${f}@${fs.statSync(f).mtimeMs}`; } catch { return `${f}@0`; }
    })
    .join('|');
}

async function cached(key, produce) {
  if (cache.has(key)) return cache.get(key);
  const value = await produce();
  cache.set(key, value);
  if (cache.size > MAX_CACHE) cache.delete(cache.keys().next().value);
  return value;
}

// --------------------------- run discovery --------------------------------

const isMetaName = (n) => n === 'meta.json' || n.endsWith('.meta.json');
const isVerdictName = (n) => /verdict/i.test(n);

// A verdicts file looks exactly like a results file apart from the per-row
// "verdict" object, so peek at the head rather than trusting the filename.
function looksLikeVerdicts(file) {
  if (isVerdictName(path.basename(file))) return true;
  try {
    const fd = fs.openSync(file, 'r');
    const buf = Buffer.alloc(8192);
    const n = fs.readSync(fd, buf, 0, 8192, 0);
    fs.closeSync(fd);
    return buf.slice(0, n).includes('"verdict"');
  } catch {
    return false;
  }
}

function walk(dir, depth = 0, acc = []) {
  if (depth > 3) return acc;
  let entries;
  try { entries = fs.readdirSync(dir, { withFileTypes: true }); } catch { return acc; }
  for (const e of entries) {
    const full = path.join(dir, e.name);
    if (e.isDirectory()) walk(full, depth + 1, acc);
    else if (e.isFile() && e.name.endsWith('.json')) acc.push(full);
  }
  return acc;
}

export function listRuns() {
  const files = walk(RESULTS_DIR);
  const runs = [];

  for (const file of files) {
    const name = path.basename(file);
    if (isMetaName(name) || looksLikeVerdicts(file)) continue;

    const dir = path.dirname(file);
    const metaPath = path.join(dir, 'meta.json');
    const verdicts = files.find(
      (f) => path.dirname(f) === dir && f !== file && looksLikeVerdicts(f)
    );

    let meta = null;
    if (fs.existsSync(metaPath)) {
      try { meta = JSON.parse(fs.readFileSync(metaPath, 'utf8')); } catch { /* unreadable */ }
    }

    const st = fs.statSync(file);
    runs.push({
      id: path.relative(RESULTS_DIR, file).split(path.sep).join('/'),
      label: name.replace(/\.json$/, ''),
      folder: path.relative(RESULTS_DIR, dir).split(path.sep).join('/') || '.',
      uploaded: file.startsWith(UPLOAD_DIR + path.sep),
      partial: /checkpoint/i.test(name),
      bytes: st.size,
      modified: st.mtimeMs,
      hasMeta: Boolean(meta),
      hasVerdicts: Boolean(verdicts),
      model: meta?.model || null,
      mPaths: meta?.m_paths ?? null,
      paperFaithful: meta?.paper_faithful ?? null,
      nResults: meta?.n_results ?? null,
      finishedAt: meta?.finished_at || null,
    });
  }

  // Newest first — the run someone just finished is the one they want.
  return runs.sort((a, b) => b.modified - a.modified);
}

// Resolve a client-supplied run id to a path INSIDE the results directory.
// Rejects anything that escapes it, so a crafted id cannot read arbitrary files.
function resolveRun(id) {
  if (!id || typeof id !== 'string') throw new Error('missing run id');
  const abs = path.resolve(RESULTS_DIR, id);
  if (abs !== RESULTS_DIR && !abs.startsWith(RESULTS_DIR + path.sep))
    throw new Error('run id outside the results directory');
  if (!abs.endsWith('.json')) throw new Error('run must be a .json file');
  if (!fs.existsSync(abs) || !fs.statSync(abs).isFile())
    throw new Error(`no such run: ${id}`);
  return abs;
}

// The judge verdicts sitting beside a run, if any.
function verdictsFor(runPath) {
  const dir = path.dirname(runPath);
  let entries;
  try { entries = fs.readdirSync(dir); } catch { return null; }
  const hit = entries
    .map((n) => path.join(dir, n))
    .find((f) => f !== runPath && f.endsWith('.json') &&
                 !isMetaName(path.basename(f)) && looksLikeVerdicts(f));
  return hit || null;
}

// ------------------------------- routes -----------------------------------

const clean = (v) => (typeof v === 'string' && v.trim() ? v.trim() : null);
const slug = (s) =>
  (s || 'upload').toLowerCase().replace(/[^a-z0-9._-]+/g, '-')
    .replace(/^-+|-+$/g, '').slice(0, 60) || 'upload';

export function register(app, express) {
  fs.mkdirSync(UPLOAD_DIR, { recursive: true });

  app.get('/api/chartqapro/runs', (req, res) => {
    try {
      res.json({ runs: listRuns(), python: PYTHON, resultsDir: RESULTS_DIR });
    } catch (e) {
      res.status(500).json({ error: e.message });
    }
  });

  // Full statistics for one run: accuracy breakdowns, the learning curve,
  // the offline aggregation ablation and any judge verdicts.
  app.get('/api/chartqapro/summary', async (req, res) => {
    try {
      const run = resolveRun(req.query.run);
      const verdicts = verdictsFor(run);
      const key = `summary:${stamp(run, verdicts)}`;
      const data = await cached(key, () =>
        runPython([SUMMARIZE, run, ...(verdicts ? ['--verdicts', verdicts] : [])])
      );
      res.json({ ...data, run: req.query.run, verdictsFile: verdicts ? path.basename(verdicts) : null });
    } catch (e) {
      res.status(400).json({ error: e.message });
    }
  });

  // Paginated per-question browsing with filters.
  app.get('/api/chartqapro/rows', async (req, res) => {
    try {
      const run = resolveRun(req.query.run);
      const verdicts = verdictsFor(run);
      const offset = Math.max(parseInt(req.query.offset) || 0, 0);
      const limit = Math.min(Math.max(parseInt(req.query.limit) || 25, 1), 200);
      const args = [SUMMARIZE, run, '--rows', '--offset', String(offset), '--limit', String(limit)];
      if (clean(req.query.type)) args.push('--type', req.query.type.trim());
      if (clean(req.query.only)) args.push('--only', req.query.only.trim());
      if (clean(req.query.q)) args.push('--q', req.query.q.trim());
      if (verdicts) args.push('--verdicts', verdicts);
      const key = `rows:${stamp(run, verdicts)}:${args.slice(2).join(' ')}`;
      res.json(await cached(key, () => runPython(args)));
    } catch (e) {
      res.status(400).json({ error: e.message });
    }
  });

  // One question in full: every sampled reasoning path and its raw generation.
  app.get('/api/chartqapro/row', async (req, res) => {
    try {
      const run = resolveRun(req.query.run);
      const verdicts = verdictsFor(run);
      const id = clean(req.query.id);
      if (!id) throw new Error('missing row id');
      const args = [SUMMARIZE, run, '--row-id', id];
      if (verdicts) args.push('--verdicts', verdicts);
      res.json(await cached(`row:${stamp(run, verdicts)}:${id}`, () => runPython(args)));
    } catch (e) {
      res.status(400).json({ error: e.message });
    }
  });

  // Hand the CSV export straight to the browser — same script the CLI uses.
  app.get('/api/chartqapro/export.csv', async (req, res) => {
    let tmp;
    try {
      const run = resolveRun(req.query.run);
      const verdicts = verdictsFor(run);
      const perPath = req.query.perPath === '1' || req.query.perPath === 'true';
      tmp = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'rose-csv-')), 'export.csv');

      const args = [EXPORT_CSV, run, '-o', tmp];
      if (verdicts) args.push('--verdicts', verdicts);
      if (perPath) args.push('--per-path');
      await runPython(args, { json: false });

      const file = perPath ? tmp.replace(/\.csv$/, '_per_path.csv') : tmp;
      const base = path.basename(run, '.json') + (perPath ? '_per_path' : '') + '.csv';
      res.setHeader('Content-Type', 'text/csv; charset=utf-8');
      res.setHeader('Content-Disposition', `attachment; filename="${base}"`);
      fs.createReadStream(file)
        .on('close', () => fs.rm(path.dirname(tmp), { recursive: true, force: true }, () => {}))
        .pipe(res);
    } catch (e) {
      if (tmp) fs.rm(path.dirname(tmp), { recursive: true, force: true }, () => {});
      res.status(400).json({ error: e.message });
    }
  });

  // Bring a Colab/Kaggle run into the dashboard. The raw file is the body —
  // results files carry every generation and get large, hence the big limit.
  app.post(
    '/api/chartqapro/upload',
    express.text({ limit: '256mb', type: '*/*' }),
    (req, res) => {
      try {
        const kind = clean(req.query.kind) || 'results';
        if (!['results', 'meta', 'verdicts'].includes(kind))
          throw new Error('kind must be results, meta or verdicts');

        const name = slug(clean(req.query.name) || 'run');
        const body = req.body;
        if (!body || !body.length) throw new Error('empty body');

        let parsed;
        try { parsed = JSON.parse(body); } catch { throw new Error('body is not valid JSON'); }
        if (kind !== 'meta' && !Array.isArray(parsed) && !Array.isArray(parsed?.results))
          throw new Error('expected a JSON array of result rows');

        const dir = path.join(UPLOAD_DIR, name);
        fs.mkdirSync(dir, { recursive: true });
        const file = path.join(dir, `${kind === 'results' ? name : kind}.json`);
        fs.writeFileSync(file, body);

        res.json({
          ok: true,
          kind,
          run: path.relative(RESULTS_DIR, file).split(path.sep).join('/'),
          rows: Array.isArray(parsed) ? parsed.length : undefined,
        });
      } catch (e) {
        res.status(400).json({ error: e.message });
      }
    }
  );

  // Remove an uploaded run. Scoped to the uploads folder — a run that was
  // produced on this machine is never touched.
  app.delete('/api/chartqapro/upload', (req, res) => {
    try {
      const abs = resolveRun(req.query.run);
      if (!abs.startsWith(UPLOAD_DIR + path.sep))
        throw new Error('only uploaded runs can be removed');
      fs.rmSync(path.dirname(abs), { recursive: true, force: true });
      res.json({ ok: true });
    } catch (e) {
      res.status(400).json({ error: e.message });
    }
  });
}

export const info = () => {
  const runs = listRuns();
  return { runs: runs.length, resultsDir: RESULTS_DIR, python: PYTHON };
};
