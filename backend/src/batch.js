// Headless batch runner for a RoSE evaluation sweep.
//
// Streams N questions through RoSE (and, optionally, the two baselines),
// maintaining the streaming experience pool across the whole run, and writes
// results incrementally so a killed/timed-out session can resume where it left
// off. Built for Kaggle "Save & Run All" commits (12h wall) and Colab.
//
// Usage (from backend/):
//   node src/batch.js
//   DATASET=train N=4000 NUM_PATHS=20 node src/batch.js
//   RESULTS_DIR=/kaggle/working/results BASELINES=1 node src/batch.js
//
// Env knobs (all optional):
//   DATASET        train | test        (default train — 9741 CSQA items)
//   N              how many questions   (default 4000)
//   NUM_PATHS      self-consistency     (default 20, matches the paper)
//   NUM_BUCKETS    orchestration buckets(default 4)
//   LAMBDA         uncertainty gate     (default 1.2)
//   BASELINES      1 to also run Zero-Shot-CoT + Auto-CoT (default 0)
//   CHECKPOINT     flush every K qs     (default 25)
//   RESULTS_DIR    output dir           (default ../results)
//   SHARD/ SHARDS  optional sharding    (SHARD=0 SHARDS=4 -> take every 4th q)
//   SEED_SHUFFLE   1 to deterministically shuffle before slicing (default 0)
//
// Outputs (in RESULTS_DIR):
//   run.jsonl            one JSON line per answered question (no embeddings)
//   pool_checkpoint.json full pool WITH embeddings — needed for exact resume
//   meta.json           { nextIndex, config } — resume pointer
//   summary.json        final accuracies (written at the end)

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { answerWithRoSE, ExperiencePool, isCorrect } from './rose.js';
import { zeroShotCoT, autoCoT } from './baselines.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ------------------------------- config ----------------------------------

const env = (k, d) => (process.env[k] != null ? process.env[k] : d);
const num = (k, d) => Number(env(k, d));

const DATASET = env('DATASET', 'train');            // train | test
const N = num('N', 4000);
const NUM_PATHS = num('NUM_PATHS', 20);
const NUM_BUCKETS = num('NUM_BUCKETS', 4);
const LAMBDA = num('LAMBDA', 1.2);
const BASELINES = env('BASELINES', '0') === '1';
const CHECKPOINT = num('CHECKPOINT', 25);
const SHARDS = num('SHARDS', 1);
const SHARD = num('SHARD', 0);
const SEED_SHUFFLE = env('SEED_SHUFFLE', '0') === '1';

const RESULTS_DIR = env('RESULTS_DIR', path.join(__dirname, '..', 'results'));
const DATA_FILE = path.join(
  __dirname, '..', 'data',
  DATASET === 'test' ? 'commonsenseqa_test.json' : 'commonsenseqa_train.json'
);

const RUN_JSONL = path.join(RESULTS_DIR, 'run.jsonl');
const POOL_CKPT = path.join(RESULTS_DIR, 'pool_checkpoint.json');
const META = path.join(RESULTS_DIR, 'meta.json');
const SUMMARY = path.join(RESULTS_DIR, 'summary.json');

const config = { DATASET, N, NUM_PATHS, NUM_BUCKETS, LAMBDA, BASELINES, SHARD, SHARDS, SEED_SHUFFLE };

// --------------------------- deterministic shuffle -------------------------
// A tiny seeded PRNG (mulberry32) so a shuffled order is reproducible across
// resumes and seeds — never Math.random(), which would reorder on restart.

function mulberry32(seed) {
  return function () {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffledCopy(arr, seed) {
  const a = [...arr];
  const rnd = mulberry32(seed);
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

// ------------------------------- load data --------------------------------

function loadQuestions() {
  const all = JSON.parse(fs.readFileSync(DATA_FILE, 'utf8'));
  let items = SEED_SHUFFLE ? shuffledCopy(all, 12345) : all;
  if (SHARDS > 1) items = items.filter((_, i) => i % SHARDS === SHARD); // strided shard
  return items.slice(0, N);
}

// ------------------------------- resume -----------------------------------

function restore() {
  if (!fs.existsSync(META) || !fs.existsSync(POOL_CKPT)) return { nextIndex: 0, pool: new ExperiencePool() };
  const meta = JSON.parse(fs.readFileSync(META, 'utf8'));
  const poolItems = JSON.parse(fs.readFileSync(POOL_CKPT, 'utf8'));
  const pool = new ExperiencePool();
  pool.items = poolItems;
  console.log(`↻ resuming from question #${meta.nextIndex} (pool size ${pool.size})`);
  return { nextIndex: meta.nextIndex, pool };
}

// Persist a consistent checkpoint: buffered results + full pool + resume pointer,
// all reflecting the SAME nextIndex so run.jsonl and the pool never disagree.
function checkpoint(buffer, pool, nextIndex) {
  if (buffer.length) {
    fs.appendFileSync(RUN_JSONL, buffer.map((r) => JSON.stringify(r)).join('\n') + '\n');
    buffer.length = 0;
  }
  fs.writeFileSync(POOL_CKPT, JSON.stringify(pool.items));            // includes embeddings
  fs.writeFileSync(META, JSON.stringify({ nextIndex, config }, null, 2));
}

// ------------------------------- main -------------------------------------

async function main() {
  fs.mkdirSync(RESULTS_DIR, { recursive: true });
  const questions = loadQuestions();
  console.log(`RoSE batch · ${DATASET} · ${questions.length} questions · ${NUM_PATHS} paths` +
    (BASELINES ? ' · +baselines' : '') + (SHARDS > 1 ? ` · shard ${SHARD}/${SHARDS}` : ''));

  const { nextIndex, pool } = restore();
  const buffer = [];
  const startedAt = Date.now();
  let roseCorrect = 0, zsCorrect = 0, acCorrect = 0, counted = 0;

  for (let i = nextIndex; i < questions.length; i++) {
    const q = questions[i];
    const t0 = Date.now();
    try {
      const rose = await answerWithRoSE(q, pool, { numPaths: NUM_PATHS, numBuckets: NUM_BUCKETS, lambda: LAMBDA });

      let zs = null, ac = null;
      if (BASELINES) {
        zs = await zeroShotCoT(q);
        ac = await autoCoT(q, pool);   // Auto-CoT selects demos from the same pool
      }

      counted++;
      if (rose.correct) roseCorrect++;
      if (zs?.correct) zsCorrect++;
      if (ac?.correct) acCorrect++;

      // One compact record per question. poolSizeAfter drives the learning curve;
      // no embeddings here (they live only in pool_checkpoint.json).
      buffer.push({
        i,
        id: q.id,
        gold: q.answer,
        rose: { pred: rose.predicted, correct: rose.correct, unc: rose.uncertainty, cplx: rose.complexity },
        zeroShot: zs ? { pred: zs.predicted, correct: zs.correct } : null,
        autoCoT: ac ? { pred: ac.predicted, correct: ac.correct } : null,
        poolSizeAfter: pool.size,
        running: {
          rose: roseCorrect / counted,
          zeroShot: BASELINES ? zsCorrect / counted : null,
          autoCoT: BASELINES ? acCorrect / counted : null,
        },
        ms: Date.now() - t0,
      });
    } catch (e) {
      // A single failure must not kill a multi-hour run — record and move on.
      buffer.push({ i, id: q.id, error: String(e.message || e) });
      console.error(`  ! q#${i} failed: ${e.message || e}`);
    }

    // progress line + periodic checkpoint
    const done = i + 1;
    if (done % CHECKPOINT === 0 || done === questions.length) {
      checkpoint(buffer, pool, done);
      const elapsed = (Date.now() - startedAt) / 1000;
      const rate = (done - nextIndex) / elapsed;                 // q/s this session
      const remaining = (questions.length - done) / (rate || 1e-9);
      const acc = counted ? (roseCorrect / counted * 100).toFixed(1) : '—';
      console.log(
        `  [${done}/${questions.length}] RoSE acc ${acc}%  ` +
        `pool ${pool.size}  ${rate.toFixed(2)} q/s  ~${(remaining / 60).toFixed(0)} min left`
      );
    }
  }

  const summary = {
    config,
    counted,
    accuracy: {
      rose: counted ? roseCorrect / counted : null,
      zeroShot: BASELINES && counted ? zsCorrect / counted : null,
      autoCoT: BASELINES && counted ? acCorrect / counted : null,
    },
    finalPoolSize: pool.size,
    wallSeconds: (Date.now() - startedAt) / 1000,
  };
  fs.writeFileSync(SUMMARY, JSON.stringify(summary, null, 2));
  console.log('\n✓ done');
  console.log(JSON.stringify(summary.accuracy, null, 2));
}

main().catch((e) => { console.error('FATAL', e); process.exit(1); });
