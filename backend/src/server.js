import express from 'express';
import cors from 'cors';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { health } from './ollama.js';
import { ExperiencePool, answerWithRoSE } from './rose.js';
import { zeroShotCoT, autoCoT } from './baselines.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DATA_DIR = path.join(__dirname, '..', 'data');

const app = express();
app.use(cors());
app.use(express.json({ limit: '1mb' }));

// ------------------------------ datasets ---------------------------------

function load(file) {
  const p = path.join(DATA_DIR, file);
  return fs.existsSync(p) ? JSON.parse(fs.readFileSync(p, 'utf8')) : [];
}

// Registry of the benchmarks the UI can explore. `train` is an optional
// warm-up source for the pool (only CommonsenseQA ships one here).
const DATASETS = {
  CommonsenseQA: {
    label: 'CommonsenseQA',
    type: 'multiple_choice',
    kind: 'Commonsense',
    test: load('commonsenseqa_test.json'),
    train: load('commonsenseqa_train.json'),
  },
  AddSub: {
    label: 'AddSub',
    type: 'numeric',
    kind: 'Arithmetic',
    test: load('addsub_test.json'),
    train: [],
  },
  GSM8K: {
    label: 'GSM8K',
    type: 'numeric',
    kind: 'Arithmetic',
    test: load('gsm8k_test.json'),
    train: [],
  },
};

// One streaming experience pool PER dataset, so each benchmark self-improves on
// its own stream (mixing math + commonsense demos would be nonsensical).
const pools = Object.fromEntries(
  Object.keys(DATASETS).map((k) => [k, new ExperiencePool()])
);

const getDataset = (name) => DATASETS[name] || DATASETS.CommonsenseQA;
const findItem = (id) => {
  for (const d of Object.values(DATASETS)) {
    const hit = d.test.find((x) => x.id === id) || d.train.find((x) => x.id === id);
    if (hit) return hit;
  }
  return null;
};

// ----------------------------- meta routes -------------------------------

app.get('/api/health', async (req, res) => {
  res.json({
    ollama: await health(),
    pools: Object.fromEntries(Object.entries(pools).map(([k, p]) => [k, p.size])),
    datasets: Object.fromEntries(
      Object.entries(DATASETS).map(([k, d]) => [k, d.test.length])
    ),
  });
});

app.get('/api/datasets', (req, res) => {
  res.json(
    Object.entries(DATASETS).map(([key, d]) => ({
      key,
      label: d.label,
      type: d.type,
      kind: d.kind,
      count: d.test.length,
      hasTrain: d.train.length > 0,
      poolSize: pools[key].size,
    }))
  );
});

app.get('/api/dataset', (req, res) => {
  const d = getDataset(req.query.dataset);
  const limit = Math.min(parseInt(req.query.limit) || 50, d.test.length);
  res.json(d.test.slice(0, limit));
});

// --------------------------- pool management -----------------------------

app.get('/api/pool', (req, res) => {
  const key = DATASETS[req.query.dataset] ? req.query.dataset : 'CommonsenseQA';
  res.json({ dataset: key, size: pools[key].size, items: pools[key].snapshot() });
});

app.post('/api/pool/reset', (req, res) => {
  const key = DATASETS[req.body?.dataset] ? req.body.dataset : 'CommonsenseQA';
  pools[key].clear();
  res.json({ dataset: key, size: 0 });
});

// Warm a dataset's pool. Uses its train split if present, otherwise later test
// items (kept disjoint from the first ones a user is likely to try).
app.post('/api/pool/warmup', async (req, res) => {
  const key = DATASETS[req.body?.dataset] ? req.body.dataset : 'CommonsenseQA';
  const d = DATASETS[key];
  const k = Math.min(parseInt(req.body?.k) || 5, 30);
  const numPaths = parseInt(req.body?.numPaths) || 5;
  const source = d.train.length ? d.train : d.test.slice(-200);
  try {
    for (let i = 0; i < k && i < source.length; i++) {
      await answerWithRoSE(source[i], pools[key], { numPaths, append: true });
    }
    res.json({ dataset: key, size: pools[key].size, added: k });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// ------------------------- single-question answer ------------------------

// Body: { id } to pick a dataset item, OR
//       { question, choices?, answer?, type?, dataset? } for a custom question.
app.post('/api/answer', async (req, res) => {
  try {
    const b = req.body || {};
    let item, poolKey;
    if (b.id) {
      item = findItem(b.id);
      if (!item) return res.status(404).json({ error: 'question id not found' });
      poolKey = DATASETS[item.dataset] ? item.dataset : 'CommonsenseQA';
    } else if (b.question) {
      const type = b.type || (Array.isArray(b.choices) && b.choices.length ? 'multiple_choice' : 'numeric');
      item = {
        id: 'custom',
        dataset: b.dataset && DATASETS[b.dataset] ? b.dataset : 'Custom',
        type,
        question: b.question,
        choices: type === 'multiple_choice' ? b.choices : null,
        answer: b.answer ?? null,
      };
      poolKey = DATASETS[item.dataset] ? item.dataset : 'CommonsenseQA';
    } else {
      return res.status(400).json({ error: 'provide {id} or {question}' });
    }

    const numPaths = parseInt(b.numPaths) || 10;
    const numBuckets = parseInt(b.numBuckets) || 4;
    const pool = pools[poolKey];

    const rose = await answerWithRoSE(item, pool, { numPaths, numBuckets, append: true });

    const out = { item, rose };
    if (b.compareBaselines) {
      out.zeroShot = await zeroShotCoT(item);
      out.autoCoT = await autoCoT(item, pool);
    }
    res.json(out);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// ----------------------------- benchmark ---------------------------------

// Streaming evaluation over N items of one dataset. RoSE builds/uses a fresh
// pool as it streams; Auto-CoT selects from the same growing pool; Zero-Shot
// ignores it. Mirrors the paper's Table 2 comparison for that benchmark.
app.post('/api/benchmark', async (req, res) => {
  try {
    const b = req.body || {};
    const key = DATASETS[b.dataset] ? b.dataset : 'CommonsenseQA';
    const d = DATASETS[key];
    const n = Math.min(parseInt(b.n) || 10, d.test.length);
    const numPaths = parseInt(b.numPaths) || 5;
    const freshPool = new ExperiencePool();

    // Optional pool warm-up: RoSE's whole advantage is a populated experience
    // pool, which barely builds up over a short streaming run. Pre-answering
    // some items (drawn from the END of the test set, disjoint from the scored
    // first-n) gives RoSE something to orchestrate from — closer to the paper,
    // which streams the full test set. Baselines are unaffected by the pool
    // except Auto-CoT, which also legitimately benefits, as in the paper.
    const warmup = Math.min(parseInt(b.warmup) || 0, 40);
    for (let i = 0; i < warmup && i < d.test.length - n; i++) {
      await answerWithRoSE(d.test[d.test.length - 1 - i], freshPool, { numPaths, append: true });
    }

    const log = [];
    const tally = {
      'Zero-Shot-CoT': { correct: 0, total: 0 },
      'Auto-CoT': { correct: 0, total: 0 },
      RoSE: { correct: 0, total: 0 },
    };

    for (let i = 0; i < n; i++) {
      const item = d.test[i];
      const zs = await zeroShotCoT(item);
      const ac = await autoCoT(item, freshPool);
      const rs = await answerWithRoSE(item, freshPool, { numPaths, append: true });

      for (const [m, r] of [['Zero-Shot-CoT', zs], ['Auto-CoT', ac], ['RoSE', rs]]) {
        if (r.correct != null) {
          tally[m].total++;
          if (r.correct) tally[m].correct++;
        }
      }

      log.push({
        i,
        question: item.question,
        gold: item.answer,
        poolSize: freshPool.size,
        predictions: {
          'Zero-Shot-CoT': { pred: zs.predicted, correct: zs.correct },
          'Auto-CoT': { pred: ac.predicted, correct: ac.correct },
          RoSE: { pred: rs.predicted, correct: rs.correct, uncertainty: rs.uncertainty },
        },
      });
    }

    const accuracy = Object.fromEntries(
      Object.entries(tally).map(([m, t]) => [
        m,
        { correct: t.correct, total: t.total, acc: t.total ? t.correct / t.total : 0 },
      ])
    );

    res.json({ dataset: key, n, numPaths, warmup, accuracy, log, finalPoolSize: freshPool.size });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => {
  console.log(`RoSE backend on http://localhost:${PORT}`);
  for (const [k, d] of Object.entries(DATASETS)) {
    console.log(`  ${k}: ${d.test.length} test${d.train.length ? `, ${d.train.length} train` : ''}`);
  }
});
