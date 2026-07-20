// Baselines that RoSE is compared against in the paper.
//
//  - Zero-Shot-CoT (Kojima et al. 2022): no demonstrations, just append
//    "Let's think step by step." and read off the answer.
//  - Auto-CoT (Zhang et al. 2023): automatically build few-shot demos by
//    clustering previously-seen questions for diversity and taking the
//    representative (nearest-to-centroid) of each cluster. Unlike RoSE it does
//    NOT use similarity-bucketing or uncertainty/complexity orchestration.

import { generate, embed } from './ollama.js';
import {
  buildPrompt,
  extractAnswer,
  cleanReasoning,
  cosine,
  isCorrect,
  systemFor,
} from './rose.js';

const isMC = (item) => item.type === 'multiple_choice' && item.choices;

const questionBlock = (item) =>
  isMC(item)
    ? `Q: ${item.question}\nAnswer Choices: ${item.choices
        .map((c) => `(${c.label}) ${c.text}`)
        .join(' ')}`
    : `Q: ${item.question}`;

export async function zeroShotCoT(item) {
  const prompt = `${questionBlock(item)}\nA: Let's think step by step.`;
  const text = await generate(prompt, { temperature: 0, system: systemFor(item) });
  const predicted = extractAnswer(text, item);
  return {
    method: 'Zero-Shot-CoT',
    predicted,
    gold: item.answer ?? null,
    correct: item.answer != null ? isCorrect(predicted, item.answer, item) : null,
    reasoning: cleanReasoning(text),
    demonstrations: [],
  };
}

// Simple k-means over embedding vectors (deterministic seeding: spread the
// initial centroids across the pool so results are reproducible).
function kmeans(vectors, k, iters = 10) {
  const n = vectors.length;
  k = Math.min(k, n);
  let centroids = Array.from({ length: k }, (_, i) =>
    vectors[Math.floor((i * n) / k)].slice()
  );
  let assign = new Array(n).fill(0);

  for (let it = 0; it < iters; it++) {
    // assign
    for (let i = 0; i < n; i++) {
      let best = 0, bestSim = -Infinity;
      for (let c = 0; c < k; c++) {
        const s = cosine(vectors[i], centroids[c]);
        if (s > bestSim) { bestSim = s; best = c; }
      }
      assign[i] = best;
    }
    // update
    const sums = Array.from({ length: k }, () => new Array(vectors[0].length).fill(0));
    const cnts = new Array(k).fill(0);
    for (let i = 0; i < n; i++) {
      const a = assign[i];
      cnts[a]++;
      const v = vectors[i];
      for (let d = 0; d < v.length; d++) sums[a][d] += v[d];
    }
    for (let c = 0; c < k; c++) {
      if (cnts[c] === 0) continue;
      for (let d = 0; d < sums[c].length; d++) centroids[c][d] = sums[c][d] / cnts[c];
    }
  }
  return { assign, centroids, k };
}

// Auto-CoT selects demos from the pool by diversity clustering.
export function autoCoTSelect(pool, numDemos = 4) {
  if (pool.length === 0) return [];
  const vectors = pool.map((e) => e.embedding);
  const { assign, centroids, k } = kmeans(vectors, numDemos);

  const demos = [];
  for (let c = 0; c < k; c++) {
    let best = -1, bestSim = -Infinity;
    for (let i = 0; i < pool.length; i++) {
      if (assign[i] !== c) continue;
      const s = cosine(vectors[i], centroids[c]);
      if (s > bestSim) { bestSim = s; best = i; }
    }
    if (best >= 0) demos.push(pool[best]);
  }
  return demos;
}

export async function autoCoT(item, pool, { numDemos = 4 } = {}) {
  const demos = autoCoTSelect(pool.items, numDemos);
  const prompt = buildPrompt(item, demos);
  const text = await generate(prompt, { temperature: 0, system: systemFor(item) });
  const predicted = extractAnswer(text, item);
  return {
    method: 'Auto-CoT',
    predicted,
    gold: item.answer ?? null,
    correct: item.answer != null ? isCorrect(predicted, item.answer, item) : null,
    reasoning: cleanReasoning(text),
    demonstrations: demos.map((d) => ({ id: d.id, question: d.question })),
  };
}
