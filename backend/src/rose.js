// RoSE — Reasoning with Orchestrated Streaming Experiences
// Faithful (local, small-model) implementation of Liu et al., EMNLP 2024.
//
// Pipeline for each new question:
//   1. Embed the question, compute cosine similarity to every pooled experience.
//   2. Sort the pool by similarity, split uniformly into K buckets (diversity —
//      avoids the "copy effect" of only using near-duplicates).
//   3. From each bucket pick the best experience: filter out ones whose
//      uncertainty exceeds a dynamic threshold (lambda x min-uncertainty in the
//      bucket), then prefer LOW uncertainty and HIGH complexity.
//   4. Build a few-shot CoT prompt from those demonstrations, ordered high->low
//      similarity, and sample N reasoning paths at temperature 1.0.
//   5. Self-consistency majority vote -> final answer.
//   6. Score the answered question's uncertainty & complexity and append it to
//      the streaming experience pool so future questions benefit.

import { generateMany, generate, embed } from './ollama.js';

// ----------------------------- math helpers ------------------------------

export function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-8);
}

// ----------------------------- prompting ---------------------------------

const isMC = (item) => item.type === 'multiple_choice' && item.choices;

function formatChoices(choices) {
  return choices.map((c) => `(${c.label}) ${c.text}`).join(' ');
}

// Multiple-choice questions include an "Answer Choices" line; numeric (math)
// questions are just the problem statement.
function questionBlock(item) {
  return isMC(item)
    ? `Q: ${item.question}\nAnswer Choices: ${formatChoices(item.choices)}`
    : `Q: ${item.question}`;
}

// How the final answer is written in a demonstration: "(A)" for MC, a bare
// number for numeric.
function answerString(item, answer) {
  return isMC(item) ? `(${answer})` : `${answer}`;
}

// A demonstration shows the question, the reasoning path, and the final answer.
function demoBlock(exp) {
  return `${questionBlock(exp)}\nA: Let's think step by step. ${exp.reasoning}\nThe answer is ${answerString(exp, exp.answer)}.`;
}

// System prompt that forces a parseable final line. Weak extraction (the model
// never stating the answer in a fixed form) is the single biggest source of
// artificially-low accuracy, so we constrain the output format explicitly.
export function systemFor(item) {
  if (isMC(item)) {
    return (
      'You are an expert at multiple-choice commonsense reasoning. ' +
      'Reason briefly step by step, then finish with a single line in exactly this format:\n' +
      'The answer is (X).\n' +
      'where X is one of the given choice letters. Do not add anything after that line.'
    );
  }
  return (
    'You are an expert at solving math word problems. ' +
    'Work through it step by step, then finish with a single line in exactly this format:\n' +
    'The answer is N.\n' +
    'where N is the final numeric value only — no units, no currency symbols, no commas, ' +
    'and no extra text after that line.'
  );
}

export function buildPrompt(item, demos) {
  const shots = demos.map(demoBlock).join('\n\n');
  const head = shots ? `${shots}\n\n` : '';
  return `${head}${questionBlock(item)}\nA: Let's think step by step.`;
}

// --------------------- answer extraction & scoring -----------------------

// Canonicalize a numeric answer string ("$1,024.0" -> "1024") for comparison
// and voting. Returns null if not a number.
export function canonicalNumber(v) {
  if (v == null) return null;
  const n = Number(String(v).replace(/[$,\s]/g, ''));
  return Number.isFinite(n) ? String(n) : null;
}

// True if a predicted answer matches the gold answer, respecting the item type.
export function isCorrect(predicted, gold, item) {
  if (predicted == null || gold == null) return false;
  if (isMC(item)) return predicted.toUpperCase() === gold.toUpperCase();
  const a = canonicalNumber(predicted), b = canonicalNumber(gold);
  return a != null && b != null && Math.abs(Number(a) - Number(b)) < 1e-4;
}

// Pull the final answer out of a free-form reasoning path. For MC this is a
// choice label (A-E); for numeric it is the last number stated.
export function extractAnswer(text, item) {
  if (!text) return null;
  return isMC(item) ? extractChoice(text, item.choices) : extractNumber(text);
}

function extractChoice(text, choices) {
  const labelSet = choices.map((c) => c.label).join('');
  const patterns = [
    `answer\\s*is\\s*:?\\s*\\(?([${labelSet}])\\)?`,
    `answer\\s*:?\\s*\\(?([${labelSet}])\\)?`,
    `\\(([${labelSet}])\\)`,
  ];
  for (const src of patterns) {
    const matches = [...text.matchAll(new RegExp(src, 'gi'))];
    if (matches.length) return matches[matches.length - 1][1].toUpperCase();
  }
  // Fallback: match the choice TEXT appearing in the reasoning.
  const lower = text.toLowerCase();
  let best = null, bestIdx = -1;
  for (const c of choices) {
    const idx = lower.lastIndexOf(c.text.toLowerCase());
    if (idx > bestIdx) { bestIdx = idx; best = c.label; }
  }
  return best;
}

function extractNumber(text) {
  const t = text.replace(/,/g, ''); // drop thousands separators: "1,024" -> "1024"
  // Prefer the LAST "answer is/=: N" statement (the final line we asked for).
  const ans = [...t.matchAll(/answer\s*(?:is|:|=)?\s*\$?(-?\d+(?:\.\d+)?)/gi)];
  if (ans.length) return canonicalNumber(ans[ans.length - 1][1]);
  // Otherwise take the last number mentioned anywhere.
  const nums = t.match(/-?\d+(?:\.\d+)?/g);
  if (nums && nums.length) return canonicalNumber(nums[nums.length - 1]);
  return null;
}

// Complexity proxy (Fu et al., complexity-based prompting): number of reasoning
// steps in the chain. More steps => more instructive as a demonstration.
export function complexityScore(reasoning) {
  const steps = reasoning
    .split(/[\n.]/)
    .map((s) => s.trim())
    .filter((s) => s.length > 8);
  return steps.length;
}

// Uncertainty from self-consistency: normalized entropy over the sampled answer
// distribution. 0 = all paths agree (confident), ->1 = maximal disagreement.
export function uncertaintyScore(answers) {
  const counts = {};
  let n = 0;
  for (const a of answers) {
    if (a == null) continue;
    counts[a] = (counts[a] || 0) + 1;
    n++;
  }
  if (n === 0) return 1;
  const keys = Object.keys(counts);
  if (keys.length <= 1) return 0;
  let entropy = 0;
  for (const k of keys) {
    const p = counts[k] / n;
    entropy -= p * Math.log(p);
  }
  return entropy / Math.log(keys.length); // normalize to [0,1]
}

// Tally the self-consistency votes. When `allLabels` is given (the full set of
// choice letters for a multiple-choice question), every option is seeded at 0
// so the distribution always shows all options — even ones no path selected.
// `unparsed` counts paths whose answer couldn't be read.
function majorityVote(answers, allLabels = null) {
  const counts = {};
  if (allLabels) for (const l of allLabels) counts[l] = 0;

  let unparsed = 0;
  for (const a of answers) {
    if (a == null) { unparsed++; continue; }
    counts[a] = (counts[a] || 0) + 1;
  }

  let best = null, bestCount = -1;
  for (const [k, v] of Object.entries(counts)) {
    if (v > bestCount) { bestCount = v; best = k; }
  }
  return { answer: bestCount > 0 ? best : null, counts, unparsed, total: answers.length };
}

// --------------------------- orchestration -------------------------------

// Given the pool (each entry already carries similarity to the current
// question), split into K buckets and select one experience per bucket.
export function orchestrate(pool, { numBuckets = 4, lambda = 1.2 } = {}) {
  if (pool.length === 0) return { demos: [], buckets: [] };

  const sorted = [...pool].sort((a, b) => b.similarity - a.similarity);
  const k = Math.min(numBuckets, sorted.length);
  const size = Math.ceil(sorted.length / k);

  const buckets = [];
  const demos = [];
  for (let i = 0; i < k; i++) {
    const bucket = sorted.slice(i * size, (i + 1) * size);
    if (bucket.length === 0) continue;

    const minU = Math.min(...bucket.map((e) => e.uncertainty));
    const threshold = lambda * (minU + 1e-6); // dynamic per-bucket threshold
    const eligible = bucket.filter((e) => e.uncertainty <= threshold);
    const pickPool = eligible.length ? eligible : bucket;

    // Prefer low uncertainty, then high complexity.
    const pick = [...pickPool].sort(
      (a, b) => a.uncertainty - b.uncertainty || b.complexity - a.complexity
    )[0];

    buckets.push({
      index: i,
      size: bucket.length,
      simRange: [bucket[bucket.length - 1].similarity, bucket[0].similarity],
      picked: {
        id: pick.id,
        question: pick.question,
        similarity: pick.similarity,
        uncertainty: pick.uncertainty,
        complexity: pick.complexity,
      },
    });
    demos.push(pick);
  }
  // Keep high->low similarity ordering for the prompt (diversity spread).
  demos.sort((a, b) => b.similarity - a.similarity);
  return { demos, buckets };
}

// ------------------------------ the pool ---------------------------------

export class ExperiencePool {
  constructor() { this.items = []; }
  get size() { return this.items.length; }
  clear() { this.items = []; }

  add(exp) { this.items.push(exp); }

  // Attach similarity of every experience to the given query embedding.
  withSimilarity(queryEmbedding) {
    return this.items.map((e) => ({
      ...e,
      similarity: cosine(queryEmbedding, e.embedding),
    }));
  }

  // Lightweight view (no embeddings) for API responses.
  snapshot() {
    return this.items.map(({ embedding, ...rest }) => rest);
  }
}

// ------------------------- the main RoSE step ----------------------------

// Answer one question with RoSE. Returns a rich trace and (by default) appends
// the new experience to the pool for future questions.
export async function answerWithRoSE(item, pool, opts = {}) {
  const {
    numPaths = 10,          // paper uses 20; default lower for local speed
    numBuckets = 4,
    lambda = 1.2,
    temperature = 1.0,
    append = true,
  } = opts;

  const queryEmbedding = await embed(item.question);
  const scored = pool.withSimilarity(queryEmbedding);
  const { demos, buckets } = orchestrate(scored, { numBuckets, lambda });

  const prompt = buildPrompt(item, demos);
  const paths = await generateMany(prompt, numPaths, { temperature, system: systemFor(item) });
  const answers = paths.map((p) => extractAnswer(p, item));

  // For multiple-choice, seed the tally with every option so all choices are
  // shown in the vote distribution (even ones that received zero votes).
  const allLabels = isMC(item) ? item.choices.map((c) => c.label) : null;
  const { answer, counts, unparsed, total } = majorityVote(answers, allLabels);
  const uncertainty = uncertaintyScore(answers);

  // Representative reasoning = a path that agrees with the majority vote.
  const idx = answers.findIndex((a) => a === answer);
  const reasoning = cleanReasoning(paths[idx >= 0 ? idx : 0]);
  const complexity = complexityScore(reasoning);

  const experience = {
    id: item.id,
    dataset: item.dataset,
    type: item.type,
    question: item.question,
    choices: item.choices,
    answer,
    reasoning,
    uncertainty,
    complexity,
    embedding: queryEmbedding,
  };
  if (append && answer != null) pool.add(experience);

  return {
    method: 'RoSE',
    question: item.question,
    choices: item.choices,
    predicted: answer,
    gold: item.answer ?? null,
    correct: item.answer != null ? isCorrect(answer, item.answer, item) : null,
    uncertainty,
    complexity,
    voteDistribution: counts,
    unparsed,            // paths whose answer could not be parsed
    numPaths: total,
    reasoning,
    demonstrations: buckets, // which experiences were orchestrated & why
    poolSizeBefore: pool.size - (append && answer != null ? 1 : 0),
  };
}

// Trim a raw completion into a compact reasoning path (drop the trailing
// "the answer is ..." so demos read cleanly).
export function cleanReasoning(text) {
  if (!text) return '';
  let t = text.trim();
  const cut = t.search(/the answer is/i);
  if (cut > 0) t = t.slice(0, cut).trim();
  return t.replace(/\s+/g, ' ').slice(0, 600);
}
