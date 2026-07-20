// Normalizes every raw dataset into one unified schema the RoSE app consumes:
//
//   {
//     id, dataset, type,            // type: "multiple_choice" | "numeric"
//     question,
//     choices: [{label, text}] | null,   // only for multiple_choice
//     answer                              // "A" for MC, "43" (string number) for numeric
//   }
//
// Sources:
//   - CommonsenseQA  (Kaggle JSONL)                        -> multiple_choice
//   - AddSub, GSM8K  (AGI-Edgerunners/LLM-Adapters JSON)   -> numeric

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const OUT_DIR = path.join(__dirname, '..', 'data');
const SRC_DIR = path.join(OUT_DIR, 'source');

// CommonsenseQA lives wherever the user downloaded the Kaggle archive.
const CSQA_DIR =
  process.env.CSQA_DIR || '/Users/kazitahsinraihan/Downloads/archive (1)';

// ------------------------- CommonsenseQA (JSONL) -------------------------

function convertCSQA(file, out) {
  const srcPath = path.join(CSQA_DIR, file);
  if (!fs.existsSync(srcPath)) {
    console.warn(`  ! missing ${srcPath} — skipping`);
    return;
  }
  const items = fs
    .readFileSync(srcPath, 'utf8')
    .split('\n')
    .filter((l) => l.trim())
    .map((line) => {
      const raw = JSON.parse(line);
      const q = raw.question;
      return {
        id: `CommonsenseQA-${raw.id}`,
        dataset: 'CommonsenseQA',
        type: 'multiple_choice',
        question: q.stem,
        choices: q.choices.map((c) => ({ label: c.label, text: c.text })),
        answer: raw.answerKey ?? null,
      };
    });
  write(out, items, file);
}

// --------------------- numeric math datasets (JSON) ----------------------

// Coerce a raw solution (string/number/array) into a canonical number string.
function canonicalNumber(sol) {
  let v = Array.isArray(sol) ? sol[0] : sol;
  if (typeof v === 'string') v = v.replace(/[$,]/g, '').trim();
  const n = Number(v);
  if (!Number.isFinite(n)) return null;
  // Drop a trailing ".0" so 18.0 -> "18" but keep real decimals.
  return Number.isInteger(n) ? String(n) : String(n);
}

function convertMath(file, datasetName, out) {
  const srcPath = path.join(SRC_DIR, file);
  if (!fs.existsSync(srcPath)) {
    console.warn(`  ! missing ${srcPath} — skipping`);
    return;
  }
  const raw = JSON.parse(fs.readFileSync(srcPath, 'utf8'));
  const items = raw
    .map((r, i) => {
      const answer = canonicalNumber(r.lSolutions ?? r.answer ?? r.output);
      if (answer == null) return null;
      return {
        id: `${datasetName}-${r.iIndex ?? i}`,
        dataset: datasetName,
        type: 'numeric',
        question: (r.sQuestion ?? r.instruction ?? '').trim(),
        choices: null,
        answer,
      };
    })
    .filter(Boolean);
  write(out, items, file);
}

// ------------------------------- helpers ---------------------------------

function write(out, items, srcLabel) {
  fs.writeFileSync(path.join(OUT_DIR, out), JSON.stringify(items, null, 0));
  console.log(`  ✓ ${srcLabel} → data/${out}  (${items.length} items)`);
}

// -------------------------------- run ------------------------------------

fs.mkdirSync(OUT_DIR, { recursive: true });
console.log('Preparing datasets…');
convertCSQA('dev_rand_split.jsonl', 'commonsenseqa_test.json');
convertCSQA('train_rand_split.jsonl', 'commonsenseqa_train.json');
convertMath('AddSub.json', 'AddSub', 'addsub_test.json');
convertMath('gsm8k.json', 'GSM8K', 'gsm8k_test.json');
console.log('Done.');
