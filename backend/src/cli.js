// Terminal client for the RoSE backend — prints the full result of answering a
// question (vote distribution per option, answer, uncertainty, complexity,
// percentages, and baselines) without needing the web UI.
//
// Usage:
//   node src/cli.js                         # random CommonsenseQA question
//   node src/cli.js GSM8K                    # random GSM8K question
//   node src/cli.js AddSub 3                 # AddSub question #3
//   node src/cli.js CommonsenseQA 0 10       # question #0, 10 self-consistency paths
//   node src/cli.js GSM8K random 8 --no-baselines
//
// Requires the backend running (npm start) and Ollama up.

const API = process.env.ROSE_API || 'http://localhost:3001';

// ---- tiny ANSI helpers ----
const c = (n) => (s) => `\x1b[${n}m${s}\x1b[0m`;
const bold = c(1), dim = c(2), red = c(91), green = c(92), yellow = c(93), rose = c(95), cyan = c(96);
const bar = (frac, width = 24) => {
  const f = Math.max(0, Math.min(1, frac));
  const filled = Math.round(f * width);
  return '█'.repeat(filled) + dim('░'.repeat(width - filled));
};
const pct = (x) => `${(x * 100).toFixed(1)}%`;

async function main() {
  const args = process.argv.slice(2);
  const noBaselines = args.includes('--no-baselines');
  const pos = args.filter((a) => !a.startsWith('--'));
  const dataset = pos[0] || 'CommonsenseQA';
  const idxArg = pos[1];
  const numPaths = parseInt(pos[2]) || 10;

  // Pick the question.
  const list = await fetch(`${API}/api/dataset?dataset=${encodeURIComponent(dataset)}&limit=500`).then((r) => r.json());
  if (!Array.isArray(list) || !list.length) {
    console.error(red(`No questions for dataset "${dataset}". Try CommonsenseQA, AddSub, or GSM8K.`));
    process.exit(1);
  }
  let idx;
  if (idxArg == null || idxArg === 'random') idx = Math.floor(list.length * Math.abs(Math.sin(Date.now())) ) % list.length;
  else idx = Math.max(0, Math.min(parseInt(idxArg) || 0, list.length - 1));
  const q = list[idx];

  console.log('\n' + bold(cyan('━'.repeat(70))));
  console.log(bold(`  ${dataset}  ·  question #${idx}  ·  ${numPaths} self-consistency paths`));
  console.log(bold(cyan('━'.repeat(70))));
  console.log(bold('\nQ: ') + q.question);
  if (q.type === 'multiple_choice') {
    for (const ch of q.choices) {
      const gold = ch.label === q.answer;
      console.log(`   ${gold ? green('(' + ch.label + ')') : '(' + ch.label + ')'} ${ch.text}${gold ? green('  ← gold') : ''}`);
    }
  } else {
    console.log(dim('   gold answer: ') + green(q.answer));
  }

  console.log(dim('\n  running… (sampling reasoning paths on the local model)\n'));

  const res = await fetch(`${API}/api/answer`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ id: q.id, numPaths, compareBaselines: !noBaselines }),
  }).then((r) => r.json());

  if (res.error) { console.error(red('Error: ' + res.error)); process.exit(1); }
  const r = res.rose;

  // ---- RoSE block ----
  console.log(bold(rose('┌─ RoSE ' + '─'.repeat(62))));
  const ok = r.correct == null ? dim('no gold') : r.correct ? green('✓ CORRECT') : red('✗ WRONG');
  const predStr = q.type === 'multiple_choice' ? `(${r.predicted})` : r.predicted;
  console.log(`│  answer: ${bold(rose(predStr))}   ${ok}   ${dim('gold=')}${r.gold}`);
  console.log('│');
  console.log(`│  ${bold('Vote distribution')} ${dim(`(all ${r.numPaths} paths across every option)`)}`);

  const dist = r.voteDistribution || {};
  let rows;
  if (q.type === 'multiple_choice') {
    rows = q.choices.map((ch) => [ch.label, dist[ch.label] || 0, ch.text]);
  } else {
    rows = Object.entries(dist).sort((a, b) => b[1] - a[1]).map(([k, v]) => [k, v, '']);
  }
  for (const [label, n, text] of rows) {
    const frac = r.numPaths ? n / r.numPaths : 0;
    const win = label === r.predicted;
    const lab = win ? bold(rose(label.padEnd(3))) : dim(label.padEnd(3));
    const cnt = String(n).padStart(2);
    const line = `│   ${lab} ${bar(frac)} ${cnt}/${r.numPaths}  ${(frac * 100).toFixed(0).padStart(3)}%  ${dim(text.slice(0, 24))}`;
    console.log(win ? line : dim(line.replace(/\x1b\[2m/g, '')));
  }
  if (r.unparsed > 0) {
    const frac = r.unparsed / r.numPaths;
    console.log(`│   ${yellow('?  ')} ${bar(frac)} ${String(r.unparsed).padStart(2)}/${r.numPaths}  ${yellow('unparsed')}`);
  }

  console.log('│');
  console.log(`│  ${bold('Metrics')}`);
  console.log(`│    uncertainty : ${uncColor(r.uncertainty)}   ${dim('(0 = all paths agree, 1 = max disagreement)')}`);
  console.log(`│    confidence  : ${green(pct(1 - r.uncertainty))}   ${dim('(1 - uncertainty)')}`);
  console.log(`│    agreement   : ${pct((dist[r.predicted] || Math.max(0, ...Object.values(dist))) / r.numPaths)}   ${dim('(winning option share)')}`);
  console.log(`│    complexity  : ${r.complexity} ${dim('reasoning steps')}`);
  console.log(`│    pool used   : ${r.poolSizeBefore} ${dim('experiences orchestrated from')}`);

  if (r.demonstrations?.length) {
    console.log('│');
    console.log(`│  ${bold('Orchestrated experiences')} ${dim(`(${r.demonstrations.length} buckets)`)}`);
    for (const b of r.demonstrations) {
      console.log(`│    bucket ${b.index}: sim=${b.picked.similarity.toFixed(2)} unc=${b.picked.uncertainty.toFixed(2)} cplx=${b.picked.complexity}  ${dim(b.picked.question.slice(0, 40))}`);
    }
  }
  console.log('│');
  console.log(`│  ${bold('Reasoning')} ${dim('(representative winning path)')}`);
  wrap(r.reasoning, 64).forEach((l) => console.log('│    ' + dim(l)));
  console.log(bold(rose('└' + '─'.repeat(69))));

  // ---- baselines ----
  if (!noBaselines) {
    console.log(bold('\n  Baselines'));
    for (const [name, b] of [['Zero-Shot-CoT', res.zeroShot], ['Auto-CoT', res.autoCoT]]) {
      if (!b) continue;
      const bok = b.correct == null ? dim('no gold') : b.correct ? green('✓') : red('✗');
      const bp = q.type === 'multiple_choice' ? `(${b.predicted})` : b.predicted;
      console.log(`    ${name.padEnd(16)} → ${bold(bp)}  ${bok}`);
    }
  }

  // ---- one-line summary ----
  console.log(bold('\n  Summary'));
  const line = (name, p, correct) => {
    const s = correct == null ? dim('—') : correct ? green('✓') : red('✗');
    return `    ${name.padEnd(16)} ${String(p).padEnd(6)} ${s}`;
  };
  console.log(line('RoSE', q.type === 'multiple_choice' ? `(${r.predicted})` : r.predicted, r.correct));
  if (!noBaselines) {
    console.log(line('Zero-Shot-CoT', q.type === 'multiple_choice' ? `(${res.zeroShot.predicted})` : res.zeroShot.predicted, res.zeroShot.correct));
    console.log(line('Auto-CoT', q.type === 'multiple_choice' ? `(${res.autoCoT.predicted})` : res.autoCoT.predicted, res.autoCoT.correct));
  }
  console.log('');
}

function uncColor(u) {
  const s = u.toFixed(3);
  if (u < 0.34) return green(s);
  if (u < 0.67) return yellow(s);
  return red(s);
}
function wrap(text, width) {
  const words = (text || '').split(/\s+/);
  const lines = [];
  let cur = '';
  for (const w of words) {
    if ((cur + ' ' + w).trim().length > width) { lines.push(cur.trim()); cur = w; }
    else cur += ' ' + w;
  }
  if (cur.trim()) lines.push(cur.trim());
  return lines.slice(0, 8);
}

main().catch((e) => { console.error(red('Failed: ' + e.message)); process.exit(1); });
