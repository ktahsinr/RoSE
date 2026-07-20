const j = (r) => r.json();
const post = (url, body) =>
  fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }).then(j);

export const api = {
  health: () => fetch('/api/health').then(j),
  datasets: () => fetch('/api/datasets').then(j),
  dataset: (dataset, limit = 40) =>
    fetch(`/api/dataset?dataset=${encodeURIComponent(dataset)}&limit=${limit}`).then(j),
  pool: (dataset) => fetch(`/api/pool?dataset=${encodeURIComponent(dataset)}`).then(j),
  resetPool: (dataset) => post('/api/pool/reset', { dataset }),
  warmup: (dataset, k = 5, numPaths = 5) => post('/api/pool/warmup', { dataset, k, numPaths }),
  answer: (body) => post('/api/answer', body),
  benchmark: (dataset, n, numPaths, warmup = 0) =>
    post('/api/benchmark', { dataset, n, numPaths, warmup }),
};
