const j = (r) => r.json();
const post = (url, body) =>
  fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }).then(j);

const qs = (params) =>
  Object.entries(params)
    .filter(([, v]) => v !== undefined && v !== null && v !== '')
    .map(([k, v]) => `${k}=${encodeURIComponent(v)}`)
    .join('&');

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

  // ChartQAPro (vision extension). Inference happens offline on a GPU; these
  // read the result files it leaves behind — every number is computed by
  // chartqapro/summarize.py, not by the browser.
  cqa: {
    runs: () => fetch('/api/chartqapro/runs').then(j),
    summary: (run) => fetch(`/api/chartqapro/summary?${qs({ run })}`).then(j),
    rows: (params) => fetch(`/api/chartqapro/rows?${qs(params)}`).then(j),
    row: (run, id) => fetch(`/api/chartqapro/row?${qs({ run, id })}`).then(j),
    csvUrl: (run, perPath) =>
      `/api/chartqapro/export.csv?${qs({ run, perPath: perPath ? 1 : undefined })}`,
    upload: (name, kind, text) =>
      fetch(`/api/chartqapro/upload?${qs({ name, kind })}`, {
        method: 'POST',
        headers: { 'Content-Type': 'text/plain' },
        body: text,
      }).then(j),
    remove: (run) =>
      fetch(`/api/chartqapro/upload?${qs({ run })}`, { method: 'DELETE' }).then(j),
  },
};
