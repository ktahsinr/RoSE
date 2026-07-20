// Thin wrapper around the local Ollama HTTP API for chat generation and
// embeddings. No external SDK — just fetch against http://localhost:11434.

const OLLAMA_URL = process.env.OLLAMA_URL || 'http://localhost:11434';
export const CHAT_MODEL = process.env.ROSE_CHAT_MODEL || 'llama3.1:8b';
export const EMBED_MODEL = process.env.ROSE_EMBED_MODEL || 'nomic-embed-text';

// Stop sequences: with few-shot demonstrations in the prompt, the model tends
// to keep going and hallucinate the NEXT "Q: … A: …" pair after it has already
// answered. That runs generation to the context limit — slow enough to look
// like a hang, especially for the numeric datasets. Cutting at the start of any
// new question keeps each answer to just its own reasoning + final line.
const STOP = ['\nQ:', '\n\nQ', '\nQuestion:', '\nAnswer Choices:'];
const MAX_TOKENS = 512; // hard cap so no single generation can run away

// Single completion. temperature controls diversity (RoSE uses T=1.0 for the
// self-consistency sampling, T=0 for deterministic single answers).
export async function generate(prompt, { temperature = 0, system } = {}) {
  const res = await fetch(`${OLLAMA_URL}/api/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: CHAT_MODEL,
      prompt,
      system,
      stream: false,
      options: { temperature, stop: STOP, num_predict: MAX_TOKENS },
    }),
  });
  if (!res.ok) throw new Error(`Ollama generate failed: ${res.status} ${await res.text()}`);
  const data = await res.json();
  return data.response;
}

// Generate `n` diverse completions concurrently (for self-consistency).
export async function generateMany(prompt, n, opts = {}) {
  return Promise.all(Array.from({ length: n }, () => generate(prompt, opts)));
}

export async function embed(text) {
  const res = await fetch(`${OLLAMA_URL}/api/embeddings`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: EMBED_MODEL, prompt: text }),
  });
  if (!res.ok) throw new Error(`Ollama embeddings failed: ${res.status} ${await res.text()}`);
  const data = await res.json();
  return data.embedding;
}

export async function health() {
  try {
    const res = await fetch(`${OLLAMA_URL}/api/tags`);
    if (!res.ok) return { ok: false, error: `status ${res.status}` };
    const data = await res.json();
    const models = (data.models || []).map((m) => m.name);
    return {
      ok: true,
      models,
      hasChat: models.some((m) => m.startsWith(CHAT_MODEL.split(':')[0])),
      hasEmbed: models.some((m) => m.startsWith(EMBED_MODEL.split(':')[0])),
      chatModel: CHAT_MODEL,
      embedModel: EMBED_MODEL,
    };
  } catch (e) {
    return { ok: false, error: e.message };
  }
}
