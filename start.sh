#!/usr/bin/env bash
# One-command launcher for the RoSE demo (backend + frontend).
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "› Ensuring Ollama is running…"
if ! curl -s http://localhost:11434/api/tags >/dev/null; then
  brew services start ollama || ollama serve &
  sleep 3
fi

echo "› Checking models…"
ollama list | grep -q "llama3.1" || { echo "  pulling llama3.1:8b…"; ollama pull llama3.1:8b; }
ollama list | grep -q "nomic-embed-text" || { echo "  pulling nomic-embed-text…"; ollama pull nomic-embed-text; }

echo "› Preparing dataset (if needed)…"
[ -f "$ROOT/backend/data/commonsenseqa_test.json" ] || (cd "$ROOT/backend" && npm run prepare-data)

echo "› Starting backend on :3001…"
(cd "$ROOT/backend" && npm start) &
BACK=$!

echo "› Starting frontend on :5173…"
(cd "$ROOT/frontend" && npm run dev) &
FRONT=$!

trap "kill $BACK $FRONT 2>/dev/null" EXIT
echo ""
echo "  RoSE running →  http://localhost:5173"
echo "  (Ctrl-C to stop)"
wait
