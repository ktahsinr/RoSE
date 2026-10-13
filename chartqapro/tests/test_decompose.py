# =============================================================
# CPU tests for the decompose arms — no downloads, no GPU
# =============================================================
# rose.call_vlm / rose.load_image / rose.ext are monkeypatched, so both
# full algorithms (decompose → sub-answers / vote → RoSE → checkpoint →
# CSV + meta) run in milliseconds. Works under pytest, or:
#
#   python tests/test_decompose.py        (from chartqapro/)
# =============================================================

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

CQA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CQA))

from decompose.decomposer import parse_subqueries                   # noqa: E402
from decompose import decomp_runner, prompts, vote                  # noqa: E402
from decompose.vote import answer_evidence, choose_decomposition    # noqa: E402
import rose_chartqapro as rose                                      # noqa: E402


# ── 1. the parser always returns two strings ─────────────────

def test_parser_always_returns_two_strings():
    q = "what is the tallest bar?"
    cases = [
        '["sub one", "sub two"]',                          # clean JSON
        'Sure! Here you go:\n["sub one", "sub two"] hope that helps',
        '```json\n["sub one", "sub two"]\n```',            # fenced
        '["a", "b", "c"]',                                 # over-long → first two
        '[1, 2]',                                          # scalars → coerced
        '[" padded ", "two"]',                             # whitespace stripped
    ]
    for raw in cases:
        subqs, fell_back = parse_subqueries(raw, q)
        assert isinstance(subqs, list) and len(subqs) == 2, raw
        assert all(isinstance(s, str) and s.strip() for s in subqs), raw
    subqs, _ = parse_subqueries('["a", "b", "c"]', q)
    assert subqs == ["a", "b"]
    subqs, fell_back = parse_subqueries('["sub one", "sub two"]', q)
    assert subqs == ["sub one", "sub two"] and fell_back is False
    print("✓ parser: always exactly two non-empty strings")


# ── 2. fallback on malformed JSON ────────────────────────────

def test_fallback_on_malformed_json():
    q = "what share of the pie is red?"
    for raw in ["", None, "not json at all", '{"a": 1}', "[unclosed",
                '["only one"]', '[""]', "[]", '[{"sub_query": "a"}, "b"]']:
        subqs, fell_back = parse_subqueries(raw, q)
        assert subqs == [q, q], f"{raw!r} → {subqs}"
        assert fell_back is True, raw
    print("✓ parser: malformed output falls back to [question, question]")


# ── 3. Algorithm 2: merge rule vs pick rule ──────────────────

def _embed_from(table):
    """embed_fn backed by a {text: vector} table."""
    def embed(texts):
        return np.array([table[t] for t in texts], dtype=float)
    return embed


def test_alg2_merge_vs_pick():
    q = "how much did revenue grow from 2019 to 2020?"

    # Agreeing proposals (same directions, above the 0.8 threshold) → merge.
    table = {
        q:        [1.0, 1.0, 0.0],
        "a-rev":  [1.0, 0.0, 0.0], "a-grow": [0.0, 1.0, 0.0],
        "b-rev":  [0.98, 0.05, 0.0], "b-grow": [0.05, 0.98, 0.0],
    }
    out = choose_decomposition(
        [("gemma3-4b", ["a-rev", "a-grow"]), ("qwen3.5-4b", ["b-rev", "b-grow"])],
        q, _embed_from(table))
    assert out["rule"] == "merge" and out["vote_rule"] == "merge"
    assert len(out["sub_queries"]) == 2
    assert min(out["pair_sims"]) >= vote.AGREE_THRESHOLD

    # Best matching is found even when one proposal is in swapped order.
    out = choose_decomposition(
        [("gemma3-4b", ["a-rev", "a-grow"]), ("qwen3.5-4b", ["b-grow", "b-rev"])],
        q, _embed_from(table))
    assert out["rule"] == "merge" and out["matching"] == "crossed"

    # Disagreeing proposals → pick the one closer to the question.
    table = {
        q:         [1.0, 0.0, 0.0],
        "a-close": [0.9, 0.1, 0.0], "a-near": [0.8, 0.2, 0.0],
        "b-off":   [0.0, 1.0, 0.0], "b-far":  [0.0, 0.0, 1.0],
    }
    out = choose_decomposition(
        [("gemma3-4b", ["a-close", "a-near"]), ("qwen3.5-4b", ["b-off", "b-far"])],
        q, _embed_from(table))
    assert out["rule"] == "pick" and out["picked_model"] == "gemma3-4b"
    assert out["vote_rule"] == "pick:gemma3-4b"
    assert out["sub_queries"] == ["a-close", "a-near"]

    # …and symmetric: the other proposal wins when it is the closer one.
    out = choose_decomposition(
        [("gemma3-4b", ["b-off", "b-far"]), ("qwen3.5-4b", ["a-close", "a-near"])],
        q, _embed_from(table))
    assert out["vote_rule"] == "pick:qwen3.5-4b"
    assert out["sub_queries"] == ["a-close", "a-near"]
    print("✓ Alg 2 vote: merge on agreement, pick-closer otherwise")


# ── 4. evidence block formatting ─────────────────────────────

def test_evidence_block_formatting():
    items = [
        {"sub_query": "what was revenue in 2019?", "answer": "4.5",
         "model": "gemma3-4b", "confidence": 1.0},
        {"sub_query": "what was revenue in 2020?", "answer": "6.1",
         "model": "qwen3.5-4b", "confidence": 2 / 3},
    ]
    block = answer_evidence(items)
    assert block.startswith(prompts.EVIDENCE_HEADER)
    for it in items:
        assert it["sub_query"] in block
        assert str(it["answer"]) in block
        assert it["model"] in block
    assert "confidence: 1.00" in block and "confidence: 0.67" in block
    assert "[Evidence 1]" in block and "[Evidence 2]" in block
    print("✓ evidence block: sub-query, answer, model, confidence all present")


# ── mocks for the end-to-end runner tests ────────────────────

class _FakeModel:
    class config:
        _commit_hash = "deadbeef"


class _FakeEmbedder:
    def encode(self, texts, convert_to_numpy=True):
        if isinstance(texts, str):
            return np.ones(8)
        return np.ones((len(texts), 8))


def _dataset(tmp, n=4):
    rows = [{"id": f"new-{i:05d}",
             "question": f"what is the tallest bar in chart {i}?",
             "choices": None, "answer": "42",
             "question_type": "factoid", "image": ""} for i in range(n)]
    p = tmp / "dataset.json"
    p.write_text(json.dumps(rows))
    return p


def _install_mocks(log):
    def fake_call_vlm(model, processor, image, prompt,
                      temperature=0.8, max_new_tokens=None):
        if "JSON list" in prompt:              # the decomposition prompt
            log.append("decompose")
            return '["sub one", "sub two"]'
        log.append("generate")
        return "Looking at the chart, the value is 42.\nFinal Answer: 42"

    saved = (rose.call_vlm, rose.load_image, rose.ext)
    rose.call_vlm = fake_call_vlm
    rose.load_image = lambda field: object()
    rose.ext = lambda name: False              # no CLIP / two-stage in tests
    loaded = {k: (_FakeModel(), object())
              for k in ("qwen2.5-vl-7b", "gemma3-4b", "qwen3.5-4b")}
    return saved, loaded


def _restore(saved):
    rose.call_vlm, rose.load_image, rose.ext = saved


def _run(algorithm, tmp, data_path, log, loaded, **kw):
    out = tmp / f"decomp_alg{algorithm}.json"
    results = decomp_runner.run_decomp(
        algorithm, data_path, out, n=4, loaded=loaded,
        embedder=_FakeEmbedder(), **kw)
    return out, results


# ── 5. Algorithm 1 never imports chartqapro.routing ──────────

def _routing_modules(modules):
    """Modules with a dotted-name segment 'routing' — the repo's package,
    not bystanders like sklearn.utils.metadata_routing."""
    return {m for m in modules if "routing" in m.lower().split(".")}


def test_alg1_never_imports_routing():
    # (a) running the full Algorithm 1 pipeline pulls in no routing module
    before = _routing_modules(sys.modules)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        data_path = _dataset(tmp)
        log = []
        saved, loaded = _install_mocks(log)
        try:
            _run(1, tmp, data_path, log, loaded)
        finally:
            _restore(saved)
    after = _routing_modules(sys.modules)
    assert after == before, f"Algorithm 1 imported: {sorted(after - before)}"

    # (b) no decompose source file imports routing, even lazily
    pat = re.compile(r"^\s*(?:from|import)\s+[\w.]*routing", re.MULTILINE)
    for src in sorted((CQA / "decompose").glob("*.py")):
        assert not pat.search(src.read_text()), f"{src.name} imports routing"

    # (c) a fresh interpreter importing the runner stays routing-free
    code = (f"import sys; sys.path.insert(0, {str(CQA)!r}); "
            "import decompose.decomp_runner; "
            "bad = [m for m in sys.modules "
            "       if 'routing' in m.lower().split('.')]; "
            "assert not bad, bad; print('clean')")
    proc = subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "clean" in proc.stdout
    print("✓ Algorithm 1 never imports chartqapro.routing (run, source, fresh import)")


# ── 6. end-to-end runner: both algorithms, checkpoint, CSVs ──

def test_runner_alg1_end_to_end():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        data_path = _dataset(tmp)
        log = []
        saved, loaded = _install_mocks(log)
        try:
            out, results = _run(1, tmp, data_path, log, loaded)
        finally:
            _restore(saved)

        assert len(results) == 4
        for r in results:
            assert r["algorithm"] == 1
            assert r["sub_queries"] == ["sub one", "sub two"]
            assert r["sub_models"] == ["gemma3-4b", "qwen3.5-4b"]
            assert r["sub_confidences"] == [1.0, 1.0]   # 3/3 identical paths
            assert r["is_correct"] is True
            assert r["vote_rule"] == ""                  # Alg 2 only
            assert r["params_b_used"] == 24.9            # 8.3+4.3+4.0+8.3
            assert set(r["stage_latency_s"]) == {
                "decompose", "sub_gemma3-4b", "sub_qwen3.5-4b", "rose_final"}
        # per question: 1 decomposition + 2 sub-queries x 3 paths + 5 RoSE paths
        assert log.count("decompose") == 4
        assert log.count("generate") == 4 * (2 * 3 + rose.M_PATHS)

        # outputs: per-question CSV, per-stage CSV, meta
        csv_text = out.with_suffix(".csv").read_text()
        assert "sub_queries" in csv_text and "params_b_used" in csv_text
        stages = out.with_name(out.stem + "_stages.csv").read_text()
        assert "rose_final" in stages and "sub_gemma3-4b" in stages
        meta = json.loads(out.with_name(out.stem + "_meta.json").read_text())
        assert meta["algorithm"] == 1 and meta["complete"] is True
        assert meta["accuracy_rescored"] == 100.0
        assert meta["algorithm_config"]["n_sub_paths"] == 3
        print("✓ Alg 1 end-to-end: evidence → RoSE, CSVs + meta written")


def test_runner_alg2_end_to_end():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        data_path = _dataset(tmp)
        log = []
        saved, loaded = _install_mocks(log)
        try:
            out, results = _run(2, tmp, data_path, log, loaded)
        finally:
            _restore(saved)

        assert len(results) == 4
        for r in results:
            assert r["algorithm"] == 2
            assert r["sub_answers"] == [] and r["sub_confidences"] == []
            assert r["vote_rule"] == "merge"     # identical fake embeddings agree
            assert r["is_correct"] is True
            assert r["params_b_used"] == 16.6    # 4.3+4.0+8.3 — no 7B decompose
        # per question: 2 proposals, no sub-answers, 5 RoSE paths
        assert log.count("decompose") == 4 * 2
        assert log.count("generate") == 4 * rose.M_PATHS
        meta = json.loads(out.with_name(out.stem + "_meta.json").read_text())
        assert meta["vote_rules"] == {"merge": 4}
        assert meta["algorithm_config"]["agree_threshold"] == 0.8
        print("✓ Alg 2 end-to-end: vote on decomposition, no small-model answers")


def test_runner_checkpoint_resume_and_budget():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        data_path = _dataset(tmp)
        log = []
        saved, loaded = _install_mocks(log)
        try:
            # budget 0: stops before the first question, still writes outputs
            out, results = _run(1, tmp, data_path, log, loaded, stop_after_s=0)
            meta = json.loads(out.with_name(out.stem + "_meta.json").read_text())
            assert meta["complete"] is False and meta["n"] == 0
            # dry run: first 2 only
            out, results = _run(1, tmp, data_path, log, loaded, limit=2)
            assert len(results) == 2
            n_decomp_first = log.count("decompose")
            # next "session": finishes the rest, nothing redone
            out, results = _run(1, tmp, data_path, log, loaded)
        finally:
            _restore(saved)

        assert n_decomp_first == 2
        assert len(results) == 4
        assert log.count("decompose") == 4                   # 2 + 2, no redo
        assert len({r["id"] for r in results}) == 4
        meta = json.loads(out.with_name(out.stem + "_meta.json").read_text())
        assert meta["complete"] is True
        print("✓ budget stop + checkpoint resume: outputs written, nothing redone")


def test_errored_rows_are_retried():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        data_path = _dataset(tmp)
        out = tmp / "decomp_alg1.json"
        ckpt = out.with_name(out.stem + "_checkpoint.json")
        ckpt.write_text(json.dumps([
            {"id": "new-00000", "question": "q", "question_type": "factoid",
             "ground_truth": "42", "prediction": "ERROR",
             "is_correct": False, "method": "error", "error": "boom"},
        ]))
        log = []
        saved, loaded = _install_mocks(log)
        try:
            results = decomp_runner.run_decomp(
                1, data_path, out, n=4, loaded=loaded, embedder=_FakeEmbedder())
        finally:
            _restore(saved)
        assert len(results) == 4
        assert all(r.get("method") != "error" for r in results)
        print("✓ errored checkpoint rows are retried, not skipped forever")


if __name__ == "__main__":
    test_parser_always_returns_two_strings()
    test_fallback_on_malformed_json()
    test_alg2_merge_vs_pick()
    test_evidence_block_formatting()
    test_alg1_never_imports_routing()
    test_runner_alg1_end_to_end()
    test_runner_alg2_end_to_end()
    test_runner_checkpoint_resume_and_budget()
    test_errored_rows_are_retried()
    print("\nall decompose tests passed")
