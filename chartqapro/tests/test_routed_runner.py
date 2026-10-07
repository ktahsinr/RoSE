# =============================================================
# CPU tests for the routed runner — no downloads, no GPU
# =============================================================
# load_vlm and rose.process_one are monkeypatched, so the full
# route-then-run loop (grouping, checkpointing, resume, token budgets,
# meta) is exercised in milliseconds. Works under pytest, or:
#
#   python tests/test_routed_runner.py    (from chartqapro/)
# =============================================================

import json
import sys
import tempfile
from pathlib import Path

CQA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CQA))

from routing.pool import build_pool, save_pool                     # noqa: E402
from routing import routed_runner                                  # noqa: E402
from routing.routed_runner import route_dataset, run_routed        # noqa: E402
import rose_chartqapro as rose                                     # noqa: E402

MODELS3 = ["gemma3-4b", "qwen3.5-4b", "qwen2.5-vl-7b"]


# ── synthetic world (mirrors test_routing.py) ────────────────
# "bar" questions: gemma3-4b always right → routed to gemma.
# "pie" questions: only the 7B right → routed to 7B.

def fake_embed(texts):
    out = []
    for t in texts:
        a = 1.0 if "bar" in t else 0.05
        b = 1.0 if "pie" in t else 0.05
        out.append([a, b, 0.1])
    return out


def _pool_file(tmp, n=20):
    files = {}
    for m in MODELS3:
        rows = []
        for i in range(n):
            kind = "bar" in _q(i)
            right = kind if m == "gemma3-4b" else (m == "qwen2.5-vl-7b")
            rows.append({
                "id": f"factoid-{i:05d}", "question": _q(i),
                "question_type": "factoid", "ground_truth": ["42"],
                "prediction": "42" if right else "7",
                "latency_s": 10.0,
            })
        p = tmp / f"{m}.json"
        p.write_text(json.dumps(rows))
        files[m] = p
    pool = build_pool(files, embed_fn=fake_embed)
    path = tmp / "pool.json"
    save_pool(pool, path)
    return path


def _q(i):
    return (f"what is the tallest bar in chart {i}?" if i % 2 == 0
            else f"what share of the pie is slice {i}?")


def _dataset(tmp, n=6):
    rows = [{
        "id": f"new-{i:05d}", "question": _q(i), "choices": None,
        "answer": "42", "question_type": "factoid", "image": "",
    } for i in range(n)]
    p = tmp / "dataset.json"
    p.write_text(json.dumps(rows))
    return p


# ── mocks ────────────────────────────────────────────────────

class _FakeModel:
    class config:
        _commit_hash = "deadbeef"


def _install_mocks(monkey_log):
    """Patch GPU entry points; log which model answered which id."""
    def fake_load_vlm(model_key, quant="nf4"):
        monkey_log.append(("load", model_key))
        return _FakeModel(), object(), f"rev-{model_key}"

    def fake_process_one(sample, model, processor, pool,
                         clip_model=None, clip_processor=None):
        monkey_log.append(("answer", sample["id"]))
        pool.add(question=sample["question"], rationale="r", answer="42",
                 qtype="factoid", uncertainty=0.0, complexity=1.0, clip_emb=None)
        return {"id": sample["id"], "question": sample["question"],
                "question_type": "factoid", "ground_truth": sample["answer"],
                "prediction": "42", "is_correct": True,
                "method": "rose_few_shot", "uncertainty": 0.0,
                "complexity": 1.0, "tokens_budget": rose.MAX_NEW_TOKENS}

    class _FakeEmbedder:
        def encode(self, texts, convert_to_numpy=True):
            return fake_embed(list(texts))

    saved = (routed_runner.load_vlm, rose.process_one, rose.ext)
    routed_runner.load_vlm = fake_load_vlm
    rose.process_one = fake_process_one
    rose.ext = lambda name: False          # no CLIP in tests
    return saved, _FakeEmbedder()


def _restore(saved):
    routed_runner.load_vlm, rose.process_one, rose.ext = saved


# ── tests ────────────────────────────────────────────────────

def test_route_then_run_groups_and_budgets():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        pool_path = _pool_file(tmp)
        data_path = _dataset(tmp, n=6)
        records = json.loads(data_path.read_text())

        dec = route_dataset(pool_path, records, embed_fn=fake_embed,
                            out_path=tmp / "dec.json")
        # bar questions → gemma, pie questions → 7B
        assert dec["decisions"]["new-00000"]["model"] == "gemma3-4b"
        assert dec["decisions"]["new-00001"]["model"] == "qwen2.5-vl-7b"

        log = []
        saved, embedder = _install_mocks(log)
        try:
            out = tmp / "routed.json"
            results = run_routed(data_path, dec, out, embedder=embedder,
                                 token_budget={"qwen3.5-4b": 768})
        finally:
            _restore(saved)

        assert len(results) == 6
        # one load per model that got routed questions, cheapest first
        loads = [m for op, m in log if op == "load"]
        assert loads == [m for m in ["gemma3-4b", "qwen2.5-vl-7b"] if m in loads]
        # every result carries its routing decision and model
        for r in results:
            assert r["model_key"] == dec["decisions"][r["id"]]["model"]
            assert r["routing"]["model"] == r["model_key"]
            assert r["tokens_budget"] == rose.MAX_NEW_TOKENS  # budget restored/none hit qwen
        # meta written, with per-model table
        meta = json.loads((tmp / "routed_meta.json").read_text())
        assert meta["n"] == 6 and meta["accuracy_run"] == 100.0
        assert meta["per_model"]["gemma3-4b"]["n"] == 3
        assert meta["per_model"]["qwen2.5-vl-7b"]["n"] == 3
        print("✓ route→run grouping, decisions on rows, meta")


def test_resume_skips_answered():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        pool_path = _pool_file(tmp)
        data_path = _dataset(tmp, n=4)
        records = json.loads(data_path.read_text())
        dec = route_dataset(pool_path, records, embed_fn=fake_embed)

        log = []
        saved, embedder = _install_mocks(log)
        try:
            out = tmp / "routed.json"
            run_routed(data_path, dec, out, embedder=embedder, limit=2)
            n_first = sum(1 for op, _ in log if op == "answer")
            results = run_routed(data_path, dec, out, embedder=embedder)
        finally:
            _restore(saved)

        assert n_first == 2
        assert len(results) == 4
        answered = [i for op, i in log if op == "answer"]
        assert len(answered) == 4 and len(set(answered)) == 4  # nothing redone
        print("✓ checkpoint resume: dry-run answers kept, not redone")


def test_session_budget_stops_cleanly_and_resumes():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        pool_path = _pool_file(tmp)
        data_path = _dataset(tmp, n=4)
        records = json.loads(data_path.read_text())
        dec = route_dataset(pool_path, records, embed_fn=fake_embed)

        log = []
        saved, embedder = _install_mocks(log)
        try:
            out = tmp / "routed.json"
            # budget 0: stops before the first question, but still writes outputs
            run_routed(data_path, dec, out, embedder=embedder, stop_after_s=0)
            assert out.exists()
            meta = json.loads((tmp / "routed_meta.json").read_text())
            assert meta["complete"] is False and meta["n"] == 0
            # next "session": no budget → finishes everything
            results = run_routed(data_path, dec, out, embedder=embedder)
        finally:
            _restore(saved)

        assert len(results) == 4
        meta = json.loads((tmp / "routed_meta.json").read_text())
        assert meta["complete"] is True
        print("✓ session budget: clean stop, outputs written, resume completes")


def test_missing_decision_raises():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        pool_path = _pool_file(tmp)
        data_path = _dataset(tmp, n=3)
        records = json.loads(data_path.read_text())
        dec = route_dataset(pool_path, records[:2], embed_fn=fake_embed)

        log = []
        saved, embedder = _install_mocks(log)
        try:
            try:
                run_routed(data_path, dec, tmp / "routed.json", embedder=embedder)
                raise AssertionError("expected ValueError for unrouted ids")
            except ValueError as e:
                assert "no routing decision" in str(e)
        finally:
            _restore(saved)
        print("✓ unrouted dataset ids are refused up front")


if __name__ == "__main__":
    test_route_then_run_groups_and_budgets()
    test_resume_skips_answered()
    test_session_budget_stops_cleanly_and_resumes()
    test_missing_decision_raises()
    print("\nall routed-runner tests passed")
