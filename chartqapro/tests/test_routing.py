# =============================================================
# CPU tests for the routing package — no downloads, no GPU
# =============================================================
# Embeddings are synthetic (two well-separated clusters), so these run
# in milliseconds anywhere. Works under pytest, or stand-alone:
#
#   python tests/test_routing.py          (from chartqapro/)
# =============================================================

import json
import sys
import tempfile
from pathlib import Path

CQA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CQA))

from routing.pool import build_pool, load_pool, save_pool          # noqa: E402
from routing.router import Router, leave_one_out, route_file       # noqa: E402
from routing.evaluate_pilot import evaluate                        # noqa: E402

MODELS3 = ["gemma3-4b", "qwen3.5-4b", "qwen2.5-vl-7b"]


# ── synthetic world ──────────────────────────────────────────
# Cluster A ("bar-chart questions", even ids): gemma3-4b always right.
# Cluster B ("pie-chart questions", odd ids): only the 7B is right.

def fake_embed(texts):
    out = []
    for t in texts:
        a = 1.0 if "bar" in t else 0.05
        b = 1.0 if "pie" in t else 0.05
        out.append([a, b, 0.1])
    return out


def _fake_results_files(tmp, n=20):
    """One results JSON per model over the same n questions."""
    files = {}
    rows_by_model = {m: [] for m in MODELS3}
    for i in range(n):
        bar = i % 2 == 0
        q = f"How tall is the {'bar' if bar else 'pie'} segment {i}?"
        gold = str(i)
        for m in MODELS3:
            right = (m == "qwen2.5-vl-7b") or (m == "gemma3-4b" and bar)
            rows_by_model[m].append({
                "id": f"factoid-{i:05d}", "question": q,
                "question_type": "factoid", "ground_truth": gold,
                "prediction": gold if right else "wrong",
                "is_correct": not right,   # deliberately WRONG flag: the pool
                                           # must rescore with scoring.py
                "latency_s": 10.0 if m != "qwen2.5-vl-7b" else 30.0,
            })
    for m, rows in rows_by_model.items():
        p = Path(tmp) / f"{m}.json"
        p.write_text(json.dumps(rows))
        files[m] = str(p)
    return files


def _build(tmp, n=20):
    files = _fake_results_files(tmp, n)
    return build_pool(files, embed_fn=fake_embed)


# ── tests ────────────────────────────────────────────────────

def test_pool_rescores_with_scoring_py():
    with tempfile.TemporaryDirectory() as tmp:
        pool = _build(tmp)
        assert pool["n"] == 20
        e0 = next(e for e in pool["entries"] if e["id"] == "factoid-00000")
        # run flags said the opposite; scoring.py must have won:
        assert e0["model_hits"] == {"gemma3-4b": 1, "qwen3.5-4b": 0, "qwen2.5-vl-7b": 1}
        assert e0["latency_s"]["qwen2.5-vl-7b"] == 30.0


def test_pool_roundtrip_and_intersection():
    with tempfile.TemporaryDirectory() as tmp:
        files = _fake_results_files(tmp)
        # drop one id from one file → it must leave the pool entirely
        rows = json.loads(Path(files["gemma3-4b"]).read_text())
        Path(files["gemma3-4b"]).write_text(json.dumps(rows[:-1]))
        pool = build_pool(files, embed_fn=fake_embed)
        assert pool["n"] == 19

        p = Path(tmp) / "pool.json"
        save_pool(pool, p)
        assert load_pool(p)["n"] == 19


def test_router_sends_easy_to_small_and_hard_to_7b():
    with tempfile.TemporaryDirectory() as tmp:
        router = Router(_build(tmp), k=5, tau=0.6, embed_fn=fake_embed)
        bar = router.route(fake_embed(["a new bar question"])[0])
        pie = router.route(fake_embed(["a new pie question"])[0])
        assert bar["model"] == "gemma3-4b", bar
        assert pie["model"] == "qwen2.5-vl-7b", pie
        assert bar["scores"]["gemma3-4b"] > 0.9
        assert pie["scores"]["gemma3-4b"] < 0.6
        assert len(bar["neighbours"]) == 5
        # neighbours of a bar question are bar questions (even ids)
        assert all(int(nb["id"].split("-")[1]) % 2 == 0 for nb in bar["neighbours"])


def test_leave_one_out_excludes_self_and_matches_world():
    with tempfile.TemporaryDirectory() as tmp:
        pool = _build(tmp)
        loo = leave_one_out(pool, k=3, tau=0.6)
        by_id = {r["id"]: r for r in loo}
        for e in pool["entries"]:
            assert e["id"] not in by_id[e["id"]]["neighbours"]   # self held out
        # in this separable world LOO routing is perfect:
        assert all(r["hit"] == 1 for r in loo)
        share = sum(r["model"] == "gemma3-4b" for r in loo) / len(loo)
        assert share == 0.5


def test_evaluate_report_numbers():
    with tempfile.TemporaryDirectory() as tmp:
        rep = evaluate(_build(tmp), k=3, tau=0.6)
        assert rep["per_model"]["qwen2.5-vl-7b"]["accuracy"] == 1.0
        assert rep["per_model"]["gemma3-4b"]["accuracy"] == 0.5
        assert rep["oracle"]["accuracy"] == 1.0
        assert rep["routing"]["accuracy"] == 1.0
        # half the queries ride the 4.3B model → mean params well under 8.3
        assert rep["routing"]["mean_params_b"] < rep["always_7b"]["mean_params_b"]
        assert abs(rep["routing"]["mean_params_b"] - (4.3 + 8.3) / 2) < 1e-6
        assert rep["per_model"]["gemma3-4b"]["mean_latency_s"] == 10.0


def test_route_file_skips_pilot_ids(tmp_path=None):
    with tempfile.TemporaryDirectory() as tmp:
        pool = _build(tmp)
        pool_path = Path(tmp) / "pool.json"
        save_pool(pool, pool_path)

        questions = [{"id": f"new-{i}", "question": f"a new {'bar' if i % 2 == 0 else 'pie'} question {i}"}
                     for i in range(6)]
        qpath = Path(tmp) / "questions.json"
        qpath.write_text(json.dumps(questions))

        # monkeypatch the lazy embedder so no model is downloaded
        import routing.pool as rpool
        orig = rpool._default_embed_fn
        rpool._default_embed_fn = lambda: fake_embed
        try:
            out = Path(tmp) / "routing_decisions.json"
            dec = route_file(pool_path, qpath, out, k=5, tau=0.6,
                             skip_ids=["new-0"])
            assert "new-0" not in dec and len(dec) == 5
            assert dec["new-2"]["model"] == "gemma3-4b"
            assert dec["new-1"]["model"] == "qwen2.5-vl-7b"
            assert json.loads(out.read_text()) == dec
        finally:
            rpool._default_embed_fn = orig


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"✓ {fn.__name__}")
    print(f"\nall {len(fns)} routing tests passed")
