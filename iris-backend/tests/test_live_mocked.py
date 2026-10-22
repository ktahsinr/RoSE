"""Live-mode run through the full API with mocked models (no GPU, no
downloads): POST /run → queue → LiveEngine.execute → SSE event sequence.
Mirrors the mocking style of chartqapro/tests/test_decompose.py."""

import time

import numpy as np
import pytest

from iris_backend.config import Settings
from iris_backend.live import LiveEngine


class _FakeModel:
    class config:
        _commit_hash = "deadbeef"


class _FakeEmbedder:
    def encode(self, texts, convert_to_numpy=True):
        if isinstance(texts, str):
            return np.ones(8)
        return np.ones((len(list(texts)), 8))


def _fake_call_vlm(model, processor, image, prompt, temperature=0.8,
                   max_new_tokens=None):
    if "JSON list" in prompt:
        return '["sub one", "sub two"]'
    if "CHART_TYPE" in prompt:
        return "CHART_TYPE: bar\nX_AXIS: Year\nY_AXIS: $M\nKEY_VALUES: 2019: 48"
    return "Looking at the chart, the peak is 2019.\nFinal Answer: 2019"


@pytest.fixture()
def live_client(sample, tmp_path, monkeypatch):
    import rose_chartqapro as rose

    def fake_load(self):
        self.rose = rose
        monkeypatch.setattr(rose, "call_vlm", _fake_call_vlm)
        monkeypatch.setattr(rose, "load_image", lambda field: object())
        monkeypatch.setattr(rose, "ext", lambda name: False)
        self.loaded = {k: (_FakeModel(), object())
                       for k in ("gemma3-4b", "qwen3.5-4b", "qwen2.5-vl-7b")}
        self.revisions = {k: "deadbeef" for k in self.loaded}
        self.embedder = _FakeEmbedder()
        self.embed_fn = lambda texts: np.ones((len(list(texts)), 8))
        self.pool = rose.ExperiencePool(self.embedder)
        self._seed_pool()

    monkeypatch.setattr(LiveEngine, "load", fake_load)

    settings = Settings(
        dataset_json=sample["dataset_json"],
        image_dir=sample["image_dir"],
        results_dir=sample["results_dir"],
        pool_path=sample["results_dir"] / "decomp_alg2.json",
        pool_stream=False, demo_mode=False,
        upload_dir=tmp_path / "uploads",
        cors_origins=[],
    )
    settings.upload_dir.mkdir(parents=True, exist_ok=True)

    from fastapi.testclient import TestClient
    from iris_backend.main import create_app
    with TestClient(create_app(settings)) as c:
        yield c


def _wait_finished(client, run_id, timeout=15.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        snap = client.get(f"/run/{run_id}").json()
        if snap["finished"]:
            return snap
        time.sleep(0.05)
    raise AssertionError("run did not finish in time")


def test_live_run_streams_every_stage(live_client):
    h = live_client.get("/health").json()
    assert h["mode"] == "live" and h["live_runs_enabled"] is True
    assert h["pool_size"] == 4            # seeded from the sample results JSON

    r = live_client.post("/run", json={"question_id": "factoid-00000",
                                       "fast_mode": True})
    assert r.status_code == 202, r.text
    run_id = r.json()["run_id"]

    snap = _wait_finished(live_client, run_id)
    names = [e["event"] for e in snap["events"]]
    assert names[0] == "queued"
    assert names.count("stage.decompose") == 2
    assert names.count("stage.vote") == 1
    assert names.count("stage.describe") == 1
    assert names.count("stage.path") == 3         # fast mode: M_PATHS=3
    assert names.count("stage.final") == 1
    assert names[-1] == "done"

    by_name = {e["event"]: e["data"] for e in snap["events"]}
    assert by_name["stage.decompose"]["sub_queries"] == ["sub one", "sub two"]
    assert by_name["stage.vote"]["vote_rule"] == "merge"
    assert "CHART_TYPE: bar" in by_name["stage.describe"]["description"]
    final = by_name["stage.final"]
    assert final["answer"] == "2019" and final["is_correct"] is True
    assert final["ground_truth"] == "2019"
    assert final["params_b_used"] == 16.6
    assert set(final["stage_latency_s"]) >= {
        "decompose_gemma3-4b", "decompose_qwen3.5-4b", "choose",
        "describe", "rose_final"}
    assert all("announce" in e["data"] for e in snap["events"])

    # Read-only pool: the run must not have grown it.
    assert live_client.get("/health").json()["pool_size"] == 4


def test_live_run_single_planner_and_sse_stream(live_client):
    r = live_client.post("/run", json={"question_id": "factoid-00002",
                                       "planners": ["gemma3-4b"]})
    assert r.status_code == 202
    run_id = r.json()["run_id"]
    snap = _wait_finished(live_client, run_id)
    votes = [e["data"] for e in snap["events"] if e["event"] == "stage.vote"]
    assert votes[0]["vote_rule"] == "single:gemma3-4b"

    # The SSE endpoint replays the finished run as text/event-stream.
    with live_client.stream("GET", f"/run/{run_id}/events") as es:
        assert es.headers["content-type"].startswith("text/event-stream")
        body = "".join(es.iter_text())
    assert "event: stage.final" in body and "event: done" in body
    assert 'data: {"' in body


def test_live_run_validation(live_client):
    assert live_client.post("/run", json={"question_id": "nope"}).status_code == 404
    assert live_client.post(
        "/run", json={"question_id": "factoid-00000",
                      "planners": []}).status_code == 422
    assert live_client.post(
        "/run", json={"question_id": "factoid-00000",
                      "algorithm": 1}).status_code == 422
