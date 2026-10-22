"""The API in demo mode: everything serves from stored files; /run refuses."""


def test_health_reports_demo_mode(client):
    h = client.get("/health").json()
    assert h["mode"] == "demo" and h["live_runs_enabled"] is False
    assert h["models_loaded"] == []
    assert h["dataset"]["count"] == 6
    assert h["replay_algorithms"] == [2]


def test_models_lists_registry_unloaded(client):
    rows = client.get("/models").json()
    keys = {r["key"] for r in rows}
    assert {"gemma3-4b", "qwen3.5-4b", "qwen2.5-vl-7b",
            "all-mpnet-base-v2"} <= keys
    assert all(r["loaded"] is False for r in rows)
    planners = [r["key"] for r in rows if r["role"] == "planner"]
    assert sorted(planners) == ["gemma3-4b", "qwen3.5-4b"]
    assert any(r["params_b"] for r in rows)


def test_questions_list_filter_search(client):
    out = client.get("/questions").json()
    assert out["total"] == 6 and len(out["items"]) == 6
    item = out["items"][0]
    assert set(item) >= {"id", "question", "question_type", "ground_truth",
                         "image_url", "has_replay"}
    assert item["has_replay"]["alg2"] is True        # first 4 have replays
    assert out["items"][5]["has_replay"]["alg2"] is False

    out = client.get("/questions", params={"search": "unemployment"}).json()
    assert out["total"] == 1
    assert "unemployment" in out["items"][0]["question"]

    out = client.get("/questions", params={"offset": 4, "limit": 1}).json()
    assert len(out["items"]) == 1 and out["total"] == 6


def test_question_detail_carries_full_replay(client):
    q = client.get("/questions/factoid-00000").json()
    r = q["replay"]["alg2"]
    assert r["algorithm"] == 2 and r["is_correct"] is True
    assert len(r["sub_queries"]) == 2
    assert set(r["proposals"]) == {"gemma3-4b", "qwen3.5-4b"}
    assert r["vote"]["pair_sims"] and r["vote"]["threshold"] == 0.8
    assert r["chart_description"]
    assert len(r["paths"]) == 3 and r["paths"][0]["raw"]
    assert r["final_answer"] == "2019"
    assert client.get("/questions/nope").status_code == 404


def test_images_served_and_traversal_blocked(client):
    q = client.get("/questions/factoid-00000").json()
    img = client.get(q["image_url"])
    assert img.status_code == 200
    assert img.headers["content-type"] == "image/png"
    assert client.get("/images/..%2Fdataset.json").status_code == 404


def test_run_refused_in_demo_mode(client):
    r = client.post("/run", json={"question_id": "factoid-00000"})
    assert r.status_code == 503
    assert "demo" in r.json()["detail"].lower()


def test_run_validation_runs_before_queueing(client):
    assert client.get("/run/run-9999/events").status_code == 404
    assert client.get("/run/run-9999").status_code == 404


def test_benchmarks_payload(client):
    b = client.get("/benchmarks").json()
    cols = b["published"]["columns"]
    assert cols[0] == "Factoid" and cols[-1] == "Overall"
    assert len(b["published"]["rows"]) == 18
    # Best-per-column is computed: Gemini-Flash-2.0 leads Overall.
    best_overall = [r["model"] for r in b["published"]["rows"] if r["best"][-1]]
    assert best_overall == ["Gemini-Flash-2.0"]
    iris = b["iris"]["rows"]
    assert [r["accuracy_pct"] for r in iris] == [38.8, 42.8, 47.6, 24.0]
    assert all(len(r["ci"]) == 2 for r in iris)
    assert "defects" in b["iris"]["footnote"]
