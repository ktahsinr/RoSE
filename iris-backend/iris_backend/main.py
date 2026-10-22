"""IRIS backend — FastAPI app.

    IRIS_DEMO_MODE=true uvicorn iris_backend.main:app --port 8000

All run state lives in app.state.iris (an AppState); tests build the app
with an explicit Settings via create_app().
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from . import benchmarks
from .config import Settings
from .pdfqa import PdfStore
from .runner import AppState

EMBEDDER = {"key": "all-mpnet-base-v2",
            "hf_id": "sentence-transformers/all-mpnet-base-v2",
            "params_b": 0.11, "label": "all-mpnet-base-v2", "role": "embedder"}


class RunRequest(BaseModel):
    question_id: Optional[str] = None
    image_id: Optional[str] = None
    question: Optional[str] = None
    algorithm: int = 2
    planners: list = Field(default=["gemma3-4b", "qwen3.5-4b"])
    final_model: str = "qwen2.5-vl-7b"
    fast_mode: bool = True


class CropRequest(BaseModel):
    page: int
    bbox: list  # [x0, y0, x1, y1] in PDF points


def create_app(settings: "Settings | None" = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state = AppState(settings or Settings.from_env())
        if not state.settings.demo_mode:
            try:
                from .live import LiveEngine
                state.engine = LiveEngine(state.settings)
                await asyncio.to_thread(state.engine.load)
            except Exception as exc:  # noqa: BLE001 — degrade visibly, not silently
                import traceback
                traceback.print_exc()
                state.settings.demo_mode = True
                state.live_load_error = str(exc)
        state.worker_task = asyncio.create_task(state.worker())
        app.state.iris = state
        app.state.pdfs = PdfStore(state.settings.upload_dir)
        yield
        state.worker_task.cancel()

    app = FastAPI(title="IRIS backend", lifespan=lifespan)

    # The frontend is the only intended caller (IRIS_API_URL / same origin).
    app.add_middleware(
        CORSMiddleware,
        allow_origins=(settings or Settings.from_env()).cors_origins,
        allow_methods=["*"], allow_headers=["*"])

    def st(request: Request) -> AppState:
        return request.app.state.iris

    # ── status ───────────────────────────────────────────────

    @app.get("/health")
    def health(request: Request):
        s = st(request)
        return {
            "mode": "demo" if s.settings.demo_mode else "live",
            "live_runs_enabled": not s.settings.demo_mode,
            "live_load_error": getattr(s, "live_load_error", None),
            "models_loaded": (list(s.engine.loaded) if s.engine else []),
            "pool_size": (s.engine.pool.size()
                          if s.engine and s.engine.pool else None),
            "pool_stream": s.settings.pool_stream,
            "dataset": {"path": str(s.settings.dataset_json or ""),
                        "count": len(s.records)},
            "replay_algorithms": s.replay.algorithms(),
            "queue_length": s.queue.qsize(),
        }

    @app.get("/models")
    def models(request: Request):
        s = st(request)
        from decompose.models import MODELS, SMALL_MODELS, FINAL_MODEL
        rows = []
        for key, spec in MODELS.items():
            rows.append({
                "key": key, "hf_id": spec["hf_id"], "label": spec["label"],
                "params_b": spec["params_b"], "quant": "nf4",
                "role": ("planner" if key in SMALL_MODELS else "final"),
                "default_final": key == FINAL_MODEL,
                "loaded": bool(s.engine and key in s.engine.loaded),
                "revision": (s.engine.revisions.get(key)
                             if s.engine else spec.get("revision")),
            })
        rows.append(dict(EMBEDDER, quant=None, default_final=False,
                         loaded=bool(s.engine), revision=None))
        return rows

    # ── questions ────────────────────────────────────────────

    def _question_payload(s: AppState, r: dict) -> dict:
        return {
            "id": r["id"],
            "question": r["question"],
            "question_type": r["question_type"],
            "ground_truth": r["ground_truth"],
            "image_url": f"/images/{r['image_name']}" if r["image_name"] else None,
            "has_replay": s.replay.has(r["id"]),
        }

    @app.get("/questions")
    def questions(request: Request, offset: int = 0, limit: int = 50,
                  type: Optional[str] = None, search: Optional[str] = None):
        s = st(request)
        rows = s.records
        if type:
            rows = [r for r in rows if r["question_type"] == type.lower()]
        if search:
            needle = search.lower()
            rows = [r for r in rows
                    if needle in r["question"].lower() or needle in r["id"]]
        limit = max(1, min(limit, 200))
        return {"total": len(rows),
                "items": [_question_payload(s, r)
                          for r in rows[offset:offset + limit]]}

    @app.get("/questions/{question_id}")
    def question(request: Request, question_id: str):
        s = st(request)
        r = s.by_id.get(question_id)
        if r is None:
            raise HTTPException(404, f"no question with id {question_id!r}")
        out = _question_payload(s, r)
        out["replay"] = {f"alg{a}": s.replay.get(question_id, a)
                         for a in s.replay.algorithms()}
        return out

    @app.get("/images/{name}")
    def image(request: Request, name: str):
        s = st(request)
        if s.settings.image_dir is None or "/" in name or name.startswith("."):
            raise HTTPException(404, "image not found")
        path = s.settings.image_dir / name
        if not path.is_file():
            raise HTTPException(404, "image not found")
        return FileResponse(path)

    @app.get("/uploads/{image_id}")
    def upload_image(request: Request, image_id: str):
        path = request.app.state.pdfs.image_path(image_id)
        if path is None:
            raise HTTPException(404, "image not found")
        return FileResponse(path)

    # ── runs ─────────────────────────────────────────────────

    @app.post("/run", status_code=202)
    async def run(request: Request, body: RunRequest):
        s = st(request)
        if s.settings.demo_mode:
            raise HTTPException(503, "Live runs are disabled: demo mode "
                                     "(no GPU). Stored results remain available.")
        if body.algorithm != 2:
            raise HTTPException(422, "only algorithm 2 runs live; algorithm 1 "
                                     "is available in replay")
        if not body.planners:
            raise HTTPException(422, "pick at least one planner model")
        if body.question_id:
            if body.question_id not in s.by_id:
                raise HTTPException(404, f"no question {body.question_id!r}")
        elif not (body.image_id and body.question):
            raise HTTPException(422, "provide question_id, or image_id with "
                                     "a question")
        run_id = await s.submit(body.model_dump())
        return {"run_id": run_id, "queue_position": s.queue_position(run_id)}

    @app.get("/run/{run_id}/events")
    async def run_events(request: Request, run_id: str):
        s = st(request)
        events = s.runs.get(run_id)
        if events is None:
            raise HTTPException(404, f"no run {run_id!r}")
        last = request.headers.get("last-event-id")
        last_id = int(last) if last and last.isdigit() else None
        return StreamingResponse(
            events.stream(last_id), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/run/{run_id}")
    def run_snapshot(request: Request, run_id: str):
        s = st(request)
        events = s.runs.get(run_id)
        if events is None:
            raise HTTPException(404, f"no run {run_id!r}")
        return {"run_id": run_id, "finished": events.finished,
                "request": {k: v for k, v in s.runs.info[run_id].items()
                            if k not in ("cancelled",)},
                "events": [{"seq": q, "event": n, "data": d}
                           for q, n, d in events.events]}

    @app.post("/run/{run_id}/cancel")
    async def run_cancel(request: Request, run_id: str):
        s = st(request)
        events = s.runs.get(run_id)
        if events is None:
            raise HTTPException(404, f"no run {run_id!r}")
        s.runs.info[run_id]["cancelled"] = True
        if not events.finished:
            await events.publish("error", {
                "stage": "cancel", "message": "Run cancelled.",
                "announce": "Run cancelled"})
        return {"cancelled": True}

    # ── benchmarks ───────────────────────────────────────────

    @app.get("/benchmarks")
    def get_benchmarks():
        return benchmarks.payload()

    # ── PDF chart QA ─────────────────────────────────────────

    @app.post("/pdf")
    async def pdf_upload(request: Request, file: UploadFile):
        data = await file.read()
        if not data[:5] == b"%PDF-":
            raise HTTPException(422, "that file does not look like a PDF")
        try:
            return request.app.state.pdfs.save(data)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(422, f"could not open the PDF: {exc}")

    @app.get("/pdf/{pdf_id}/thumb/{page}")
    def pdf_thumb(request: Request, pdf_id: str, page: int):
        path = request.app.state.pdfs.thumb_path(pdf_id, page)
        if path is None:
            raise HTTPException(404, "no such page thumbnail")
        return FileResponse(path)

    @app.get("/pdf/{pdf_id}/page/{page}")
    def pdf_page(request: Request, pdf_id: str, page: int, dpi: int = 150):
        try:
            png = request.app.state.pdfs.render_page(pdf_id, page, dpi)
        except (FileNotFoundError, IndexError) as exc:
            raise HTTPException(404, str(exc))
        return Response(png, media_type="image/png")

    @app.post("/pdf/{pdf_id}/extract")
    def pdf_extract(request: Request, pdf_id: str):
        try:
            cands = request.app.state.pdfs.extract_candidates(pdf_id)
        except FileNotFoundError:
            raise HTTPException(404, f"no PDF {pdf_id!r}")
        return {"candidates": cands}

    @app.post("/pdf/{pdf_id}/crop")
    def pdf_crop(request: Request, pdf_id: str, body: CropRequest):
        try:
            return request.app.state.pdfs.crop(pdf_id, body.page, body.bbox)
        except FileNotFoundError:
            raise HTTPException(404, f"no PDF {pdf_id!r}")
        except (ValueError, IndexError) as exc:
            raise HTTPException(422, str(exc))

    return app


app = create_app()
