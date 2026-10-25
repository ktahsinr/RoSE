"""Application state and the single-worker run queue.

The GPU is shared, so live runs execute strictly one at a time: POST /run
enqueues, a single asyncio worker pops, and the blocking model work happens
in a thread while events stream out through RunStore. In demo mode there is
no engine and POST /run is rejected before anything is queued.
"""

import asyncio
import time

from .config import Settings
from .datasets import load_dataset
from .events import RunStore
from .replay import ReplayStore


class AppState:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.records = []            # dataset records, in dataset order
        self.by_id = {}
        self.replay = ReplayStore(settings.results_dir)
        self.runs = RunStore()
        self.queue = asyncio.Queue()
        self.worker_task = None
        self.engine = None           # LiveEngine when not in demo mode
        self.started_at = time.time()

        if settings.dataset_json and settings.dataset_json.exists():
            self.records = load_dataset(settings.dataset_json)
            self.by_id = {r["id"]: r for r in self.records}
        elif settings.dataset_json:
            print(f"⚠  IRIS_DATASET_JSON not found: {settings.dataset_json} "
                  "— serving 0 questions. Check the path (it must be "
                  "absolute or relative to the server's working directory).")
        else:
            print("⚠  IRIS_DATASET_JSON is not set — serving 0 questions.")
        if not self.replay.runs:
            print(f"ℹ  no stored run files in {settings.results_dir} — "
                  "replay disabled.")

    # ── queue ────────────────────────────────────────────────

    def queue_position(self, run_id: str) -> int:
        """0 = running next / now; -1 = not queued (finished or unknown)."""
        ids = [item[0] for item in self.queue._queue]  # peek, asyncio.Queue
        return ids.index(run_id) if run_id in ids else -1

    async def submit(self, request: dict) -> str:
        run_id = self.runs.create(request)
        events = self.runs.get(run_id)
        await events.publish("queued", {
            "position": self.queue.qsize(),
            "announce": ("Run queued" if self.queue.qsize() == 0 else
                         f"Run queued, {self.queue.qsize()} ahead of you"),
        })
        await self.queue.put((run_id, request))
        return run_id

    async def worker(self):
        loop = asyncio.get_running_loop()
        while True:
            run_id, request = await self.queue.get()
            events = self.runs.get(run_id)
            info = self.runs.info.get(run_id, {})
            try:
                if info.get("cancelled"):
                    await events.publish("done", {"cancelled": True,
                                                  "announce": "Run cancelled"})
                    continue
                if self.engine is None:
                    await events.publish("error", {
                        "stage": "startup", "message": "Live runs are disabled "
                        "(demo mode, no GPU).",
                        "announce": "Live runs are disabled in demo mode."})
                    continue
                # Blocking model work runs in a thread; it publishes its own
                # stage events through events.publish_threadsafe(loop, …).
                await asyncio.to_thread(
                    self.engine.execute, run_id, request, events, loop, info)
                await events.publish("done", {"announce": "Run finished"})
            except Exception as exc:  # noqa: BLE001 — surfaced to the client
                import traceback
                traceback.print_exc()
                await events.publish("error", {
                    "stage": "run", "message": str(exc),
                    "announce": f"The run failed: {exc}"})
            finally:
                self.queue.task_done()
