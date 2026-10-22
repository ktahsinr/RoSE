"""Per-run event buffers for Server-Sent Events.

Every event a run emits is kept, so a client that connects late or
reconnects (Last-Event-ID) replays the missed events and then follows live.
Thread-safe on the publish side: live runs execute in a worker thread and
publish via the event loop.
"""

import asyncio
import json
import time


class RunEvents:
    def __init__(self):
        self.events = []                 # [(seq, name, data_dict)]
        self.finished = False
        self._cond = asyncio.Condition()

    async def publish(self, name: str, data: dict):
        async with self._cond:
            self.events.append((len(self.events), name, data))
            if name in ("done", "error"):
                self.finished = True
            self._cond.notify_all()

    def publish_threadsafe(self, loop: asyncio.AbstractEventLoop,
                           name: str, data: dict):
        asyncio.run_coroutine_threadsafe(self.publish(name, data), loop).result()

    async def stream(self, last_event_id: "int | None" = None):
        """Yield SSE-formatted events from after last_event_id, then live."""
        next_seq = (last_event_id + 1) if last_event_id is not None else 0
        while True:
            async with self._cond:
                while next_seq >= len(self.events) and not self.finished:
                    try:
                        await asyncio.wait_for(self._cond.wait(), timeout=15)
                    except asyncio.TimeoutError:
                        break
                batch = self.events[next_seq:]
                done = self.finished and next_seq + len(batch) >= len(self.events)
            if not batch:
                if self.finished:
                    return
                yield ": keep-alive\n\n"      # comment frame, ignored by clients
                continue
            for seq, name, data in batch:
                yield f"id: {seq}\nevent: {name}\ndata: {json.dumps(data)}\n\n"
                next_seq = seq + 1
            if done:
                return


class RunStore:
    def __init__(self):
        self.runs = {}        # {run_id: RunEvents}
        self.info = {}        # {run_id: request summary}
        self._n = 0

    def create(self, info: dict) -> str:
        self._n += 1
        run_id = f"run-{self._n:04d}-{int(time.time())}"
        self.runs[run_id] = RunEvents()
        self.info[run_id] = dict(info, created_at=time.time(), cancelled=False)
        return run_id

    def get(self, run_id: str) -> "RunEvents | None":
        return self.runs.get(run_id)
