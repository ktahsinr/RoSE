"use client";

/** Accessible chart image viewer: zoom to 4×, pan (drag, touch or arrow
 *  keys), invert-colors toggle, reset — plus the text alternative: IRIS's
 *  own chart description under the image, readable and read-aloud-able. */

import { useCallback, useEffect, useRef, useState } from "react";
import ReadAloud from "./ReadAloud";

const ZOOM_STEPS = [1, 1.5, 2, 3, 4];
const PAN_STEP = 40;

export default function ChartViewer({
  src,
  description,
  title,
}: {
  src: string | null;
  description: string;
  title: string;
}) {
  const [zoomIdx, setZoomIdx] = useState(0);
  const [inverted, setInverted] = useState(false);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const dragRef = useRef<{ x: number; y: number; px: number; py: number } | null>(
    null,
  );
  const zoom = ZOOM_STEPS[zoomIdx];

  useEffect(() => {
    // New chart: reset the view.
    setZoomIdx(0);
    setPan({ x: 0, y: 0 });
  }, [src]);

  const reset = useCallback(() => {
    setZoomIdx(0);
    setPan({ x: 0, y: 0 });
  }, []);

  const onKeyDown = (e: React.KeyboardEvent) => {
    const move: Record<string, [number, number]> = {
      ArrowLeft: [PAN_STEP, 0],
      ArrowRight: [-PAN_STEP, 0],
      ArrowUp: [0, PAN_STEP],
      ArrowDown: [0, -PAN_STEP],
    };
    if (move[e.key] && zoom > 1) {
      e.preventDefault();
      setPan((p) => ({ x: p.x + move[e.key][0], y: p.y + move[e.key][1] }));
    } else if (e.key === "+" || e.key === "=") {
      e.preventDefault();
      setZoomIdx((i) => Math.min(i + 1, ZOOM_STEPS.length - 1));
    } else if (e.key === "-") {
      e.preventDefault();
      setZoomIdx((i) => Math.max(i - 1, 0));
    } else if (e.key === "0") {
      e.preventDefault();
      reset();
    }
  };

  const onPointerDown = (e: React.PointerEvent) => {
    if (zoom === 1) return;
    (e.target as Element).setPointerCapture(e.pointerId);
    dragRef.current = { x: e.clientX, y: e.clientY, px: pan.x, py: pan.y };
  };
  const onPointerMove = (e: React.PointerEvent) => {
    const d = dragRef.current;
    if (!d) return;
    setPan({ x: d.px + (e.clientX - d.x), y: d.py + (e.clientY - d.y) });
  };
  const onPointerUp = () => {
    dragRef.current = null;
  };

  return (
    <section aria-label="Chart" className="flex flex-col gap-3">
      <div
        role="toolbar"
        aria-label="Chart viewer controls"
        className="flex flex-wrap items-center gap-2"
      >
        <button
          type="button"
          aria-label="Zoom out"
          disabled={zoomIdx === 0 || !src}
          onClick={() => setZoomIdx((i) => Math.max(i - 1, 0))}
          className="rounded border-2 border-line px-3 py-1 font-bold hover:bg-surface2 disabled:opacity-40"
        >
          −
        </button>
        <span aria-live="off" className="min-w-14 text-center font-bold">
          {Math.round(zoom * 100)}%
        </span>
        <button
          type="button"
          aria-label="Zoom in"
          disabled={zoomIdx === ZOOM_STEPS.length - 1 || !src}
          onClick={() => setZoomIdx((i) => Math.min(i + 1, ZOOM_STEPS.length - 1))}
          className="rounded border-2 border-line px-3 py-1 font-bold hover:bg-surface2 disabled:opacity-40"
        >
          +
        </button>
        <button
          type="button"
          aria-pressed={inverted}
          disabled={!src}
          onClick={() => setInverted((v) => !v)}
          className={`rounded border-2 px-3 py-1 font-bold disabled:opacity-40 ${
            inverted
              ? "border-accent bg-accent-soft text-accent"
              : "border-line hover:bg-surface2"
          }`}
        >
          Invert colors
        </button>
        <button
          type="button"
          onClick={reset}
          disabled={!src}
          className="rounded border-2 border-line px-3 py-1 font-bold hover:bg-surface2 disabled:opacity-40"
        >
          Reset view
        </button>
        <span className="text-sm text-muted">
          When zoomed: drag or arrow keys to pan, + / − / 0 keys to zoom.
        </span>
      </div>

      <div
        tabIndex={src ? 0 : -1}
        role="group"
        aria-label={`${title}. Zoom ${Math.round(zoom * 100)} percent${
          inverted ? ", colors inverted" : ""
        }. Use plus, minus and arrow keys.`}
        onKeyDown={onKeyDown}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        className="relative min-h-64 overflow-hidden rounded-lg border-2 border-line bg-surface"
        style={{ touchAction: zoom > 1 ? "none" : "auto" }}
      >
        {src ? (
          // eslint-disable-next-line @next/next/no-img-element — the chart
          // comes from the IRIS backend; next/image optimization adds nothing.
          <img
            src={src}
            alt={description || title}
            draggable={false}
            className="w-full select-none"
            style={{
              transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
              transformOrigin: "center center",
              filter: inverted ? "invert(1) hue-rotate(180deg)" : undefined,
              cursor: zoom > 1 ? "grab" : "default",
            }}
          />
        ) : (
          <p className="p-6 text-muted">No chart selected yet.</p>
        )}
      </div>

      <div className="rounded-lg border-2 border-line bg-surface p-3">
        <div className="mb-1 flex flex-wrap items-center justify-between gap-2">
          <h3 className="font-bold">Chart description</h3>
          {description && (
            <ReadAloud text={description} label="the chart description" />
          )}
        </div>
        <p className="text-fg">
          {description ||
            "IRIS writes its own description of the chart here during a run (it is also the image's alt text)."}
        </p>
      </div>
    </section>
  );
}
