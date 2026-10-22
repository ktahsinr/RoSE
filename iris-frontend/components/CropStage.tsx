"use client";

/** Pick the chart region on a rendered PDF page: click a detected candidate,
 *  draw a box with the pointer, or adjust with keyboard — the crop box is
 *  focusable (arrows move, Shift+arrows resize) and the exact coordinates
 *  are editable as numbers. Coordinates are PDF points (72 per inch). */

import { useRef, useState } from "react";
import { apiUrl } from "@/lib/api";

export interface Candidate {
  page: number;
  source: string;
  bbox: [number, number, number, number];
  page_size: [number, number];
}

export type Box = [number, number, number, number];

const NUDGE = 8; // points

export default function CropStage({
  pdfId,
  page,
  candidates,
  crop,
  onCrop,
}: {
  pdfId: string;
  page: number;
  candidates: Candidate[];
  crop: Box | null;
  onCrop: (box: Box | null) => void;
}) {
  const imgRef = useRef<HTMLImageElement>(null);
  const [pageSizePt, setPageSizePt] = useState<[number, number] | null>(null);
  const dragRef = useRef<{ x0: number; y0: number } | null>(null);

  const pageCandidates = candidates.filter((c) => c.page === page);

  const toPoints = (e: React.PointerEvent): [number, number] | null => {
    const img = imgRef.current;
    if (!img || !pageSizePt) return null;
    const rect = img.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * pageSizePt[0];
    const y = ((e.clientY - rect.top) / rect.height) * pageSizePt[1];
    return [
      Math.max(0, Math.min(x, pageSizePt[0])),
      Math.max(0, Math.min(y, pageSizePt[1])),
    ];
  };

  const onPointerDown = (e: React.PointerEvent) => {
    const pt = toPoints(e);
    if (!pt) return;
    (e.currentTarget as Element).setPointerCapture(e.pointerId);
    dragRef.current = { x0: pt[0], y0: pt[1] };
    onCrop([pt[0], pt[1], pt[0], pt[1]]);
  };
  const onPointerMove = (e: React.PointerEvent) => {
    const d = dragRef.current;
    const pt = toPoints(e);
    if (!d || !pt) return;
    onCrop([
      Math.min(d.x0, pt[0]),
      Math.min(d.y0, pt[1]),
      Math.max(d.x0, pt[0]),
      Math.max(d.y0, pt[1]),
    ]);
  };
  const onPointerUp = () => {
    dragRef.current = null;
  };

  const onBoxKeyDown = (e: React.KeyboardEvent) => {
    if (!crop || !pageSizePt) return;
    let [x0, y0, x1, y1] = crop;
    const resize = e.shiftKey;
    const map: Record<string, () => void> = {
      ArrowLeft: () => {
        if (resize) x1 = Math.max(x0 + NUDGE, x1 - NUDGE);
        else {
          x0 -= NUDGE;
          x1 -= NUDGE;
        }
      },
      ArrowRight: () => {
        if (resize) x1 = Math.min(pageSizePt[0], x1 + NUDGE);
        else {
          x0 += NUDGE;
          x1 += NUDGE;
        }
      },
      ArrowUp: () => {
        if (resize) y1 = Math.max(y0 + NUDGE, y1 - NUDGE);
        else {
          y0 -= NUDGE;
          y1 -= NUDGE;
        }
      },
      ArrowDown: () => {
        if (resize) y1 = Math.min(pageSizePt[1], y1 + NUDGE);
        else {
          y0 += NUDGE;
          y1 += NUDGE;
        }
      },
    };
    if (map[e.key]) {
      e.preventDefault();
      map[e.key]();
      onCrop([Math.max(0, x0), Math.max(0, y0), x1, y1]);
    }
  };

  const setCoord = (i: number, v: number) => {
    if (!crop) return;
    const next = [...crop] as Box;
    next[i] = v;
    onCrop(next);
  };

  const pct = (v: number, axis: 0 | 1) =>
    pageSizePt ? `${(v / pageSizePt[axis]) * 100}%` : "0%";

  return (
    <div className="flex flex-col gap-3">
      {pageCandidates.length > 0 ? (
        <div>
          <h3 className="font-bold">Detected charts on this page</h3>
          <div className="mt-1 flex flex-wrap gap-2">
            {pageCandidates.map((c, i) => (
              <button
                key={i}
                type="button"
                onClick={() => onCrop(c.bbox)}
                className="rounded border-2 border-accent bg-accent-soft px-3 py-1 font-semibold text-accent hover:opacity-90"
              >
                Candidate {i + 1} ({c.source === "embedded" ? "image" : "drawing"})
              </button>
            ))}
          </div>
        </div>
      ) : (
        <p className="text-muted">
          No charts were auto-detected on this page — draw a box around the
          chart instead.
        </p>
      )}

      <div
        className="relative touch-none select-none overflow-hidden rounded-lg border-2 border-line"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          ref={imgRef}
          src={apiUrl(`/pdf/${pdfId}/page/${page}?dpi=150`)}
          alt={`PDF page ${page + 1}, rendered. Draw a box around the chart with the pointer, or use the candidate buttons and coordinate fields.`}
          className="w-full"
          draggable={false}
          onLoad={(e) => {
            const img = e.currentTarget;
            setPageSizePt([
              (img.naturalWidth * 72) / 150,
              (img.naturalHeight * 72) / 150,
            ]);
          }}
        />
        {crop && pageSizePt && (
          <div
            tabIndex={0}
            role="group"
            aria-label={`Crop box at ${crop.map((v) => Math.round(v)).join(", ")} points. Arrow keys move it, Shift plus arrows resize it.`}
            onKeyDown={onBoxKeyDown}
            className="absolute border-4 border-amber bg-amber/10"
            style={{
              left: pct(crop[0], 0),
              top: pct(crop[1], 1),
              width: pct(crop[2] - crop[0], 0),
              height: pct(crop[3] - crop[1], 1),
            }}
          />
        )}
      </div>

      <fieldset className="flex flex-wrap items-end gap-2">
        <legend className="text-sm font-semibold">
          Crop box (PDF points — editable)
        </legend>
        {(["Left", "Top", "Right", "Bottom"] as const).map((name, i) => (
          <label key={name} className="flex flex-col text-sm font-semibold">
            {name}
            <input
              type="number"
              value={crop ? Math.round(crop[i]) : ""}
              onChange={(e) => setCoord(i, Number(e.target.value))}
              disabled={!crop}
              className="w-24 rounded border-2 border-line bg-bg px-2 py-1"
            />
          </label>
        ))}
        <button
          type="button"
          onClick={() => onCrop(null)}
          disabled={!crop}
          className="rounded border-2 border-line px-3 py-1 font-semibold hover:bg-surface2 disabled:opacity-40"
        >
          Clear box
        </button>
      </fieldset>
    </div>
  );
}
