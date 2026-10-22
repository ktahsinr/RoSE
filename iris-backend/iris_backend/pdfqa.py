"""PDF chart extraction — upload, render, candidate figures, crop.

PyMuPDF only (local, free). Candidate detection is deliberately simple:
embedded raster images plus clusters of vector drawings (charts in papers
are usually vector line-art). Recall over precision — the frontend always
offers a manual crop box, so a bad candidate costs one click.
"""

import json
import secrets
from pathlib import Path

import pymupdf

# rose_chartqapro.MAX_PIXELS caps what the VLM processor sees; crops respect
# it. Imported lazily so demo mode never pulls the GPU stack's module.
_FALLBACK_MAX_PIXELS = 1280 * 28 * 28


def _max_pixels() -> int:
    try:
        import rose_chartqapro as rose
        return int(rose.MAX_PIXELS)
    except Exception:
        return _FALLBACK_MAX_PIXELS


THUMB_DPI = 40
MIN_CANDIDATE_SIDE_PT = 60        # ignore specks: candidates ≥ ~0.8 inch


class PdfStore:
    """Uploaded PDFs and the crops made from them, under the upload dir."""

    def __init__(self, upload_dir: Path):
        self.root = Path(upload_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    # ── paths ────────────────────────────────────────────────

    def _pdf_path(self, pdf_id: str) -> Path:
        return self.root / f"{pdf_id}.pdf"

    def image_path(self, image_id: str) -> "Path | None":
        p = self.root / f"{image_id}.png"
        return p if p.exists() else None

    # ── operations ───────────────────────────────────────────

    def save(self, data: bytes) -> dict:
        pdf_id = "pdf-" + secrets.token_hex(6)
        path = self._pdf_path(pdf_id)
        path.write_bytes(data)
        with pymupdf.open(path) as doc:
            n = doc.page_count
            for i in range(n):
                pix = doc[i].get_pixmap(dpi=THUMB_DPI)
                pix.save(self.root / f"{pdf_id}-thumb-{i}.png")
        return {"pdf_id": pdf_id, "pages": n,
                "thumbnail_urls": [f"/pdf/{pdf_id}/thumb/{i}" for i in range(n)]}

    def thumb_path(self, pdf_id: str, page: int) -> "Path | None":
        p = self.root / f"{pdf_id}-thumb-{page}.png"
        return p if p.exists() else None

    def render_page(self, pdf_id: str, page: int, dpi: int = 150) -> bytes:
        dpi = max(36, min(int(dpi), 300))
        with pymupdf.open(self._pdf_path(pdf_id)) as doc:
            if not 0 <= page < doc.page_count:
                raise IndexError(f"page {page} out of range")
            return doc[page].get_pixmap(dpi=dpi).tobytes("png")

    def extract_candidates(self, pdf_id: str) -> list:
        """Candidate figure regions per page, in PDF points (72/inch)."""
        out = []
        with pymupdf.open(self._pdf_path(pdf_id)) as doc:
            for pno in range(doc.page_count):
                page = doc[pno]
                seen = []

                def _add(rect, source):
                    r = pymupdf.Rect(rect) & page.rect
                    if r.is_empty or min(r.width, r.height) < MIN_CANDIDATE_SIDE_PT:
                        return
                    for other in seen:
                        if (r & other).get_area() > 0.8 * r.get_area():
                            return                      # mostly duplicate
                    seen.append(r)
                    out.append({
                        "page": pno, "source": source,
                        "bbox": [round(v, 1) for v in (r.x0, r.y0, r.x1, r.y1)],
                        "page_size": [round(page.rect.width, 1),
                                      round(page.rect.height, 1)],
                    })

                for img in page.get_images(full=True):
                    for rect in page.get_image_rects(img[0]):
                        _add(rect, "embedded")
                try:
                    for rect in page.cluster_drawings():
                        _add(rect, "layout")
                except Exception:
                    pass        # heuristic only; manual crop always remains
        return out

    def crop(self, pdf_id: str, page: int, bbox: list) -> dict:
        """Render a page region to a PNG no larger than rose.MAX_PIXELS."""
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            raise ValueError("bbox must be [x0, y0, x1, y1] in PDF points")
        with pymupdf.open(self._pdf_path(pdf_id)) as doc:
            if not 0 <= page < doc.page_count:
                raise IndexError(f"page {page} out of range")
            rect = pymupdf.Rect(*map(float, bbox)) & doc[page].rect
            if rect.is_empty:
                raise ValueError("bbox is empty or outside the page")
            # Pick the largest zoom that keeps the pixel count in budget.
            budget = _max_pixels()
            area_pt = rect.width * rect.height
            zoom = min(4.0, max(0.5, (budget / max(area_pt, 1.0)) ** 0.5))
            pix = doc[page].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom),
                                       clip=rect)
        image_id = "crop-" + secrets.token_hex(6)
        path = self.root / f"{image_id}.png"
        pix.save(path)
        meta = {"pdf_id": pdf_id, "page": page, "bbox": list(map(float, bbox))}
        (self.root / f"{image_id}.json").write_text(json.dumps(meta))
        return {"image_id": image_id, "url": f"/uploads/{image_id}",
                "width": pix.width, "height": pix.height}
