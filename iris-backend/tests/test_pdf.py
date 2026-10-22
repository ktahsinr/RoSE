"""PDF upload → thumbnails → candidates → crop, all through the API."""

import io

import pymupdf


def _make_pdf() -> bytes:
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)        # A4 portrait
    page.insert_text((72, 72), "A sample paper page with a chart below.")
    # A vector "chart": axes plus a few bars (clusters as a drawing region).
    page.draw_line((100, 500), (100, 300))
    page.draw_line((100, 500), (400, 500))
    for i, h in enumerate((60, 120, 90, 150)):
        x = 120 + i * 70
        page.draw_rect(pymupdf.Rect(x, 500 - h, x + 40, 500),
                       color=(0.2, 0.2, 0.6), fill=(0.4, 0.35, 0.8))
    data = doc.tobytes()
    doc.close()
    return data


def test_pdf_flow(client):
    up = client.post("/pdf", files={"file": ("paper.pdf", _make_pdf(),
                                             "application/pdf")})
    assert up.status_code == 200, up.text
    meta = up.json()
    assert meta["pages"] == 1 and len(meta["thumbnail_urls"]) == 1
    pdf_id = meta["pdf_id"]

    thumb = client.get(meta["thumbnail_urls"][0])
    assert thumb.status_code == 200
    assert thumb.headers["content-type"] == "image/png"

    page = client.get(f"/pdf/{pdf_id}/page/0", params={"dpi": 96})
    assert page.status_code == 200
    assert page.content[:8] == b"\x89PNG\r\n\x1a\n"
    assert client.get(f"/pdf/{pdf_id}/page/7").status_code == 404

    cands = client.post(f"/pdf/{pdf_id}/extract").json()["candidates"]
    assert cands, "the drawn bar chart should be detected"
    assert all(c["source"] in ("embedded", "layout") for c in cands)
    box = cands[0]

    crop = client.post(f"/pdf/{pdf_id}/crop",
                       json={"page": box["page"], "bbox": box["bbox"]})
    assert crop.status_code == 200, crop.text
    c = crop.json()
    assert c["width"] * c["height"] <= 1280 * 28 * 28    # rose.MAX_PIXELS
    img = client.get(c["url"])
    assert img.status_code == 200

    bad = client.post(f"/pdf/{pdf_id}/crop", json={"page": 0, "bbox": [0, 0, 0, 0]})
    assert bad.status_code == 422


def test_pdf_rejects_non_pdf(client):
    r = client.post("/pdf", files={"file": ("x.pdf", io.BytesIO(b"hello").read(),
                                            "application/pdf")})
    assert r.status_code == 422
