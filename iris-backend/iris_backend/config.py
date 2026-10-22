"""Every IRIS_* environment variable, read once into a Settings object."""

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from . import REPO_ROOT


def _bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


@dataclass
class Settings:
    dataset_json: "Path | None"
    image_dir: "Path | None"
    results_dir: Path
    pool_path: "Path | None"
    pool_stream: bool
    demo_mode: bool
    upload_dir: Path
    cors_origins: list = field(default_factory=list)

    @classmethod
    def from_env(cls) -> "Settings":
        dataset = os.environ.get("IRIS_DATASET_JSON")
        images = os.environ.get("IRIS_IMAGE_DIR")
        results = os.environ.get(
            "IRIS_RESULTS_DIR", str(REPO_ROOT / "chartqapro/results/decompose"))
        pool = os.environ.get("IRIS_POOL_PATH")
        upload = os.environ.get("IRIS_UPLOAD_DIR") or tempfile.mkdtemp(
            prefix="iris-uploads-")

        # Demo mode: explicit env wins; otherwise live only when CUDA exists.
        if "IRIS_DEMO_MODE" in os.environ:
            demo = _bool("IRIS_DEMO_MODE")
        else:
            demo = not _cuda_available()

        origins = [o.strip() for o in os.environ.get(
            "IRIS_CORS_ORIGINS",
            "http://localhost:3000,http://127.0.0.1:3000").split(",") if o.strip()]

        s = cls(
            dataset_json=Path(dataset) if dataset else None,
            image_dir=Path(images) if images else None,
            results_dir=Path(results),
            pool_path=Path(pool) if pool else None,
            pool_stream=_bool("IRIS_POOL_STREAM", False),
            demo_mode=demo,
            upload_dir=Path(upload),
            cors_origins=origins,
        )
        s.upload_dir.mkdir(parents=True, exist_ok=True)
        return s
