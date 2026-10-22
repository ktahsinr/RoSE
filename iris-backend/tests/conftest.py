import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from iris_backend.config import Settings  # noqa: E402
from iris_backend.sampledata import make_sample_data  # noqa: E402


@pytest.fixture(scope="session")
def sample(tmp_path_factory):
    dest = tmp_path_factory.mktemp("sample")
    return make_sample_data(dest)


@pytest.fixture()
def demo_settings(sample, tmp_path):
    return Settings(
        dataset_json=sample["dataset_json"],
        image_dir=sample["image_dir"],
        results_dir=sample["results_dir"],
        pool_path=None, pool_stream=False, demo_mode=True,
        upload_dir=tmp_path / "uploads",
        cors_origins=["http://localhost:3000"],
    )


@pytest.fixture()
def client(demo_settings):
    from fastapi.testclient import TestClient
    from iris_backend.main import create_app
    demo_settings.upload_dir.mkdir(parents=True, exist_ok=True)
    with TestClient(create_app(demo_settings)) as c:
        yield c
