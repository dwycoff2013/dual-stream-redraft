from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from dualstream.api import app, service
from dualstream.service import JobRecord


def test_root_route_serves_ui() -> None:
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "DualStream" in response.text


def test_static_assets_are_reachable() -> None:
    client = TestClient(app)
    css = client.get("/static/styles.css")
    js = client.get("/static/app.js")
    api_js = client.get("/static/api-client.js")
    assert css.status_code == 200
    assert js.status_code == 200
    assert api_js.status_code == 200


def test_jobs_endpoints_still_work() -> None:
    client = TestClient(app)
    response = client.get("/jobs")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_artifact_links_are_same_origin_and_well_formed(tmp_path: Path) -> None:
    outdir = tmp_path / "job-output"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "result.json").write_text("{}", encoding="utf-8")
    job = JobRecord(id="job-ui-test", kind="generate", status="completed", output_dir=str(outdir))

    with service._lock:  # noqa: SLF001
        service._jobs[job.id] = job  # noqa: SLF001

    client = TestClient(app)
    response = client.get(f"/artifacts/{job.id}")
    assert response.status_code == 200
    artifacts = response.json()
    assert artifacts
    url = artifacts[0]["url"]
    assert url.startswith(f"/artifacts/{job.id}/file/")
    assert not url.startswith("http")

    file_response = client.get(url)
    assert file_response.status_code == 200
