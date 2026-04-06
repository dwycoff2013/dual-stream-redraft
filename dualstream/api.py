from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from .service import DualStreamService

app = FastAPI(title="DualStream Desktop API", version="0.1.0")
service = DualStreamService()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/generate")
def generate(payload: dict) -> dict:
    job = service.start_generate(payload)
    return {"job_id": job.id, "status": job.status}


@app.post("/arc/solve-task")
def arc_solve_task(payload: dict) -> dict:
    job = service.start_arc_solve_task(payload)
    return {"job_id": job.id, "status": job.status}


@app.post("/arc/solve-dataset")
def arc_solve_dataset(payload: dict) -> dict:
    job = service.start_arc_solve_dataset(payload)
    return {"job_id": job.id, "status": job.status}


@app.post("/arc/kaggle-submit")
def arc_kaggle_submit(payload: dict) -> dict:
    job = service.start_kaggle_submit(payload)
    return {"job_id": job.id, "status": job.status}


@app.get("/jobs")
def list_jobs() -> list[dict]:
    return [asdict(job) for job in service.list_jobs()]


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return asdict(job)


@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict[str, bool]:
    ok = service.cancel_job(job_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"ok": True}


@app.get("/artifacts/{job_id}")
def get_artifacts(job_id: str) -> list[dict]:
    job = service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.output_dir:
        return []
    from .artifacts import discover_artifacts

    return discover_artifacts(job.output_dir)


@app.get("/artifacts/{job_id}/file/{artifact_path:path}")
def get_artifact_file(job_id: str, artifact_path: str) -> FileResponse:
    job = service.get_job(job_id)
    if not job or not job.output_dir:
        raise HTTPException(status_code=404, detail="Job or output directory not found")
    root = Path(job.output_dir).resolve()
    target = (root / artifact_path).resolve()
    if root not in target.parents and target != root:
        raise HTTPException(status_code=400, detail="Invalid artifact path")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(path=target)
