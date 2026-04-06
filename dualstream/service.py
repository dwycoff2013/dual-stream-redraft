from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
import threading
import traceback
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from .artifacts import discover_artifacts, load_arc_artifacts, load_generation_artifacts
from .arc_solver import ArcSolver, SolverConfig, write_submission, write_task_artifacts
from .arc_task import load_task, load_tasks_from_dir
from .cli import _run_generation
from .generator import DualStreamGenerator, GenerationConfig


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class JobRecord:
    id: str
    kind: str
    status: str = "queued"
    created_at: str = field(default_factory=_utcnow)
    started_at: str | None = None
    ended_at: str | None = None
    progress: float = 0.0
    message: str = "Queued"
    output_dir: str | None = None
    error: str | None = None
    result: dict[str, Any] | None = None
    logs: list[str] = field(default_factory=list)


class DualStreamService:
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._jobs: dict[str, JobRecord] = {}
        self._cancel_flags: dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    def create_job(self, kind: str, runner: Callable[[JobRecord, threading.Event], dict[str, Any]]) -> JobRecord:
        job = JobRecord(id=str(uuid4()), kind=kind)
        cancel_flag = threading.Event()
        with self._lock:
            self._jobs[job.id] = job
            self._cancel_flags[job.id] = cancel_flag

        def wrapped() -> None:
            self._update(job.id, status="running", started_at=_utcnow(), progress=0.01, message="Starting")
            try:
                result = runner(job, cancel_flag)
                if cancel_flag.is_set() and job.status != "completed":
                    self._update(job.id, status="cancelled", message="Cancelled")
                else:
                    self._update(job.id, status="completed", progress=1.0, message="Completed", result=result)
            except Exception as exc:  # pragma: no cover
                self._update(
                    job.id,
                    status="failed",
                    message="Failed",
                    error=f"{exc}\n{traceback.format_exc()}",
                )
            finally:
                self._update(job.id, ended_at=_utcnow())

        self._executor.submit(wrapped)
        return job

    def _update(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            job = self._jobs[job_id]
            for k, v in changes.items():
                setattr(job, k, v)

    def _append_log(self, job_id: str, message: str) -> None:
        with self._lock:
            self._jobs[job_id].logs.append(message)

    def get_job(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self) -> list[JobRecord]:
        with self._lock:
            return list(self._jobs.values())

    def cancel_job(self, job_id: str) -> bool:
        with self._lock:
            flag = self._cancel_flags.get(job_id)
            job = self._jobs.get(job_id)
            if not flag or not job:
                return False
            flag.set()
            if job.status in {"queued", "running"}:
                job.message = "Cancellation requested"
            return True

    def start_generate(self, payload: dict[str, Any]) -> JobRecord:
        def run(job: JobRecord, cancelled: threading.Event) -> dict[str, Any]:
            prompt = payload.get("prompt")
            prompt_file = payload.get("prompt_file")
            if not prompt and prompt_file:
                prompt = Path(prompt_file).read_text(encoding="utf-8").strip()
            if not prompt:
                raise ValueError("Either prompt or prompt_file must be provided")

            outdir = Path(payload.get("outdir", ".")).resolve()
            outdir.mkdir(parents=True, exist_ok=True)
            self._update(job.id, output_dir=str(outdir), progress=0.05, message="Loading model")

            cfg = GenerationConfig(
                model=payload.get("model", "gpt2"),
                max_new_tokens=int(payload.get("max_new_tokens", 128)),
                top_k=int(payload.get("top_k", 5)),
                temperature=float(payload.get("temperature", 1.0)),
                top_p=float(payload.get("top_p", 1.0)),
                do_sample=not bool(payload.get("greedy", False)),
                seed=payload.get("seed"),
                include_attn=bool(payload.get("include_attn", False)),
                include_probes=bool(payload.get("include_probes", False)),
                probe_pack_path=payload.get("probe_pack"),
                enable_heuristics=not bool(payload.get("no_heuristics", False)),
                include_crc32=not bool(payload.get("no_crc32", False)),
                include_running_hash=not bool(payload.get("no_running_hash", False)),
                device=payload.get("device"),
                local_files_only=bool(payload.get("offline", False)),
                cache_dir=payload.get("cache_dir"),
            )

            gen = DualStreamGenerator(
                cfg.model,
                device=cfg.device,
                local_files_only=cfg.local_files_only,
                cache_dir=cfg.cache_dir,
            )
            if cancelled.is_set():
                return {"cancelled": True}

            self._update(job.id, progress=0.5, message="Generating")
            _run_generation(gen, cfg, prompt, outdir)
            self._update(job.id, progress=0.9, message="Loading artifacts")
            return load_generation_artifacts(outdir)

        return self.create_job("generate", run)

    def _solver_config(self, payload: dict[str, Any]) -> SolverConfig:
        return SolverConfig(
            max_program_depth=int(payload.get("max_program_depth", 2)),
            max_candidates=int(payload.get("max_candidates", 128)),
            beam_width=int(payload.get("beam_width", 24)),
            diversity_penalty=float(payload.get("diversity_penalty", 0.10)),
            emit_trace=not bool(payload.get("no_trace", False)),
            require_integrity=not bool(payload.get("no_integrity", False)),
            write_candidate_rankings=bool(payload.get("emit_candidate_rankings", False)),
        )

    def start_arc_solve_task(self, payload: dict[str, Any]) -> JobRecord:
        def run(job: JobRecord, cancelled: threading.Event) -> dict[str, Any]:
            task_path = payload["task"]
            outdir = Path(payload["outdir"]).resolve()
            outdir.mkdir(parents=True, exist_ok=True)
            self._update(job.id, output_dir=str(outdir), progress=0.1, message="Loading task")

            task = load_task(task_path)
            solver = ArcSolver(self._solver_config(payload))
            self._update(job.id, progress=0.45, message="Solving")
            result = solver.solve_task(task)
            if cancelled.is_set():
                return {"cancelled": True}

            write_task_artifacts(result, outdir, include_rankings=bool(payload.get("emit_candidate_rankings", False)))
            write_submission([result], outdir / "submission.json")
            self._update(job.id, progress=0.9, message="Loading artifacts")
            return load_arc_artifacts(outdir)

        return self.create_job("arc_solve_task", run)

    def start_arc_solve_dataset(self, payload: dict[str, Any]) -> JobRecord:
        def run(job: JobRecord, cancelled: threading.Event) -> dict[str, Any]:
            tasks_dir = payload["tasks_dir"]
            outdir = Path(payload["outdir"]).resolve()
            outdir.mkdir(parents=True, exist_ok=True)
            self._update(job.id, output_dir=str(outdir), progress=0.05, message="Loading dataset")

            tasks = load_tasks_from_dir(tasks_dir)
            solver = ArcSolver(self._solver_config(payload))
            results = []
            for idx, task in enumerate(tasks, start=1):
                if cancelled.is_set():
                    return {"cancelled": True, "completed_tasks": idx - 1}
                result = solver.solve_task(task)
                results.append(result)
                write_task_artifacts(result, outdir / task.task_id, include_rankings=bool(payload.get("emit_candidate_rankings", False)))
                self._update(job.id, progress=idx / max(len(tasks), 1), message=f"Solved {idx}/{len(tasks)} tasks")

            write_submission(results, outdir / "submission.json")
            summary = {
                "num_tasks": len(results),
                "tasks": [{"task_id": r.task_id, "metrics": r.metrics, "audit_summary": r.audit_summary} for r in results],
                "artifacts": discover_artifacts(outdir),
            }
            return summary

        return self.create_job("arc_solve_dataset", run)

    def start_kaggle_submit(self, payload: dict[str, Any]) -> JobRecord:
        def run(job: JobRecord, cancelled: threading.Event) -> dict[str, Any]:
            tasks = load_tasks_from_dir(payload["tasks_dir"])
            output = Path(payload["output"]).resolve()
            self._update(job.id, output_dir=str(output.parent), progress=0.1, message="Solving tasks")
            solver = ArcSolver(self._solver_config(payload))

            results = []
            for idx, task in enumerate(tasks, start=1):
                if cancelled.is_set():
                    return {"cancelled": True, "completed_tasks": idx - 1}
                results.append(solver.solve_task(task))
                self._update(job.id, progress=idx / max(len(tasks), 1), message=f"Solved {idx}/{len(tasks)} tasks")

            write_submission(results, output)
            return {
                "submission_path": str(output),
                "num_tasks": len(results),
                "preview": output.read_text(encoding="utf-8")[:4000],
                "artifacts": discover_artifacts(output.parent),
            }

        return self.create_job("kaggle_submit", run)
