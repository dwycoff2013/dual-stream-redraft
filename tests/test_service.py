from __future__ import annotations

import json
import time

from dualstream.service import DualStreamService


class DummyTask:
    task_id = "task1"


class DummyResult:
    task_id = "task1"
    attempts = []
    candidates = []
    frames = []
    findings = []
    audit_summary = {"num_frames": 0}
    metrics = {"coherence_score": 1.0}


def test_start_arc_solve_task(monkeypatch, tmp_path) -> None:
    service = DualStreamService()

    monkeypatch.setattr("dualstream.service.load_task", lambda _: DummyTask())

    class DummySolver:
        def __init__(self, *_args, **_kwargs):
            pass

        def solve_task(self, _task):
            return DummyResult()

    monkeypatch.setattr("dualstream.service.ArcSolver", DummySolver)

    def fake_write_task_artifacts(_result, outdir, include_rankings=True):
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "predictions.json").write_text("[]", encoding="utf-8")
        (outdir / "summary_metrics.json").write_text('{"coherence_score": 1.0}', encoding="utf-8")
        (outdir / "audit.json").write_text('{"summary": {"num_frames": 0}}', encoding="utf-8")
        (outdir / "trace.jsonl").write_text("", encoding="utf-8")

    monkeypatch.setattr("dualstream.service.write_task_artifacts", fake_write_task_artifacts)

    def fake_write_submission(_results, output):
        output.write_text(json.dumps({"ok": True}), encoding="utf-8")

    monkeypatch.setattr("dualstream.service.write_submission", fake_write_submission)

    task_path = tmp_path / "task.json"
    task_path.write_text("{}", encoding="utf-8")
    job = service.start_arc_solve_task({"task": str(task_path), "outdir": str(tmp_path)})

    # wait for completion
    for _ in range(200):
        state = service.get_job(job.id)
        assert state is not None
        if state.status in {"completed", "failed", "cancelled"}:
            break
        time.sleep(0.01)
    final = service.get_job(job.id)
    assert final is not None
    assert final.status == "completed"
    assert final.result is not None
    assert "metrics" in final.result
