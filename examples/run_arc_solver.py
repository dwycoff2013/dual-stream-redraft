from __future__ import annotations

from pathlib import Path

from dualstream.arc_solver import ArcSolver, SolverConfig, write_submission, write_task_artifacts
from dualstream.arc_task import load_tasks_from_dir


def main() -> int:
    tasks_dir = Path("examples/arc_tasks")
    outdir = Path("runs/arc_baseline")

    if not tasks_dir.exists():
        print(f"No tasks found at {tasks_dir}; add ARC task JSON files there.")
        return 0

    solver = ArcSolver(SolverConfig())
    results = []
    for task in load_tasks_from_dir(tasks_dir):
        result = solver.solve_task(task)
        results.append(result)
        write_task_artifacts(result, outdir / task.task_id, include_rankings=True)
        print(f"Solved {task.task_id}: coherence={result.metrics['coherence_score']:.3f}")

    write_submission(results, outdir / "submission.json")
    print(f"Wrote submission to {outdir / 'submission.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
