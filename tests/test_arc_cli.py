import argparse
import json

from dualstream import cli


def _write_task(path):
    payload = {
        "train": [{"input": [[1, 0], [0, 0]], "output": [[0, 0], [1, 0]]}],
        "test": [{"input": [[2, 0], [0, 0]]}],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_solve_task_cli_smoke(tmp_path):
    task = tmp_path / "toy.json"
    _write_task(task)
    outdir = tmp_path / "out"

    args = argparse.Namespace(
        task=str(task),
        outdir=str(outdir),
        max_program_depth=2,
        max_candidates=64,
        beam_width=16,
        diversity_penalty=0.1,
        no_trace=False,
        no_integrity=False,
        emit_candidate_rankings=True,
    )
    rc = cli.cmd_solve_task(args)
    assert rc == 0
    assert (outdir / "submission.json").exists()
    assert (outdir / "trace.jsonl").exists()


def test_solve_dataset_cli_smoke(tmp_path):
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    _write_task(tasks_dir / "toy1.json")
    _write_task(tasks_dir / "toy2.json")
    outdir = tmp_path / "dataset_out"

    args = argparse.Namespace(
        tasks_dir=str(tasks_dir),
        outdir=str(outdir),
        max_program_depth=2,
        max_candidates=64,
        beam_width=16,
        diversity_penalty=0.1,
        no_trace=False,
        no_integrity=False,
        emit_candidate_rankings=False,
    )
    rc = cli.cmd_solve_dataset(args)
    assert rc == 0
    assert (outdir / "submission.json").exists()
