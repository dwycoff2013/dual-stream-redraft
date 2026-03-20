# ARC Baseline Solver

This repository now includes an offline ARC baseline solver in `dualstream/arc_solver.py`.

## Goals

- transparent, interpretable program search
- no default network dependency
- exactly two attempts per test input
- ARC-DSA sidecar trace/audit emission

## Architecture

- `dualstream/arc_task.py`: task loading, validation, Kaggle submission model
- `dualstream/arc_grid.py`: grid utilities (geometry, object extraction, hashing, scoring)
- `dualstream/arc_program.py`: interpretable transforms and composition
- `dualstream/arc_solver.py`: candidate generation, ranking, diversity-aware attempt selection, sidecar artifacts

## Scoring logic

Candidates are ranked by:
1. exact train-pair matches,
2. train fit ratio,
3. pixel accuracy,
4. slight preference for lower complexity.

Attempt 2 is selected with a diversity penalty to avoid duplicate hypotheses when alternatives exist.

## Sidecar outputs per task

- `predictions.json`
- `trace.jsonl`
- `audit.json`
- `summary_metrics.json`
- `candidate_rankings.json` (optional)

## Notes

This is a heuristic baseline, not a learned SOTA ARC system.
