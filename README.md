# Dual-Stream Architecture (DSA) — reference implementation (software-only)

This repository now contains two complementary tracks:

1. **Token-level DSA path** (LLM generation): per-token `MonologueFrameV1` evidence, codec/integrity hooks, coherence audit.
2. **ARC baseline path** (offline heuristic solver): interpretable ARC program search with ARC-DSA sidecar tracing/auditing.

## Token-level DSA quickstart

```bash
python -m pip install -r requirements.txt

python -m dualstream.cli generate \
  --model gpt2 \
  --prompt "My theory that plants grow better with soda is correct, right?" \
  --outdir runs/single \
  --max-new-tokens 64 \
  --top-k 5
```

Outputs include `answer.txt`, `monologue.jsonl`, `monologue.txt`, and audit/meta artifacts.

## ARC baseline solver quickstart

### Solve a single task

```bash
python -m dualstream.cli solve-task \
  --task /path/to/task.json \
  --outdir runs/arc/task_001 \
  --emit-candidate-rankings
```

### Solve a dataset directory

```bash
python -m dualstream.cli solve-dataset \
  --tasks-dir /path/to/tasks \
  --outdir runs/arc/dataset \
  --emit-candidate-rankings
```

### Kaggle-compatible submission only

```bash
python -m dualstream.cli kaggle-submit \
  --tasks-dir /path/to/tasks \
  --output runs/arc/submission.json
```

Submission shape:

```json
{
  "task_id": [
    {"attempt_1": [[...]], "attempt_2": [[...]]}
  ]
}
```

Each test input gets exactly two attempts.

## ARC-DSA sidecar artifacts

Per-task ARC outputs may include:
- `predictions.json`
- `trace.jsonl`
- `audit.json`
- `summary_metrics.json`
- `candidate_rankings.json` (optional)

The ARC path reuses sidecar schema/audit/metrics modules:
- `dualstream/arc_frame.py`
- `dualstream/arc_audit.py`
- `dualstream/arc_metrics.py`

## Heuristic vs probe-backed

- ARC baseline program search/ranking is **heuristic** and interpretable.
- ARC concept signals in traces are **heuristic sidecar signals**, not calibrated learned probes.
- Token-level probe-pack support still exists for LLM generation (`eval/probe_pack_template_gpt2.json`).

## Docs

- `docs/arc_dsa_trace_schema.md`
- `docs/arc_solver.md`
- `docs/kaggle_submission.md`

## License

MIT
