# Kaggle Submission Output

Use CLI:

```bash
python -m dualstream.cli kaggle-submit \
  --tasks-dir /path/to/arc/tasks \
  --output /path/to/submission.json
```

The emitted JSON shape is:

```json
{
  "task_id": [
    {"attempt_1": [[...]], "attempt_2": [[...]]},
    {"attempt_1": [[...]], "attempt_2": [[...]]}
  ]
}
```

Each test input gets exactly one object containing `attempt_1` and `attempt_2`.

For richer artifacts (trace/audit/metrics per task), use `solve-task` or `solve-dataset`.
