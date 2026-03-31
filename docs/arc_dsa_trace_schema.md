# ARC-DSA Trace Schema (Sidecar Benchmark)

ARC-DSA is a **sidecar** benchmark and debugging harness that adapts Dual-Stream Architecture ideas to ARC/Kaggle solver decision tracing.

It differs from token-level DSA in this repository:
- Token-level DSA (`MonologueFrameV1`) traces answer token generation.
- ARC-DSA (`ArcDecisionFrameV1`) traces solver decisions (e.g., hypothesis selection, candidate rendering, and attempt finalization).

## Scope and intent

- ARC-DSA is **not** Kaggle's official scoring path.
- ARC-DSA is for traceability, diagnostics, and benchmark-style coherence checks.
- Concept probes are optional and probabilistic.
- Coherence findings do not prove correctness.
- ARC-DSA aligns to the repository assurance split:
  - **DSA-R**: software-only trace/debug mode.
  - **DSA-P**: production-assurance path requiring device-bound attestation (outside this ARC baseline harness).

## `ArcDecisionFrameV1`

Fields:
- `task_id: str`
- `attempt_id: int`
- `step_index: int`
- `prompt_nonce: int`
- `decision_type: "hypothesis_select" | "candidate_render" | "attempt_finalize"`
- `chosen_hypothesis_id: str | None`
- `topk_hypotheses: list[HypothesisScore]`
- `concepts: list[ArcConceptScore]`
- `artifacts: dict[str, Any]`
- `crc32: int | None`
- `running_hash: str | None`
- framing tags: `magic`, `version`

`HypothesisScore` contains `id` and `score`.
`ArcConceptScore` contains `concept_id` and `score`.

## Expected sidecar artifacts

- `submission.json`: solver submissions (if produced by a caller harness)
- `trace.jsonl`: per-decision ARC-DSA frames
- `audit.json`: coherence findings + summary
- `summary_metrics.json`: benchmark metric aggregates

## Probe pack plumbing

`eval/probe_pack_arc_v0.json` is a template vocabulary for ARC-oriented concepts/rules/conflicts/states. It is intentionally scaffold-level and should be replaced/tuned for production.

## AST-1 and frame metadata compatibility

Token-level `MonologueFrameV1` includes additive optional metadata for schema/probe/capture provenance:
- `signal_schema_id`, `signal_schema_hash`
- `probe_pack_id`, `probe_pack_hash`
- `capture_stage`, `decode_controls_applied`

These fields are additive and do not break existing frame consumers.

## Coherence checks (initial scaffold)

ARC audit checks include:
- structural checks (step continuity, top-k presence/consistency, optional integrity requirements)
- probabilistic checks (weak selected support, close alternative mass, redundant attempt 2)
- concept checks (conflict/uncertainty vs confidence)
- cross-stream checks (hash alignment, train-support label mismatch, finalize contradiction)

These are intentionally lightweight and expected to be tuned by downstream evaluators.
