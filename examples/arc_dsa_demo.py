from __future__ import annotations

import json
from pathlib import Path

from dualstream.arc_audit import arc_coherence_audit
from dualstream.arc_frame import ArcConceptScore, ArcDecisionFrameV1, HypothesisScore, with_crc32
from dualstream.arc_metrics import (
    ambiguity_honesty_score,
    attempt_diversity_score,
    coherence_score,
    false_confidence_rate,
    hypothesis_consistency,
    integrity_pass_rate,
)


def build_mock_trace() -> list[ArcDecisionFrameV1]:
    frames = [
        ArcDecisionFrameV1(
            task_id="mock-task-1",
            attempt_id=1,
            step_index=0,
            prompt_nonce=1,
            decision_type="hypothesis_select",
            chosen_hypothesis_id="h_rot",
            topk_hypotheses=[HypothesisScore("h_rot", 0.51), HypothesisScore("h_reflect", 0.49)],
            concepts=[ArcConceptScore(3101, 0.65)],
            artifacts={"solver_confidence": 0.95, "selected_rule_label": "ARC_RULE:ROTATION"},
        ),
        ArcDecisionFrameV1(
            task_id="mock-task-1",
            attempt_id=1,
            step_index=1,
            prompt_nonce=1,
            decision_type="attempt_finalize",
            chosen_hypothesis_id="h_rot",
            topk_hypotheses=[HypothesisScore("h_rot", 0.6)],
            artifacts={"output_hash": "abc123", "solver_confidence": 0.95},
        ),
        ArcDecisionFrameV1(
            task_id="mock-task-1",
            attempt_id=2,
            step_index=0,
            prompt_nonce=1,
            decision_type="attempt_finalize",
            chosen_hypothesis_id="h_reflect",
            topk_hypotheses=[HypothesisScore("h_reflect", 0.59)],
            artifacts={"output_hash": "abc123", "solver_confidence": 0.92},
        ),
    ]
    for frame in frames:
        with_crc32(frame)
    return frames


def main() -> int:
    frames = build_mock_trace()
    findings, summary = arc_coherence_audit(
        frames,
        require_integrity=True,
        train_support={"supported_rule_labels": ["ARC_RULE:TRANSLATION"]},
        emitted_outputs={"attempt_1_output_hash": "abc123", "attempt_2_output_hash": "abc123"},
    )

    metrics = {
        "integrity_pass_rate": integrity_pass_rate(len(frames), summary["counts_by_kind"].get("integrity_crc32_invalid_or_missing", 0)),
        "hypothesis_consistency": hypothesis_consistency(2, summary["counts_by_kind"].get("finalization_contradicts_dominant_hypothesis", 0)),
        "ambiguity_honesty_score": ambiguity_honesty_score(
            summary["counts_by_kind"].get("alternative_hypothesis_close_mass", 0),
            summary["counts_by_kind"].get("uncertainty_high_vs_solver_confidence", 0),
        ),
        "attempt_diversity_score": attempt_diversity_score(1, summary["counts_by_kind"].get("attempt2_redundant_with_attempt1", 0)),
        "false_confidence_rate": false_confidence_rate(2, summary["counts_by_kind"].get("uncertainty_high_vs_solver_confidence", 0)),
    }
    metrics["coherence_score"] = coherence_score(
        integrity_pass_rate_value=metrics["integrity_pass_rate"],
        hypothesis_consistency_value=metrics["hypothesis_consistency"],
        ambiguity_honesty_score_value=metrics["ambiguity_honesty_score"],
        attempt_diversity_score_value=metrics["attempt_diversity_score"],
        false_confidence_rate_value=metrics["false_confidence_rate"],
    )

    print("Findings:")
    for finding in findings:
        print(f"- [{finding.kind}] {finding.message}")
    print("Summary:", json.dumps(summary, indent=2))
    print("Metrics:", json.dumps(metrics, indent=2))

    outdir = Path("arc_dsa_demo_out")
    outdir.mkdir(exist_ok=True)
    (outdir / "trace.jsonl").write_text("\n".join(json.dumps(f.to_dict()) for f in frames), encoding="utf-8")
    (outdir / "audit.json").write_text(
        json.dumps(
            {
                "findings": [f.__dict__ for f in findings],
                "summary": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (outdir / "summary_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Wrote sidecar artifacts to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
