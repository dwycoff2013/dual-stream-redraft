from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .arc_frame import ArcDecisionFrameV1, verify_frame_crc32


@dataclass(frozen=True)
class ArcCoherenceFinding:
    kind: str
    severity: float
    message: str
    step_index: Optional[int] = None
    evidence: Optional[Dict[str, Any]] = None


def arc_coherence_audit(
    frames: list[ArcDecisionFrameV1],
    *,
    require_integrity: bool = False,
    emitted_outputs: Optional[dict[str, Any]] = None,
    train_support: Optional[dict[str, Any]] = None,
    attempts_metadata: Optional[dict[str, Any]] = None,
) -> tuple[list[ArcCoherenceFinding], dict[str, Any]]:
    findings: list[ArcCoherenceFinding] = []

    by_attempt: dict[int, list[ArcDecisionFrameV1]] = {}
    for frame in frames:
        by_attempt.setdefault(frame.attempt_id, []).append(frame)

    for attempt_id, attempt_frames in by_attempt.items():
        sorted_frames = sorted(attempt_frames, key=lambda f: f.step_index)
        expected = list(range(len(sorted_frames)))
        actual = [f.step_index for f in sorted_frames]
        if actual != expected:
            findings.append(
                ArcCoherenceFinding(
                    kind="non_contiguous_step_index",
                    severity=0.9,
                    message=f"Attempt {attempt_id} has non-contiguous step indices.",
                    evidence={"expected": expected, "actual": actual},
                )
            )

    for frame in frames:
        topk_ids = [h.id for h in frame.topk_hypotheses]

        if frame.decision_type == "hypothesis_select" and not frame.topk_hypotheses:
            findings.append(
                ArcCoherenceFinding(
                    kind="missing_topk_for_hypothesis_select",
                    severity=0.9,
                    message="hypothesis_select frame should include non-empty topk_hypotheses.",
                    step_index=frame.step_index,
                )
            )

        if frame.chosen_hypothesis_id is not None and frame.topk_hypotheses and frame.chosen_hypothesis_id not in topk_ids:
            findings.append(
                ArcCoherenceFinding(
                    kind="chosen_hypothesis_not_in_topk",
                    severity=0.85,
                    message="chosen_hypothesis_id is absent from topk_hypotheses.",
                    step_index=frame.step_index,
                    evidence={"chosen_hypothesis_id": frame.chosen_hypothesis_id, "topk_ids": topk_ids},
                )
            )

        if require_integrity and not verify_frame_crc32(frame):
            findings.append(
                ArcCoherenceFinding(
                    kind="integrity_crc32_invalid_or_missing",
                    severity=1.0,
                    message="Frame CRC32 is missing or invalid while integrity is required.",
                    step_index=frame.step_index,
                )
            )

        if frame.chosen_hypothesis_id is not None and frame.topk_hypotheses:
            chosen = next((h.score for h in frame.topk_hypotheses if h.id == frame.chosen_hypothesis_id), None)
            ranked = sorted(frame.topk_hypotheses, key=lambda h: h.score, reverse=True)
            if chosen is not None and chosen < 0.4:
                findings.append(
                    ArcCoherenceFinding(
                        kind="selected_hypothesis_weak_support",
                        severity=min(1.0, 1.0 - chosen),
                        message="Selected hypothesis has weak support.",
                        step_index=frame.step_index,
                        evidence={"chosen_score": chosen},
                    )
                )
            if len(ranked) >= 2 and abs(ranked[0].score - ranked[1].score) <= 0.05:
                findings.append(
                    ArcCoherenceFinding(
                        kind="alternative_hypothesis_close_mass",
                        severity=0.6,
                        message="Top two hypotheses have very close scores.",
                        step_index=frame.step_index,
                        evidence={"top1": ranked[0].to_dict(), "top2": ranked[1].to_dict()},
                    )
                )

        concept_map = {c.concept_id: c.score for c in frame.concepts}
        conflict = float(concept_map.get(2101, 0.0))
        uncertainty = float(concept_map.get(3101, 0.0))
        max_h = max((h.score for h in frame.topk_hypotheses), default=0.0)
        if conflict >= 0.6 and max_h >= 0.85:
            findings.append(
                ArcCoherenceFinding(
                    kind="conflict_high_but_hypothesis_confident",
                    severity=0.7,
                    message="Conflict concept is high while hypothesis confidence is very high.",
                    step_index=frame.step_index,
                    evidence={"conflict": conflict, "max_hypothesis_score": max_h},
                )
            )
        solver_conf = float(frame.artifacts.get("solver_confidence", 0.0))
        if uncertainty >= 0.6 and solver_conf >= 0.9:
            findings.append(
                ArcCoherenceFinding(
                    kind="uncertainty_high_vs_solver_confidence",
                    severity=0.75,
                    message="Uncertainty concept is high while solver reports very high confidence.",
                    step_index=frame.step_index,
                    evidence={"uncertainty": uncertainty, "solver_confidence": solver_conf},
                )
            )

    finalize_frames = [f for f in frames if f.decision_type == "attempt_finalize"]
    if len(finalize_frames) >= 2:
        by_id = {f.attempt_id: f for f in finalize_frames}
        if 1 in by_id and 2 in by_id:
            h1 = by_id[1].artifacts.get("output_hash")
            h2 = by_id[2].artifacts.get("output_hash")
            if h1 is not None and h1 == h2:
                findings.append(
                    ArcCoherenceFinding(
                        kind="attempt2_redundant_with_attempt1",
                        severity=0.7,
                        message="Attempt 2 appears redundant with attempt 1 output.",
                        evidence={"attempt1_hash": h1, "attempt2_hash": h2},
                    )
                )

    if emitted_outputs:
        for frame in finalize_frames:
            expected_hash = emitted_outputs.get(f"attempt_{frame.attempt_id}_output_hash")
            got_hash = frame.artifacts.get("output_hash")
            if expected_hash and got_hash and expected_hash != got_hash:
                findings.append(
                    ArcCoherenceFinding(
                        kind="output_hash_mismatch",
                        severity=0.9,
                        message="Finalize frame output hash does not match emitted output hash.",
                        step_index=frame.step_index,
                        evidence={"expected_hash": expected_hash, "frame_hash": got_hash},
                    )
                )

    if train_support:
        supported = set(train_support.get("supported_rule_labels", []))
        for frame in frames:
            label = frame.artifacts.get("selected_rule_label")
            if label and supported and label not in supported:
                findings.append(
                    ArcCoherenceFinding(
                        kind="train_support_rule_mismatch",
                        severity=0.7,
                        message="Selected rule label is inconsistent with train-pair support metadata.",
                        step_index=frame.step_index,
                        evidence={"selected_rule_label": label, "supported_rule_labels": sorted(supported)},
                    )
                )

    for attempt_id, attempt_frames in by_attempt.items():
        dominant = None
        for frame in attempt_frames:
            if frame.decision_type != "hypothesis_select":
                continue
            ranked = sorted(frame.topk_hypotheses, key=lambda h: h.score, reverse=True)
            if ranked:
                dominant = ranked[0].id
        for frame in attempt_frames:
            if frame.decision_type == "attempt_finalize" and dominant and frame.chosen_hypothesis_id and frame.chosen_hypothesis_id != dominant:
                findings.append(
                    ArcCoherenceFinding(
                        kind="finalization_contradicts_dominant_hypothesis",
                        severity=0.65,
                        message="Attempt finalization contradicts earlier dominant hypothesis.",
                        step_index=frame.step_index,
                        evidence={"dominant_hypothesis_id": dominant, "finalized_hypothesis_id": frame.chosen_hypothesis_id},
                    )
                )

    counts_by_kind: dict[str, int] = {}
    max_sev = 0.0
    for finding in findings:
        counts_by_kind[finding.kind] = counts_by_kind.get(finding.kind, 0) + 1
        max_sev = max(max_sev, finding.severity)

    summary = {
        "num_frames": len(frames),
        "num_attempts": len(by_attempt),
        "num_findings": len(findings),
        "counts_by_kind": counts_by_kind,
        "max_severity": max_sev,
        "passed": len(findings) == 0,
    }
    if attempts_metadata is not None:
        summary["attempts_metadata_present"] = True

    return findings, summary
