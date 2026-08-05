from dualstream.arc_audit import arc_coherence_audit
from dualstream.arc_frame import ArcDecisionFrameV1, ArcConceptScore, HypothesisScore, with_crc32


def _frame(attempt_id: int, step_index: int, *, decision_type: str = "hypothesis_select", chosen: str | None = "h1", topk=None, concepts=None, artifacts=None):
    fr = ArcDecisionFrameV1(
        task_id="task-a",
        attempt_id=attempt_id,
        step_index=step_index,
        prompt_nonce=1,
        decision_type=decision_type,  # type: ignore[arg-type]
        chosen_hypothesis_id=chosen,
        topk_hypotheses=topk if topk is not None else [HypothesisScore("h1", 0.6), HypothesisScore("h2", 0.4)],
        concepts=concepts or [],
        artifacts=artifacts or {},
    )
    with_crc32(fr)
    return fr


def test_contiguous_step_index_passes():
    frames = [_frame(1, 0), _frame(1, 1, decision_type="candidate_render", chosen=None)]
    findings, summary = arc_coherence_audit(frames)
    assert "non_contiguous_step_index" not in summary["counts_by_kind"]
    assert summary["num_frames"] == 2


def test_missing_topk_and_chosen_not_in_topk_and_ambiguity_warning():
    frames = [
        _frame(1, 0, topk=[]),
        _frame(1, 1, chosen="h3", topk=[HypothesisScore("h1", 0.51), HypothesisScore("h2", 0.49)]),
    ]
    findings, summary = arc_coherence_audit(frames)
    kinds = {f.kind for f in findings}
    assert "missing_topk_for_hypothesis_select" in kinds
    assert "chosen_hypothesis_not_in_topk" in kinds
    assert "alternative_hypothesis_close_mass" in kinds
    assert summary["num_findings"] >= 3


def test_attempt_redundancy_detection():
    frames = [
        _frame(1, 0, decision_type="attempt_finalize", artifacts={"output_hash": "same"}),
        _frame(2, 0, decision_type="attempt_finalize", artifacts={"output_hash": "same"}),
    ]
    findings, _ = arc_coherence_audit(frames)
    assert any(f.kind == "attempt2_redundant_with_attempt1" for f in findings)


def test_integrity_required_behavior():
    frame = ArcDecisionFrameV1(
        task_id="task-a",
        attempt_id=1,
        step_index=0,
        prompt_nonce=1,
        decision_type="hypothesis_select",
        chosen_hypothesis_id="h1",
        topk_hypotheses=[HypothesisScore("h1", 0.9)],
    )
    findings, _ = arc_coherence_audit([frame], require_integrity=True)
    assert any(f.kind == "integrity_crc32_invalid_or_missing" for f in findings)


def test_concept_uncertainty_vs_confidence_warning():
    frames = [
        _frame(
            1,
            0,
            concepts=[ArcConceptScore(3101, 0.8)],
            artifacts={"solver_confidence": 0.95},
        )
    ]
    findings, _ = arc_coherence_audit(frames)
    assert any(f.kind == "uncertainty_high_vs_solver_confidence" for f in findings)
