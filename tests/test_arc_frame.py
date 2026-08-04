from dualstream.arc_frame import ArcDecisionFrameV1, ArcConceptScore, HypothesisScore, verify_frame_crc32, with_crc32


def test_arc_decision_frame_to_dict_and_crc32():
    frame = ArcDecisionFrameV1(
        task_id="task-1",
        attempt_id=1,
        step_index=0,
        prompt_nonce=777,
        decision_type="hypothesis_select",
        chosen_hypothesis_id="h1",
        topk_hypotheses=[HypothesisScore(id="h1", score=0.8)],
        concepts=[ArcConceptScore(concept_id=3101, score=0.3)],
        artifacts={"candidate_hash": "abc"},
    )
    with_crc32(frame)
    d = frame.to_dict()
    assert d["task_id"] == "task-1"
    assert d["topk_hypotheses"][0]["id"] == "h1"
    assert d["crc32"] is not None
    assert verify_frame_crc32(frame)
