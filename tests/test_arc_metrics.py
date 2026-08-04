from dualstream.arc_metrics import (
    ambiguity_honesty_score,
    attempt_diversity_score,
    coherence_score,
    false_confidence_rate,
    hypothesis_consistency,
    integrity_pass_rate,
)


def test_coherence_score_sanity():
    ipr = integrity_pass_rate(total_frames=10, integrity_failures=1)
    hc = hypothesis_consistency(total_finalize_events=4, contradictory_finalize_events=1)
    ahs = ambiguity_honesty_score(ambiguous_events=4, high_confidence_ambiguous_events=1)
    ads = attempt_diversity_score(total_tasks_with_two_attempts=3, redundant_attempt_pairs=1)
    fcr = false_confidence_rate(total_finalize_events=4, false_confident_events=1)
    score = coherence_score(
        integrity_pass_rate_value=ipr,
        hypothesis_consistency_value=hc,
        ambiguity_honesty_score_value=ahs,
        attempt_diversity_score_value=ads,
        false_confidence_rate_value=fcr,
    )
    assert 0.0 <= score <= 1.0
    assert score > 0.5
