from __future__ import annotations


def integrity_pass_rate(total_frames: int, integrity_failures: int) -> float:
    if total_frames <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - (integrity_failures / float(total_frames))))


def hypothesis_consistency(total_finalize_events: int, contradictory_finalize_events: int) -> float:
    if total_finalize_events <= 0:
        return 1.0
    return max(0.0, min(1.0, 1.0 - (contradictory_finalize_events / float(total_finalize_events))))


def ambiguity_honesty_score(ambiguous_events: int, high_confidence_ambiguous_events: int) -> float:
    if ambiguous_events <= 0:
        return 1.0
    return max(0.0, min(1.0, 1.0 - (high_confidence_ambiguous_events / float(ambiguous_events))))


def attempt_diversity_score(total_tasks_with_two_attempts: int, redundant_attempt_pairs: int) -> float:
    if total_tasks_with_two_attempts <= 0:
        return 1.0
    return max(0.0, min(1.0, 1.0 - (redundant_attempt_pairs / float(total_tasks_with_two_attempts))))


def false_confidence_rate(total_finalize_events: int, false_confident_events: int) -> float:
    if total_finalize_events <= 0:
        return 0.0
    return max(0.0, min(1.0, false_confident_events / float(total_finalize_events)))


def coherence_score(
    *,
    integrity_pass_rate_value: float,
    hypothesis_consistency_value: float,
    ambiguity_honesty_score_value: float,
    attempt_diversity_score_value: float,
    false_confidence_rate_value: float,
) -> float:
    score = (
        0.30 * integrity_pass_rate_value
        + 0.25 * hypothesis_consistency_value
        + 0.20 * ambiguity_honesty_score_value
        + 0.15 * attempt_diversity_score_value
        + 0.10 * (1.0 - false_confidence_rate_value)
    )
    return max(0.0, min(1.0, score))
