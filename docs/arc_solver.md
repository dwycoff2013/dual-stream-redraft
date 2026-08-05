from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
from typing import Any

from .arc_audit import ArcCoherenceFinding, arc_coherence_audit
from .arc_frame import ArcConceptScore, ArcDecisionFrameV1, HypothesisScore, with_crc32, update_running_hash
from .arc_grid import color_histogram, grids_equal, hash_grid, pixel_accuracy
from .arc_metrics import (
    ambiguity_honesty_score,
    attempt_diversity_score,
    coherence_score,
    false_confidence_rate,
    hypothesis_consistency,
    integrity_pass_rate,
)
from .arc_program import (
    Compose,
    CropToBoundingBox,
    FilterColor,
    Identity,
    KeepLargestObject,
    KeepSmallestObject,
    Program,
    RecolorMap,
    ReflectHorizontal,
    ReflectVertical,
    Rotate90,
    Rotate180,
    Rotate270,
    TilePattern,
    TranslateWholeGrid,
)
from .arc_task import ArcTask, PredictionAttempt, submission_dict_from_predictions


@dataclass(frozen=True)
class SolverConfig:
    max_program_depth: int = 2
    max_candidates: int = 128
    beam_width: int = 24
    enable_color_rules: bool = True
    enable_object_rules: bool = True
    enable_geometric_rules: bool = True
    enable_compositions: bool = True
    diversity_penalty: float = 0.10
    emit_trace: bool = True
    require_integrity: bool = True
    write_candidate_rankings: bool = True


@dataclass(frozen=True)
class CandidateScore:
    program: Program
    exact_train_match_count: int
    train_fit_score: float
    pixel_score: float
    complexity_penalty: float
    final_score: float
    support_features: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "program": self.program.to_dict(),
            "program_repr": self.program.stable_repr(),
            "exact_train_match_count": self.exact_train_match_count,
            "train_fit_score": self.train_fit_score,
            "pixel_score": self.pixel_score,
            "complexity_penalty": self.complexity_penalty,
            "final_score": self.final_score,
            "support_features": self.support_features,
        }


@dataclass(frozen=True)
class ArcSolveResult:
    task_id: str
    attempts: list[PredictionAttempt]
    candidates: list[CandidateScore]
    frames: list[ArcDecisionFrameV1]
    findings: list[ArcCoherenceFinding]
    audit_summary: dict[str, Any]
    metrics: dict[str, float]


class ArcSolver:
    def __init__(self, config: SolverConfig | None = None):
        self.config = config or SolverConfig()

    @staticmethod
    def _candidate_sort_key(c: CandidateScore) -> tuple[float, float, float, float, float]:
        return (
            c.exact_train_match_count,
            c.train_fit_score,
            c.pixel_score,
            -c.complexity_penalty,
            c.final_score,
        )

    @staticmethod
    def _sequence_output_hash(grids: list[list[list[int]]]) -> str:
        h = hashlib.sha256()
        for grid in grids:
            h.update(hash_grid(grid).encode("utf-8"))
        return h.hexdigest()

    def _build_atomic_programs(self, task: ArcTask) -> list[Program]:
        programs: list[Program] = [Identity()]

        if self.config.enable_geometric_rules:
            programs.extend([Rotate90(), Rotate180(), Rotate270(), ReflectHorizontal(), ReflectVertical()])
            programs.extend(
                [
                    TranslateWholeGrid(1, 0),
                    TranslateWholeGrid(-1, 0),
                    TranslateWholeGrid(0, 1),
                    TranslateWholeGrid(0, -1),
                ]
            )

        if self.config.enable_object_rules:
            programs.extend([CropToBoundingBox(), KeepLargestObject(), KeepSmallestObject()])

        if self.config.enable_color_rules:
            all_colors = set()
            for pair in task.train:
                all_colors.update(color_histogram(pair.input).keys())
                all_colors.update(color_histogram(pair.output).keys())
            for c in sorted(all_colors):
                programs.append(FilterColor(c))

            mapping = self._infer_color_map(task)
            if mapping:
                programs.append(RecolorMap(mapping))

        programs.extend([TilePattern(1, 2), TilePattern(2, 1), TilePattern(2, 2)])

        seen: set[str] = set()
        deduped: list[Program] = []
        for program in programs:
            key = program.stable_repr()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(program)
        return deduped

    def _infer_color_map(self, task: ArcTask) -> dict[int, int]:
        mapping: dict[int, int] = {}
        for pair in task.train:
            in_hist = color_histogram(pair.input)
            out_hist = color_histogram(pair.output)
            if len(in_hist) != len(out_hist):
                continue
            in_colors = sorted(in_hist.keys())
            out_colors = sorted(out_hist.keys())
            for src, dst in zip(in_colors, out_colors):
                existing = mapping.get(src)
                if existing is not None and existing != dst:
                    return {}
                mapping[src] = dst
        return mapping

    def _score_program(self, program: Program, task: ArcTask) -> CandidateScore:
        exact = 0
        pixel_scores: list[float] = []
        shape_matches = 0

        for pair in task.train:
            pred = program.apply(pair.input)
            if grids_equal(pred, pair.output):
                exact += 1
            if len(pred) == len(pair.output) and len(pred[0]) == len(pair.output[0]):
                shape_matches += 1
            pixel_scores.append(pixel_accuracy(pred, pair.output))

        train_fit = exact / len(task.train)
        mean_pixel = sum(pixel_scores) / len(pixel_scores)
        complexity = program.complexity_cost()
        final_score = 2.0 * train_fit + mean_pixel + 0.15 * (shape_matches / len(task.train)) - 0.05 * complexity

        return CandidateScore(
            program=program,
            exact_train_match_count=exact,
            train_fit_score=train_fit,
            pixel_score=mean_pixel,
            complexity_penalty=complexity,
            final_score=final_score,
            support_features={"shape_match_ratio": shape_matches / len(task.train)},
        )

    def _rank_candidates(self, task: ArcTask) -> list[CandidateScore]:
        atomic_programs = self._build_atomic_programs(task)
        scored_all: list[CandidateScore] = []
        seen_reprs: set[str] = set()

        def add_scored(candidate: CandidateScore) -> bool:
            key = candidate.program.stable_repr()
            if key in seen_reprs:
                return False
            seen_reprs.add(key)
            scored_all.append(candidate)
            return True

        scored_atoms = [self._score_program(program, task) for program in atomic_programs]
        scored_atoms.sort(key=self._candidate_sort_key, reverse=True)

        for candidate in scored_atoms:
            add_scored(candidate)
            if len(scored_all) >= self.config.max_candidates:
                break

        if not self.config.enable_compositions or self.config.max_program_depth <= 1:
            scored_all.sort(key=self._candidate_sort_key, reverse=True)
            return scored_all[: self.config.beam_width]

        seed_count = max(1, min(self.config.beam_width, len(scored_atoms)))
        extension_atoms = [c.program for c in scored_atoms[:seed_count]]
        current_layer = [c.program for c in scored_atoms[:seed_count]]

        for depth in range(2, self.config.max_program_depth + 1):
            if len(scored_all) >= self.config.max_candidates:
                break

            next_layer_scored: list[CandidateScore] = []

            for prefix in current_layer:
                prefix_steps = prefix.steps if isinstance(prefix, Compose) else (prefix,)
                for atom in extension_atoms:
                    composed = Compose(prefix_steps + (atom,))
                    candidate = self._score_program(composed, task)
                    if add_scored(candidate):
                        next_layer_scored.append(candidate)
                    if len(scored_all) >= self.config.max_candidates:
                        break
                if len(scored_all) >= self.config.max_candidates:
                    break

            if not next_layer_scored:
                break

            next_layer_scored.sort(key=self._candidate_sort_key, reverse=True)
            current_layer = [c.program for c in next_layer_scored[: self.config.beam_width]]

        scored_all.sort(key=self._candidate_sort_key, reverse=True)
        return scored_all[: self.config.beam_width]

    def _choose_diverse_second(self, ranked: list[CandidateScore], first: CandidateScore) -> CandidateScore:
        first_repr = first.program.stable_repr()
        first_type = first.program.to_dict().get("type")
        best_alt = first
        best_alt_score = -1e9

        for cand in ranked:
            penalty = 0.0
            if cand.program.stable_repr() == first_repr:
                penalty += 1.0
            if cand.program.to_dict().get("type") == first_type:
                penalty += 0.5
            diverse_score = cand.final_score - self.config.diversity_penalty * penalty
            if diverse_score > best_alt_score:
                best_alt_score = diverse_score
                best_alt = cand
        return best_alt

    def solve_task(self, task: ArcTask, *, prompt_nonce: int = 1) -> ArcSolveResult:
        ranked = self._rank_candidates(task)
        if not ranked:
            ranked = [self._score_program(Identity(), task)]

        topk = [HypothesisScore(id=c.program.stable_repr(), score=float(c.final_score)) for c in ranked[:5]]
        best = ranked[0]
        second = self._choose_diverse_second(ranked, best)

        frames: list[ArcDecisionFrameV1] = []
        if self.config.emit_trace:
            select = ArcDecisionFrameV1(
                task_id=task.task_id,
                attempt_id=1,
                step_index=0,
                prompt_nonce=prompt_nonce,
                decision_type="hypothesis_select",
                chosen_hypothesis_id=best.program.stable_repr(),
                topk_hypotheses=topk,
                concepts=self._concepts_from_ranked(ranked),
                artifacts={"solver_confidence": min(0.99, max(0.0, best.train_fit_score + 0.2))},
            )
            if self.config.require_integrity:
                with_crc32(select)
            frames.append(select)

        attempts: list[PredictionAttempt] = []
        for test_index, test_grid in enumerate(task.test):
            out1 = best.program.apply(test_grid)
            out2 = second.program.apply(test_grid)
            attempts.append(PredictionAttempt(attempt_1=out1, attempt_2=out2))

            if self.config.emit_trace:
                render = ArcDecisionFrameV1(
                    task_id=task.task_id,
                    attempt_id=1,
                    step_index=1 + test_index,
                    prompt_nonce=prompt_nonce,
                    decision_type="candidate_render",
                    chosen_hypothesis_id=best.program.stable_repr(),
                    topk_hypotheses=topk,
                    artifacts={
                        "test_index": test_index,
                        "candidate_program": best.program.to_dict(),
                        "candidate_output_hash": hash_grid(out1),
                    },
                )
                if self.config.require_integrity:
                    with_crc32(render)
                frames.append(render)

        attempt1_outputs = [attempt.attempt_1 for attempt in attempts]
        attempt2_outputs = [attempt.attempt_2 for attempt in attempts]
        attempt1_output_hash = self._sequence_output_hash(attempt1_outputs)
        attempt2_output_hash = self._sequence_output_hash(attempt2_outputs)

        if self.config.emit_trace:
            attempt1_next_step = max((f.step_index for f in frames if f.attempt_id == 1), default=-1) + 1
            attempt2_next_step = max((f.step_index for f in frames if f.attempt_id == 2), default=-1) + 1

            finalize1 = ArcDecisionFrameV1(
                task_id=task.task_id,
                attempt_id=1,
                step_index=attempt1_next_step,
                prompt_nonce=prompt_nonce,
                decision_type="attempt_finalize",
                chosen_hypothesis_id=best.program.stable_repr(),
                topk_hypotheses=topk,
                artifacts={
                    "output_hash": attempt1_output_hash,
                    "per_test_output_hashes": [hash_grid(grid) for grid in attempt1_outputs],
                    "num_test_outputs": len(attempt1_outputs),
                    "selected_rule_label": best.program.to_dict().get("type"),
                    "solver_confidence": min(0.99, max(0.0, best.train_fit_score + 0.2)),
                },
            )
            finalize2 = ArcDecisionFrameV1(
                task_id=task.task_id,
                attempt_id=2,
                step_index=attempt2_next_step,
                prompt_nonce=prompt_nonce,
                decision_type="attempt_finalize",
                chosen_hypothesis_id=second.program.stable_repr(),
                topk_hypotheses=topk,
                artifacts={
                    "output_hash": attempt2_output_hash,
                    "per_test_output_hashes": [hash_grid(grid) for grid in attempt2_outputs],
                    "num_test_outputs": len(attempt2_outputs),
                    "selected_rule_label": second.program.to_dict().get("type"),
                    "solver_confidence": min(0.99, max(0.0, second.train_fit_score + 0.2)),
                },
            )
            if self.config.require_integrity:
                with_crc32(finalize1)
                with_crc32(finalize2)
            frames.extend([finalize1, finalize2])

            if self.config.require_integrity:
                digest = update_running_hash(frames)
                for frame in frames:
                    frame.running_hash = digest

        findings, summary = arc_coherence_audit(
            frames,
            require_integrity=self.config.require_integrity,
            train_support={"supported_rule_labels": [c.program.to_dict().get("type") for c in ranked[:3]]},
            emitted_outputs={
                "attempt_1_output_hash": attempt1_output_hash,
                "attempt_2_output_hash": attempt2_output_hash,
            }
            if attempts
            else None,
        )
        metrics = self._metrics_from_findings(summary)

        return ArcSolveResult(
            task_id=task.task_id,
            attempts=attempts,
            candidates=ranked,
            frames=frames,
            findings=findings,
            audit_summary=summary,
            metrics=metrics,
        )

    def _concepts_from_ranked(self, ranked: list[CandidateScore]) -> list[ArcConceptScore]:
        concepts: list[ArcConceptScore] = []
        if len(ranked) >= 2 and abs(ranked[0].final_score - ranked[1].final_score) <= 0.05:
            concepts.append(ArcConceptScore(3104, 0.75))
        if ranked and ranked[0].train_fit_score < 1.0:
            concepts.append(ArcConceptScore(2104, 0.65))
            concepts.append(ArcConceptScore(3101, 0.6))
        return concepts

    def _metrics_from_findings(self, summary: dict[str, Any]) -> dict[str, float]:
        counts = summary.get("counts_by_kind", {})
        total_frames = int(summary.get("num_frames", 0))
        total_finalize = 2

        ipr = integrity_pass_rate(total_frames, counts.get("integrity_crc32_invalid_or_missing", 0))
        hc = hypothesis_consistency(total_finalize, counts.get("finalization_contradicts_dominant_hypothesis", 0))
        ahs = ambiguity_honesty_score(
            counts.get("alternative_hypothesis_close_mass", 0),
            counts.get("uncertainty_high_vs_solver_confidence", 0),
        )
        ads = attempt_diversity_score(1, counts.get("attempt2_redundant_with_attempt1", 0))
        fcr = false_confidence_rate(total_finalize, counts.get("uncertainty_high_vs_solver_confidence", 0))

        return {
            "integrity_pass_rate": ipr,
            "hypothesis_consistency": hc,
            "ambiguity_honesty_score": ahs,
            "attempt_diversity_score": ads,
            "false_confidence_rate": fcr,
            "coherence_score": coherence_score(
                integrity_pass_rate_value=ipr,
                hypothesis_consistency_value=hc,
                ambiguity_honesty_score_value=ahs,
                attempt_diversity_score_value=ads,
                false_confidence_rate_value=fcr,
            ),
        }


def write_task_artifacts(result: ArcSolveResult, outdir: str | Path, *, include_rankings: bool = True) -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    predictions = [attempt.to_dict() for attempt in result.attempts]
    (out / "predictions.json").write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    (out / "trace.jsonl").write_text("\n".join(json.dumps(frame.to_dict()) for frame in result.frames), encoding="utf-8")
    (out / "audit.json").write_text(
        json.dumps(
            {
                "findings": [asdict(f) for f in result.findings],
                "summary": result.audit_summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out / "summary_metrics.json").write_text(json.dumps(result.metrics, indent=2), encoding="utf-8")

    if include_rankings:
        (out / "candidate_rankings.json").write_text(
            json.dumps([c.to_dict() for c in result.candidates], indent=2),
            encoding="utf-8",
        )


def write_submission(results: list[ArcSolveResult], output_path: str | Path) -> None:
    data = submission_dict_from_predictions({r.task_id: r.attempts for r in results})
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
