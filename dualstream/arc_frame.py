from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional
import binascii
import hashlib
import json

MAGIC_ARCD = 0x41524344  # 'ARCD'
VERSION_V1 = 0x0001
DecisionType = Literal["hypothesis_select", "candidate_render", "attempt_finalize"]


@dataclass(frozen=True)
class HypothesisScore:
    id: str
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return {"id": str(self.id), "score": float(self.score)}


@dataclass(frozen=True)
class ArcConceptScore:
    concept_id: int
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return {"concept_id": int(self.concept_id), "score": float(self.score)}


@dataclass
class ArcDecisionFrameV1:
    """ARC-side decision frame emitted for solver decisions (sidecar trace path)."""

    task_id: str
    attempt_id: int
    step_index: int
    prompt_nonce: int
    decision_type: DecisionType
    chosen_hypothesis_id: Optional[str]
    topk_hypotheses: list[HypothesisScore]
    concepts: list[ArcConceptScore] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    crc32: Optional[int] = None
    running_hash: Optional[str] = None

    magic: int = MAGIC_ARCD
    version: int = VERSION_V1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "magic": int(self.magic),
            "version": int(self.version),
            "task_id": str(self.task_id),
            "attempt_id": int(self.attempt_id),
            "step_index": int(self.step_index),
            "prompt_nonce": int(self.prompt_nonce),
            "decision_type": self.decision_type,
            "chosen_hypothesis_id": None if self.chosen_hypothesis_id is None else str(self.chosen_hypothesis_id),
            "topk_hypotheses": [h.to_dict() for h in self.topk_hypotheses],
            "concepts": [c.to_dict() for c in self.concepts],
            "artifacts": self.artifacts,
            "crc32": None if self.crc32 is None else int(self.crc32),
            "running_hash": self.running_hash,
        }


def canonical_payload_dict(frame: ArcDecisionFrameV1) -> Dict[str, Any]:
    d = frame.to_dict().copy()
    d.pop("crc32", None)
    d.pop("running_hash", None)
    return d


def canonical_payload_bytes(frame: ArcDecisionFrameV1) -> bytes:
    return json.dumps(canonical_payload_dict(frame), sort_keys=True, separators=(",", ":")).encode("utf-8")


def compute_crc32(frame: ArcDecisionFrameV1) -> int:
    return binascii.crc32(canonical_payload_bytes(frame)) & 0xFFFFFFFF


def with_crc32(frame: ArcDecisionFrameV1) -> ArcDecisionFrameV1:
    frame.crc32 = compute_crc32(frame)
    return frame


def update_running_hash(frames: list[ArcDecisionFrameV1], algo: str = "sha256") -> str:
    h = hashlib.new(algo)
    for frame in frames:
        h.update(canonical_payload_bytes(frame))
    return h.hexdigest()


def verify_frame_crc32(frame: ArcDecisionFrameV1) -> bool:
    if frame.crc32 is None:
        return False
    return int(frame.crc32) == compute_crc32(frame)


def verify_running_hash(frames: list[ArcDecisionFrameV1], expected_hex: str, algo: str = "sha256") -> bool:
    return update_running_hash(frames, algo=algo).lower() == expected_hex.lower()
