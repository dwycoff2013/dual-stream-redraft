"""Dual-Stream Architecture (DSA) software-only reference implementation.

This package is intentionally modular: low-level codecs (frames) can be imported without requiring
heavy ML dependencies. The generation wrapper requires torch + transformers.
"""

from .frame import MonologueFrameV1, AttnSummary, ConceptScore, TopKToken, encode_frame, decode_frame
from .render import render_monologue_text
from .audit import coherence_audit, CoherenceFinding

__all__ = [
    "MonologueFrameV1",
    "TopKToken",
    "AttnSummary",
    "ConceptScore",
    "encode_frame",
    "decode_frame",
    "render_monologue_text",
    "coherence_audit",
    "CoherenceFinding",
]

# Optional: generation wrapper (torch/transformers)
try:
    from .generator import DualStreamGenerator, GenerationConfig  # noqa: F401

    __all__ += ["DualStreamGenerator", "GenerationConfig"]
except Exception:
    # Allow importing codecs/render/audit without ML stack installed.
    pass
