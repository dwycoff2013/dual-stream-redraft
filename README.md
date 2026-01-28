# dual-stream-redraft

**Dual-Stream Architecture (DSA)** reference implementation: an output contract that couples each user-facing token (Answer Stream) with a synchronized **Monologue evidence frame** (Monologue Stream), enabling automated **Coherence Audits** and CI-style regression tests.

> This repo is an engineering implementation of the v2.2 redraft whitepaper.  
> It does **not** claim solved alignment or proven guarantees. It aims to make **auditable inner-alignment signals** measurable under explicit assumptions.

---

## What this repo provides

### Core
- **DSA token-commit contract** (one evidence frame per answer token)
- **MonologueFrameV1** schema (typed telemetry; not natural-language reasoning)
- **Frame encoders**:
  - `JSONL` frames for inspection/logging
  - optional compact **binary** frames for high-throughput pipelines
- **Integrity hooks** (optional):
  - per-frame CRC32
  - optional running hash over the stream

### Tooling
- **CLI**: generate answer + monologue, render frames, run audits
- **Coherence Audit**: minimal structural + evidence/answer consistency checks
- **promptfoo harness**: a provider and JS assertions to run audit checks in eval/CI

### Optional instrumentation
- **Probe packs**: load lightweight concept/probe outputs into frames (template included)
- **Attention summaries**: compact attention head summaries (model/implementation dependent)

---

## Conceptual model (DSA in one paragraph)

DSA produces two synchronized outputs:

- **Answer Stream (A):** normal text tokens.
- **Monologue Stream (B):** typed evidence frames, one per committed token.

**DSA compliance invariant (core):**  
For every Answer token index `t`, there must exist exactly one `MonologueFrameV1` for that same token commit. Missing/malformed frames are verifier errors and must not be silently accepted.

The Monologue Stream is **telemetry evidence**, not “chain-of-thought.”

---

## Repository layout

