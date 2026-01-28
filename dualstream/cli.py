from __future__ import annotations

import argparse
import json
from pathlib import Path

from .generator import DualStreamGenerator, GenerationConfig
from .render import render_monologue_text
from .audit import coherence_audit


def cmd_generate(args: argparse.Namespace) -> int:
    cfg = GenerationConfig(
        model=args.model,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=not args.greedy,
        seed=args.seed,
        include_attn=args.include_attn,
        include_probes=args.include_probes,
        probe_pack_path=args.probe_pack,
        enable_heuristics=not args.no_heuristics,
        include_crc32=not args.no_crc32,
        include_running_hash=not args.no_running_hash,
        device=args.device,
    )

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    gen = DualStreamGenerator(cfg.model, device=cfg.device)
    result = gen.generate(args.prompt, cfg)

    frames = result["frames"]
    monologue_text = render_monologue_text(frames, tokenizer_decode=gen.tokenizer.decode)

    (outdir / "answer.txt").write_text(result["answer"], encoding="utf-8")
    (outdir / "monologue.txt").write_text(monologue_text, encoding="utf-8")

    # JSONL evidence frames
    with (outdir / "monologue.jsonl").open("w", encoding="utf-8") as f:
        for fr in frames:
            f.write(json.dumps(fr.to_dict(), ensure_ascii=False) + "\n")

    # Raw binary stream (concatenated frames) for low-level consumers
    with (outdir / "monologue.bin").open("wb") as f:
        for fb in result["frame_bytes"]:
            f.write(fb)

    meta = {
        "prompt_nonce": result["prompt_nonce"],
        "model": result["model"],
        "running_hash": result["running_hash"],
        "config": {
            "model": cfg.model,
            "max_new_tokens": cfg.max_new_tokens,
            "top_k": cfg.top_k,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "do_sample": cfg.do_sample,
            "include_attn": cfg.include_attn,
            "include_probes": cfg.include_probes,
            "probe_pack_path": cfg.probe_pack_path,
        },
    }
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Optional coherence audit
    findings = coherence_audit(result["answer"], frames, decode_token=lambda tid: gen.tokenizer.decode([tid]))
    (outdir / "audit.json").write_text(
        json.dumps([f.__dict__ for f in findings], indent=2),
        encoding="utf-8",
    )

    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dualstream", description="Dual-Stream Architecture reference implementation")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="Generate answer + monologue evidence for a prompt")
    g.add_argument("--model", default="gpt2", help="HF model name or path")
    g.add_argument("--prompt", required=True, help="User prompt")
    g.add_argument("--outdir", default=".", help="Output directory")
    g.add_argument("--max-new-tokens", type=int, default=128)
    g.add_argument("--top-k", type=int, default=5, help="Top-K evidence tokens per step (pre-sampling)")
    g.add_argument("--temperature", type=float, default=1.0)
    g.add_argument("--top-p", type=float, default=1.0)
    g.add_argument("--greedy", action="store_true", help="Disable sampling (argmax)")
    g.add_argument("--seed", type=int, default=None)

    g.add_argument("--include-attn", action="store_true", help="Emit attention summaries (slow)")
    g.add_argument("--include-probes", action="store_true", help="Run concept probes (requires hidden states; slow)")
    g.add_argument("--probe-pack", default=None, help="Path to probe pack JSON")
    g.add_argument("--no-heuristics", action="store_true", help="Disable heuristic concepts")

    g.add_argument("--no-crc32", action="store_true", help="Do not append crc32 to frames")
    g.add_argument("--no-running-hash", action="store_true", help="Disable running hash accumulation")
    g.add_argument("--device", default=None, help="Override device, e.g. cpu/cuda")

    g.set_defaults(func=cmd_generate)
    return p


def main() -> None:
    p = build_parser()
    args = p.parse_args()
    rc = args.func(args)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
