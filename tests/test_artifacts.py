from __future__ import annotations

import json

from dualstream.artifacts import load_arc_artifacts, load_generation_artifacts, summarize_audit, summarize_meta


def test_summarize_meta_and_audit() -> None:
    meta = {
        "model": "gpt2",
        "prompt_nonce": 123,
        "running_hash": "abc",
        "config": {"top_k": 5, "max_new_tokens": 10, "temperature": 1.0, "top_p": 0.95, "do_sample": True},
    }
    audit = [{"kind": "foo"}, {"kind": "foo"}, {"kind": "bar"}]
    meta_summary = summarize_meta(meta)
    audit_summary = summarize_audit(audit)
    assert meta_summary["model"] == "gpt2"
    assert audit_summary["num_findings"] == 3
    assert audit_summary["counts_by_kind"]["foo"] == 2


def test_load_generation_artifacts(tmp_path) -> None:
    out = tmp_path
    (out / "answer.txt").write_text("answer", encoding="utf-8")
    (out / "monologue.txt").write_text("mono", encoding="utf-8")
    (out / "monologue.jsonl").write_text('{"topk_tokens": [1]}\n', encoding="utf-8")
    (out / "meta.json").write_text(json.dumps({"model": "m", "config": {}}), encoding="utf-8")
    (out / "audit.json").write_text("[]", encoding="utf-8")
    data = load_generation_artifacts(out)
    assert data["answer_text"] == "answer"
    assert data["frames_summary"]["num_frames"] == 1


def test_load_arc_artifacts(tmp_path) -> None:
    out = tmp_path
    (out / "predictions.json").write_text("[]", encoding="utf-8")
    (out / "summary_metrics.json").write_text('{"coherence_score": 0.9}', encoding="utf-8")
    (out / "candidate_rankings.json").write_text("[]", encoding="utf-8")
    (out / "audit.json").write_text('{"summary": {"num_frames": 1}}', encoding="utf-8")
    (out / "trace.jsonl").write_text('{"decision_type": "x"}\n', encoding="utf-8")
    data = load_arc_artifacts(out)
    assert data["metrics"]["coherence_score"] == 0.9
    assert data["trace_summary"]["num_frames"] == 1
