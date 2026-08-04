from __future__ import annotations

import os
import sys
import types

from dualstream.offline import enforce_offline_env, preflight_model_assets
from dualstream.service import DualStreamService


def test_offline_preflight_succeeds_with_local_model_assets(tmp_path) -> None:
    model_dir = tmp_path / "local-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_text("x", encoding="utf-8")

    service = DualStreamService()
    result = service.preflight_generate(
        {
            "prompt": "hello",
            "model": str(model_dir),
            "offline": True,
            "outdir": str(tmp_path / "out"),
        }
    )

    assert result["ok"] is True
    assert result["model_source"] == "local_path"


def test_offline_preflight_fails_when_model_assets_missing(tmp_path) -> None:
    model_dir = tmp_path / "broken-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    service = DualStreamService()
    result = service.preflight_generate(
        {
            "prompt": "hello",
            "model": str(model_dir),
            "offline": True,
            "outdir": str(tmp_path / "out"),
        }
    )

    assert result["ok"] is False
    assert any("Missing required local model assets" in error for error in result["errors"])


def test_start_generate_forces_local_files_only_in_offline_mode(monkeypatch, tmp_path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_text("x", encoding="utf-8")

    observed: dict[str, bool] = {}

    class DummyGenerator:
        def __init__(self, _model, *, local_files_only=False, **_kwargs):
            observed["local_files_only"] = bool(local_files_only)
            observed["hf_hub_offline"] = os.environ.get("HF_HUB_OFFLINE")
            self.tokenizer = type("Tok", (), {"decode": staticmethod(lambda _ids: "tok")})()

        def generate(self, _prompt, _cfg):
            return {
                "answer": "ok",
                "frames": [],
                "frame_bytes": [],
                "prompt_nonce": 1,
                "model": "x",
                "running_hash": "y",
            }

    monkeypatch.setattr("dualstream.service.DualStreamGenerator", DummyGenerator)
    def fake_run_generation(*_args, **_kwargs):
        observed["hf_hub_offline_during_generate"] = os.environ.get("HF_HUB_OFFLINE") == "1"

    monkeypatch.setattr("dualstream.service._run_generation", fake_run_generation)
    monkeypatch.setattr("dualstream.service.load_generation_artifacts", lambda _outdir: {"ok": True})

    service = DualStreamService()
    job = service.start_generate(
        {
            "prompt": "hi",
            "model": str(model_dir),
            "offline": True,
            "outdir": str(tmp_path / "out"),
        }
    )

    import time

    for _ in range(200):
        state = service.get_job(job.id)
        assert state is not None
        if state.status in {"completed", "failed", "cancelled"}:
            break
        time.sleep(0.01)

    assert observed["local_files_only"] is True
    assert observed["hf_hub_offline"] is None
    assert observed["hf_hub_offline_during_generate"] is False


def test_enforce_offline_env_does_not_mutate_process_environment() -> None:
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    with enforce_offline_env(True):
        assert os.environ.get("HF_HUB_OFFLINE") is None
        assert os.environ.get("TRANSFORMERS_OFFLINE") is None


def test_offline_preflight_treats_cached_no_exist_sentinel_as_missing(monkeypatch) -> None:
    sentinel = object()

    hub_module = types.ModuleType("huggingface_hub")
    file_download_module = types.ModuleType("huggingface_hub.file_download")
    file_download_module._CACHED_NO_EXIST = sentinel

    def fake_try_to_load_from_cache(_model, filename, cache_dir=None):
        del cache_dir
        if filename == "config.json":
            return sentinel
        return "/tmp/fake-cache-hit"

    hub_module.try_to_load_from_cache = fake_try_to_load_from_cache

    monkeypatch.setitem(sys.modules, "huggingface_hub", hub_module)
    monkeypatch.setitem(sys.modules, "huggingface_hub.file_download", file_download_module)

    result = preflight_model_assets("org/model", offline=True)

    assert result["ok"] is False
    assert any("config.json" in error for error in result["errors"])
