from __future__ import annotations

import os

from dualstream.offline import enforce_offline_env
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
            observed["hf_hub_offline"] = os.environ.get("HF_HUB_OFFLINE") == "1"
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
    assert observed["hf_hub_offline"] is True
    assert observed["hf_hub_offline_during_generate"] is False


def test_enforce_offline_env_sets_expected_variables() -> None:
    with enforce_offline_env(True):
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_enforce_offline_env_serializes_contexts_across_modes() -> None:
    import threading
    import time

    order: list[str] = []
    entered = threading.Event()

    def offline_worker() -> None:
        with enforce_offline_env(True):
            order.append("offline-enter")
            entered.set()
            time.sleep(0.05)
            order.append("offline-exit")

    def online_worker() -> None:
        entered.wait(timeout=1)
        with enforce_offline_env(False):
            order.append("online-enter")

    t1 = threading.Thread(target=offline_worker)
    t2 = threading.Thread(target=online_worker)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert order == ["offline-enter", "offline-exit", "online-enter"]
