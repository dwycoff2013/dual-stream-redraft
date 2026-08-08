from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any

def _is_local_model_path(model: str) -> Path | None:
    candidate = Path(model).expanduser()
    if candidate.exists():
        return candidate.resolve()
    return None


def _check_local_model_dir(model_dir: Path) -> list[str]:
    missing: list[str] = []
    if not model_dir.exists() or not model_dir.is_dir():
        return [f"Model path '{model_dir}' must be an existing directory."]

    if not (model_dir / "config.json").exists():
        missing.append("config.json")

    tokenizer_ok = any(
        (model_dir / name).exists()
        for name in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "spiece.model"]
    )
    if not tokenizer_ok:
        missing.append("tokenizer files (tokenizer.json/tokenizer_config.json/vocab.json/spiece.model)")

    weights_ok = any(
        any(model_dir.glob(pattern))
        for pattern in ["*.safetensors", "pytorch_model*.bin", "model*.safetensors", "model*.bin"]
    )
    if not weights_ok:
        missing.append("model weights (*.safetensors or *model*.bin)")

    return [f"Missing required local model assets in '{model_dir}': {', '.join(missing)}"] if missing else []


def _check_cached_model(model: str, cache_dir: str | None) -> list[str]:
    try:
        from huggingface_hub import try_to_load_from_cache
        from huggingface_hub.file_download import _CACHED_NO_EXIST
    except Exception:
        return [
            "huggingface_hub is required to validate cached models. Install dependencies and retry."
        ]

    def _get_cached_path(filename: str) -> str | None:
        try:
            cached = try_to_load_from_cache(model, filename, cache_dir=cache_dir)
            if cached is not None and cached is not _CACHED_NO_EXIST:
                return cached
            return None
        except Exception:
            return None

    def _exists(filename: str) -> bool:
        return _get_cached_path(filename) is not None

    missing: list[str] = []
    if not _exists("config.json"):
        missing.append("config.json")

    tokenizer_ok = any(_exists(name) for name in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "spiece.model"])
    if not tokenizer_ok:
        missing.append("tokenizer files")

    weights_ok = False
    for base_weight in ["model.safetensors", "pytorch_model.bin"]:
        if _exists(base_weight):
            weights_ok = True
            break
        index_file = base_weight + ".index.json"
        index_path = _get_cached_path(index_file)
        if index_path:
            try:
                import json
                with open(index_path, "r", encoding="utf-8") as f:
                    index_data = json.load(f)
                weight_map = index_data.get("weight_map", {})
                shards = set(weight_map.values())
                if shards and all(_exists(shard) for shard in shards):
                    weights_ok = True
                    break
            except Exception:
                pass

    if not weights_ok:
        missing.append("model weights")

    if missing:
        return [
            "Offline cache for model "
            f"'{model}' is incomplete. Missing: {', '.join(missing)}. "
            "Populate the cache while online, or provide a local model directory path."
        ]
    return []


def preflight_model_assets(model: str, offline: bool, cache_dir: str | None = None) -> dict[str, Any]:
    model_path = _is_local_model_path(model)
    source = "local_path" if model_path else "cached_model"

    errors: list[str] = []
    if offline:
        if model_path:
            errors.extend(_check_local_model_dir(model_path))
        else:
            errors.extend(_check_cached_model(model, cache_dir))

    return {
        "ok": not errors,
        "model_source": source,
        "model_path": str(model_path) if model_path else None,
        "errors": errors,
    }


@contextmanager
def enforce_offline_env(enabled: bool):
    """
    Keep API compatibility without mutating process-global environment state.

    Generation already enforces offline behavior via explicit `local_files_only=True`
    arguments passed to Transformers/Hugging Face loaders. Toggling `os.environ`
    in a multi-threaded service can create cross-job races because environment
    variables are process-wide, not request-scoped.
    """
    del enabled
    yield
