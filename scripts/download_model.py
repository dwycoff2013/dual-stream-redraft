from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a Hugging Face model for offline use.")
    parser.add_argument("--model", default="gpt2", help="HF model ID to download")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional cache directory (defaults to HF_HOME/transformers cache)",
    )
    parser.add_argument(
        "--local-dir",
        default=None,
        help="Optional local directory to materialize a full copy of the repo snapshot",
    )
    args = parser.parse_args()

    local_dir = Path(args.local_dir).expanduser().resolve() if args.local_dir else None
    if local_dir is not None:
        local_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=args.model,
        cache_dir=args.cache_dir,
        local_dir=str(local_dir) if local_dir else None,
        local_dir_use_symlinks=False if local_dir else None,
    )


if __name__ == "__main__":
    main()
