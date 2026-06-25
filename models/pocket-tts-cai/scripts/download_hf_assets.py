#!/usr/bin/env python
"""Download Pocket TTS Hugging Face files for offline conversion attempts."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="kyutai/pocket-tts")
    parser.add_argument("--language", default="english")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--local-dir", type=Path, default=Path("models/hf/pocket-tts"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    allow_patterns = [
        "README.md",
        "tokenizer.model",
        f"languages/{args.language}/model.safetensors",
        f"languages/{args.language}/tokenizer.model",
        f"languages/{args.language}/embeddings/**",
        f"languages/{args.language}/embeddings_v*/**",
    ]
    path = snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=args.local_dir,
        allow_patterns=allow_patterns,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
