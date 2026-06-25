#!/usr/bin/env python
"""Check whether this machine can run the Pocket TTS Core AI conversion."""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import sys


def version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def main() -> int:
    print("Python:", sys.version.split()[0])
    print("Platform:", platform.platform())
    print("Machine:", platform.machine())
    for package in ("torch", "pocket-tts", "huggingface-hub", "coreai-torch", "coreai-core"):
        print(f"{package}:", version(package))

    try:
        import coreai_torch  # noqa: F401

        print("Core AI conversion import: ok")
    except Exception as exc:
        print("Core AI conversion import: unavailable")
        print(f"  {type(exc).__name__}: {exc}")

    try:
        completed = subprocess.run(
            ["hf", "auth", "whoami"],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0:
            print("Hugging Face auth: ok")
            print(completed.stdout.strip())
        else:
            print("Hugging Face auth: not logged in or unavailable")
            print((completed.stderr or completed.stdout).strip())
    except FileNotFoundError:
        print("Hugging Face auth: hf CLI not found")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
