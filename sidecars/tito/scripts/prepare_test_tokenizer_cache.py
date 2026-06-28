"""Prepare tokenizer-only HF cache assets for copied Miles upstream tests."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence

from huggingface_hub import snapshot_download


TOKENIZER_REPOS: tuple[str, ...] = (
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen3-4B-Thinking-2507",
    "Qwen/Qwen3-Next-80B-A3B-Thinking",
    "Qwen/Qwen3.5-0.8B",
    "zai-org/GLM-4.7-Flash",
)

ALLOW_PATTERNS: tuple[str, ...] = (
    "added_tokens.json",
    "chat_template*.jinja",
    "config.json",
    "configuration*.py",
    "generation_config.json",
    "merges.txt",
    "modeling*.py",
    "special_tokens_map.json",
    "tokenization*.py",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab*.json",
)

IGNORE_PATTERNS: tuple[str, ...] = (
    "*.bin",
    "*.gguf",
    "*.h5",
    "*.msgpack",
    "*.onnx",
    "*.pt",
    "*.safetensors",
    "*.tflite",
    "*.th",
    "*.weights",
)


def prepare_tokenizer_cache(repos: Sequence[str], *, endpoint: str | None = None) -> None:
    for repo_id in repos:
        print(f"Preparing tokenizer cache for {repo_id}")
        snapshot_download(
            repo_id=repo_id,
            endpoint=endpoint,
            allow_patterns=ALLOW_PATTERNS,
            ignore_patterns=IGNORE_PATTERNS,
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("HF_ENDPOINT"),
        help="HF endpoint to use. Defaults to HF_ENDPOINT when set.",
    )
    parser.add_argument(
        "--repo",
        action="append",
        dest="repos",
        help="Override repo list; may be passed multiple times.",
    )
    args = parser.parse_args(argv)

    prepare_tokenizer_cache(tuple(args.repos or TOKENIZER_REPOS), endpoint=args.endpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
