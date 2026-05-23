"""mitmproxy CLI entrypoint."""

from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Sequence
from pathlib import Path

ADDON_SCRIPT = Path(__file__).resolve().parent / "addon.py"


def mitmdump_args(argv: Sequence[str] | None = None) -> list[str]:
    args = list(argv if argv is not None else sys.argv[1:])
    if any(arg in {"-h", "--help", "--version", "--options", "--commands"} for arg in args):
        return args
    if not _has_option(args, "--mode", "-m"):
        args = ["--mode", os.getenv("ABRIDGE_MITM_MODE", "wireguard"), *args]
    if not _has_option(args, "--scripts", "-s"):
        args = ["-s", str(ADDON_SCRIPT), *args]
    return args


def _has_option(args: Sequence[str], *names: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in args for name in names)


def main(argv: Sequence[str] | None = None) -> int:
    """Run mitmdump with the bridge addon script."""
    try:
        mitm_main = importlib.import_module("mitmproxy.tools.main")
    except ModuleNotFoundError as exc:
        if exc.name == "mitmproxy":
            print(
                "mitmproxy is required for abridge-mitm. "
                "Install the mitm extra or run with `uv run --extra mitm abridge-mitm`.",
                file=sys.stderr,
            )
            return 2
        raise

    mitmdump = getattr(mitm_main, "mitmdump")
    mitmdump(mitmdump_args(argv))
    return 0
