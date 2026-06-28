"""Dependency-gated entrypoint for Miles session-server TITO e2e verification."""

from __future__ import annotations

from typing import Any


def run_from_args(args: Any) -> int:
    """Run the optional session TITO verifier when its e2e stack is available."""
    try:
        from miles.utils.test_utils.session_verify_runner import run_session_verify
    except Exception as exc:
        print(
            "verify-session-tito-tokenizer requires the optional Miles/SGLang "
            "session e2e runner.",
        )
        print(f"Missing runner detail: {type(exc).__name__}: {exc}")
        return 2

    try:
        run_session_verify(args=args)
    except AssertionError as exc:
        print("verify-session-tito-tokenizer failed TITO/session verification.")
        print(f"Verification detail: {exc}")
        return 1
    except (ImportError, ModuleNotFoundError, RuntimeError, FileNotFoundError, OSError, ValueError) as exc:
        print(
            "verify-session-tito-tokenizer requires the optional Miles/SGLang "
            "training stack and a valid verifier configuration.",
        )
        print(f"Dependency/runtime detail: {type(exc).__name__}: {exc}")
        return 2
    return 0
