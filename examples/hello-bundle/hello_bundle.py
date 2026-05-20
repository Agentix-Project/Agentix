"""Minimal remote target for the `agentix build` end-to-end test.

A bundle of this project is the smallest thing that still exercises the
real pipeline: a uv venv with the framework + a plugin, the Nix
toolchain, and a plugin-contributed system closure.
"""

from __future__ import annotations


def run(name: str = "world") -> dict[str, str]:
    """A trivial remote callable — `c.remote(run, name=...)`."""
    return {"greeting": f"hello, {name}"}
