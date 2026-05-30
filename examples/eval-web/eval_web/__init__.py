"""Agentix eval-web — a live web dashboard for batch rollouts.

A FastAPI server + a single dark dashboard page. It drives the *same* engine
the Textual TUI (`agentix-eval-tui`) uses — the demo provider, the
phase-tracing adapters, and `agentix.runner.run_rollouts` — and streams live
phase / result events to the browser over a WebSocket.
"""

from __future__ import annotations

from eval_web.app import create_app

__all__ = ["create_app"]
