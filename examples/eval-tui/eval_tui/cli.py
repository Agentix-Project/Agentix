"""CLI for the Agentix eval dashboard.

`agentix-eval-tui --demo 40` runs a synthetic, no-Docker demo. For real runs,
resolve a dataset/agent/provider exactly like `agentix-run`::

    agentix-eval-tui --dataset my_pkg:dataset --agent my_pkg:agent \\
        --provider docker --bundle eval:0.1.0 --n-concurrent 8
"""

from __future__ import annotations

import argparse
import importlib
import sys
from typing import Any

from .app import EvalDashboard


def _load(path: str) -> Any:
    module_name, sep, attr = path.partition(":")
    if not module_name or not sep or not attr:
        raise SystemExit(f"expected 'module:attr', got {path!r}")
    obj = getattr(importlib.import_module(module_name), attr)
    return obj() if isinstance(obj, type) else obj


def _load_provider(name_or_path: str) -> Any:
    if ":" in name_or_path:
        return _load(name_or_path)
    module = importlib.import_module(f"agentix.provider.{name_or_path}")
    classes = [
        value
        for key, value in vars(module).items()
        if isinstance(value, type) and key.endswith("Provider") and value.__module__ == module.__name__
    ]
    if len(classes) != 1:
        raise SystemExit(f"could not find a single *Provider class in agentix.provider.{name_or_path}")
    return classes[0]()


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="agentix-eval-tui", description="Live TUI for Agentix batch rollouts.")
    parser.add_argument("--demo", type=int, metavar="N", default=None, help="Run N synthetic instances (no Docker).")
    parser.add_argument("--dataset", help="Dataset adapter as 'module:attr'.")
    parser.add_argument("--agent", help="Agent adapter as 'module:attr'.")
    parser.add_argument("--provider", default="docker", help="Provider backend name or 'module:attr'.")
    parser.add_argument("--bundle", help="Agentix bundle reference (from `agentix build`).")
    parser.add_argument("--model", default=None)
    parser.add_argument("--n-concurrent", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    if args.demo is not None:
        from .demo import DemoAgent, DemoDataset, DemoProvider

        app = EvalDashboard(
            dataset=DemoDataset(args.demo),
            agent=DemoAgent(),
            provider=DemoProvider(),
            bundle="demo",
            n_concurrent=args.n_concurrent,
            run_title="Agentix · eval (demo)",
        )
    else:
        missing = [name for name in ("dataset", "agent", "bundle") if not getattr(args, name)]
        if missing:
            raise SystemExit(f"--{', --'.join(missing)} required (or use --demo N)")
        dataset = _load(args.dataset)
        instances = list(dataset.instances())
        if args.limit is not None:
            instances = instances[: args.limit]
        app = EvalDashboard(
            dataset=dataset,
            agent=_load(args.agent),
            provider=_load_provider(args.provider),
            bundle=args.bundle,
            model=args.model,
            instances=instances,
            n_concurrent=args.n_concurrent,
        )

    app.run()
    return 0
