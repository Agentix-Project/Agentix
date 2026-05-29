"""Thin CLI over `agentix.runner.run_rollouts`.

Resolves a dataset adapter, an agent adapter, and a provider backend, then
runs the agent over the dataset and writes a `Rollout` summary per
instance. The library API (`run_rollouts`) is the real interface; this
wrapper is for manual eval runs — an RL loop calls the function directly.

Example::

    agentix-run \\
        --dataset my_pkg.recipes:swe_dataset \\
        --agent my_pkg.recipes:claude_agent \\
        --provider docker \\
        --bundle eval:0.1.0 \\
        --model claude-3-5-sonnet-latest \\
        --n-concurrent 8 --out runs/
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

from .core import Rollout, run_rollouts

logger = logging.getLogger("agentix.runner.cli")


def _load(path: str) -> Any:
    """Resolve a `module:attr` adapter. If `attr` names a class or zero-arg
    factory it is called; an already-built object is used as-is."""
    module_name, sep, attr = path.partition(":")
    if not module_name or not sep or not attr:
        raise SystemExit(f"expected 'module:attr', got {path!r}")
    obj = getattr(importlib.import_module(module_name), attr)
    return obj() if isinstance(obj, type) else obj


def _load_provider(name_or_path: str) -> Any:
    """Resolve a provider. A bare name (`docker`) imports
    `agentix.provider.<name>` and instantiates its single `*Provider` class;
    a `module:attr` path resolves explicitly."""
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
    parser = argparse.ArgumentParser(prog="agentix-run", description="Batch rollout runner for Agentix.")
    parser.add_argument("--dataset", required=True, help="Dataset adapter as 'module:attr'.")
    parser.add_argument("--agent", required=True, help="Agent adapter as 'module:attr'.")
    parser.add_argument("--provider", default="docker", help="Provider backend name or 'module:attr'.")
    parser.add_argument("--bundle", required=True, help="Agentix bundle reference (from `agentix build`).")
    parser.add_argument("--model", default=None, help="Model id passed to the agent.")
    parser.add_argument("--platform", default=None, help="Container platform, e.g. linux/amd64.")
    parser.add_argument("--n-concurrent", type=int, default=1, help="Max instances in flight.")
    parser.add_argument("--limit", type=int, default=None, help="Run at most N instances.")
    parser.add_argument("--no-score", action="store_true", help="Skip the scoring phase.")
    parser.add_argument("--out", default="runs", help="Directory for per-instance summaries.")
    return parser.parse_args(argv)


async def _amain(argv: list[str]) -> int:
    args = _parse_args(argv)
    if args.n_concurrent < 1:
        print("error: --n-concurrent must be >= 1", file=sys.stderr)
        return 2

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    dataset = _load(args.dataset)
    agent = _load(args.agent)
    provider = _load_provider(args.provider)

    instances = list(dataset.instances())
    if args.limit is not None:
        instances = instances[: args.limit]
    if not instances:
        print("error: dataset produced no instances", file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _persist(rollout: Rollout) -> None:
        (out_dir / f"{rollout.instance_id}.json").write_text(json.dumps(rollout.to_dict(), indent=2))
        verdict = "PASS" if rollout.resolved else (rollout.skipped or rollout.error or "FAIL")
        print(f"[{rollout.instance_id}] {verdict} ({rollout.duration_s:.1f}s)")

    rollouts = await run_rollouts(
        dataset=dataset,
        agent=agent,
        provider=provider,
        bundle=args.bundle,
        model=args.model,
        instances=instances,
        n_concurrent=args.n_concurrent,
        platform=args.platform,
        score=not args.no_score,
        on_result=_persist,
    )

    resolved = sum(1 for rollout in rollouts if rollout.resolved)
    (out_dir / "summary.json").write_text(json.dumps([r.to_dict() for r in rollouts], indent=2))
    print(f"\n{resolved}/{len(rollouts)} resolved")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Console-script entry point (`agentix-run`)."""
    return asyncio.run(_amain(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    raise SystemExit(main())
