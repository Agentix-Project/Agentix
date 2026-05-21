"""Fail-fast SWE-bench Verified ground-truth runner for nightly CI.

This is intentionally narrower than ``runner.py``:

* ground-truth patches only, no agent phase or OpenAI configuration
* sequential execution inside each shard
* exit immediately on the first unresolved instance, patch-apply failure,
  skip, or runner exception
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

from datasets import load_dataset
from runner import _selected_instances, _should_fail, evaluate_ground_truth_one

logger = logging.getLogger("eval_cc_swe.ci_runner")


def _write_run_summary(out_dir: Path, summaries: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    failures = [s for s in summaries if _should_fail(s)]
    (out_dir / "summary.json").write_text(json.dumps(summaries, indent=2))
    if failures:
        (out_dir / "failures.json").write_text(json.dumps(failures, indent=2))


def _failure_reason(summary: dict[str, Any]) -> str:
    if summary.get("error"):
        return str(summary["error"])
    if summary.get("skipped"):
        return f"skipped: {summary['skipped']}"
    if not summary.get("patch_applied"):
        return "patch did not apply"
    if not summary.get("resolved"):
        ftp = summary.get("fail_to_pass", {}) or {}
        ptp = summary.get("pass_to_pass", {}) or {}
        return (
            "unresolved: "
            f"FAIL_TO_PASS failures={len(ftp.get('failure', []))}, "
            f"PASS_TO_PASS failures={len(ptp.get('failure', []))}"
        )
    return "unknown failure"


async def run_ci(args: argparse.Namespace) -> int:
    ds = load_dataset(args.dataset, split=args.split)
    instances = _selected_instances(
        ds,
        instance_ids=args.instance_id,
        limit=args.limit,
        ground_truth=True,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )
    if not instances:
        print("error: no instances selected", file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"selected {len(instances)} instance(s) (shard {args.shard_index + 1}/{args.num_shards}, fail_fast=True)")

    summaries: list[dict[str, Any]] = []
    for offset, inst in enumerate(instances, start=1):
        iid = inst["instance_id"]
        print(f"[ci] {offset}/{len(instances)} {iid}")
        try:
            summary = await evaluate_ground_truth_one(
                inst,
                bundle_image=args.bundle_image,
                swebench_namespace=args.swebench_namespace,
                swebench_tag=args.swebench_tag,
                arch=args.arch,
                docker_platform=args.docker_platform,
                eval_timeout=args.eval_timeout,
                out_dir=out_dir,
            )
        except Exception as exc:
            logger.exception("[%s] crashed", iid)
            summary = {
                "instance_id": iid,
                "mode": "ground_truth",
                "error": f"{type(exc).__name__}: {exc}",
            }
            (out_dir / f"{iid}.json").write_text(json.dumps(summary, indent=2))

        summaries.append(summary)
        _write_run_summary(out_dir, summaries)

        if _should_fail(summary):
            print(
                f"[ci] fail-fast: {iid} failed: {_failure_reason(summary)}",
                file=sys.stderr,
            )
            return 1

    print(f"\n{len(summaries)}/{len(summaries)} resolved")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bundle-image", default="eval-cc-swe:0.2.0")
    parser.add_argument("--swebench-namespace", default="swebench")
    parser.add_argument("--swebench-tag", default="latest")
    parser.add_argument("--arch", default="x86_64", choices=["x86_64", "arm64"])
    parser.add_argument(
        "--docker-platform",
        default=None,
        help="Docker platform for the runtime and task containers, e.g. linux/amd64.",
    )
    parser.add_argument("--dataset", default="princeton-nlp/SWE-bench_Verified")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of dataset rows to consider before sharding. Defaults to all rows.",
    )
    parser.add_argument("--instance-id", action="append", default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--eval-timeout", type=float, default=1800)
    parser.add_argument("--out", default="runs/ci-ground-truth")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
    )
    args = build_parser().parse_args(argv)
    return asyncio.run(run_ci(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
