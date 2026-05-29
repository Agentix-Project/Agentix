"""Batch SWE-bench rollouts on Agentix, driven by `agentix.runner`.

This is the `examples/eval-cc-swe` flow expressed through the reusable
runner: two small adapters plus one `run_rollouts(...)` call replace the
hand-written per-instance orchestration.

- `SweDataset` enumerates SWE-bench rows, builds each task image, resets
  `/testbed` to the base commit (`agentix.plugins.datasets.swe.prepare_env`),
  and scores a patch with the official harness
  (`agentix.plugins.datasets.swe.score`).
- `ClaudeCodeAgent` starts the in-sandbox Anthropic<->OpenAI bridge, runs the
  `claude` CLI against it, and extracts the diff with `agentix.bash.run`. The
  real provider call stays on the host (the bridge gateway owns the
  `openai.AsyncOpenAI` client).
- `--ground-truth` swaps in `GroundTruthAgent`, which submits each row's gold
  patch — reusing the identical scoring path for harness validation.

Run as::

    python main.py --bundle <ref> --openai-api-key sk-... [--limit N] [--concurrency K]
    python main.py --bundle <ref> --ground-truth --fail-on-unresolved
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import agentix.agents.claude_code as cc
import agentix.bridge.anthropic
import agentix.plugins.datasets.swe as swe
from agentix.bash import run as bash_run
from agentix.provider.docker import DockerProvider
from agentix.runner import AgentResult, run_rollouts
from datasets import load_dataset
from openai import AsyncOpenAI

WORKDIR = "/testbed"
_DIFF_CMD = (
    "cd /testbed && "
    "git -c core.fileMode=false add -A && "
    "git -c core.fileMode=false diff --cached --no-color --binary"
)


def _instance_image(instance: dict[str, Any], *, namespace: str, tag: str, arch: str) -> str:
    from swebench.harness.test_spec.test_spec import make_test_spec

    image = make_test_spec(instance, namespace=namespace).instance_image_key
    image = image.replace("arm64", arch).replace("x86_64", arch)
    if tag != "latest":
        image = f"{image.rsplit(':', 1)[0]}:{tag}"
    return image


class SweDataset:
    """A `agentix.runner.Dataset` over a SWE-bench split."""

    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        namespace: str,
        tag: str,
        arch: str,
        eval_timeout: float,
    ) -> None:
        self._rows = rows
        self._namespace = namespace
        self._tag = tag
        self._arch = arch
        self._eval_timeout = eval_timeout

    def instances(self) -> list[dict[str, Any]]:
        return list(self._rows)

    def image(self, instance: dict[str, Any]) -> str:
        return _instance_image(instance, namespace=self._namespace, tag=self._tag, arch=self._arch)

    async def setup(self, sandbox: Any, instance: dict[str, Any]) -> bool:
        prepared = await sandbox.remote(swe.prepare_env, workdir=WORKDIR, base_commit=instance["base_commit"])
        return bool(prepared.ok)

    async def score(self, sandbox: Any, instance: dict[str, Any], patch: str) -> dict[str, Any]:
        report = await sandbox.remote(
            swe.score,
            instance=instance,
            patch=patch,
            workdir=WORKDIR,
            eval_timeout=self._eval_timeout,
        )
        return dict(report)


class ClaudeCodeAgent:
    """A `agentix.runner.Agent` that runs Claude Code through the bridge."""

    def __init__(
        self,
        *,
        openai_base_url: str,
        openai_api_key: str,
        upstream_model: str,
        response_model: str,
        cc_timeout: float,
        max_turns: int | None,
    ) -> None:
        self._openai_base_url = openai_base_url
        self._openai_api_key = openai_api_key
        self._upstream_model = upstream_model
        self._response_model = response_model
        self._cc_timeout = cc_timeout
        self._max_turns = max_turns

    async def solve(self, sandbox: Any, instance: dict[str, Any], *, model: str | None) -> AgentResult:
        response_model = model or self._response_model
        gateway = agentix.bridge.anthropic.OpenAIGateway(
            client=AsyncOpenAI(base_url=self._openai_base_url, api_key=self._openai_api_key),
            upstream_model=self._upstream_model,
        )
        sandbox.register_namespace(gateway)

        svc = await sandbox.remote(agentix.bridge.anthropic.start_service, response_model=response_model)
        result = await sandbox.remote(
            cc.run,
            instruction=instance["problem_statement"],
            workdir=WORKDIR,
            timeout=self._cc_timeout,
            max_turns=self._max_turns,
            anthropic_base_url=svc.url,
            anthropic_model=response_model,
        )
        diff = await sandbox.remote(bash_run, command=_DIFF_CMD)
        try:
            await sandbox.remote(agentix.bridge.anthropic.stop_service, handle=svc)
        except Exception:
            pass
        return AgentResult(patch=diff.stdout, exit_code=result.exit_code)


class GroundTruthAgent:
    """A `agentix.runner.Agent` that submits each row's gold patch."""

    async def solve(self, sandbox: Any, instance: dict[str, Any], *, model: str | None) -> AgentResult:
        return AgentResult(patch=str(instance.get("patch") or ""), exit_code=0)


def _select_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.dataset_file:
        dataset = load_dataset(args.dataset, data_files=args.dataset_file, split=args.split)
    else:
        dataset = load_dataset(args.dataset, split=args.split)

    if args.instance_id:
        wanted = set(args.instance_id)
        rows = [dict(row) for row in dataset if row["instance_id"] in wanted]
        missing = sorted(wanted - {row["instance_id"] for row in rows})
        if missing:
            raise SystemExit(f"unknown --instance-id value(s): {', '.join(missing)}")
        return rows

    limit = args.limit
    if limit is None:
        limit = len(dataset) if args.ground_truth else 1
    return [dict(dataset[i]) for i in range(min(limit, len(dataset)))]


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch SWE-bench rollouts via agentix.runner.")
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--swebench-namespace", default="swebench")
    parser.add_argument("--swebench-tag", default="latest")
    parser.add_argument("--arch", default="x86_64", choices=["x86_64", "arm64"])
    parser.add_argument("--docker-platform", default=None)
    parser.add_argument("--dataset", default="princeton-nlp/SWE-bench_Verified")
    parser.add_argument("--dataset-file", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--instance-id", action="append", default=None)
    parser.add_argument("--ground-truth", action="store_true")
    parser.add_argument("--fail-on-unresolved", action="store_true")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--openai-base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--openai-api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--upstream-model", default=os.environ.get("UPSTREAM_MODEL", "gpt-4o-mini"))
    parser.add_argument("--response-model", default=os.environ.get("RESPONSE_MODEL", "claude-3-5-sonnet-latest"))
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--cc-timeout", type=float, default=1800)
    parser.add_argument("--eval-timeout", type=float, default=1800)
    parser.add_argument("--out", default="runs")
    return parser.parse_args(argv)


async def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if args.concurrency < 1:
        print("error: --concurrency must be >= 1", file=sys.stderr)
        return 2
    if not args.ground_truth and not args.openai_api_key:
        print("error: --openai-api-key (or OPENAI_API_KEY) is required", file=sys.stderr)
        return 2

    rows = _select_rows(args)
    if not rows:
        print("error: no instances selected", file=sys.stderr)
        return 2

    dataset = SweDataset(
        rows,
        namespace=args.swebench_namespace,
        tag=args.swebench_tag,
        arch=args.arch,
        eval_timeout=args.eval_timeout,
    )
    if args.ground_truth:
        agent: Any = GroundTruthAgent()
    else:
        agent = ClaudeCodeAgent(
            openai_base_url=args.openai_base_url,
            openai_api_key=args.openai_api_key,
            upstream_model=args.upstream_model,
            response_model=args.response_model,
            cc_timeout=args.cc_timeout,
            max_turns=args.max_turns,
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _persist(rollout: Any) -> None:
        (out_dir / f"{rollout.instance_id}.json").write_text(json.dumps(rollout.to_dict(), indent=2))
        verdict = "PASS" if rollout.resolved else (rollout.skipped or rollout.error or "FAIL")
        print(f"[{rollout.instance_id}] {verdict} ({rollout.duration_s:.1f}s)")

    print(f"selected {len(rows)} instance(s) (concurrency={args.concurrency}, ground_truth={args.ground_truth})")
    rollouts = await run_rollouts(
        dataset=dataset,
        agent=agent,
        provider=DockerProvider(),
        bundle=args.bundle,
        model=args.response_model,
        instances=rows,
        n_concurrent=args.concurrency,
        platform=args.docker_platform,
        on_result=_persist,
    )

    resolved = sum(1 for rollout in rollouts if rollout.resolved)
    (out_dir / "summary.json").write_text(json.dumps([r.to_dict() for r in rollouts], indent=2))
    print(f"\n{resolved}/{len(rollouts)} resolved")

    failures = [r for r in rollouts if not r.resolved]
    if failures:
        print(f"{len(failures)} unresolved: {', '.join(r.instance_id for r in failures)}", file=sys.stderr)
    if args.fail_on_unresolved and failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
