"""Host-side orchestrator: evaluate Claude Code on SWE-bench Verified.

Per instance, two sandboxes back-to-back:

    1. Agent sandbox (base = `swebench/sweb.eval.x86_64.<id>:latest`)
         - register `agentix.bridge.anthropic.OpenAIGateway` on the host RuntimeClient
         - c.remote(agentix.plugins.datasets.swe.prepare_env, /testbed, base_commit)
         - c.remote(agentix.bridge.anthropic.start_service, ...)
         - c.remote(agentix.agents.claude_code.run, ..., anthropic_base_url=svc.url)
         - c.remote(runner.get_patch, /testbed)

    2. Score sandbox (fresh container, no LLM gateway)
         - c.remote(agentix.plugins.datasets.swe.score, instance=..., patch=...)

Host wires an `openai.AsyncOpenAI` client into `AnthropicGateway` so
the actual provider call (model-eval, OpenRouter, etc.) lives on the
host. The sandbox-side `abridge` service just translates Anthropic ↔
OpenAI and routes via the `/abridge` SIO namespace.

Run as `python -m runner --limit N` or `python runner.py --limit N`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

import agentix.agents.claude_code as cc
import agentix.bridge.anthropic
import agentix.plugins.datasets.swe as swe
from agentix.deployment.docker import DockerDeployment
from datasets import load_dataset
from openai import AsyncOpenAI

from agentix import RuntimeClient
from agentix.deployment.base import SandboxConfig, session

WORKDIR = "/testbed"
logger = logging.getLogger("eval_cc_swe.runner")


def _instance_image(instance: dict, *, namespace: str, tag: str, arch: str) -> str:
    from swebench.harness.test_spec.test_spec import make_test_spec

    image = make_test_spec(instance, namespace=namespace).instance_image_key
    image = image.replace("arm64", arch).replace("x86_64", arch)
    if tag != "latest":
        image = f"{image.rsplit(':', 1)[0]}:{tag}"
    return image


def _load_instances_dataset(dataset: str, *, split: str, dataset_file: str | None = None):
    if dataset_file:
        return load_dataset(dataset, data_files=dataset_file, split=split)
    return load_dataset(dataset, split=split)


def _make_openai_client(*, base_url: str, api_key: str) -> AsyncOpenAI:
    return AsyncOpenAI(base_url=base_url, api_key=api_key)


async def get_patch(workdir: str = WORKDIR) -> str:
    """Return all `workdir` changes, including new files, as a unified diff."""
    proc = await asyncio.create_subprocess_shell(
        "git -c core.fileMode=false add -A && "
        "git -c core.fileMode=false diff --cached --no-color --binary",
        cwd=workdir,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, _ = await proc.communicate()
    return out.decode(errors="replace") if proc.returncode == 0 else ""


async def _run_agent_phase(
    inst: dict,
    *,
    cfg: SandboxConfig,
    openai_base_url: str,
    openai_api_key: str,
    upstream_model: str,
    response_model: str,
    cc_timeout: float,
    max_turns: int | None,
) -> dict:
    """Spin the agent sandbox: prepare env → start abridge → cc.run → get patch."""
    iid = inst["instance_id"]

    gateway = agentix.bridge.anthropic.OpenAIGateway(
        client=_make_openai_client(base_url=openai_base_url, api_key=openai_api_key),
        upstream_model=upstream_model,
    )

    async with session(DockerDeployment(), cfg) as sandbox:
        client = RuntimeClient(sandbox.runtime_url)
        client.register_namespace(gateway)
        async with client as c:
            prepared = await c.remote(
                swe.prepare_env,
                workdir=WORKDIR,
                base_commit=inst["base_commit"],
            )
            if not prepared.ok:
                print(f"[{iid}] prepare-env failed:\n{prepared.log[-500:]}")
                return {"instance_id": iid, "skipped": "prepare_env_failed"}
            print(f"[{iid}] HEAD={prepared.head[:12]}")

            svc = await c.remote(
                agentix.bridge.anthropic.start_service,
                response_model=response_model,
            )
            print(f"[{iid}] abridge service at {svc.url}")

            print(f"[{iid}] running claude (model={response_model})")
            cc_res = await c.remote(
                cc.run,
                instruction=inst["problem_statement"],
                workdir=WORKDIR,
                timeout=cc_timeout,
                max_turns=max_turns,
                anthropic_base_url=svc.url,
                anthropic_model=response_model,
            )
            print(f"[{iid}] claude exit={cc_res.exit_code}")
            if cc_res.stderr_tail:
                print(f"[{iid}] stderr_tail:\n{cc_res.stderr_tail.rstrip()}")

            patch = await c.remote(get_patch, workdir=WORKDIR)
            try:
                await c.remote(agentix.bridge.anthropic.stop_service, handle=svc)
            except Exception:
                pass

    return {"instance_id": iid, "patch": patch, "claude_exit": cc_res.exit_code}


async def _score_patch(
    inst: dict,
    *,
    patch: str,
    mode: str,
    bundle: str,
    swebench_namespace: str,
    swebench_tag: str,
    arch: str,
    docker_platform: str | None,
    eval_timeout: float,
    out_dir: Path,
) -> dict:
    """Run client2: prepare a fresh SWE image and score `patch`."""
    iid = inst["instance_id"]
    base_image = _instance_image(
        inst,
        namespace=swebench_namespace,
        tag=swebench_tag,
        arch=arch,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    (out_dir / f"{iid}.patch").write_text(patch)
    if not patch.strip():
        summary = {
            "instance_id": iid,
            "mode": mode,
            "skipped": "empty_patch",
            "resolved": False,
            "patch_applied": False,
            "timed_out": False,
            "test_status": {},
            "duration_s": round(time.time() - started, 1),
        }
        (out_dir / f"{iid}.json").write_text(json.dumps(summary, indent=2))
        return summary

    cfg = SandboxConfig(image=base_image, bundle=bundle, platform=docker_platform)
    print(f"[{iid}] score sandbox: {base_image}")
    print(f"[{iid}] patch_bytes={len(patch)}")
    async with session(DockerDeployment(), cfg) as sandbox:
        async with RuntimeClient(sandbox.runtime_url) as c:
            ev = await c.remote(
                swe.score,
                instance=inst,
                patch=patch,
                workdir=WORKDIR,
                eval_timeout=eval_timeout,
            )

    result = dict(ev)
    summary = {
        "instance_id": iid,
        "mode": mode,
        **result,
        "duration_s": round(time.time() - started, 1),
    }
    (out_dir / f"{iid}.json").write_text(json.dumps(summary, indent=2))

    verdict = "PASS" if summary["resolved"] else "FAIL"
    print(
        f"[{iid}] {verdict}  patch_applied={summary['patch_applied']}  "
        f"tests={len(summary['test_status'])}  "
        f"({summary['duration_s']}s)"
    )
    return summary


async def solve_one(
    inst: dict,
    *,
    bundle: str,
    swebench_namespace: str,
    swebench_tag: str,
    arch: str,
    docker_platform: str | None,
    openai_base_url: str,
    openai_api_key: str,
    upstream_model: str,
    response_model: str,
    cc_timeout: float,
    eval_timeout: float,
    max_turns: int | None,
    out_dir: Path,
) -> dict:
    iid = inst["instance_id"]
    base_image = _instance_image(
        inst,
        namespace=swebench_namespace,
        tag=swebench_tag,
        arch=arch,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = SandboxConfig(image=base_image, bundle=bundle, platform=docker_platform)

    print(f"[{iid}] agent sandbox: {base_image}")
    agent = await _run_agent_phase(
        inst,
        cfg=cfg,
        openai_base_url=openai_base_url,
        openai_api_key=openai_api_key,
        upstream_model=upstream_model,
        response_model=response_model,
        cc_timeout=cc_timeout,
        max_turns=max_turns,
    )
    if "patch" not in agent:
        return agent

    patch: str = agent["patch"]
    return await _score_patch(
        inst,
        patch=patch,
        mode="agent",
        bundle=bundle,
        swebench_namespace=swebench_namespace,
        swebench_tag=swebench_tag,
        arch=arch,
        docker_platform=docker_platform,
        eval_timeout=eval_timeout,
        out_dir=out_dir,
    )


def _selected_instances(
    ds,
    *,
    instance_ids: list[str] | None,
    limit: int | None,
    ground_truth: bool,
) -> list[dict]:
    if limit is not None and limit < 1:
        raise SystemExit("--limit must be >= 1")

    if instance_ids:
        wanted = set(instance_ids)
        instances = [dict(row) for row in ds if row["instance_id"] in wanted]
        found = {inst["instance_id"] for inst in instances}
        missing = sorted(wanted - found)
        if missing:
            raise SystemExit(f"unknown --instance-id value(s): {', '.join(missing)}")
    else:
        selected_limit = limit
        if selected_limit is None:
            selected_limit = len(ds) if ground_truth else 1
        instances = [dict(ds[i]) for i in range(min(selected_limit, len(ds)))]

    return instances


def _should_fail(summary: dict) -> bool:
    return not summary.get("patch_applied") or not summary.get("resolved")


async def _run_selected_instance(inst: dict, args: argparse.Namespace, out_dir: Path) -> dict:
    iid = inst["instance_id"]
    try:
        if args.ground_truth:
            return await _score_patch(
                inst,
                patch=inst.get("patch") or "",
                mode="ground_truth",
                bundle=args.bundle,
                swebench_namespace=args.swebench_namespace,
                swebench_tag=args.swebench_tag,
                arch=args.arch,
                docker_platform=args.docker_platform,
                eval_timeout=args.eval_timeout,
                out_dir=out_dir,
            )
        return await solve_one(
            inst,
            bundle=args.bundle,
            swebench_namespace=args.swebench_namespace,
            swebench_tag=args.swebench_tag,
            arch=args.arch,
            docker_platform=args.docker_platform,
            openai_base_url=args.openai_base_url,
            openai_api_key=args.openai_api_key,
            upstream_model=args.upstream_model,
            response_model=args.response_model,
            cc_timeout=args.cc_timeout,
            eval_timeout=args.eval_timeout,
            max_turns=args.max_turns,
            out_dir=out_dir,
        )
    except Exception as exc:
        # One instance blowing up (sandbox error, lost connection, ...)
        # must not abort the whole run; record and keep other tasks going.
        logger.exception("[%s] crashed", iid)
        return {"instance_id": iid, "error": f"{type(exc).__name__}: {exc}"}


async def _run_instances(instances: list[dict], args: argparse.Namespace, out_dir: Path) -> list[dict]:
    semaphore = asyncio.Semaphore(args.concurrency)

    async def run_with_slot(inst: dict) -> dict:
        async with semaphore:
            return await _run_selected_instance(inst, args, out_dir)

    return await asyncio.gather(*(run_with_slot(inst) for inst in instances))


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bundle", default="eval-cc-swe:0.2.0")
    parser.add_argument("--swebench-namespace", default="swebench")
    parser.add_argument("--swebench-tag", default="latest")
    parser.add_argument("--arch", default="x86_64", choices=["x86_64", "arm64"])
    parser.add_argument(
        "--docker-platform",
        default=None,
        help="Docker platform for the runtime and task containers, e.g. linux/amd64.",
    )
    parser.add_argument("--dataset", default="princeton-nlp/SWE-bench_Verified")
    parser.add_argument(
        "--dataset-file",
        default=None,
        help="Optional data file for dataset loaders such as parquet.",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of dataset rows to run. Defaults to 1 for agent mode and all rows for --ground-truth.",
    )
    parser.add_argument("--instance-id", action="append", default=None)
    parser.add_argument(
        "--ground-truth",
        action="store_true",
        help="Skip the agent phase and evaluate each row's SWE-bench gold patch.",
    )
    parser.add_argument(
        "--fail-on-unresolved",
        action="store_true",
        help="Exit non-zero if any selected instance is unresolved, fails patch apply, is skipped, or errors.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Maximum number of instances to run concurrently.",
    )
    parser.add_argument(
        "--openai-base-url",
        default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    parser.add_argument(
        "--openai-api-key",
        default=os.environ.get("OPENAI_API_KEY", ""),
    )
    parser.add_argument(
        "--upstream-model",
        default=os.environ.get("UPSTREAM_MODEL", "gpt-4o-mini"),
        help="Model id sent to the upstream OpenAI-compatible provider.",
    )
    parser.add_argument(
        "--response-model",
        default=os.environ.get("RESPONSE_MODEL", "claude-3-5-sonnet-latest"),
        help="Model id echoed back to the agent (claude CLI).",
    )
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--cc-timeout", type=float, default=1800)
    parser.add_argument("--eval-timeout", type=float, default=1800)
    parser.add_argument("--out", default="runs")
    args = parser.parse_args(argv)

    if not args.ground_truth and not args.openai_api_key:
        print(
            "error: --openai-api-key (or OPENAI_API_KEY) is required",
            file=sys.stderr,
        )
        return 2
    if args.concurrency < 1:
        print("error: --concurrency must be >= 1", file=sys.stderr)
        return 2

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
    )

    ds = _load_instances_dataset(args.dataset, split=args.split, dataset_file=args.dataset_file)
    instances = _selected_instances(
        ds,
        instance_ids=args.instance_id,
        limit=args.limit,
        ground_truth=args.ground_truth,
    )
    if not instances:
        print("error: no instances selected", file=sys.stderr)
        return 2
    print(
        f"selected {len(instances)} instance(s) "
        f"(concurrency={args.concurrency}, ground_truth={args.ground_truth})"
    )

    out_dir = Path(args.out)
    summaries = await _run_instances(instances, args, out_dir)

    resolved = sum(1 for s in summaries if s.get("resolved"))
    failures = [s for s in summaries if _should_fail(s)]
    print(f"\n{resolved}/{len(summaries)} resolved")
    if failures:
        failed_ids = ", ".join(str(s.get("instance_id")) for s in failures)
        print(f"{len(failures)} failed: {failed_ids}", file=sys.stderr)
    (out_dir / "summary.json").write_text(json.dumps(summaries, indent=2))
    if args.fail_on_unresolved and failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(sys.argv[1:])))
