"""Evaluate mini-swe-agent on SWE-bench Verified / Lite via Agentix.

Per instance, two sandboxes back-to-back:

    1. Agent sandbox (base = `swebench/sweb.eval.x86_64.<id>:latest`)
       - c.remote(agentix.plugins.datasets.swe.prepare_env, /testbed, base_commit)
       - construct a `DefaultAgent` host-side (mini-swe-agent + LiteLLM)
       - c.remote(agentix.agents.mini_swe_agent.run, agent=<built_agent>)
       - c.remote(get_patch, /testbed)

    2. Score sandbox (fresh container, no LLM access required)
       - c.remote(agentix.plugins.datasets.swe.score, instance=..., patch=...)

The agent is constructed on the host because `DefaultAgent` carries
a `LitellmModel` instance pre-wired with the upstream API key — we do
not want the key inside the sandbox. The whole `DefaultAgent` object
pickles cleanly across `client.remote(...)`.

Run:
    python runner.py --limit 1                    # smoke test
    python runner.py --split verified             # full Verified (500 instances)
    python runner.py --split verified --limit 50  # first 50

CLI mirrors `examples/eval-cc-swe/runner.py` so the two harnesses
feel the same to operate.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import agentix.agents.mini_swe_agent as mini_swe
import agentix.plugins.datasets.swe as swe
from datasets import load_dataset
from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.config import get_config_from_spec
from minisweagent.environments.local import LocalEnvironment, LocalEnvironmentConfig
from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig

from agentix import RuntimeClient, bash
from agentix.deployment.base import SandboxConfig, session
from agentix.log import configure_logging

WORKDIR = "/testbed"
DEFAULT_DATASET = "princeton-nlp/SWE-bench_Verified"
DEFAULT_SPLIT = "test"
DEFAULT_AGENT_TIMEOUT = 1200.0
DEFAULT_SCORE_TIMEOUT = 600.0

logger = logging.getLogger("eval_mini_swe_swebench.runner")


# ── CLI ───────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--split", default=DEFAULT_SPLIT)
    p.add_argument("--deployment", default="local", help="`local` or `apptainer`.")
    p.add_argument(
        "--bundle",
        default="eval-mini-swe-swebench:0.1.0",
        help=(
            "Docker image tag (local) or tar-bundle path (apptainer)."
            " Build with `agentix build . --format <oci-image|tar>`."
        ),
    )
    p.add_argument(
        "--instance-namespace",
        default="swebench",
        help="Docker namespace for SWE-bench eval images (e.g. `swebench` for Docker Hub).",
    )
    p.add_argument("--instance-tag", default="latest")
    p.add_argument("--instance-arch", default="x86_64")
    p.add_argument(
        "--score-image",
        default="ubuntu:24.04",
        help="Base image for the score sandbox (no LLM access needed).",
    )
    p.add_argument("--platform", default=None)
    p.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Run at most N instances. Default 1 (smoke test). Use 500 for full Verified.",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="How many instances to run concurrently. Bump for cluster runs.",
    )
    p.add_argument("--model", default=os.getenv("MINI_SWE_MODEL", "openai/gpt-4o-mini"))
    p.add_argument("--cost-limit", type=float, default=0.0)
    p.add_argument(
        "--agent-timeout",
        type=float,
        default=DEFAULT_AGENT_TIMEOUT,
        help="Per-instance agent timeout in seconds.",
    )
    p.add_argument("--output", default="eval-results", help="Output directory for results.")
    p.add_argument(
        "--instances",
        nargs="*",
        default=None,
        help="Explicit instance_id allow-list (overrides --limit ordering).",
    )
    return p.parse_args()


# ── helpers ───────────────────────────────────────────────────────────────


def _instance_image(instance: dict, *, namespace: str, tag: str, arch: str) -> str:
    """Resolve SWE-bench's per-instance eval image (host-architecture aware)."""
    from swebench.harness.test_spec.test_spec import make_test_spec

    image = make_test_spec(instance, namespace=namespace).instance_image_key
    image = image.replace("arm64", arch).replace("x86_64", arch)
    if tag != "latest":
        image = f"{image.rsplit(':', 1)[0]}:{tag}"
    return image


def _load_instances(dataset: str, *, split: str, limit: int, allow_ids: list[str] | None) -> list[dict]:
    ds = load_dataset(dataset, split=split)
    rows = list(ds)
    if allow_ids:
        rows = [r for r in rows if r["instance_id"] in allow_ids]
    if limit > 0:
        rows = rows[:limit]
    return rows


def build_agent(*, model_name: str, api_key: str, api_base: str | None, workdir: str = WORKDIR) -> DefaultAgent:
    base_cfg = get_config_from_spec("mini.yaml")

    model_raw = dict(base_cfg.get("model", {}))
    model_raw["model_name"] = model_name
    model_kwargs = dict(model_raw.get("model_kwargs", {}))
    model_kwargs["api_key"] = api_key
    if api_base:
        model_kwargs["api_base"] = api_base
    model_raw["model_kwargs"] = model_kwargs
    model_raw["cost_tracking"] = "ignore_errors"
    model_config = LitellmModelConfig.model_validate(model_raw)

    environment_raw = dict(base_cfg.get("environment", {}))
    environment_raw["cwd"] = workdir
    environment_config = LocalEnvironmentConfig.model_validate(environment_raw)

    agent_raw = dict(base_cfg.get("agent", {}))
    agent_raw.pop("mode", None)
    agent_raw.pop("confirm_exit", None)
    agent_config = AgentConfig.model_validate(agent_raw)

    return DefaultAgent(
        LitellmModel(**model_config.model_dump(mode="python")),
        LocalEnvironment(**environment_config.model_dump(mode="python")),
        **agent_config.model_dump(mode="python"),
    )


async def get_patch(workdir: str = WORKDIR) -> str:
    """Return all `workdir` changes (including new files) as a unified diff.

    This is sandbox-side: invoked via `c.remote(get_patch, workdir)`.
    """
    proc = await asyncio.create_subprocess_shell(
        "git -c core.fileMode=false add -A && "
        "git -c core.fileMode=false diff --cached --no-color --binary",
        cwd=workdir,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, _ = await proc.communicate()
    return out.decode(errors="replace") if proc.returncode == 0 else ""


# ── per-instance run ──────────────────────────────────────────────────────


async def _run_agent_phase(
    instance: dict,
    *,
    deployment: Any,
    bundle: str,
    image: str,
    platform: str | None,
    api_key: str,
    api_base: str | None,
    model: str,
    timeout: float,
) -> dict[str, Any]:
    iid = instance["instance_id"]
    cfg = SandboxConfig(image=image, bundle=bundle, platform=platform)
    started_at = time.time()
    try:
        async with session(deployment, cfg) as sandbox:
            async with RuntimeClient(sandbox.runtime_url, timeout=timeout + 60) as client:
                prep = await client.remote(swe.prepare_env, WORKDIR, instance.get("base_commit"))
                if not prep.ok:
                    return {
                        "phase": "prepare_env",
                        "instance_id": iid,
                        "ok": False,
                        "error": prep.log[-2000:],
                    }
                agent = build_agent(
                    model_name=model,
                    api_key=api_key,
                    api_base=api_base,
                    workdir=WORKDIR,
                )
                mini_result = await client.remote(
                    mini_swe.run,
                    task=instance["problem_statement"],
                    workdir=WORKDIR,
                    agent=agent,
                )
                patch = await client.remote(get_patch, WORKDIR)
        return {
            "phase": "agent",
            "instance_id": iid,
            "ok": True,
            "patch": patch,
            "mini_exit_status": mini_result.get("exit_status") if isinstance(mini_result, dict) else "unknown",
            "mini_submission": (mini_result.get("submission") if isinstance(mini_result, dict) else "") or "",
            "duration_s": time.time() - started_at,
        }
    except Exception as exc:
        return {
            "phase": "agent",
            "instance_id": iid,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "duration_s": time.time() - started_at,
        }


async def _run_score_phase(
    instance: dict,
    patch: str,
    *,
    deployment: Any,
    bundle: str,
    image: str,
    platform: str | None,
    timeout: float = DEFAULT_SCORE_TIMEOUT,
) -> dict[str, Any]:
    iid = instance["instance_id"]
    cfg = SandboxConfig(image=image, bundle=bundle, platform=platform)
    started_at = time.time()
    try:
        async with session(deployment, cfg) as sandbox:
            async with RuntimeClient(sandbox.runtime_url, timeout=timeout + 60) as client:
                await client.remote(bash.run, command=f"mkdir -p {WORKDIR}", timeout=10)
                score = await client.remote(
                    swe.score,
                    instance=instance,
                    patch=patch,
                )
        return {
            "phase": "score",
            "instance_id": iid,
            "ok": True,
            "result": _serialise_score(score),
            "duration_s": time.time() - started_at,
        }
    except Exception as exc:
        return {
            "phase": "score",
            "instance_id": iid,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "duration_s": time.time() - started_at,
        }


def _serialise_score(score: Any) -> dict[str, Any]:
    if hasattr(score, "to_dict"):
        return dict(score.to_dict())
    if hasattr(score, "__dict__"):
        return {k: v for k, v in score.__dict__.items() if not k.startswith("_")}
    return {"value": str(score)}


async def run_one_instance(
    instance: dict,
    *,
    deployment: Any,
    bundle: str,
    score_image: str,
    instance_namespace: str,
    instance_tag: str,
    instance_arch: str,
    platform: str | None,
    api_key: str,
    api_base: str | None,
    model: str,
    agent_timeout: float,
) -> dict[str, Any]:
    image = _instance_image(
        instance, namespace=instance_namespace, tag=instance_tag, arch=instance_arch
    )
    agent_phase = await _run_agent_phase(
        instance,
        deployment=deployment,
        bundle=bundle,
        image=image,
        platform=platform,
        api_key=api_key,
        api_base=api_base,
        model=model,
        timeout=agent_timeout,
    )
    if not agent_phase["ok"]:
        return {"instance_id": instance["instance_id"], "agent": agent_phase, "score": None}
    score_phase = await _run_score_phase(
        instance,
        patch=agent_phase["patch"],
        deployment=deployment,
        bundle=bundle,
        image=score_image,
        platform=platform,
    )
    return {
        "instance_id": instance["instance_id"],
        "agent": agent_phase,
        "score": score_phase,
    }


# ── orchestrator ──────────────────────────────────────────────────────────


async def amain(args: argparse.Namespace) -> int:
    configure_logging(default_context="host")
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    instances = _load_instances(
        args.dataset,
        split=args.split,
        limit=args.limit,
        allow_ids=args.instances,
    )
    if not instances:
        logger.error("no instances matched the filter; check --dataset/--split/--instances")
        return 2

    from agentix.deployment.base import load_deployment

    deployment_cls = load_deployment(args.deployment)
    deployment = deployment_cls()

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("MINI_SWE_API_KEY") or ""
    api_base = os.environ.get("OPENAI_BASE_URL")
    if not api_key:
        logger.error("OPENAI_API_KEY (or MINI_SWE_API_KEY) must be set")
        return 2

    semaphore = asyncio.Semaphore(max(1, args.concurrency))

    async def _bounded(instance: dict) -> dict[str, Any]:
        async with semaphore:
            res = await run_one_instance(
                instance,
                deployment=deployment,
                bundle=args.bundle,
                score_image=args.score_image,
                instance_namespace=args.instance_namespace,
                instance_tag=args.instance_tag,
                instance_arch=args.instance_arch,
                platform=args.platform,
                api_key=api_key,
                api_base=api_base,
                model=args.model,
                agent_timeout=args.agent_timeout,
            )
            (out_dir / f"{instance['instance_id']}.json").write_text(json.dumps(res, indent=2, default=str))
            return res

    results = await asyncio.gather(*[_bounded(inst) for inst in instances])
    _summarise(results, out_dir=out_dir)
    return 0


def _summarise(results: Iterable[dict[str, Any]], *, out_dir: Path) -> None:
    rows = list(results)
    total = len(rows)
    agent_ok = sum(1 for r in rows if r["agent"]["ok"])
    scored = sum(1 for r in rows if r["score"] and r["score"]["ok"])
    resolved = sum(
        1
        for r in rows
        if r["score"]
        and r["score"]["ok"]
        and (r["score"]["result"] or {}).get("resolved")
    )
    summary = {
        "total": total,
        "agent_ok": agent_ok,
        "scored_ok": scored,
        "resolved": resolved,
        "resolved_rate": (resolved / total) if total else 0.0,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main() -> int:
    return asyncio.run(amain(parse_args()))


if __name__ == "__main__":
    sys.exit(main())
