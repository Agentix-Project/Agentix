"""Structured trajectory model for mini-swe-agent runs.

Mirrors the ATIF (Agent Trial Interaction Format) shape used by
harbor's `MiniSweAgent.populate_context_post_run`: one
`Trajectory` per agent run, made of ordered `Step`s, each carrying
the message, optional tool calls, observations, and per-call
metrics. `FinalMetrics` aggregates the totals used downstream by
schedulers / eval harnesses / RL buffers.

The actual format is decoupled from any external schema package so
the plugin can be consumed by the in-tree examples and tests without
pulling in harbor.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

SCHEMA_VERSION = "ATIF-v1.2"

StepSource = Literal["agent", "user", "system"]


@dataclass(slots=True)
class ToolCall:
    tool_call_id: str
    function_name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class ObservationResult:
    content: str


@dataclass(slots=True)
class Observation:
    results: list[ObservationResult] = field(default_factory=list)


@dataclass(slots=True)
class Metrics:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    cost_usd: float | None = None
    extra: dict[str, Any] | None = None


@dataclass(slots=True)
class FinalMetrics:
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cached_tokens: int | None = None
    total_cost_usd: float | None = None
    extra: dict[str, Any] | None = None


@dataclass(slots=True)
class Step:
    step_id: int
    timestamp: str
    source: StepSource
    message: str = ""
    model_name: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] | None = None
    observation: Observation | None = None
    metrics: Metrics | None = None


@dataclass(slots=True)
class AgentInfo:
    name: str
    version: str | None = None
    model_name: str | None = None
    extra: dict[str, Any] | None = None


@dataclass(slots=True)
class Trajectory:
    """Structured representation of one mini-swe-agent run."""

    schema_version: str
    session_id: str
    agent: AgentInfo
    steps: list[Step]
    final_metrics: FinalMetrics
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _strip_none(asdict(self))

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ── mini-swe-agent v2 -> Trajectory ───────────────────────────────────────


def from_mini_swe_agent(
    trajectory: dict[str, Any],
    *,
    session_id: str,
    now: datetime | None = None,
) -> Trajectory:
    """Convert a mini-swe-agent v2 trajectory dict into our `Trajectory`.

    Expects the v2 native tool-calling format where assistant messages
    contain a `tool_calls` array and tool results use `role: "tool"`.
    Unknown shapes degrade gracefully: text content is preserved,
    tool calls are best-effort parsed, message role mapping mirrors
    harbor's `convert_mini_swe_agent_to_atif`.
    """
    info = trajectory.get("info") or {}
    config = info.get("config") or {}
    model_config = config.get("model") or {}
    agent_config = config.get("agent") or {}
    model_name = model_config.get("model_name") or "unknown"
    mini_version = info.get("mini_version") or "unknown"
    original_format = trajectory.get("trajectory_format", "unknown")

    messages = trajectory.get("messages") or []
    total_cost_usd = float((info.get("model_stats") or {}).get("instance_cost") or 0.0)

    total_completion_tokens = 0
    for message in messages:
        usage = _usage_of(message)
        total_completion_tokens += int(usage.get("completion_tokens") or 0)

    base_now = now or datetime.now(UTC)
    steps: list[Step] = []
    step_id = 1
    total_prompt = 0
    total_cached = 0
    total_reasoning = 0

    for i, message in enumerate(messages):
        role = message.get("role")
        content = _normalize_content(message.get("content"))
        usage = _usage_of(message)
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        prompt_details = usage.get("prompt_tokens_details") or {}
        completion_details = usage.get("completion_tokens_details") or {}
        cached_tokens = (
            int(prompt_details.get("cached_tokens") or 0)
            if isinstance(prompt_details, dict)
            else 0
        )
        reasoning_tokens = (
            int(completion_details.get("reasoning_tokens") or 0)
            if isinstance(completion_details, dict)
            else 0
        )
        total_prompt += prompt_tokens
        total_cached += cached_tokens
        total_reasoning += reasoning_tokens

        timestamp = _isoformat(base_now)

        if role == "system":
            steps.append(Step(step_id=step_id, timestamp=timestamp, source="system", message=content))
            step_id += 1
        elif role == "user":
            if i == 1:
                steps.append(Step(step_id=step_id, timestamp=timestamp, source="user", message=content))
                step_id += 1
            else:
                _attach_observation(steps, content)
        elif role == "tool":
            _attach_observation(steps, content)
        elif role == "assistant":
            tool_calls, reasoning = _parse_tool_calls(message, content, step_id)
            metrics = _build_step_metrics(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cached_tokens=cached_tokens,
                prompt_tokens_details=prompt_details if isinstance(prompt_details, dict) else {},
                completion_tokens_details=completion_details
                if isinstance(completion_details, dict)
                else {},
                total_cost_usd=total_cost_usd,
                total_completion_tokens=total_completion_tokens,
            )
            steps.append(
                Step(
                    step_id=step_id,
                    timestamp=timestamp,
                    source="agent",
                    model_name=model_name,
                    message=content,
                    reasoning_content=reasoning,
                    tool_calls=tool_calls,
                    metrics=metrics,
                )
            )
            step_id += 1

    final_extra: dict[str, Any] = {}
    if total_reasoning > 0:
        final_extra["total_reasoning_tokens"] = total_reasoning

    final = FinalMetrics(
        total_prompt_tokens=total_prompt,
        total_completion_tokens=total_completion_tokens,
        total_cached_tokens=total_cached if total_cached > 0 else None,
        total_cost_usd=total_cost_usd if total_cost_usd > 0 else None,
        extra=final_extra or None,
    )

    return Trajectory(
        schema_version=SCHEMA_VERSION,
        session_id=session_id,
        agent=AgentInfo(
            name="mini-swe-agent",
            version=mini_version,
            model_name=model_name,
            extra={"original_format": original_format, "agent_config": agent_config},
        ),
        steps=steps,
        final_metrics=final,
        notes="Converted from mini-swe-agent v2 trajectory",
    )


def aggregate_usage(trajectory: dict[str, Any]) -> dict[str, Any]:
    """Light-weight summary: total tokens + cost from a raw mini-swe-agent trajectory.

    Cheaper than `from_mini_swe_agent` for callers that only need the
    metrics summary (e.g. populating `AgentContext.n_*` fields).
    """
    info = trajectory.get("info") or {}
    total_cost_usd = float((info.get("model_stats") or {}).get("instance_cost") or 0.0)
    prompt = 0
    completion = 0
    cached = 0
    for message in trajectory.get("messages") or []:
        usage = _usage_of(message)
        prompt += int(usage.get("prompt_tokens") or 0)
        completion += int(usage.get("completion_tokens") or 0)
        details = usage.get("prompt_tokens_details") or {}
        if isinstance(details, dict):
            cached += int(details.get("cached_tokens") or 0)
    return {
        "n_input_tokens": prompt,
        "n_output_tokens": completion,
        "n_cache_tokens": cached,
        "cost_usd": total_cost_usd,
    }


# ── internals ─────────────────────────────────────────────────────────────


def _isoformat(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _usage_of(message: dict[str, Any]) -> dict[str, Any]:
    extra = message.get("extra") or {}
    response = extra.get("response") or {} if isinstance(extra, dict) else {}
    usage = response.get("usage") or {} if isinstance(response, dict) else {}
    return usage if isinstance(usage, dict) else {}


def _normalize_content(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts: list[str] = []
        for part in raw:
            if isinstance(part, dict):
                parts.append(str(part.get("text", part)))
            else:
                parts.append(str(part))
        return "\n".join(parts)
    return str(raw)


def _attach_observation(steps: list[Step], content: str) -> None:
    if not steps or steps[-1].source != "agent":
        # Message has no preceding agent step (rare).
        return
    prev = steps[-1]
    if prev.observation is None:
        prev.observation = Observation(results=[ObservationResult(content=content)])
    else:
        prev.observation.results.append(ObservationResult(content=content))


def _parse_tool_calls(
    message: dict[str, Any], content: str, step_id: int
) -> tuple[list[ToolCall] | None, str | None]:
    raw_calls = message.get("tool_calls")
    if not isinstance(raw_calls, list) or not raw_calls:
        return None, content if content else None
    parsed: list[ToolCall] = []
    for tc in raw_calls:
        if not isinstance(tc, dict):
            continue
        tc_id = str(tc.get("id") or f"call_{step_id}_{len(parsed) + 1}")
        function = tc.get("function") or {}
        name = str(function.get("name", "bash")) if isinstance(function, dict) else "bash"
        raw_args = function.get("arguments", "{}") if isinstance(function, dict) else "{}"
        if isinstance(raw_args, dict):
            arguments = raw_args
        elif isinstance(raw_args, str):
            try:
                arguments = json.loads(raw_args)
            except (json.JSONDecodeError, TypeError):
                arguments = {"command": raw_args}
        else:
            arguments = {"command": str(raw_args)}
        if not isinstance(arguments, dict):
            arguments = {"_raw": arguments}
        parsed.append(ToolCall(tool_call_id=tc_id, function_name=name, arguments=arguments))
    reasoning = content if content else None
    return (parsed or None), reasoning


def _build_step_metrics(
    *,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int,
    prompt_tokens_details: dict[str, Any],
    completion_tokens_details: dict[str, Any],
    total_cost_usd: float,
    total_completion_tokens: int,
) -> Metrics | None:
    if prompt_tokens == 0 and completion_tokens == 0:
        return None

    step_cost: float | None = None
    if total_cost_usd > 0 and total_completion_tokens > 0 and completion_tokens > 0:
        step_cost = (completion_tokens / total_completion_tokens) * total_cost_usd

    extra: dict[str, Any] = {}
    if prompt_tokens_details:
        extra["prompt_tokens_details"] = prompt_tokens_details
    if completion_tokens_details:
        extra["completion_tokens_details"] = completion_tokens_details
    return Metrics(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cached_tokens=cached_tokens if cached_tokens > 0 else None,
        cost_usd=step_cost if step_cost and step_cost > 0 else None,
        extra=extra or None,
    )


def _strip_none(obj: Any) -> Any:
    """Recursively drop keys whose value is `None`, so the serialised
    Trajectory is compact and stable for diff comparisons."""
    if isinstance(obj, dict):
        return {k: _strip_none(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_strip_none(v) for v in obj]
    return obj


def trajectory_records(trajectory: Trajectory) -> Iterable[Step]:
    """Convenience iterator for callers that want to stream steps."""
    return iter(trajectory.steps)


__all__ = [
    "AgentInfo",
    "FinalMetrics",
    "Metrics",
    "Observation",
    "ObservationResult",
    "SCHEMA_VERSION",
    "Step",
    "StepSource",
    "ToolCall",
    "Trajectory",
    "aggregate_usage",
    "from_mini_swe_agent",
    "trajectory_records",
]
