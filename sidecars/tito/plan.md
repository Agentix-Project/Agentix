# TITO Gateway Package Extraction Plan

## Goal Description

把 Miles 文档中 Agentic Chat Template / TITO session-server 路径抽成一个独立 Python package。这个项目必须明确 ack Miles 同学/团队的原创工作，定位为“对 Miles TITO 工作的封装、复用和使用层”，不是重新发明或改写 Miles 的算法。核心算法尽可能原样复用 Miles upstream 代码，不重写、不改测试逻辑。新 package 需要同时支持两种使用方式：

1. Python import 方式：应用在已有 server 旁边包裹一层 gateway，自动发现或接收后端 OpenAI-compatible server 地址，捕捉 `/v1/chat/completions` 调用并维护 TITO session/token 轨迹。
2. CLI 方式：一行命令启动 gateway，参数与 Miles 对应 TITO/session-server 参数保持兼容，尤其是 `--hf-checkpoint`、`--chat-template-path`、`--apply-chat-template-kwargs`、`--tito-model`、`--tito-allowed-append-roles`、`--session-server-ip`、`--session-server-port`、router/backend URL 相关参数。

## Source Exploration Summary

- 本 package 的技术来源与核心能力来自 Miles 项目；计划、文档和代码注释中需要明确 acknowledgement：TITO tokenizer、fixed chat templates、session trajectory、session-server proxy 和验证体系均基于 Miles 同学/团队已有工作。
- 文档页 `Agentic Chat Templates (TITO)` 明确了运行不变量：messages 必须 append-only；只允许最新 assistant checkpoint 的单步 rollback；`--tito-allowed-append-roles` 必须准确声明追加角色；`tool` 总是隐含允许。
- 文档页指向的核心验证脚本是 `scripts/tools/verify_chat_template.py` 和 `scripts/tools/verify_session_tito_tokenizer.py`，它们分别验证固定模板 append-only 与真实 session-server TITO e2e。
- 代码核心集中在 Miles:
  - `miles/utils/chat_template_utils/tito_tokenizer.py`
  - `miles/utils/chat_template_utils/template.py`
  - `miles/utils/chat_template_utils/token_seq_comparator.py`
  - `miles/utils/chat_template_utils/templates/*.jinja`
  - `miles/rollout/session/session_server.py`
  - `miles/rollout/session/sessions.py`
  - `miles/rollout/session/linear_trajectory.py`
  - `miles/rollout/session/session_errors.py`
  - `miles/rollout/session/session_types.py`
- 现有 session server 是 FastAPI + httpx proxy：创建 `/sessions`，然后通过 `/sessions/{session_id}/v1/chat/completions` 将请求代理到后端，并注入 `logprobs=True`、`return_meta_info=True`、`input_ids`，再从 SGLang/OpenAI-compatible 响应中取 `meta_info.output_token_logprobs` 更新 token checkpoint。
- 现有测试应作为迁移合同原样保留，优先复制并运行以下测试簇：
  - `tests/fast/utils/chat_template_utils/test_tito_tokenizer.py`
  - `tests/fast/utils/chat_template_utils/test_pretokenized_via_tito.py`
  - `tests/fast/router/test_sessions.py`
  - `tests/fast/router/test_session_race_conditions.py`
  - `tests/fast/router/test_session_pretokenized_e2e.py`
  - `tests/fast/utils/test_utils/test_session_verify_runner.py`
  - 可选 GPU/e2e: `tests/e2e/sglang/test_session_server_multi_role/*`

## Acceptance Criteria

- AC-1: Package import surface works.
  - Positive Tests (expected to PASS):
    - `python -c "import tito_gateway; from tito_gateway import TITOGateway, SessionServer, get_tito_tokenizer"` succeeds.
    - A test constructs `TITOGateway(...)` with an explicit backend URL and starts the same FastAPI route behavior as Miles `SessionServer`.
  - Negative Tests (expected to FAIL):
    - Constructing gateway without `hf_checkpoint` raises the same skip/error behavior defined by the wrapper contract, without silently enabling broken TITO tracking.

- AC-2: Miles TITO core logic is reused with minimal source edits.
  - Positive Tests (expected to PASS):
    - Upstream copied tests for `TITOTokenizer`, fixed-template resolution, decode-roundtrip verifier, and session routes pass without test body edits.
    - A source audit shows `tito_tokenizer.py`, fixed templates, `template.py`, `token_seq_comparator.py`, `linear_trajectory.py`, `sessions.py`, and session error/type models are copied verbatim except import path rewrites required by package namespace.
  - Negative Tests (expected to FAIL):
    - Any implementation that rewrites TITO merge/tokenize algorithms instead of vendoring upstream code fails review.

- AC-3: Existing Miles tests are preserved.
  - Positive Tests (expected to PASS):
    - Migrated tests keep assertions, parametrization, expected failures, and mock trajectory behavior identical to upstream.
    - Compatibility shims make original import paths usable where practical, e.g. `miles.utils.chat_template_utils...` can resolve to vendored modules during tests.
  - Negative Tests (expected to FAIL):
    - Changing upstream test assertions, deleting negative tests like buggy Qwen3 boundary tests, or weakening expected mismatch checks is not allowed.

- AC-4: CLI starts gateway with Miles-compatible arguments.
  - Positive Tests (expected to PASS):
    - `tito-gateway --hf-checkpoint Qwen/Qwen3-0.6B --tito-model qwen3 --tito-allowed-append-roles tool user --backend-url http://127.0.0.1:8000 --session-server-port 30000` starts the FastAPI gateway.
    - CLI supports JSON parsing for `--apply-chat-template-kwargs` in the same convention as Miles.
    - `tito-gateway verify-chat-template ...` delegates to the migrated `verify_chat_template` logic.
  - Negative Tests (expected to FAIL):
    - Invalid `--tito-model` exits non-zero with argparse/typer validation.
    - Unsupported append roles fail before server startup.

- AC-5: Backend server address can be auto-detected for wrapper usage.
  - Positive Tests (expected to PASS):
    - Explicit `backend_url` always wins.
    - Environment variables are checked in deterministic order, e.g. `TITO_BACKEND_URL`, `OPENAI_BASE_URL`, `SGLANG_BASE_URL`.
    - If a common local backend port is configured for probing, `/health` or `/v1/models` detection selects a live backend and logs the selected URL.
  - Negative Tests (expected to FAIL):
    - If no backend can be found, startup fails with a clear error instead of binding a gateway that cannot proxy calls.

- AC-6: Session proxy behavior matches Miles.
  - Positive Tests (expected to PASS):
    - `/health`, `/sessions`, `/sessions/{session_id}`, `DELETE /sessions/{session_id}`, and `/sessions/{session_id}/v1/chat/completions` behave like upstream tests.
    - Proxied chat requests inject `input_ids`, `logprobs=True`, `return_meta_info=True`, and `no_stop_trim=False`.
    - Concurrent same-session, different-session, and delete-while-inflight race tests pass unchanged.
  - Negative Tests (expected to FAIL):
    - Missing upstream `meta_info.output_token_logprobs` returns the same upstream-response error behavior.
    - Non-append-only messages or forbidden appended roles return 400.

- AC-7: Verification commands remain available.
  - Positive Tests (expected to PASS):
    - `tito-gateway verify-chat-template` prints the same PASS/FAIL verdicts as Miles `scripts/tools/verify_chat_template.py`.
    - `tito-gateway verify-session-tito-tokenizer` exists as an optional command and either runs the migrated runner when Miles/SGLang training dependencies are installed, or exits with a clear dependency error.
  - Negative Tests (expected to FAIL):
    - The package must not pretend GPU/e2e verification passed when optional heavy dependencies are unavailable.

## Path Boundaries

### Upper Bound (Maximum Scope)

- New package scaffold with `pyproject.toml`, importable `tito_gateway` package, CLI entrypoints, vendored Miles TITO/session code, compatibility imports, copied tests, and CI commands for CPU-fast tests.
- Optional command namespace for e2e verification that preserves Miles arguments but documents dependency requirements.
- Minimal docs: import usage, CLI usage, backend discovery order, and test commands.

### Lower Bound (Minimum Scope)

- Importable package exposing the TITO tokenizer factory and session gateway.
- CLI that starts FastAPI gateway with explicit `--backend-url`.
- Upstream fast tests copied and passing with only import-path compatibility changes outside the test bodies.

### Allowed Choices

- Can use FastAPI, httpx, uvicorn, transformers, huggingface_hub, jinja2, pydantic, pytest, requests, typer or argparse.
- Can add a thin namespace compatibility layer so upstream test imports keep working.
- Can add wrapper-only modules such as `tito_gateway.cli`, `tito_gateway.gateway`, `tito_gateway.config`, and `tito_gateway.discovery`.
- Can vendor upstream Miles source with attribution and an upstream commit marker.

### Disallowed Choices

- Cannot rewrite TITO tokenization/merge behavior when upstream code can be copied.
- Cannot weaken or edit upstream test assertions.
- Cannot remove negative tests that prove broken templates/subclasses fail.
- Cannot require full Miles training stack for basic package import or gateway startup.
- Cannot silently auto-detect a backend when multiple candidates are alive without deterministic precedence.

## Proposed Package Layout

```text
tito_gateway/
  __init__.py
  cli.py
  config.py
  discovery.py
  gateway.py
  server.py
  vendor/
    miles_compat/
      utils/
        chat_template_utils/
          __init__.py
          template.py
          token_seq_comparator.py
          tito_tokenizer.py
          deepseek_v32.py
          deepseek_v4.py
          templates/
      rollout/
        session/
          session_server.py
          sessions.py
          linear_trajectory.py
          session_errors.py
          session_types.py
      utils/
        processing_utils.py
        http_utils.py
        test_utils/
  miles/
    __init__.py
    ... optional compatibility re-export modules for unchanged tests ...
scripts/
  verify_chat_template.py
  verify_session_tito_tokenizer.py
tests/
  upstream/
    ... copied Miles tests, unchanged ...
  package/
    test_import_surface.py
    test_cli_args.py
    test_backend_discovery.py
```

## Dependencies and Sequence

### Milestone 1: Baseline Scaffold

- Create `pyproject.toml` with package metadata, runtime dependencies, optional test/e2e extras, and console script `tito-gateway`.
- Add `tito_gateway.__init__` export surface.
- Add an upstream metadata file recording Miles repository URL and commit SHA used for extraction.

### Milestone 2: Vendor Core Miles Logic

- Copy TITO tokenizer, chat template helpers, fixed jinja templates, token comparator, session types/errors, linear trajectory, sessions route setup, and session server.
- Rewrite only import paths or provide compatibility modules so upstream code remains functionally unchanged.
- Copy required lightweight utility helpers used by tests, especially tokenizer loading, port discovery, mock SGLang server, uvicorn thread server, and mock trajectories.

### Milestone 3: Python Wrapper API

- Implement `TITOGatewayConfig` with Miles-compatible names.
- Implement `TITOGateway.from_server(...)` / `TITOGateway(...)` that accepts explicit backend URL or uses discovery.
- Expose `app`, `run()`, and helper methods so users can mount/run beside an existing server.

### Milestone 4: CLI

- Implement `tito-gateway serve` and default command alias for one-line startup.
- Preserve Miles-compatible argument names.
- Implement `tito-gateway verify-chat-template` by delegating to copied verifier.
- Implement `tito-gateway verify-session-tito-tokenizer` as optional heavy command with explicit dependency checks.

### Milestone 5: Test Migration Without Test Edits

- Copy selected upstream tests into `tests/upstream`.
- Prefer compatibility shims so copied tests import `miles.*` unchanged.
- If import path edits are absolutely unavoidable, perform mechanical path rewrites only and document each changed line in a migration ledger; do not alter assertions, cases, expected exceptions, or parametrization.

### Milestone 6: New Wrapper Tests

- Add package-specific tests for import surface, CLI parsing, backend discovery precedence, and explicit backend startup.
- Use Miles mock server utilities to avoid requiring a real SGLang server for CPU-fast tests.

### Milestone 7: Verification

- Run CPU-fast subset:
  - `pytest tests/upstream/fast/utils/chat_template_utils/test_tito_tokenizer.py`
  - `pytest tests/upstream/fast/utils/chat_template_utils/test_pretokenized_via_tito.py`
  - `pytest tests/upstream/fast/router/test_sessions.py`
  - `pytest tests/upstream/fast/router/test_session_race_conditions.py`
  - `pytest tests/upstream/fast/router/test_session_pretokenized_e2e.py`
  - `pytest tests/upstream/fast/utils/test_utils/test_session_verify_runner.py`
  - `pytest tests/package`
- Run CLI smoke tests:
  - `tito-gateway --help`
  - `tito-gateway serve --help`
  - `tito-gateway verify-chat-template --help`
- Document optional e2e command separately because it requires model/GPU/SGLang dependencies.

## Implementation Notes

- The package should treat Miles as the source of truth. Add wrapper code around it; do not “simplify” the TITO algorithm.
- Public docs, README, package metadata, and copied source headers should clearly state that this is a standalone packaging/wrapper effort around Miles TITO work, with attribution to Miles and its contributors.
- Keep upstream test files as a contract. The test migration should be boring and traceable.
- Backend auto-discovery must be deterministic and observable in logs.
- The default import path should be `tito_gateway`, but a `miles` compatibility namespace is acceptable for tests and copied code.
- Preserve Apache-2.0 license notices and upstream attribution when copying Miles code.
