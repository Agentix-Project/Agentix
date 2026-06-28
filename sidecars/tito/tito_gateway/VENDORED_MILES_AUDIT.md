# Vendored Miles Source Audit

This package is a standalone wrapper/packaging layer around Miles TITO work. It is not a rewrite of Miles TITO algorithms. Core Miles files below were audited against:

- Repository: `https://github.com/radixark/miles`
- Commit: `9437366e0aa3a25294720f70d18b081067595f85`
- Local upstream checkout used for audit: `/tmp/miles-explore`

## Audit Result

| Upstream file | Vendored file | Status | Allowed differences |
|---|---|---|---|
| `miles/utils/chat_template_utils/tito_tokenizer.py` | `tito_gateway/vendor/miles_compat/utils/chat_template_utils/tito_tokenizer.py` | Import-path rewrite plus Python 3.10 enum compatibility | `miles.*` imports rewritten to `tito_gateway.vendor.miles_compat.*`; upstream `StrEnum` replaced by `str, Enum` because this package supports Python 3.10. TITO tokenization, merge, fixed-template resolution, and factory logic are otherwise preserved. |
| `miles/utils/chat_template_utils/template.py` | `tito_gateway/vendor/miles_compat/utils/chat_template_utils/template.py` | Import-path rewrite only | `miles.utils.chat_template_utils` import rewritten to `tito_gateway.vendor.miles_compat.utils.chat_template_utils`. |
| `miles/utils/chat_template_utils/token_seq_comparator.py` | `tito_gateway/vendor/miles_compat/utils/chat_template_utils/token_seq_comparator.py` | Byte-identical | None. |
| `miles/utils/chat_template_utils/templates/*.jinja` | `tito_gateway/vendor/miles_compat/utils/chat_template_utils/templates/*.jinja` | Byte-identical | None. Audited templates: `kimi_k25_fixed.jinja`, `minimax_m25_fixed.jinja`, `minimax_m27_fixed.jinja`, `qwen3.5_fixed.jinja`, `qwen3_fixed.jinja`, `qwen3_thinking_2507_and_next_fixed.jinja`. |
| `miles/rollout/session/linear_trajectory.py` | `tito_gateway/vendor/miles_compat/rollout/session/linear_trajectory.py` | Import-path rewrite only | `miles.*` imports rewritten to `tito_gateway.vendor.miles_compat.*`. |
| `miles/rollout/session/sessions.py` | `tito_gateway/vendor/miles_compat/rollout/session/sessions.py` | Import-path rewrite only | `miles.*` imports rewritten to `tito_gateway.vendor.miles_compat.*`. |
| `miles/rollout/session/session_errors.py` | `tito_gateway/vendor/miles_compat/rollout/session/session_errors.py` | Byte-identical | None. |
| `miles/rollout/session/session_types.py` | `tito_gateway/vendor/miles_compat/rollout/session/session_types.py` | Byte-identical | None. |
| `miles/rollout/session/session_server.py` | `tito_gateway/vendor/miles_compat/rollout/session/session_server.py` | Import-path rewrite only | `miles.rollout.session.sessions` import rewritten to `tito_gateway.vendor.miles_compat.rollout.session.sessions`. |

## Compatibility Test Path

The copied upstream tokenizer test computes a repository-root-relative template path from its own `tests/upstream/...` location. To preserve the unchanged test body, this package provides:

- `tests/miles/utils/chat_template_utils/templates/qwen3_thinking_2507_and_next_fixed.jinja`

That file is byte-identical to the vendored/upstream template.
