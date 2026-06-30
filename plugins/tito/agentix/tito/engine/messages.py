"""Message-level helpers used by the session state machine.

`message_matches` compares only the fields that affect chat-template tokenization,
so a stored assistant message and a resent one are considered equal iff they
tokenize identically. `assert_messages_append_only_with_allowed_role` enforces that
each turn extends the stored history without rewriting it.
"""

from __future__ import annotations

from typing import Any

# Keys a chat template actually reads. Extra client-injected keys
# (provider_specific_fields, etc.) don't affect tokenization, so we ignore them.
TEMPLATE_RELEVANT_KEYS = ("role", "content", "reasoning_content", "tool_calls")

DEFAULT_APPEND_ROLES: list[str] = ["tool"]


def normalize_value(value: Any) -> Any:
    """Collapse the falsy sentinels that render identically in Jinja2 (None, "", [])
    to None. Non-falsy content — including whitespace like trailing newlines — is
    returned as-is, because boundary characters must tokenize identically."""
    if value is None or value == "" or value == []:
        return None
    return value


def message_matches(stored: dict[str, Any], new: dict[str, Any]) -> bool:
    for key in TEMPLATE_RELEVANT_KEYS:
        if normalize_value(stored.get(key)) != normalize_value(new.get(key)):
            return False
    return True


def assert_messages_append_only_with_allowed_role(
    stored_messages: list[dict[str, Any]],
    new_messages: list[dict[str, Any]],
    allowed_append_roles: list[str] = DEFAULT_APPEND_ROLES,
) -> None:
    """Assert *new_messages* is an append-only extension of *stored_messages*: the
    stored prefix matches (by template-relevant keys) and each appended message's
    role is in *allowed_append_roles*. Raises ValueError otherwise."""
    if not stored_messages:
        return

    if len(new_messages) < len(stored_messages):
        raise ValueError(
            f"new messages ({len(new_messages)}) are fewer than stored messages ({len(stored_messages)})",
            new_messages,
            stored_messages,
        )

    for i, stored_msg in enumerate(stored_messages):
        if not message_matches(stored_msg, new_messages[i]):
            diffs = {
                key: {"stored": repr(stored_msg.get(key))[:200], "new": repr(new_messages[i].get(key))[:200]}
                for key in TEMPLATE_RELEVANT_KEYS
                if stored_msg.get(key) != new_messages[i].get(key)
            }
            raise ValueError(
                f"message mismatch at index {i} "
                f"(role: stored={stored_msg.get('role')}, new={new_messages[i].get('role')}). "
                f"Diffs: {diffs}"
            )

    for j, msg in enumerate(new_messages[len(stored_messages):]):
        if msg.get("role") not in allowed_append_roles:
            raise ValueError(
                f"appended message at index {len(stored_messages) + j} "
                f"has role={msg.get('role')!r}, allowed={allowed_append_roles}"
            )
