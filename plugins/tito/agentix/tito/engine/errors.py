"""Session error hierarchy. Each carries the HTTP status the gateway returns."""

from __future__ import annotations


class SessionError(Exception):
    """Base class for all session-related errors."""

    status_code: int = 500


class SessionNotFoundError(SessionError):
    """The requested session ID does not exist."""

    status_code: int = 404


class MessageValidationError(SessionError):
    """Request messages aren't a valid append-only extension (or a rollback failed)."""

    status_code: int = 400


class TokenizationError(SessionError):
    """A TITO tokenization invariant was violated (e.g. pretokenized prefix mismatch)."""

    status_code: int = 500


class UpstreamResponseError(SessionError):
    """The upstream backend response is invalid or unexpected (missing
    meta_info / token_ids, malformed logprobs, etc.)."""

    status_code: int = 502
