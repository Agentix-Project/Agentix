"""Forward — a schema-agnostic JSON POST handler for a sidecar URL.

The sandbox tunnel decodes an inbound JSON object and sends that object to
the host. `Forward` serializes it as a new JSON POST to `target_url + path`,
then buffers the complete sidecar response and returns its body, media type,
and HTTP status. Shape translation, recording, and routing can therefore live
*behind* `target_url` in any JSON service without teaching abridge those
schemas.

This is intentionally not an HTTP-transparent byte ferry: the original
method, query, headers, and JSON encoding are not preserved. An SSE response
keeps its media type and payload for compatibility, but is fully buffered;
there is no chunk streaming or backpressure on this wire contract.

    proxy = Proxy(Forward(sidecar_url, paths=["/v1/messages"]))
    async with proxy.session(sandbox) as handle:
        await sandbox.remote(agent, env={"ANTHROPIC_BASE_URL": handle.url, ...})

`Forward` carries the rollout identity: it stamps `x-session-id` (the
rollout this forwarder serves) and a per-call `x-request-id` on every
upstream POST, so the sidecar can group a token-level trajectory.

Routing is dynamic — paths are chosen at construction, not via the
class-level `@on` tag — so `Forward` exposes them through
`abridge_routes()`, which `Proxy` merges alongside any `@on` handlers.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from collections.abc import Mapping

import httpx

from .proxy import AbridgeError, ClientResponse, Handler, Request

logger = logging.getLogger(__name__)


class _OwnedHandler:
    """A `Handler` that keeps a lifecycle link to the forwarder that owns it.

    `Forward.handler()` hands a bare callable to a wrapper (e.g.
    `AnthropicToOpenAI`), which hides the forwarder from the `Proxy` that
    closes clients on stop. Exposing `aclose()` here lets the wrapper delegate
    close through the seam, so the owner's HTTP pool doesn't outlive the proxy.
    """

    __slots__ = ("_handler", "_owner")

    def __init__(self, handler: Handler, owner: Forward) -> None:
        self._handler = handler
        self._owner = owner

    async def __call__(self, request: Request) -> ClientResponse:
        return await self._handler(request)

    async def aclose(self) -> None:
        await self._owner.aclose()


class Forward:
    """Forward JSON POSTs on `paths` to `target_url` over httpx.

    `session_id` identifies the rollout this forwarder serves (auto-gen if
    not passed); stamped as `x-session-id` on every upstream call. Reusing
    a `Forward` across sequential proxy sessions keeps that identity. A
    `Proxy` closes the forwarder's HTTP pool when its lifecycle stops; the
    pool is recreated lazily if the forwarder is used again. `headers` are
    merged into every upstream request (e.g. an upstream auth token the
    sidecar expects).

    Responses, including 4xx and 5xx responses, remain normal
    `ClientResponse` values so their status and body survive the tunnel.
    `AbridgeError(503)` signals a failure to obtain any HTTP response from the
    sidecar (connection refused, DNS, timeout) — a distinct code from a real
    upstream 502 the sidecar relays, so the agent can tell "sidecar down" from
    "sidecar returned bad gateway".
    """

    def __init__(
        self,
        target_url: str,
        *,
        paths: list[str],
        # Strictly under Proxy's default 600s tunnel window: the window's
        # timer starts before this forward does, so an equal deadline always
        # loses the race and turns a slow-but-successful sidecar call into a
        # tunnel 504.
        timeout: float = 540.0,
        session_id: str | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        if not paths:
            raise ValueError("Forward requires at least one path")
        self._target = target_url.rstrip("/")
        self._paths = tuple(paths)
        self.session_id = session_id or uuid.uuid4().hex
        self._headers = dict(headers or {})
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = httpx.AsyncClient(timeout=timeout)

    def abridge_routes(self) -> dict[str, Handler]:
        return {path: self._make_handler(path) for path in self._paths}

    def _make_handler(self, path: str) -> Handler:
        async def handler(request: Request) -> ClientResponse:
            return await self._forward(path, request)

        return handler

    def handler(self, path: str | None = None) -> Handler:
        """The bound forwarder for `path` (or the sole path if there's exactly
        one) as a plain `Handler`. This is the composition seam: it lets a
        converter/wrapper use this Forward as a transparent downstream —
        `AnthropicToOpenAI(SessionForward(gw).handler())` — without knowing it's a
        Forward, a SessionForward, or anything else. The returned handler also
        carries `aclose()` (delegating to this forwarder), so a wrapper can
        propagate `Proxy.stop`'s client cleanup through the seam."""
        routes = self.abridge_routes()
        if path is None:
            if len(routes) != 1:
                raise ValueError(
                    f"handler() needs an explicit path; forwarder has {sorted(routes)}"
                )
            (path,) = routes
        if path not in routes:
            raise ValueError(f"no route for {path!r}; have {sorted(routes)}")
        return _OwnedHandler(routes[path], self)

    async def _forward(self, path: str, request: Request) -> ClientResponse:
        record_id = uuid.uuid4().hex
        headers = {
            **self._headers,
            "x-session-id": self.session_id,
            "x-request-id": record_id,
            "content-type": "application/json",
        }
        url = self._url_for(path)
        try:
            resp = await self._get_client().post(url, json=request.body, headers=headers)
        except httpx.HTTPError as exc:
            logger.warning("abridge forward %s: %s", url, exc)
            raise AbridgeError(f"forward to {url}: {exc}", status_code=503) from exc

        media_type = resp.headers.get("content-type", "application/json").split(";")[0].strip()
        return ClientResponse(
            body=resp.content,
            media_type=media_type or "application/json",
            status_code=resp.status_code,
        )

    def _url_for(self, path: str) -> str:
        """Upstream URL for an inbound `path`. Override to remap — e.g. a
        session-scoped sidecar prefixes `/sessions/{id}`."""
        return self._target + path

    def _get_client(self) -> httpx.AsyncClient:
        client = self._client
        if client is None or client.is_closed:
            client = httpx.AsyncClient(timeout=self._timeout)
            self._client = client
        return client

    async def aclose(self) -> None:
        """Close the HTTP pool. Safe to call repeatedly and reusable later."""
        client = self._client
        self._client = None
        if client is not None and not client.is_closed:
            await client.aclose()


class SessionForward(Forward):
    """Forward to a *session-scoped* sidecar that keys a trajectory by URL path.

    Some sidecars don't accept a bare `/v1/chat/completions` — they require a
    session created up front and then addressed by path: `POST {create_path}`
    returns `{"session_id": ...}`, and every later call goes to
    `{create_path}/{session_id}{path}`. The TITO gateway is the motivating case:
    its recording route is `/sessions/{id}/v1/chat/completions`, so a plain
    `Forward` (which posts straight to `{target}{path}`) can't reach it, and the
    id isn't known until the gateway assigns it.

    `SessionForward` creates the session lazily on the first forwarded request
    (or eagerly via `open()`), remembers the assigned id, and rewrites every
    inbound `path` to the session-scoped URL. So the in-sandbox agent keeps
    calling an unmodified `/v1/chat/completions` and the whole rollout still
    lands in one session.

        fwd = SessionForward(gateway_url, paths=["/v1/chat/completions"])
        async with Proxy(fwd).session(sandbox) as handle:
            # the agent just POSTs /v1/chat/completions at handle.url; it never
            # sees a session id — SessionForward creates + scopes it host-side.
            await sandbox.remote(agent, base_url=handle.url)
        sid = fwd.session_id                       # assigned after the run (or `await fwd.open()` up front)
        trajectory = (await httpx.AsyncClient().get(
            f"{gateway_url}/sessions/{sid}")).json()
        await fwd.delete_session()                 # optional: reap the server-side session

    `.session_id` is only valid once the session exists — reading it before the
    first request (or `open()`) raises, so a premature harvest fails loudly rather
    than hitting a fabricated id. Assigning `fwd.session_id = existing_id`
    *attaches* to an already-created gateway session: requests scope to it and
    no create call runs. One instance == one gateway session: all of an
    instance's calls accumulate into the same session, so use a fresh
    `SessionForward` per rollout (or call `delete_session()` to reap and reset).
    The session is intentionally *not* deleted on `aclose()` — the trajectory is
    the point and must survive the proxy teardown for harvesting.
    """

    def __init__(
        self,
        target_url: str,
        *,
        paths: list[str],
        create_path: str = "/sessions",
        session_id_field: str = "session_id",
        # Strictly under Proxy's default 600s tunnel window: the window's
        # timer starts before this forward does, so an equal deadline always
        # loses the race and turns a slow-but-successful sidecar call into a
        # tunnel 504.
        timeout: float = 540.0,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(target_url, paths=paths, timeout=timeout, headers=headers)
        # Forward.__init__ stamped a throwaway uuid via the setter below; discard
        # it — the gateway assigns the real id on open()/first call. Until then a
        # read of `.session_id` raises instead of returning a meaningless value.
        self._session_id: str | None = None
        self._create_path = "/" + create_path.strip("/")
        self._session_field = session_id_field
        self._session_ready = False
        self._session_lock = asyncio.Lock()

    @property
    def session_id(self) -> str:
        if self._session_id is None:
            raise RuntimeError(
                "SessionForward.session_id is unavailable until the gateway session "
                "is created — call `await fwd.open()` (or make one forwarded request) "
                "before harvesting."
            )
        return self._session_id

    @session_id.setter
    def session_id(self, value: str) -> None:
        # Attach semantics: assigning an id means "this session already exists
        # on the gateway" — later requests scope to it and no create call runs.
        # Without the ready flag the first request would create a fresh session
        # and silently overwrite the assigned id.
        self._session_id = value
        self._session_ready = True

    async def open(self) -> str:
        """Create the sidecar session now (idempotent) and return its id.

        Lazy creation also happens on the first forwarded request, so `open()`
        is only needed when the host wants the id before the agent runs.
        """
        await self._ensure_session()
        return self.session_id

    async def delete_session(self) -> None:
        """Reap the server-side session (e.g. after the trajectory is harvested).

        No-op if no session was created. Kept separate from `aclose()` / `Proxy.stop`
        so the default flow preserves the trajectory; after this the next request
        opens a fresh session. Transport errors are suppressed — best-effort reap.
        """
        async with self._session_lock:
            sid = self._session_id
            if sid is None:
                return
            url = f"{self._target}{self._create_path}/{sid}"
            with contextlib.suppress(httpx.HTTPError):
                client = self._client
                if client is not None and not client.is_closed:
                    await client.delete(url, headers=dict(self._headers))
                else:
                    # Don't resurrect the pool for one reap: the documented
                    # harvest flow calls this after `Proxy.stop` closed it, and
                    # a lazily recreated pool here would have no owner left to
                    # close it.
                    async with httpx.AsyncClient(timeout=self._timeout) as one_shot:
                        await one_shot.delete(url, headers=dict(self._headers))
            self._session_ready = False
            self._session_id = None

    async def _ensure_session(self) -> None:
        if self._session_ready:
            return
        async with self._session_lock:
            if self._session_ready:
                return
            url = self._target + self._create_path
            headers = {**self._headers, "content-type": "application/json"}
            try:
                resp = await self._get_client().post(url, json={}, headers=headers)
            except httpx.HTTPError as exc:
                logger.warning("abridge session create %s: %s", url, exc)
                raise AbridgeError(f"create session at {url}: {exc}", status_code=503) from exc
            if not resp.is_success:
                raise AbridgeError(
                    f"create session at {url}: HTTP {resp.status_code}", status_code=502
                )
            try:
                session_id = resp.json()[self._session_field]
            except (ValueError, KeyError, TypeError) as exc:
                raise AbridgeError(
                    f"create session at {url}: response missing {self._session_field!r}",
                    status_code=502,
                ) from exc
            self.session_id = str(session_id)
            logger.info("abridge session created at %s: %s", url, self.session_id)

    def _url_for(self, path: str) -> str:
        return f"{self._target}{self._create_path}/{self.session_id}{path}"

    async def _forward(self, path: str, request: Request) -> ClientResponse:
        await self._ensure_session()
        return await super()._forward(path, request)


__all__ = ["Forward", "SessionForward"]
