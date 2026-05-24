# Agentix RPC Protocol

The runtime wire contract for `RuntimeClient.remote(fn, *args, **kwargs)`.
Tests in `tests/test_rpc_protocol.py` enforce these rules.

## Callable Reference

The client encodes the callable as a `RemoteCallable` — a `str`
subclass holding `module::qualname`. The same single-line identifier
appears in the SIO event payload, in worker frames, and on the wire
generally.

Remote call targets must be importable by the worker:

  - top-level functions
  - builtin functions such as `len`
  - top-level functions defined in a `python script.py` entrypoint,
    encoded as the script module path when that module is importable

Lambdas, local closures, bound methods, partials, and callable
instances are intentionally outside the callable boundary. Put remote
code behind an importable top-level function instead.

Args and kwargs travel separately as `arguments = pickle.dumps((args, kwargs))`.
Return values travel as `value = pickle.dumps(result)`.

```python
from my_project.tasks import run

await client.remote(run, seed=42)
```

## Transports

| Path | Carries | Wire |
| --- | --- | --- |
| `GET /health` | health probe | HTTP JSON |
| `POST /call` | internal short-call fast path | HTTP msgpack |
| Socket.IO `/rpc` | `c.remote()` RPC | msgpack-wrapped `call` / `call:result` / `call:error` / `cancel` |
| Socket.IO `/trace`, `/log`, `/<plugin>` | side channels | plugin-defined events (msgpack payloads) |
| worker private pipe | runtime ↔ worker | length-prefixed msgpack frames |

HTTP covers health plus the internal `/call` fast path for short
remote calls. Socket.IO `/rpc` remains the RPC event channel when a
call is submitted over SIO or an accepted HTTP call completes
asynchronously. The worker pipe is the runtime-to-worker edge inside
the sandbox. The current implementation uses one worker subprocess per
runtime.

## Socket.IO Events (RPC on `/rpc`)

```text
call          {call_id, callable, arguments}
call:result   {call_id, value}                  # value is pickle.dumps(result)
call:error    {call_id, error}
cancel        {call_id}
```

`call_id` correlates request ↔ response. Cancellation produces a
`call:error` with `error.cancelled=True`.

Trace, log, and plugin traffic use their own namespaces on the same
Socket.IO connection. Sandbox plugins emit through `agentix.sio`; the
worker forwards `sio_emit` / `sio_open` frames to the server, which
registers matching server namespaces and relays inbound host events
back as `sio_inbound` frames.

## Worker Frames

Length-prefixed msgpack dicts on a private pipe (stdio fds are dup'd
away from user subprocesses).

| Direction | Frame | Payload fields |
| --- | --- | --- |
| server → worker | `call` | `call_id`, `callable`, `arguments` |
| server → worker | `cancel` | `call_id` |
| server → worker | `shutdown` | — |
| server → worker | `sio_inbound` | `namespace`, `event`, `data` |
| worker → server | `ready` | — |
| worker → server | `boot_error` | `error` |
| worker → server | `result` | `call_id`, `value` |
| worker → server | `error` | `call_id`, `error` |
| worker → server | `sio_emit` | `namespace`, `event`, `data` |
| worker → server | `sio_open` | `namespace` |

## Invariants

1. **One terminal result per call.** Each `call_id` ends with exactly
   one `result` or `error` frame (worker pipe) and one matching
   `call:result` or `call:error` event (Socket.IO).
2. **Closed calls are quiet.** After a terminal result, later frames
   for the same `call_id` are dropped.
3. **Cancellation closes the call.** `cancel` produces
   `error(type="Cancelled", cancelled=True)`.
4. **Worker death closes calls.** If the worker subprocess exits, the
   runtime fails every in-flight call with `WorkerExited` so the
   client never hangs.

## Error Model

`error` payload:

```python
{
    "type": "ValueError",
    "message": "...",
    "traceback": "...",
    "cancelled": False,
}
```

Client mapping:

- `cancelled=True` → `asyncio.CancelledError`
- everything else → `agentix.RemoteCallError`

## Lifecycle

| Edge | Connect | Cleanup |
| --- | --- | --- |
| host → runtime HTTP | per health check | httpx closes |
| host → runtime Socket.IO | `RuntimeClient.__aenter__` | `RuntimeClient.close()` disconnects |
| runtime → worker | lazy on first call | `shutdown`, wait, terminate/kill fallback |

## Out of Scope

- Per-call timeouts; callers use `asyncio.wait_for(...)`.
- Retries; calls are at-most-once.
- Auth/TLS; deployments own that layer.
- Annotation-driven validation on the wire (args/return are pickle today).
