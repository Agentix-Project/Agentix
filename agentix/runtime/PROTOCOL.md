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

```python
from my_project.tasks import run

await client.remote(run, seed=42)
```

## Transports

| Path | Carries | Wire |
| --- | --- | --- |
| `GET /health` | health probe | HTTP JSON |
| `/socket.io/` | `call` / `call:result` / `call:error` / `cancel` / `trace:event` | msgpack-payload Socket.IO events |
| worker stdin/stdout | runtime ↔ worker | length-prefixed msgpack frames |

HTTP is only for `/health`. Socket.IO is the host-to-runtime remote-call
edge. Stdin/stdout is the runtime-to-worker edge inside the sandbox.
The current implementation uses one worker subprocess per runtime.

## Socket.IO Events

```text
call          {call_id, callable, arguments}
call:result   {call_id, value}                  # value is pickle.dumps(result)
call:error    {call_id, error}
cancel        {call_id}
trace:event   <TraceFrame dict>                 # broadcast to all sessions
```

`call_id` correlates request ↔ response. Cancellation produces a
`call:error` with `error.cancelled=True`.

## Worker Frames

Length-prefixed msgpack dicts.

| Direction | Frame | Payload fields |
| --- | --- | --- |
| server → worker | `call` | `call_id`, `callable`, `arguments` |
| server → worker | `cancel` | `call_id` |
| server → worker | `shutdown` | — |
| worker → server | `ready` | — |
| worker → server | `boot_error` | `error` |
| worker → server | `result` | `call_id`, `value` |
| worker → server | `error` | `call_id`, `error` |
| worker → server | `trace` | `frame` (opaque dict; broadcast as-is) |

## Invariants

1. **One terminal result per call.** Each `call_id` ends with exactly
   one `RESULT` or `ERROR` frame.
2. **Closed calls are quiet.** After a terminal result, later frames
   for the same `call_id` are dropped.
3. **Cancellation closes the call.** `CANCEL` produces
   `ERROR(type="Cancelled", cancelled=True)`.
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
| runtime → worker | lazy on first call | `SHUTDOWN`, wait, terminate/kill fallback |

## Out of Scope

- Per-call timeouts; callers use `asyncio.wait_for(...)`.
- Retries; calls are at-most-once.
- Auth/TLS; deployments own that layer.
