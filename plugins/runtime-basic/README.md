# agentix-runtime-basic

Shell + file I/O primitives for [Agentix](https://github.com/Agentix-Project/Agentix)
sandboxes. One wheel, two namespaces:

| Namespace | Purpose |
|---|---|
| `agentix.bash` | execute shell commands |
| `agentix.files` | upload/download/list files inside the sandbox |

These two used to ship as separate `agentix-bash` and `agentix-files`
distributions. They consolidated here because every realistic sandbox
image needs both — splitting them was friction without isolation
benefit (neither has any non-stdlib runtime deps).

## Install

```bash
pip install agentix-runtime-basic
```

## Use

Call `run` from the host and await one `BashResult`:

```python
from agentix import RuntimeClient
from agentix.bash import run as bash_run
from agentix.files import upload, download

async with RuntimeClient(sandbox.runtime_url) as c:
    await c.remote(upload, path="data.json", content=blob)
    result = await c.remote(bash_run, command="cat data.json | jq .")
```

Each namespace is "the package IS the namespace": top-level async
functions are the remote-callable surface, dataclasses/types like
`BashResult` and `UploadResult` are importable for return-type
annotations.

## Building a sandbox image

`runtime/Dockerfile` is the base image bundle builds extend from. Most
users invoke it indirectly:

```bash
agentix build runtime-basic -o my-agent:0.1.0
```

## License

MIT — see [LICENSE](LICENSE).
