# agentix-provider-e2b

[E2B](https://e2b.dev/) provider backend for
[Agentix](https://github.com/Agentix-Project/Agentix).

> Status: CLI surface in place; the managed-sandbox integration is
> still a stub. Tracking parity with `DockerProvider` (live runtime
> URL, `session()` context manager) before promoting to a 1.0
> release.

## Install

```bash
pip install agentix-provider-e2b
```

Set `E2B_API_KEY` in the environment.

## Use

```bash
agentix deploy e2b dist/my-agent.bundle.tar
```

```python
from agentix import RuntimeClient, SandboxConfig
from agentix.provider.e2b import E2BProvider

async with E2BProvider().session(
    SandboxConfig(image="python:3.13-slim", bundle="<backend bundle ref>")
) as sandbox:
    async with RuntimeClient(sandbox.runtime_url) as c:
        ...
```

## License

MIT — see [LICENSE](LICENSE).
