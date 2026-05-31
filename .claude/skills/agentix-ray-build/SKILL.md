---
name: agentix-ray-build
description: Build and run Agentix bundles on a restricted remote Ray cluster (rootless podman, no-Docker pod, egress only via a corporate HTTP proxy). Covers the working podman build recipe, sandbox run-args, and how to read job logs when `ray job logs` is blocked. Use when building/testing an Agentix bundle on a Ray box that lacks Docker.
---

# Building & running Agentix bundles on a restricted Ray cluster

Context: a remote Ray pod with **no Docker**, only **rootless podman** (4.x, crun),
**read-only cgroups**, and **egress only through a corporate HTTP proxy**. Jobs are
**ephemeral** (a `uv sync` in one job doesn't persist) and **logs are gateway-blocked**,
so build + run must happen in **one self-contained job** and progress is read via the
dashboard state API. Replace every `<placeholder>` with your environment's value.

## Submitting (one job, ephemeral)

Stage a clean repo and submit `python my_script.py` with `--working-dir` = repo root:

```
git archive HEAD | tar -x -C <ship-dir>          # clean tree, no .venv/.git
# write <ship-dir>/my_script.py (the driver below), then submit via the cluster's
# ray job submit (e.g. `ray job submit --address http://<head-ip>:8081 \
#   --working-dir <ship-dir> --no-wait -- python my_script.py`)
```

The driver must **stream** subprocess output (NOT `capture_output`) so it lands in the
job's `driver.log`, and end with a sentinel like `::RESULT rc=<n>` + `sys.exit(rc)`.

## The podman build recipe (what actually works)

1. **Stage as a git repo.** `agentix build` copies the *whole git repo* so a project's
   `../../plugins/*` path deps resolve. A non-git tree becomes a standalone context and
   those deps break (`Distribution not found at file:///plugins/...`). If you shipped via
   `git archive`, run `git init -q && git add -A` in the working dir first.
2. **Build RUN steps** must skip the read-only-cgroup + netns setup:
   `--container-arg --isolation=chroot --container-arg --network=host`.
3. **Nix egress.** Public `cache.nixos.org` stalls/throttles through a corp proxy. Put a
   fast mirror PRIMARY and keep `cache.nixos.org` as a coverage fallback — two separate
   options (`substituters` replaces the default, `extra-substituters` appends):
   `--nix-arg "--option substituters https://mirrors.ustc.edu.cn/nix-channels/store"`
   `--nix-arg "--option extra-substituters https://cache.nixos.org"`
   (TUNA `https://mirrors.tuna.tsinghua.edu.cn/nix-channels/store` works as primary too.
   The CN mirrors mirror channel snapshots and may miss a path — hence the cache fallback.)
4. **Export step.** The bundle is extracted with `podman create --network none` + copy —
   it never starts, so it needs NO cgroup/netns workaround. Do **NOT** pass
   `--container-run-arg --network=host` here (clashes with the default `--network none` →
   "cannot set multiple networks").
5. **Trim host-only deps** from the bundle project — provider backends (`agentix-provider-*`)
   are host-side and don't belong in the sandbox bundle (their `default.nix` can also drag
   heavy system binaries into the closure).

```
git init -q && git add -A
HTTP_PROXY=<corp-proxy> HTTPS_PROXY=<corp-proxy> NO_PROXY=127.0.0.1,localhost \
agentix build <project> --container-engine podman --platform linux/amd64 \
  --container-arg --isolation=chroot --container-arg --network=host \
  --nix-arg "--option substituters https://mirrors.ustc.edu.cn/nix-channels/store" \
  --nix-arg "--option extra-substituters https://cache.nixos.org"
```

## Running the sandbox

The sandbox container *does* start, so it **does** need the cgroup/netns workaround as
**run-args**: `--runtime=crun --cgroups=disabled --network=host`. With `--network=host`
the runtime server binds a host port (reach it at `127.0.0.1:<port>`, no mapping). Wire
them via `agentix deploy podman <tar> --run-arg --runtime=crun --run-arg --cgroups=disabled
--run-arg --network=host`, or the provider's run-arg config when orchestrating in-process.

> Trade-offs of these run-args: `--network=host` removes network isolation; `--cgroups=disabled`
> removes resource limits. Fine for trusted single-tenant eval/RL; not for multi-tenant.

## Reading logs when `ray job logs` is blocked

The dashboard proxies `ray job logs` (and `/api/jobs/<id>/logs`) to a per-node job-agent on
an ephemeral internal port that isn't reachable through `:8081` (device-auth gateway) →
`ConnectionRefusedError`. Use the status endpoint + state API on `:8081` instead:

```
H=http://<head-ip>:8081 ; JID=raysubmit_...
curl -s "$H/api/jobs/$JID" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['status'],'|',(d.get('message') or '')[:200])"
NID=$(curl -s "$H/api/v0/nodes" | python3 -c "import sys,json;print(json.load(sys.stdin)['data']['result']['result'][0]['node_id'])")
curl -s "$H/api/v0/logs?node_id=$NID&glob=*$JID*"                                  # confirm job-driver-$JID.log exists
curl -s "$H/api/v0/logs/file?node_id=$NID&filename=job-driver-$JID.log&lines=200" | tr -d '\000' | tail -60
```

Poll status until `SUCCEEDED`/`FAILED`/`STOPPED`. Device-auth challenges are intermittent
(transient `401`); just retry. If logs are *fully* unreachable, have the driver
`raise RuntimeError(tail)` so the tail surfaces in the job `message`.
