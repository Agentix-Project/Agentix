# Releasing cc_convert to PyPI

## One-time setup (only first time)

### 1. Create PyPI + TestPyPI accounts

- Real:  https://pypi.org/account/register/
- Test:  https://test.pypi.org/account/register/  (separate account, separate password)

Enable 2FA on both (PyPI requires it for publishing).

### 2. Reserve the project name + configure Trusted Publishing

The Trusted Publisher needs a *project* on PyPI to attach to. There are two
ways to bootstrap:

**Option A — Reserve the name yourself first (recommended).**
Manually `pip install twine && twine upload` a tiny placeholder wheel once,
then add the trusted-publisher record. After that, every release happens
via GitHub Actions with no token.

**Option B — Use Pending Publisher.**
On https://pypi.org/manage/account/publishing/ add a *pending* trusted
publisher *before* the project exists. The first GitHub Actions run that
matches the spec will create the project automatically.

Settings to enter (either option):

| Field | Value |
|---|---|
| PyPI Project Name | `cc-convert` |
| Owner | `yitianlian` |
| Repository name | `cc_convert` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

Do the same on TestPyPI:
https://test.pypi.org/manage/account/publishing/

| Field | Value |
|---|---|
| PyPI Project Name | `cc-convert` |
| Owner | `yitianlian` |
| Repository name | `cc_convert` |
| Workflow name | `release.yml` |
| Environment name | `testpypi` |

### 3. Create the GitHub Environments

GitHub side, on https://github.com/yitianlian/cc_convert/settings/environments
create two environments:

- `pypi` (no protection rules needed for now; you can add "Required reviewers" later if you want a manual approval gate per release)
- `testpypi` (no protection rules)

These names must match the `environment:` field in `release.yml`.

---

## Releasing a new version

### Dry-run to TestPyPI (recommended every time)

```bash
# Go to:
# https://github.com/yitianlian/cc_convert/actions/workflows/release.yml
# Click "Run workflow" → choose "testpypi" → Run.

# Or via gh CLI:
gh workflow run release.yml -f target=testpypi
```

This builds wheels for Linux x64 + Linux aarch64 + macOS x64 + macOS ARM +
Windows x64 + sdist, then uploads them to https://test.pypi.org/project/cc-convert/.

Verify it installs:

```bash
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            cc-convert
cc_convert --version
```

### Cut a real release

```bash
# 1. Bump version
sed -i 's/version = "0.1.0"/version = "0.2.0"/' python/pyproject.toml Cargo.toml
git add -A && git commit -m "release: v0.2.0"
git push

# 2. Tag and push the tag — that triggers the publish job.
git tag v0.2.0
git push origin v0.2.0
```

The tag push fires the `release.yml` workflow. It will:

1. Build wheels in parallel on 5 runners (Linux x64, Linux aarch64,
   macOS Intel, macOS ARM, Windows x64) plus an sdist.
2. Upload all artifacts.
3. Publish to https://pypi.org/project/cc-convert/ via the `pypi` environment
   (which is tied to the trusted publisher you configured in step 2 above).

Watch progress at https://github.com/yitianlian/cc_convert/actions

### After the workflow completes

Anyone in the world can now:

```bash
pip install cc-convert     # imports as: import cc_convert
cc_convert --version
```

---

## Troubleshooting

**"unable to upload: trusted publisher not configured"**
→ The PyPI Trusted Publisher record doesn't match what GitHub sent. Check
that Owner / Repository / Workflow / Environment all match exactly. The
workflow filename is `release.yml`, not `release` or `.github/workflows/release.yml`.

**"file already exists" on PyPI**
→ You can't re-upload the same version. Bump `version =` in
`python/pyproject.toml` AND `Cargo.toml`, retag, repush.

**One platform's wheel build failed**
→ The workflow uses `needs: [build-linux, build-macos, build-windows, build-sdist]`
on the publish job, so if any platform fails the whole release stops (no
half-published version on PyPI). Fix the failing job and push the tag again
— but first bump the version, because the same version can't be reuploaded.

**Manual rescue / emergency upload**
→ Generate a PyPI API token, then locally:

```bash
maturin upload --username __token__ --password "pypi-AgEIcHl..." \
    target/wheels/cc_convert-*.whl
```

---

## What the workflow is doing under the hood

- **abi3-py38 wheel**: each (OS, arch) gets *one* wheel with file name
  `cc_convert-X.Y.Z-cp38-abi3-<platform>.whl` that works on Python 3.8
  through any future 3.x. This is enabled in `crates/cc_convert_py/Cargo.toml`
  via `pyo3 = { ..., features = ["extension-module", "abi3-py38"] }`.

- **manylinux 2.34**: built inside the official `quay.io/pypa/manylinux_2_34`
  container so the resulting wheel works on any reasonably-modern Linux
  distro (glibc >= 2.34). For older glibc we'd switch to `manylinux_2_28`
  or 2014 — bump only if someone reports they need it.

- **sccache**: the workflow caches Rust build artifacts across runs via
  `sccache: "true"` on the `PyO3/maturin-action` step. First release takes
  ~15min; later ones are faster.

- **Trusted Publishing (OIDC)**: instead of a long-lived API token in
  GitHub secrets, each workflow run gets a short-lived OIDC identity from
  GitHub that PyPI cryptographically verifies came from our exact
  workflow file in our exact repo. No secrets to rotate or leak.
