"""Sandbox-side: SWE-bench Verified primitives.

Three remote-call entry points, mirroring the flow in
`swebench.harness.run_evaluation.run_instance` (L151+):

    1. `swe.clean(workdir, base_commit)`
       Reset `/testbed` to `base_commit`, drop any extra commits, wipe
       the working tree. Run before `cc.run` so the model starts from a
       known state regardless of what the prebuilt image happened to
       ship at HEAD.

    2. `swe.get_patch(workdir)`
       Capture the model's diff against `base_commit` — staged adds +
       working-tree edits — as a single unified diff string.

    3. `swe.eval(instance, patch)`
       Reproduce SWE-bench's evaluation:
         (a) write `patch` to `/tmp/patch.diff`, try the GIT_APPLY_CMDS
             fallback chain; emit APPLY_PATCH_PASS / APPLY_PATCH_FAIL.
         (b) write `test_spec.eval_script` to `/eval.sh` and run it.
         (c) grade the log via `get_eval_report`.
       Returns `{resolved, patch_applied, tests_status}`.

The intended host flow is two sandboxes per instance: one for the cc
agent (clean → cc.run → get_patch), tear down, then a fresh sandbox
for swe.eval. Both sandboxes use the per-instance SWE-bench eval
image (`swebench/sweb.eval.x86_64.<id>:latest`) as the base.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

WORKROOT = Path(os.environ.get("AGENTIX_UPLOAD_ROOT", "/tmp")) / ".cache" / "swebench-eval"
TESTBED = "/testbed"
PATCH_DOCKER_PATH = "/tmp/patch.diff"

# Same fallback chain as swebench.harness.run_evaluation.GIT_APPLY_CMDS.
GIT_APPLY_CMDS = [
    "git apply --verbose",
    "git apply --verbose --reject",
    "patch --batch --fuzz=5 -p1 -i",
]

ASTROPY_PYTEST_6_INSTANCES = {
    "astropy__astropy-8707",
    "astropy__astropy-8872",
}

ASTROPY_STDLIB_DISTUTILS_INSTANCES = {
    "astropy__astropy-8872",
}

PSF_REQUESTS_LOCAL_HTTPBIN_INSTANCES = {
    "psf__requests-1724",
    "psf__requests-1766",
    "psf__requests-2317",
}

PSF_REQUESTS_TARPIT_TIMEOUT_INSTANCES = {
    "psf__requests-2317",
    "psf__requests-2931",
    "psf__requests-5414",
}

ASTROPY_7606_EMPTY_PARAM = "astropy/units/tests/test_units.py::test_compose_roundtrip[]"
ASTROPY_7606_EMPTY_PARAM_BASE = ASTROPY_7606_EMPTY_PARAM[:-2]


@dataclass
class CleanResult:
    ok: bool
    head: str
    log: str


@dataclass
class EvalResult:
    resolved: bool
    patch_applied: bool
    apply_cmd: str | None
    known_fixes: list[str] = field(default_factory=list)
    fail_to_pass: dict[str, list[str]] = field(default_factory=dict)
    pass_to_pass: dict[str, list[str]] = field(default_factory=dict)
    git_diff_before: str = ""
    git_diff_after: str = ""
    apply_log: str = ""
    test_log: str = ""


async def clean(workdir: str = TESTBED, base_commit: str | None = None) -> CleanResult:
    """Reset `workdir` to `base_commit` and drop everything else.

    Mirrors SWE-bench's expectation that /testbed enters the eval at
    exactly base_commit with a clean working tree.
    """
    parts = [
        f"cd {workdir}",
        "git -c core.fileMode=false update-index --refresh >/dev/null 2>&1 || true",
    ]
    if base_commit:
        parts.append(f"git reset --hard {base_commit}")
    else:
        parts.append("git reset --hard HEAD")
    parts.append("git clean -fdx")
    parts.append("git rev-parse HEAD")

    proc = await asyncio.create_subprocess_shell(
        " && ".join(parts),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    out, _ = await proc.communicate()
    text = out.decode(errors="replace")
    head = text.strip().splitlines()[-1] if proc.returncode == 0 else ""
    return CleanResult(ok=proc.returncode == 0, head=head, log=text)


async def get_patch(workdir: str = TESTBED) -> str:
    """Return all `workdir` changes (including new files) as a unified diff.

    `git add -A` first so untracked files appear in `--cached`; this
    keeps parity with SWE-bench's expected prediction format.
    """
    proc = await asyncio.create_subprocess_shell(
        (
            f"cd {workdir} && "
            "git -c core.fileMode=false add -A && "
            "git -c core.fileMode=false diff --cached --no-color --binary"
        ),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, _ = await proc.communicate()
    return out.decode(errors="replace") if proc.returncode == 0 else ""


async def eval(
    *,
    instance: dict[str, Any],
    patch: str,
    workdir: str = TESTBED,
    apply_timeout: float = 120,
    eval_timeout: float = 1800,
) -> EvalResult:
    """Apply `patch` in `workdir`, run the instance's eval_script, grade."""
    from swebench.harness.constants import (
        APPLY_PATCH_FAIL,
        APPLY_PATCH_PASS,
        KEY_INSTANCE_ID,
        KEY_MODEL,
        KEY_PREDICTION,
    )
    from swebench.harness.grading import get_eval_report
    from swebench.harness.test_spec.test_spec import make_test_spec

    spec = make_test_spec(instance)
    workroot = WORKROOT / spec.instance_id
    if workroot.exists():
        shutil.rmtree(workroot)
    workroot.mkdir(parents=True)

    # (a) Stage the patch and try GIT_APPLY_CMDS in order — same chain
    # as swebench/harness/run_evaluation.py.
    Path(PATCH_DOCKER_PATH).write_text(patch or "")
    apply_log_parts: list[str] = []
    applied_with: str | None = None
    for cmd in GIT_APPLY_CMDS:
        proc = await asyncio.create_subprocess_shell(
            f"cd {workdir} && {cmd} {PATCH_DOCKER_PATH}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            out, _ = await asyncio.wait_for(
                proc.communicate(),
                timeout=apply_timeout,
            )
        except TimeoutError:
            proc.kill()
            await proc.communicate()
            apply_log_parts.append(f"[{cmd}] timed out\n")
            continue
        apply_log_parts.append(f"[{cmd}] exit={proc.returncode}\n{out.decode(errors='replace')}\n")
        if proc.returncode == 0:
            applied_with = cmd
            break

    if applied_with:
        apply_log_parts.append(f"{APPLY_PATCH_PASS}\n")
    else:
        apply_log_parts.append(f"{APPLY_PATCH_FAIL}\n")

    apply_log = "".join(apply_log_parts)
    (workroot / "apply.log").write_text(apply_log)

    git_diff_before = await _capture_diff(workdir)

    # (b) Run the eval script. test_spec.eval_script handles applying
    # test_patch and invoking the test command with START/END markers.
    eval_script = workroot / "eval.sh"
    eval_script_text, known_fixes = _patch_eval_script_for_known_image_issues(spec, spec.eval_script)
    eval_script.write_text(eval_script_text)
    eval_script.chmod(0o755)

    test_log_path = workroot / "test_output.log"
    test_log = await _run_script(eval_script, test_log_path, timeout=eval_timeout)

    # The grading function expects APPLY_PATCH_PASS / APPLY_PATCH_FAIL
    # in the same log file it parses for test results.
    combined_log = apply_log + test_log
    test_log_path.write_text(combined_log)

    git_diff_after = await _capture_diff(workdir)

    # (c) Grade with the official report function.
    pred = {
        KEY_INSTANCE_ID: spec.instance_id,
        KEY_MODEL: "eval-cc-swe",
        KEY_PREDICTION: patch,
    }
    report = get_eval_report(
        test_spec=spec,
        prediction=pred,
        test_log_path=str(test_log_path),
        include_tests_status=True,
    )
    entry, grading_fixes = _apply_known_grading_fixes(spec, report.get(spec.instance_id, {}), combined_log)
    known_fixes.extend(grading_fixes)
    tests = entry.get("tests_status", {}) or {}
    ftp = tests.get("FAIL_TO_PASS", {"success": [], "failure": []})
    ptp = tests.get("PASS_TO_PASS", {"success": [], "failure": []})

    return EvalResult(
        resolved=bool(entry.get("resolved", False)),
        patch_applied=bool(entry.get("patch_successfully_applied", False)),
        apply_cmd=applied_with,
        known_fixes=known_fixes,
        fail_to_pass={
            "success": list(ftp.get("success", [])),
            "failure": list(ftp.get("failure", [])),
        },
        pass_to_pass={
            "success": list(ptp.get("success", [])),
            "failure": list(ptp.get("failure", [])),
        },
        git_diff_before=git_diff_before,
        git_diff_after=git_diff_after,
        apply_log=apply_log,
        test_log=test_log,
    )


async def _capture_diff(workdir: str) -> str:
    proc = await asyncio.create_subprocess_shell(
        f"cd {workdir} && git -c core.fileMode=false diff",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    out, _ = await proc.communicate()
    return out.decode(errors="replace").strip()


def _patch_eval_script_for_known_image_issues(spec: Any, eval_script: str) -> tuple[str, list[str]]:
    """Patch instance-scoped SWE-bench environment drift before tests run."""
    known_fixes: list[str] = []
    if spec.instance_id in ASTROPY_PYTEST_6_INSTANCES:
        eval_script = _fix_astropy_pytest_7_incompat(spec.instance_id, eval_script)
        known_fixes.append(f"{spec.instance_id}:pytest<7")
    if spec.instance_id in ASTROPY_STDLIB_DISTUTILS_INSTANCES:
        eval_script = _fix_astropy_distutils_version_warning(spec.instance_id, eval_script)
        known_fixes.append(f"{spec.instance_id}:stdlib-distutils")
    if spec.instance_id == "django__django-10097":
        eval_script = _fix_django_10097_sqlite_legacy_alter_table(eval_script)
        known_fixes.append("django__django-10097:sqlite-legacy-alter-table")
    if spec.instance_id.startswith("sphinx-doc__"):
        eval_script, fixed = _fix_sphinx_tox_pytest_report_sections(eval_script)
        if fixed:
            known_fixes.append("sphinx-doc:tox-pytest-report-sections")
    if spec.instance_id in PSF_REQUESTS_LOCAL_HTTPBIN_INSTANCES:
        eval_script = _fix_psf_requests_local_httpbin(eval_script)
        known_fixes.append(f"{spec.instance_id}:local-httpbin")
    if spec.instance_id in PSF_REQUESTS_TARPIT_TIMEOUT_INSTANCES:
        eval_script = _fix_psf_requests_tarpit_connect_timeout(eval_script)
        known_fixes.append(f"{spec.instance_id}:tarpit-connect-timeout")
    return eval_script, known_fixes


def _fix_astropy_pytest_7_incompat(instance_id: str, eval_script: str) -> str:
    # These 2019 Astropy snapshots use nose-style setup hooks and a warning policy
    # that is incompatible with pytest 7.x installed by the regenerated image.
    return _insert_before_once(
        eval_script,
        "python -m pip install -e .[test] --verbose",
        (f": 'agentix known image fix: {instance_id} requires pytest<7'\npython -m pip install 'pytest<7' --verbose\n"),
    )


def _fix_astropy_distutils_version_warning(instance_id: str, eval_script: str) -> str:
    # The image's setuptools shim redirects distutils.version to
    # setuptools._distutils, whose Version classes warn even on Python 3.9.
    # Astropy 4.0-dev turns deprecations into errors during collection.
    return _insert_before_once(
        eval_script,
        "pytest -rA astropy/units/tests/test_quantity.py",
        (
            f": 'agentix known image fix: {instance_id} uses stdlib distutils under old warning policy'\n"
            "export SETUPTOOLS_USE_DISTUTILS=stdlib\n"
        ),
    )


def _fix_django_10097_sqlite_legacy_alter_table(eval_script: str) -> str:
    # Django 2.2-dev predates SQLite 3.26's ALTER TABLE rename behavior. The
    # eval image has SQLite 3.45, which rewrites M2M foreign keys to
    # django_site__old during migrations unless legacy_alter_table is enabled.
    return _insert_before_once(
        eval_script,
        "python setup.py install",
        r'''python - <<'PY'
from pathlib import Path

path = Path('django/db/backends/sqlite3/schema.py')
text = path.read_text()
old = """    def __enter__(self):
        # Some SQLite schema alterations need foreign key constraints to be
        # disabled. Enforce it here for the duration of the transaction.
        self.connection.disable_constraint_checking()
        return super().__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        self.connection.enable_constraint_checking()
"""
new = """    def __enter__(self):
        # Some SQLite schema alterations need foreign key constraints to be
        # disabled. Enforce it here for the duration of the transaction.
        self.connection.disable_constraint_checking()
        with self.connection.cursor() as cursor:
            cursor.execute('PRAGMA legacy_alter_table = ON')
        return super().__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            with self.connection.cursor() as cursor:
                cursor.execute('PRAGMA legacy_alter_table = OFF')
            self.connection.enable_constraint_checking()
"""
if old not in text:
    raise SystemExit('django__django-10097 SQLite schema fix target not found')
path.write_text(text.replace(old, new))
PY
''',
    )


def _fix_sphinx_tox_pytest_report_sections(eval_script: str) -> tuple[str, bool]:
    # Some Sphinx eval scripts run pytest through tox without -rA, so a passing
    # targeted test can produce only "." plus "1 passed". SWE-bench grading
    # needs per-test PASSED lines; PYTEST_ADDOPTS is already threaded through
    # Sphinx tox.ini, so this changes report verbosity without changing tests.
    needle = "tox --current-env"
    insertion = 'export PYTEST_ADDOPTS="${PYTEST_ADDOPTS:-} -rA"\n'
    if insertion in eval_script or needle not in eval_script:
        return eval_script, False
    return eval_script.replace(needle, insertion + needle, 1), True


def _fix_psf_requests_local_httpbin(eval_script: str) -> str:
    # requests 2.0-era tests depend on public httpbin.org, which makes Oracle
    # grading depend on Docker DNS/proxy/routing state. Current Requests tests
    # use pytest-httpbin/httpbin fixtures; mirror that by routing these legacy
    # URLs to a local stdlib httpbin-compatible shim.
    return _insert_before_once(
        eval_script,
        "pytest -rA test_requests.py",
        r"""openssl req -x509 -newkey rsa:2048 -days 1 -nodes \
  -keyout /tmp/agentix-httpbin.key \
  -out /tmp/agentix-httpbin.crt \
  -subj /CN=httpbin.org \
  -addext subjectAltName=DNS:localhost,DNS:httpbin.org >/tmp/agentix-httpbin-openssl.log 2>&1
python - <<'PY'
import requests.certs

with open(requests.certs.where(), 'ab') as out, open('/tmp/agentix-httpbin.crt', 'rb') as cert:
    out.write(b'\n')
    out.write(cert.read())
PY
if ! grep -q ' httpbin.org' /etc/hosts; then
  echo '127.0.0.1 httpbin.org' >> /etc/hosts
fi
cat > /tmp/agentix_httpbin.py <<'PY'
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse
import base64
import gzip
import hashlib
import json
import ssl
import threading
import time

REALM = 'testrealm'
NONCE = 'agentix'
OPAQUE = 'agentix'


def _one(query):
    return {key: values if len(values) != 1 else values[0] for key, values in query.items()}


def _headers(headers):
    return {key: value for key, value in headers.items()}


def _cookies(header):
    cookie = SimpleCookie()
    if header:
        cookie.load(header)
    return {key: morsel.value for key, morsel in cookie.items()}


def _parse_digest(value):
    if not value or not value.lower().startswith('digest '):
        return {}
    parsed = {}
    for part in value[7:].split(','):
        if '=' not in part:
            continue
        key, raw_value = part.strip().split('=', 1)
        parsed[key] = raw_value.strip().strip('"')
    return parsed


def _digest_ok(method, authorization, password='pass'):
    parsed = _parse_digest(authorization)
    if parsed.get('username') != 'user':
        return False
    if parsed.get('realm') != REALM or parsed.get('nonce') != NONCE:
        return False

    uri = parsed.get('uri', '')
    qop = parsed.get('qop')
    ha1 = hashlib.md5(f'user:{REALM}:{password}'.encode()).hexdigest()
    ha2 = hashlib.md5(f'{method}:{uri}'.encode()).hexdigest()
    if qop:
        digest_source = f"{ha1}:{NONCE}:{parsed.get('nc')}:{parsed.get('cnonce')}:{qop}:{ha2}"
    else:
        digest_source = f'{ha1}:{NONCE}:{ha2}'
    return parsed.get('response') == hashlib.md5(digest_source.encode()).hexdigest()


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    server_version = 'agentix-httpbin'

    def log_message(self, fmt, *args):
        return

    def _send(self, status=200, body=b'', headers=None):
        self.send_response(status)
        for key, value in (headers or {}).items():
            if isinstance(value, (list, tuple)):
                for item in value:
                    self.send_header(key, item)
            else:
                self.send_header(key, value)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        if self.command != 'HEAD':
            self.wfile.write(body)

    def _json(self, payload, status=200, headers=None):
        response_headers = {'Content-Type': 'application/json'}
        response_headers.update(headers or {})
        self._send(status, json.dumps(payload).encode(), response_headers)

    def _redirect(self, location, status=302, headers=None):
        response_headers = {'Location': location}
        response_headers.update(headers or {})
        self._send(status, b'', response_headers)

    def _basic_auth(self):
        expected = 'Basic ' + base64.b64encode(b'user:pass').decode()
        if self.headers.get('Authorization') == expected:
            self._json({'authenticated': True, 'user': 'user'})
        else:
            self._send(401, b'', {'WWW-Authenticate': 'Basic realm=Fake Realm'})

    def _digest_auth(self):
        if _digest_ok(self.command, self.headers.get('Authorization')):
            self._send(200, b'authenticated', {'Set-Cookie': 'fake=fake_value'})
            return
        challenge = f'Digest realm="{REALM}", nonce="{NONCE}", qop="auth", opaque="{OPAQUE}"'
        self._send(401, b'', {'WWW-Authenticate': challenge, 'Set-Cookie': 'fake=fake_value'})

    def do_HEAD(self):
        self.do_GET()

    def do_PUT(self):
        self.do_POST()

    def do_POST(self):
        parsed = urlparse(self.path)
        length = int(self.headers.get('Content-Length', '0') or '0')
        body = b''
        if length:
            body = self.rfile.read(length)
        payload = {
            'args': _one(parse_qs(parsed.query, keep_blank_values=True)),
            'data': body.decode(errors='replace'),
            'headers': _headers(self.headers),
            'json': None,
        }
        if 'application/json' in self.headers.get('Content-Type', ''):
            payload['json'] = json.loads(body.decode() or 'null')
        self._json(payload)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        args = _one(parse_qs(parsed.query, keep_blank_values=True))
        if path in ('', '/') or path == '/get' or path.startswith('/get/'):
            self._json({'args': args, 'headers': _headers(self.headers), 'url': self.path})
        elif path.startswith('/delay/'):
            time.sleep(1.0)
            self._json({'args': args, 'headers': _headers(self.headers), 'url': self.path})
        elif path == '/headers':
            self._json({'headers': _headers(self.headers)})
        elif path == '/user-agent':
            self._json({'user-agent': self.headers.get('User-Agent', '')})
        elif path == '/cookies':
            self._json({'cookies': _cookies(self.headers.get('Cookie'))})
        elif path == '/cookies/set':
            cookies = [f'{key}={value}; Path=/' for key, value in args.items()]
            self._redirect('/cookies', headers={'Set-Cookie': cookies})
        elif path.startswith('/redirect/'):
            try:
                count = int(path.rsplit('/', 1)[1])
            except ValueError:
                count = 1
            self._redirect('/get' if count <= 1 else f'/redirect/{count - 1}')
        elif path == '/redirect-to':
            self._redirect(args.get('url', '/get'))
        elif path == '/response-headers':
            self._json({'headers': args}, headers={key: value for key, value in args.items()})
        elif path == '/basic-auth/user/pass':
            self._basic_auth()
        elif path == '/digest-auth/auth/user/pass':
            self._digest_auth()
        elif path.startswith('/status/'):
            self._send(int(path.rsplit('/', 1)[1]), b'')
        elif path == '/gzip':
            body = gzip.compress(json.dumps({'gzipped': True, 'headers': _headers(self.headers)}).encode())
            self._send(200, body, {'Content-Type': 'application/json', 'Content-Encoding': 'gzip'})
        elif path == '/html':
            self._send(200, b'<html><body>ok</body></html>', {'Content-Type': 'text/html'})
        else:
            self._json({'args': args, 'headers': _headers(self.headers), 'url': self.path})


def _serve(port, cert=None, key=None):
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    if cert:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(cert, key)
        server.socket = context.wrap_socket(server.socket, server_side=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()


_serve(80)
_serve(443, '/tmp/agentix-httpbin.crt', '/tmp/agentix-httpbin.key')
print('agentix local httpbin ready', flush=True)
threading.Event().wait()
PY
python /tmp/agentix_httpbin.py >/tmp/agentix-httpbin.log 2>&1 &
for i in $(seq 1 50); do
  python - <<'PY' >/dev/null 2>&1 && break || sleep 0.1
import urllib.request

urllib.request.urlopen('http://localhost/get', timeout=1).read()
PY
done
export HTTPBIN_URL=http://localhost/
export NO_PROXY="httpbin.org,localhost,127.0.0.1${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"
""",
    )


def _fix_psf_requests_tarpit_connect_timeout(eval_script: str) -> str:
    # Several requests tests assume 10.255.255.1 blackholes TCP connect. Docker
    # Desktop currently routes it to a live-but-silent endpoint, producing read
    # timeouts or responses instead of ConnectTimeout. Restore the intended
    # sentinel behavior at the socket boundary for that exact address.
    return _insert_before_once(
        eval_script,
        "pytest -rA",
        r'''python - <<'PY'
from pathlib import Path

path = Path('conftest.py')
addition = r"""
import socket

_agentix_socket_connect = socket.socket.connect


def _agentix_tarpit_connect(self, address):
    if isinstance(address, tuple) and address and address[0] == "10.255.255.1":
        raise socket.timeout("timed out")
    return _agentix_socket_connect(self, address)


socket.socket.connect = _agentix_tarpit_connect
"""
text = path.read_text() if path.exists() else ''
if '_agentix_tarpit_connect' not in text:
    path.write_text(text.rstrip() + '\n\n' + addition.lstrip())
PY
''',
    )


def _insert_before_once(eval_script: str, needle: str, insertion: str) -> str:
    if insertion in eval_script:
        return eval_script
    if needle not in eval_script:
        raise ValueError(f"cannot patch eval script; missing line: {needle}")
    return eval_script.replace(needle, insertion + needle, 1)


def _apply_known_grading_fixes(spec: Any, entry: dict[str, Any], combined_log: str) -> tuple[dict[str, Any], list[str]]:
    known_fixes: list[str] = []
    if spec.instance_id == "astropy__astropy-7606":
        entry, fixed = _fix_astropy_7606_empty_param_node_id(entry, combined_log)
        if fixed:
            known_fixes.append("astropy__astropy-7606:empty-param-node-id")
    return entry, known_fixes


def _fix_astropy_7606_empty_param_node_id(entry: dict[str, Any], combined_log: str) -> tuple[dict[str, Any], bool]:
    # test_compose_roundtrip uses ids=str(unit); the dimensionless unit's string
    # id is empty. Pytest fills an empty generated id as unitN, so accept only
    # the unique unitN node that the log proves passed for this exact test.
    tests = entry.get("tests_status", {}) or {}
    ptp = tests.get("PASS_TO_PASS", {}) or {}
    failures = list(ptp.get("failure", []))
    if failures != [ASTROPY_7606_EMPTY_PARAM]:
        return entry, False

    passed_aliases = sorted(
        set(re.findall(rf"({re.escape(ASTROPY_7606_EMPTY_PARAM_BASE)}\[unit\d+\]) PASSED", combined_log)),
    )
    if len(passed_aliases) != 1:
        return entry, False

    fixed = dict(entry)
    fixed_tests = dict(tests)
    fixed_ptp = {
        "success": list(ptp.get("success", [])) + [ASTROPY_7606_EMPTY_PARAM],
        "failure": [],
    }
    fixed_tests["PASS_TO_PASS"] = fixed_ptp
    fixed["tests_status"] = fixed_tests

    ftp = fixed_tests.get("FAIL_TO_PASS", {}) or {}
    fixed["resolved"] = not ftp.get("failure") and not fixed_ptp["failure"]
    return fixed, True


async def _run_script(path: Path, log_path: Path, *, timeout: float) -> str:
    """Run `path` under bash, return combined stdout+stderr text."""
    from agentix.runtime.env import get_env_without_agentix

    proc = await asyncio.create_subprocess_exec(
        "/bin/bash",
        str(path),
        env=get_env_without_agentix(),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return out.decode(errors="replace")
    except TimeoutError:
        from swebench.harness.constants import TESTS_TIMEOUT

        proc.kill()
        await proc.communicate()
        return f"{TESTS_TIMEOUT}\nscript {path.name} timed out after {timeout}s\n"


__all__ = ["clean", "get_patch", "eval", "CleanResult", "EvalResult"]
