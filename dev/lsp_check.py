"""Talk LSP to pylsp and ruff the way the editor does and check that they understand this project.

Run from the project root inside the dev env (direnv or `source dev/env.sh`):  python dev/lsp_check.py
Checks: with the settings Emacs lsp-mode sends here (.dir-locals.el turns pylsp's linters off), go-to-definition
of a LangChain import lands in .venv and of a project import lands in src/; pylsp and `ruff server` (lsp-mode's
ruff add-on client, config in pyproject.toml) report no diagnostics for the files under src/. Exit 1 on failure.
"""

import json
import os
import queue
import shutil
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILE = ROOT / "src/lab/graph.py"


class Server:
    def __init__(self, cmd):
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.id, self.diagnostics, self.inbox = 0, {}, queue.Queue()
        threading.Thread(target=self._read_loop, daemon=True).start()

    def _write(self, msg):
        body = json.dumps({"jsonrpc": "2.0", **msg}).encode()
        self.p.stdin.write(b"Content-Length: %d\r\n\r\n%s" % (len(body), body))
        self.p.stdin.flush()

    def _read_loop(self):
        while True:
            n = 0
            while (line := self.p.stdout.readline().strip()):
                n = int(line.split(b":")[1]) if line.lower().startswith(b"content-length") else n
            if not n:
                return
            self.inbox.put(json.loads(self.p.stdout.read(n)))

    def send(self, method, params, notify=False):
        if notify:
            return self._write({"method": method, "params": params})
        self.id += 1
        self._write({"id": self.id, "method": method, "params": params})
        while (m := self._next()).get("id") != self.id or "method" in m:
            pass
        return m.get("result")

    def drain(self, uris):
        """Wait until every uri has published diagnostics (pylsp publishes them asynchronously)."""
        while set(uris) - set(self.diagnostics):
            self._next()
        while True:  # pylsp may publish again (e.g. after a config change); wait for 3 s of quiet
            try:
                self._handle(self.inbox.get(timeout=3))
            except queue.Empty:
                return

    def _next(self, timeout=60):
        try:
            m = self.inbox.get(timeout=timeout)
        except queue.Empty:
            sys.exit("timeout waiting for pylsp")
        return self._handle(m)

    def _handle(self, m):
        if m.get("method") == "textDocument/publishDiagnostics":
            self.diagnostics[m["params"]["uri"]] = m["params"]["diagnostics"]
        elif "id" in m and "method" in m:  # server->client request, e.g. workspace/configuration
            self._write({"id": m["id"], "result": None})
        return m


def position(text, needle):
    line_no = next(i for i, line in enumerate(text.splitlines()) if needle in line)
    return {"line": line_no, "character": text.splitlines()[line_no].index(needle) + 1}


# What lsp-mode's lsp-pylsp sends with .dir-locals.el applied: every lint plugin off, ruff lints instead.
PYLSP = {"pylsp": {"configurationSources": ["flake8"], "plugins": {
    p: {"enabled": False} for p in ["flake8", "pycodestyle", "pyflakes", "pydocstyle", "mccabe"]}}}


def check(label, cmd, settings=None, definitions=True):
    print(f"== {label}")
    s = Server([shutil.which(cmd[0]) or sys.exit(f"{cmd[0]} not on PATH"), *cmd[1:]])
    s.send("initialize", {"processId": os.getpid(), "rootUri": ROOT.as_uri(), "capabilities": {}})
    s.send("initialized", {}, notify=True)
    if settings:
        s.send("workspace/didChangeConfiguration", {"settings": settings}, notify=True)
    failures = []
    files = sorted((ROOT / "src/lab").glob("*.py"))
    for f in files:
        s.send("textDocument/didOpen", {"textDocument": {"uri": f.as_uri(), "languageId": "python",
                                                         "version": 1, "text": f.read_text()}}, notify=True)
    text = FILE.read_text()
    doc = {"uri": FILE.as_uri()}
    for needle, want in [] if not definitions else [("create_agent", ROOT / ".venv"), ("db.frontier", ROOT / "src")]:
        pos = position(text, needle if needle != "db.frontier" else "frontier(conn")
        res = s.send("textDocument/definition", {"textDocument": doc, "position": pos}) or []
        paths = [Path(r["uri"].removeprefix("file://")) for r in res]
        ok = any(str(p).startswith(str(want)) for p in paths)
        print(f"{'ok ' if ok else 'BAD'} definition of {needle}: {[str(p) for p in paths] or 'nothing'}")
        ok or failures.append(needle)
    s.drain([f.as_uri() for f in files])
    for uri, diags in sorted(s.diagnostics.items()):
        for d in diags:
            print(f"BAD {Path(uri).name}:{d['range']['start']['line'] + 1} [{d.get('source')}] {d['message']}")
            failures.append(uri)
    print(f"{len(s.diagnostics)} files checked, VIRTUAL_ENV={os.environ.get('VIRTUAL_ENV')}")
    s.send("shutdown", {})
    return failures


def main():
    failures = check("pylsp (Emacs lsp-mode settings)", ["pylsp"], PYLSP) + \
        check("ruff server", ["ruff", "server"], definitions=False)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
