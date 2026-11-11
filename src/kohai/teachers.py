"""Teacher adapters: anything that turns a prompt into text.

Spec strings select a teacher:
    claude                       Claude Code CLI (`claude -p`)
    opencode                     OpenCode CLI (`opencode run`)
    gemini                       Gemini CLI
    codex                        OpenAI Codex CLI (`codex exec`)
    ollama:<model>               local Ollama  (http://localhost:11434/v1)
    lmstudio:<model>             local LM Studio (http://localhost:1234/v1)
    openrouter:<model>           OpenRouter (needs OPENROUTER_API_KEY)
    api:<model>                  any OpenAI-compatible server (KOHAI_API_BASE, KOHAI_API_KEY)
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")

READY = "ready"
INSTALLED = "installed"  # present, but auth can't be checked without a call
MISSING = "missing"
NEEDS_SETUP = "needs-setup"


class TeacherError(Exception):
    pass


@dataclass
class Teacher:
    id: str
    name: str

    def status(self) -> tuple[str, str]:
        raise NotImplementedError

    def complete(self, prompt: str, timeout: float = 300) -> str:
        raise NotImplementedError


@dataclass
class CLITeacher(Teacher):
    cmd: tuple[str, ...] = ()
    setup_hint: str = ""

    def status(self) -> tuple[str, str]:
        if not shutil.which(self.cmd[0]):
            return MISSING, f"`{self.cmd[0]}` not found on PATH"
        return INSTALLED, self.setup_hint

    def complete(self, prompt: str, timeout: float = 300) -> str:
        exe = shutil.which(self.cmd[0])
        if not exe:
            raise TeacherError(f"{self.name}: `{self.cmd[0]}` is not installed")
        try:
            # The prompt goes through stdin: Windows shims (.cmd) would mangle
            # regex metacharacters like | ^ < > if passed as arguments.
            r = subprocess.run(
                [exe, *self.cmd[1:]],
                input=prompt,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                creationflags=_NO_WINDOW,
            )
        except subprocess.TimeoutExpired as e:
            raise TeacherError(f"{self.name}: timed out after {timeout:.0f}s") from e
        out = _ANSI.sub("", r.stdout or "").strip()
        if r.returncode != 0 or not out:
            err = _ANSI.sub("", r.stderr or "").strip().splitlines()
            tail = " ".join(err[-3:])[-400:] or f"exit code {r.returncode}"
            raise TeacherError(f"{self.name}: {tail}")
        return out


@dataclass
class APITeacher(Teacher):
    base_url: str = ""
    model: str = ""
    key_env: str | None = None

    def _key(self) -> str | None:
        return os.environ.get(self.key_env) if self.key_env else None

    def status(self) -> tuple[str, str]:
        if not self.model:
            return NEEDS_SETUP, f"pick a model, e.g. `{self.id.split(':')[0]}:<model>`"
        if self.key_env and not self._key():
            return NEEDS_SETUP, f"set {self.key_env}"
        try:
            req = urllib.request.Request(f"{self.base_url}/models", headers=self._headers())
            urllib.request.urlopen(req, timeout=2).read()
        except Exception:
            return MISSING, f"no server at {self.base_url}"
        return READY, self.base_url

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self._key():
            h["Authorization"] = f"Bearer {self._key()}"
        return h

    def complete(self, prompt: str, timeout: float = 300) -> str:
        body = json.dumps(
            {"model": self.model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.8}
        ).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body, headers=self._headers(), method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            raise TeacherError(f"{self.name}: HTTP {e.code} {e.read()[:300]!r}") from e
        except Exception as e:
            raise TeacherError(f"{self.name}: {e}") from e
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as e:
            raise TeacherError(f"{self.name}: unexpected response {str(data)[:300]}") from e


CLI_TEACHERS = {
    "claude": ("Claude Code", ("claude", "-p", "--output-format", "text"), "uses your Claude Code login"),
    "opencode": ("OpenCode", ("opencode", "run"), "uses OpenCode's configured model"),
    "gemini": ("Gemini CLI", ("gemini",), "needs GEMINI_API_KEY or a Google login"),
    "codex": ("Codex CLI", ("codex", "exec", "--skip-git-repo-check", "-"), "needs `codex login`"),
}

API_PRESETS = {
    "ollama": ("Ollama", "http://localhost:11434/v1", None),
    "lmstudio": ("LM Studio", "http://localhost:1234/v1", None),
    "openrouter": ("OpenRouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "api": ("OpenAI-compatible", os.environ.get("KOHAI_API_BASE", "http://localhost:8000/v1"), "KOHAI_API_KEY"),
}


def get_teacher(spec: str) -> Teacher:
    kind, _, model = spec.partition(":")
    if kind in CLI_TEACHERS and not model:
        name, cmd, hint = CLI_TEACHERS[kind]
        return CLITeacher(id=kind, name=name, cmd=cmd, setup_hint=hint)
    if kind in API_PRESETS:
        name, base, key_env = API_PRESETS[kind]
        if kind == "api":
            base = os.environ.get("KOHAI_API_BASE", base)
            key_env = key_env if os.environ.get(key_env or "") else None
        label = f"{name} · {model}" if model else name
        return APITeacher(id=spec, name=label, base_url=base.rstrip("/"), model=model, key_env=key_env)
    raise ValueError(
        f"Unknown teacher {spec!r}. Try: {', '.join(CLI_TEACHERS)}, "
        + ", ".join(f"{k}:<model>" for k in API_PRESETS)
    )


def all_teachers() -> list[Teacher]:
    """Every teacher kind, for listing/detection (API ones without a model)."""
    return [get_teacher(k) for k in CLI_TEACHERS] + [get_teacher(k) for k in API_PRESETS]
