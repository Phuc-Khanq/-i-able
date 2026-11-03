"""The trained clean-up model in llama.cpp: our adapter merged into Qwen3.5-4B, one GGUF file, served by
llama-server on this PC only. Used by rewrite.Rewriter when settings.rewrite_backend is "llamacpp".

Route: Khoi and Khoa (Khoi leads)
APIs: llama.cpp's llama-server (MIT, build b11960) over HTTP on 127.0.0.1. Nothing leaves this PC
      (CONTRIBUTING rule 10): the server listens on 127.0.0.1 only, runs --offline, needs a random key made for
      this run (so a web page open on the PC can't call it), and its log stays in memory.

The prompt is the one the model was trained on (prompts/cleanup_t1_short.txt, the same file as the eval's),
written in Qwen's chat format with thinking off, and decoding is greedy: the same words always give the same
answer. The model answers {"text", "alternatives", "uncertain"}; see parse_answer.
"""

from __future__ import annotations

import json
import re
import secrets
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from string import Template

ROOT = Path(__file__).resolve().parents[2]
PROMPT_FILE = Path(__file__).resolve().parent / "prompts" / "cleanup_t1_short.txt"
GREEDY = {"top_k": 1, "repeat_penalty": 1.0, "samplers": ["top_k"]}     # only the most likely token
_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)


# ------------------------------------------------------------------ the prompt and the answer
def load_prompt(path: Path = PROMPT_FILE) -> tuple[str, Template]:
    """(system message, user template) from a prompt file: the system text, "=== user ===", the user template."""
    system, sep, user = path.read_text(encoding="utf-8").partition("\n=== user ===\n")
    if not sep:
        raise ValueError(f"{path.name} has no '=== user ===' line")
    return system.strip(), Template(user.strip())


def chat_text(system: str, user: str) -> str:
    """Qwen3.5's chat template for one system and one user message, generation prompt on, thinking off.
    tests/test_cleanup_model.py checks it against the Hugging Face template."""
    return (f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n")


def build_prompt(heard: str, words: list[str] = (), app: str = "", before: str = "",
                 alternatives: list[str] = (), prompt: tuple[str, Template] | None = None) -> str:
    """The text llama-server gets: the fields as the eval and the training data fill them (empty -> "(none)")."""
    system, user = prompt or load_prompt()
    fields = {"asr_best": heard,
              "alternatives": "; ".join(a for a in alternatives if a.strip()) or "(none)",
              "app": app.strip() or "(none)",
              "context": before.strip() or "(none)",
              "personal_words": ", ".join(w.strip() for w in words if w.strip()) or "(none)"}
    return chat_text(system, user.substitute(fields))


@dataclass
class Answer:
    text: str
    alternatives: list[str] = field(default_factory=list)
    uncertain: bool = False


def parse_answer(raw: str) -> Answer | None:
    """The model's JSON answer, or None if it isn't one (the caller falls back to the user's own words)."""
    body = raw.strip()
    m = _FENCE.match(body)
    if m:
        body = m.group(1)
    try:
        data = json.loads(body)
    except ValueError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("text"), str) or not data["text"].strip():
        return None
    alts = data.get("alternatives", [])
    alts = [a for a in alts if isinstance(a, str) and a.strip()] if isinstance(alts, list) else []
    return Answer(data["text"].strip(), alts, data.get("uncertain") is True)


# ------------------------------------------------------------------ the server
def free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _kill_with_engine(proc) -> object | None:
    """Windows: put the server in a job that closes with the engine, so it never outlives a crashed engine.
    Returns the job handle (keep it), or None if it can't be set up (the server still stops in stop())."""
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    class Basic(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class Io(ctypes.Structure):
        _fields_ = [(n, ctypes.c_uint64) for n in ("Read", "Write", "Other", "ReadBytes", "WriteBytes", "OtherBytes")]

    class Extended(ctypes.Structure):
        _fields_ = [("Basic", Basic), ("Io", Io), ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    job = k32.CreateJobObjectW(None, None)
    info = Extended()
    info.Basic.LimitFlags = 0x2000                           # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    ok = job and k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)) \
        and k32.AssignProcessToJobObject(job, wintypes.HANDLE(int(proc._handle)))
    return job if ok else None


class LlamaServer:
    """One llama-server child process. start() waits until it answers; complete() restarts it once if it died."""

    def __init__(self, server: Path | str, model: Path | str, ctx: int = 4096, log=print,
                 popen=subprocess.Popen, start_timeout_s: float = 120.0) -> None:
        import urllib.request

        self.server, self.model, self.ctx = Path(server), Path(model), ctx
        self._log, self._popen, self.start_timeout_s = log, popen, start_timeout_s
        self._http = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # never through a proxy
        self._proc = self._job = self.url = None
        self._key = ""
        self.lines: deque[str] = deque(maxlen=40)          # the server's last log lines, in memory only

    def start(self) -> None:
        for path, what in ((self.server, "llama-server"), (self.model, "clean-up model file")):
            if not path.exists():
                raise FileNotFoundError(f"{what} not found: {path}")
        port, self._key = free_port(), secrets.token_urlsafe(24)
        cmd = [str(self.server), "-m", str(self.model), "--host", "127.0.0.1", "--port", str(port),
               "--api-key", self._key, "-ngl", "99", "-c", str(self.ctx), "-np", "1",
               "--no-webui", "--no-slots", "--offline"]
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._proc = self._popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                 creationflags=flags)
        self._job = _kill_with_engine(self._proc)
        threading.Thread(target=self._read_log, name="llama-server-log", daemon=True).start()
        self.url = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + self.start_timeout_s
        while not self._ready():
            if self._proc.poll() is not None:
                raise RuntimeError(f"llama-server stopped (exit {self._proc.returncode}): {self.last_lines()}")
            if time.monotonic() > deadline:
                self.stop()
                raise TimeoutError(f"llama-server not ready after {self.start_timeout_s:.0f} s")
            time.sleep(0.2)

    def _read_log(self) -> None:
        stream = self._proc.stdout
        if stream is None:
            return
        for raw in stream:
            self.lines.append(raw.decode("utf-8", "replace").rstrip() if isinstance(raw, bytes) else raw.rstrip())

    def last_lines(self, n: int = 3) -> str:
        return " | ".join(list(self.lines)[-n:]) or "no output"

    def _ready(self) -> bool:
        try:
            with self._http.open(self.url + "/health", timeout=2) as r:
                return r.status == 200
        except OSError:                                    # refused, or 503 while the model loads
            return False

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def post(self, path: str, body: dict, timeout_s: float) -> dict:
        import urllib.request

        req = urllib.request.Request(self.url + path, data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {self._key}"})
        with self._http.open(req, timeout=timeout_s) as r:
            return json.loads(r.read().decode("utf-8"))

    def complete(self, prompt: str, n_predict: int, timeout_s: float, sampling: dict = GREEDY) -> dict:
        """POST /completion. A server that died is started again once before the request."""
        if not self.running:
            self._log(f"[cleanup] llama-server stopped ({self.last_lines(1)}); starting it again")
            self.start()
        return self.post("/completion", {"prompt": prompt, "n_predict": n_predict, "seed": 1234,
                                         "cache_prompt": True, "stream": False, **sampling}, timeout_s)

    def stop(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None


# ------------------------------------------------------------------ what the Rewriter uses
class CleanupModel:
    """The clean-up as the Rewriter calls it: clean(text) -> (Answer or None, cut_off)."""

    def __init__(self, server: LlamaServer, words: list[str] = ()) -> None:
        self.server, self.words = server, list(words)
        self._prompt = load_prompt()

    @classmethod
    def from_settings(cls, settings, log=print) -> "CleanupModel":
        def path(p: str) -> Path:
            p = Path(p)
            return p if p.is_absolute() else ROOT / p

        words = [w.strip() for w in str(getattr(settings, "rewrite_words", "")).split(",") if w.strip()]
        server = LlamaServer(path(settings.llama_server_path), path(settings.rewrite_gguf), log=log)
        return cls(server, words)

    def start(self) -> None:
        if not self.server.running:
            self.server.start()

    def clean(self, text: str, max_s: float, max_tokens: int | None = None, before: str = "",
              app: str = "") -> tuple[Answer | None, bool]:
        """(the parsed answer or None if it isn't valid JSON, cut_off: it hit the token cap). `before` is what was
        typed earlier in this dictation, `app` the kind of app in front: the prompt's "Before" and "App"."""
        n = max_tokens or min(900, 64 + 4 * len(text.split()))   # the JSON repeats the text, maybe twice
        r = self.server.complete(build_prompt(text, self.words, app=app, before=before, prompt=self._prompt), n, max_s)
        cut_off = r.get("stop_type") == "limit" or bool(r.get("truncated"))
        return parse_answer(r.get("content", "")), cut_off

    def stop(self) -> None:
        self.server.stop()
