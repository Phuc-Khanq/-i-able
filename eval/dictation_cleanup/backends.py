"""Clean-up backends for the evaluation, behind one small interface.

Every backend has: model_label, settings() (saved with the run), load(), warm_up(), clean(case, prompt)
-> Reply, sync() (wait for the GPU, so timing is honest) and peak_gpu_gb(). Adding Ollama or llama.cpp
later (for low-end laptop timing) is one more class here.

- CurrentBackend: today's clean-up exactly as the app runs it (engine/voice/rewrite.py: the model from
  engine/config.py, the guard and the light_clean fallback). The baseline. Plain-text answers.
- TransformersBackend: any model by name on the GPU, answering in JSON. For Qwen3.5: thinking off.
- LlamaCppBackend: a GGUF file (engine/voice/cleanup_train/export.py) in llama.cpp's server on 127.0.0.1.
- FakeBackend: scripted answers, for tests (no GPU, no downloads).

Runs locally; nothing is sent anywhere (first-run model downloads aside).
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from eval.dictation_cleanup.scoring import Answer, parse_answer


@dataclass
class Reply:
    raw: str                      # the model's own text, before any guard or parsing
    answer: Answer
    tokens: int | None = None     # tokens generated
    extra: dict = field(default_factory=dict)


def _peak_gpu_gb() -> float | None:
    import sys

    torch = sys.modules.get("torch")                 # only if a backend already imported it
    if torch is None or not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated() / 1e9, 2)


def _reset_peak() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _sync() -> None:
    import sys

    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


def _gpu_used_mib() -> int | None:
    """Whole-card memory in use (nvidia-smi). Windows doesn't report it per process."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return int(out.split()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def _chat_template(model_id: str):
    """(tokenizer, the object that holds the chat template): Qwen3.5 keeps the template in its processor."""
    from transformers import AutoTokenizer

    from engine.core import hf

    with hf.LOAD_LOCK:
        tok = AutoTokenizer.from_pretrained(model_id)
    if getattr(tok, "chat_template", None) is not None:
        return tok, tok
    from transformers import AutoProcessor

    return tok, AutoProcessor.from_pretrained(model_id)


# ------------------------------------------------------------------ today's clean-up
class CurrentBackend:
    """engine.voice.rewrite.Rewriter.rewrite_full, as the app calls it. Tests pass a fake `rewriter`."""

    name = "current"

    def __init__(self, model: str | None = None, device: str | None = None, rewriter=None) -> None:
        self._model, self._device, self._rw = model, device, rewriter

    def rewriter(self):
        if self._rw is None:
            import dataclasses

            from engine.config import settings
            from engine.voice.rewrite import Rewriter

            over = {k: v for k, v in (("rewrite_model", self._model), ("rewrite_device", self._device)) if v}
            self._rw = Rewriter(settings=dataclasses.replace(settings, **over) if over else settings)
        return self._rw

    @property
    def model_label(self) -> str:
        return self.rewriter().model_id

    def settings(self) -> dict:
        rw = self.rewriter()
        return {"backend": self.name, "model": rw.model_id, "device": getattr(rw, "device", None),
                "time_cap_s": getattr(rw, "max_s", None), "decoding": "greedy, as in the app",
                "prompt": "the app's own (rewrite.SYSTEM_PROMPT and EXAMPLES)",
                "guard": "check_rewrite, then light_clean, then the user's own words"}

    def load(self) -> None:
        if not hasattr(self.rewriter(), "_load"):        # a fake
            return
        _reset_peak()
        self.rewriter().load()

    def warm_up(self) -> None:
        self.rewriter().warm_up()

    def clean(self, case: dict, prompt=None) -> Reply:
        r = self.rewriter().rewrite_full(case["asr_best"])
        path = "model" if r.used and r.reason is None else "light_clean" if r.used else "original"
        tok = getattr(self.rewriter(), "_tokenizer", None)
        tokens = len(tok(r.output, add_special_tokens=False)["input_ids"]) if tok is not None and r.output else None
        answer = Answer(r.clean, [], False, None, r.output)
        return Reply(r.output, answer, tokens, {"path": path, "guard_reason": r.reason})

    def sync(self) -> None:
        _sync()

    def peak_gpu_gb(self) -> float | None:
        return _peak_gpu_gb()


# ------------------------------------------------------------------ any model on the GPU
class TransformersBackend:
    """A chat model loaded by name with transformers, text only. Answers in JSON (scoring.parse_answer).

    Qwen3.5 is listed as a vision-language model; AutoModelForCausalLM maps it to Qwen3_5ForCausalLM,
    the text part only, so the vision encoder isn't loaded. Thinking is off (enable_thinking=False).
    """

    name = "transformers"
    # Qwen3.5 model card, non-thinking mode, general tasks (checked 2026-11-02). The card also says
    # presence_penalty=1.5, but transformers' generate() has no presence penalty: not applied, and
    # settings.json says so.
    QWEN35_SAMPLING = {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
                       "repetition_penalty": 1.0}
    GREEDY = {"do_sample": False}                           # always the most likely token
    NOT_APPLIED = {"presence_penalty": 1.5}

    def __init__(self, model: str, device: str = "cuda", seed: int = 1234, max_new_tokens: int = 256,
                 sampling: dict | None = None, adapter: str | None = None) -> None:
        self.model_id, self.device, self.seed, self.max_new_tokens = model, device, seed, max_new_tokens
        self.adapter = adapter                              # a LoRA folder (checkpoints/cleanup/<run>/adapter)
        self.sampling = dict(self.QWEN35_SAMPLING if sampling is None else sampling)
        self._tok = self._model = self._template = self._revision = None

    @property
    def model_label(self) -> str:
        if self.adapter:            # .../pilot/adapter -> "Qwen3.5-4B+pilot"; .../full1/epoch1 -> "...+full1-epoch1"
            from pathlib import Path

            folder = Path(self.adapter)
            run = folder.parent.name + ("" if folder.name == "adapter" else f"-{folder.name}")
            return f"{self.model_id}+{run}"
        return self.model_id

    def settings(self) -> dict:
        return {"backend": self.name, "model": self.model_id,
                "revision": self._revision, "adapter": self.adapter, "adapter_merged": bool(self.adapter),
                "device": self.device, "dtype": "bfloat16", "seed": self.seed, "seed_rule": "set before every case",
                "max_new_tokens": self.max_new_tokens, "sampling": self.sampling, "enable_thinking": False,
                "card_settings_not_applied": self.NOT_APPLIED,
                "model_class": type(self._model).__name__ if self._model is not None else None}

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from engine.core import hf

        _reset_peak()
        with hf.LOAD_LOCK:
            tok = AutoTokenizer.from_pretrained(self.model_id)
            model, info = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=torch.bfloat16,
                                                               output_loading_info=True)
        missing = info.get("missing_keys") or []
        if missing:                                         # the text weights didn't map: fail loudly
            raise RuntimeError(f"{len(missing)} weights missing when loading {self.model_id} as text only, "
                               f"e.g. {sorted(missing)[:3]}")
        template = tok
        if getattr(tok, "chat_template", None) is None:     # some vision-language repos keep it in the processor
            from transformers import AutoProcessor

            template = AutoProcessor.from_pretrained(self.model_id)
        if self.adapter:
            from peft import PeftModel

            # Merged into the weights, as the app would ship it: same answers, no adapter cost per token.
            model = PeftModel.from_pretrained(model, self.adapter).merge_and_unload()
        model.to(self.device)
        model.eval()
        self._tok, self._model, self._template = tok, model, template
        self._revision = self._cached_revision()

    def _cached_revision(self) -> str | None:
        """The snapshot the weights came from (…/snapshots/<commit>/config.json in the local cache)."""
        from pathlib import Path

        from huggingface_hub import try_to_load_from_cache

        path = try_to_load_from_cache(self.model_id, "config.json")
        return Path(path).parent.name if isinstance(path, str) else None

    def _generate(self, messages: list[dict]) -> tuple[str, int, bool]:
        import torch
        from transformers import set_seed

        tok, model = self._tok, self._model
        text = self._template.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                                  enable_thinking=False)
        enc = tok(text, return_tensors="pt", add_special_tokens=False).to(self.device)
        n_in = enc["input_ids"].shape[1]
        set_seed(self.seed)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=self.max_new_tokens,
                                 pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
                                 **self.sampling)
        new = out[0, n_in:].tolist()
        eos = getattr(model.generation_config, "eos_token_id", None)
        eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos]) | {tok.eos_token_id}
        eos_ids.discard(None)
        cut_off = not any(t in eos_ids for t in new)
        return tok.decode(new, skip_special_tokens=True), len(new), cut_off

    def warm_up(self) -> None:
        self._generate([{"role": "system", "content": "Answer in JSON."},
                        {"role": "user", "content": "um so this is uh a test"}])

    def clean(self, case: dict, prompt) -> Reply:
        raw, tokens, cut_off = self._generate(prompt.messages(case))
        return Reply(raw, parse_answer(raw), tokens, {"cut_off": cut_off})

    def sync(self) -> None:
        _sync()

    def peak_gpu_gb(self) -> float | None:
        return _peak_gpu_gb()


# ------------------------------------------------------------------ a GGUF file in llama.cpp
class LlamaCppBackend:
    """A GGUF model in llama.cpp's llama-server, on 127.0.0.1 only and offline, answering in JSON.

    The prompt is the Hugging Face chat template (thinking off) turned into token ids by the Hugging Face
    tokenizer, so llama.cpp gets exactly the tokens the training and the transformers backend used. Same
    Qwen3.5 sampling, in transformers' order (temperature, top-k, top-p, min-p); llama.cpp's sampler isn't
    bit-identical to transformers', so the wording can differ at the same seed. A case's time is the whole
    request, as an app calling the server would see it. The server starts in load() and stops in close().
    """

    name = "llamacpp"
    SAMPLING = {"temperature": 0.7, "top_k": 20, "top_p": 0.8, "min_p": 0.0, "repeat_penalty": 1.0,
                "samplers": ["temperature", "top_k", "top_p", "min_p"]}
    GREEDY = {"top_k": 1, "repeat_penalty": 1.0, "samplers": ["top_k"]}          # only the most likely token
    NOT_APPLIED = {"presence_penalty": 1.5}               # as in TransformersBackend, so the runs compare

    def __init__(self, gguf: str, server: str, tokenizer: str = "Qwen/Qwen3.5-4B", seed: int = 1234,
                 max_new_tokens: int = 256, ctx: int = 4096, sampling: dict | None = None) -> None:
        self.gguf, self.server, self.tokenizer_id = gguf, server, tokenizer
        self.seed, self.max_new_tokens, self.ctx = seed, max_new_tokens, ctx
        self.sampling = dict(self.SAMPLING if sampling is None else sampling)
        self._tok = self._template = self._proc = self._url = self._log = None
        self._version = self._gpu_before = None
        import urllib.request

        self._http = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # never through a proxy

    @property
    def model_label(self) -> str:
        return f"llama.cpp/{Path(self.gguf).stem}"           # .../gguf/full1-q8_0.gguf -> llama.cpp/full1-q8_0

    def settings(self) -> dict:
        size = Path(self.gguf).stat().st_size if Path(self.gguf).exists() else None
        return {"backend": self.name, "model": self.model_label, "gguf": self.gguf,
                "gguf_gb": round(size / 1e9, 2) if size else None, "server": self.server,
                "server_version": self._version, "tokenizer": self.tokenizer_id, "ctx": self.ctx,
                "gpu_layers": "all (-ngl 99)", "seed": self.seed, "seed_rule": "sent with every case",
                "max_new_tokens": self.max_new_tokens, "sampling": self.sampling, "enable_thinking": False,
                "prompt_sent_as": "token ids from the Hugging Face tokenizer",
                "card_settings_not_applied": self.NOT_APPLIED,
                "peak_gpu_gb_measure": "nvidia-smi, whole card: in use at the end of the run minus before the server"}

    def load(self) -> None:
        import atexit
        import socket

        self._tok, self._template = _chat_template(self.tokenizer_id)
        v = subprocess.run([self.server, "--version"], capture_output=True, text=True, timeout=60)
        self._version = next((line for line in (v.stdout + v.stderr).splitlines() if line.startswith("version")), None)
        self._gpu_before = _gpu_used_mib()
        with socket.socket() as s:                          # a free port on this machine only
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        log = Path(self.gguf).with_suffix(".server.log")
        self._log = log.open("wb")
        self._proc = subprocess.Popen([self.server, "-m", self.gguf, "--host", "127.0.0.1", "--port", str(port),
                                       "-ngl", "99", "-c", str(self.ctx), "-np", "1", "--no-webui", "--offline"],
                                      stdout=self._log, stderr=subprocess.STDOUT)
        atexit.register(self.close)
        self._url = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + 300
        while not self._ready():
            if self._proc.poll() is not None:
                raise RuntimeError(f"llama-server stopped (exit {self._proc.returncode}); its log: {log}")
            if time.monotonic() > deadline:
                raise TimeoutError(f"llama-server not ready after 300 s; its log: {log}")
            time.sleep(0.25)

    def _ready(self) -> bool:
        try:
            with self._http.open(self._url + "/health", timeout=2) as r:
                return r.status == 200
        except OSError:                                     # refused, or 503 while the model loads
            return False

    def _post(self, path: str, body: dict) -> dict:
        import urllib.request

        req = urllib.request.Request(self._url + path, data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        with self._http.open(req, timeout=120) as r:
            return json.loads(r.read().decode("utf-8"))

    def _generate(self, messages: list[dict]) -> tuple[str, int | None, bool, dict]:
        text = self._template.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                                  enable_thinking=False)
        ids = self._tok(text, add_special_tokens=False)["input_ids"]
        r = self._post("/completion", {"prompt": ids, "n_predict": self.max_new_tokens, "seed": self.seed,
                                       "cache_prompt": True, "stream": False, **self.sampling})
        cut_off = r.get("stop_type") == "limit" or bool(r.get("truncated"))
        return r["content"], r.get("tokens_predicted"), cut_off, r.get("timings") or {}

    def warm_up(self) -> None:
        self._generate([{"role": "system", "content": "Answer in JSON."},
                        {"role": "user", "content": "um so this is uh a test"}])

    def clean(self, case: dict, prompt) -> Reply:
        raw, tokens, cut_off, t = self._generate(prompt.messages(case))
        server_ms = round(t["prompt_ms"] + t["predicted_ms"], 1) if "prompt_ms" in t and "predicted_ms" in t else None
        return Reply(raw, parse_answer(raw), tokens, {"cut_off": cut_off, "server_ms": server_ms})

    def sync(self) -> None:
        pass                                                # the answer is back: nothing still running

    def peak_gpu_gb(self) -> float | None:
        now = _gpu_used_mib()
        if now is None or self._gpu_before is None:
            return None
        return round((now - self._gpu_before) * 2**20 / 1e9, 2)

    def close(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None
        if self._log is not None:
            self._log.close()
            self._log = None


# ------------------------------------------------------------------ tests
class FakeBackend:
    """Answers from a script ({case id: raw text}); any other case comes back unchanged, as JSON."""

    name = "fake"
    model_label = "fake"

    def __init__(self, replies: dict[str, str] | None = None) -> None:
        self.replies = replies or {}
        self.prompts_seen: list[list[dict]] = []

    def settings(self) -> dict:
        return {"backend": self.name, "scripted_cases": sorted(self.replies)}

    def load(self) -> None:
        pass

    def warm_up(self) -> None:
        pass

    def clean(self, case: dict, prompt) -> Reply:
        if prompt is not None:
            self.prompts_seen.append(prompt.messages(case))
        raw = self.replies.get(case["id"]) or json.dumps({"text": case["asr_best"], "alternatives": [],
                                                          "uncertain": False})
        return Reply(raw, parse_answer(raw), len(raw.split()))

    def sync(self) -> None:
        pass

    def peak_gpu_gb(self) -> float | None:
        return None
