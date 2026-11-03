"""Clean-up backends for the evaluation, behind one small interface.

Every backend has: model_label, settings() (saved with the run), load(), warm_up(), clean(case, prompt)
-> Reply, sync() (wait for the GPU, so timing is honest) and peak_gpu_gb(). Adding Ollama or llama.cpp
later (for low-end laptop timing) is one more class here.

- CurrentBackend: today's clean-up exactly as the app runs it (engine/voice/rewrite.py: the model from
  engine/config.py, the guard and the light_clean fallback). The baseline. Plain-text answers.
- TransformersBackend: any model by name on the GPU, answering in JSON. For Qwen3.5: thinking off.
- FakeBackend: scripted answers, for tests (no GPU, no downloads).

Runs locally; nothing is sent anywhere (first-run model downloads aside).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

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
    NOT_APPLIED = {"presence_penalty": 1.5}

    def __init__(self, model: str, device: str = "cuda", seed: int = 1234, max_new_tokens: int = 256,
                 sampling: dict | None = None, adapter: str | None = None) -> None:
        self.model_id, self.device, self.seed, self.max_new_tokens = model, device, seed, max_new_tokens
        self.adapter = adapter                              # a LoRA folder (checkpoints/cleanup/<run>/adapter)
        self.sampling = dict(self.QWEN35_SAMPLING if sampling is None else sampling)
        self._tok = self._model = self._template = self._revision = None

    @property
    def model_label(self) -> str:
        if self.adapter:                                    # Qwen3.5-4B + adapter "pilot" -> "Qwen3.5-4B+pilot"
            from pathlib import Path

            return f"{self.model_id}+{Path(self.adapter).parent.name}"
        return self.model_id

    def settings(self) -> dict:
        return {"backend": self.name, "model": self.model_id,
                "revision": self._revision, "adapter": self.adapter,
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

            model = PeftModel.from_pretrained(model, self.adapter)
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
