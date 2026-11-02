"""Clean-up prompts, kept as versioned text files in prompts/ (never inline in code).

A prompt file is the system message, a line "=== user ===", then the user message as a template:
$asr_best, $alternatives, $app, $context, $personal_words. Empty fields become "(none)".
Change a prompt by adding a new version (cleanup_v2_full.txt), so old runs stay comparable.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from string import Template

PROMPTS = Path(__file__).resolve().parent / "prompts"
SPLIT_LINE = "=== user ==="


@dataclass
class Prompt:
    name: str
    system: str
    user: Template
    sha256: str

    def fields(self, case: dict) -> dict:
        alts = [a for a in (case.get("asr_alt1", ""), case.get("asr_alt2", "")) if a.strip()]
        words = [w.strip() for w in case.get("personal_words", "").split("|") if w.strip()]
        return {"asr_best": case["asr_best"],
                "alternatives": "; ".join(alts) or "(none)",
                "app": case.get("app", "").strip() or "(none)",
                "context": case.get("context", "").strip() or "(none)",
                "personal_words": ", ".join(words) or "(none)"}

    def messages(self, case: dict) -> list[dict]:
        return [{"role": "system", "content": self.system},
                {"role": "user", "content": self.user.substitute(self.fields(case))}]


def load_prompt(name: str) -> Prompt:
    """"v1_full" or "cleanup_v1_full" -> prompts/cleanup_v1_full.txt."""
    stem = name if name.startswith("cleanup_") else f"cleanup_{name}"
    path = PROMPTS / f"{stem}.txt"
    if not path.exists():
        raise FileNotFoundError(f"no prompt {path.name} in {PROMPTS}")
    raw = path.read_text(encoding="utf-8")
    system, sep, user = raw.partition(f"\n{SPLIT_LINE}\n")
    if not sep:
        raise ValueError(f"{path.name} has no '{SPLIT_LINE}' line")
    return Prompt(stem, system.strip(), Template(user.strip()), hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12])
