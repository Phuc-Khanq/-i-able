---
paths:
  - "engine/voice/**"
  - "eval/**"
  - "tests/test_rewrite*.py"
  - "tests/test_rewriter.py"
  - "tests/test_refine.py"
  - "tests/test_dictation*.py"
  - "tests/test_live_preview.py"
---

# Voice, dictation and evaluation

## The pipeline today

mic (`audio_in.py`, Silero VAD) → faster-whisper large-v3-turbo on the GPU (`dictation.py`, word confidences, hotwords) → mode manager splits commands from dictation (`engine/core/mode_manager.py`) → dictation card preview (`engine/eyes/dictation_card.py`) → clean-up (`rewrite.py`: Qwen2.5-3B with a guard, a rule-based `light_clean` fallback, and a whole-passage refine) → the user confirms (eyebrow raise, nod or "yes") or drops it (head shake or "no") → typed. The mouth gate keeps only words said while the user's own mouth moved. A CPU "lite" path uses Whisper small.

Measured at the hackathon (README "Results"): the TORGO LoRA fine-tune of Whisper cut word errors on 801 dysarthric clips from 2 unseen speakers from 41.4% to 19.4%, with typical speech unchanged (LibriSpeech 1.98% to 2.01%). Only 5 sentences had unseen wording, so no claim that it generalises.

## Where the clean-up is going

- **Goal:** the clean-up must interpret, not just tidy. Users may not form full sentences or may mispronounce words. Recover what they meant without inventing anything.
- **Model:** move from Qwen2.5-3B (research licence, not sellable) to the **Qwen3.5 family** (Apache 2.0). Test **Qwen3.5-4B** first. The light ladder for weaker laptops is Qwen3.5-2B, then Qwen3.5-0.8B, then Qwen3-1.7B. Qwen3.5-9B is a stretch goal only. Stay in the Qwen family.
- **Thinking must be off** (`enable_thinking=False`). Qwen3.5 4B and 9B think by default, which is far too slow for dictation. 2B is off by default.
- **Behind one interface.** The model name and its prompt template come from config, so swapping models is a config change, not a rewrite.
- **Inputs to use** (planned): the top few Whisper candidates and word confidences, the last text typed, the app, and the user's personal word list.
- **Safety stays:** the preview and confirm step never goes away; names, numbers, dates and negations are never changed without evidence; when two readings are plausible, show both instead of guessing.

## The test set (`eval/dictation_cleanup/`)

- `testset.csv` has 100 original cases: mispronounced, broken structure, mixed, ambiguous, clean controls. `README.md` has the columns and the scoring rules.
- **Never tune prompts or train on `split == holdout` rows.** Use `dev` rows for that; run holdout only to report.
- **Never change a case's `intended` or `expected_behavior` to make a model pass.** Fixing a wrong case needs Khang's OK and a DECISIONS.md line.
- Report per model and per category: pass rate, invented-fact rate, over-edit rate on clean controls, silent-guess rate on ambiguous cases, and latency p50/p90/p99 on the reference machine.
- The older `tests/rewrite_cases.csv` belongs to the existing guard tests; leave it as is.

## Fine-tuning notes

- Prompt-only baseline first. Fine-tune only the gap the test set shows.
- Unsloth's Qwen3.5 guide lists bf16 LoRA memory as about 5 GB (2B), 10 GB (4B), 22 GB (9B), and advises against QLoRA for Qwen3.5. So 0.8B, 2B and 4B can train on the 16 GB card; 9B would need a rented GPU.
- Qwen3.5 needs transformers 5 (the lock file has 5.20.2). Its training uses custom kernels; if they fail on Windows with the RTX 5080, try WSL2 before anything else and tell Khang.
- Training data must be ours or commercially licensed. If another AI model generates training pairs, check that provider's terms first.
- TORGO adapters and data never leave the machine (CLAUDE.md hard rule 3).
