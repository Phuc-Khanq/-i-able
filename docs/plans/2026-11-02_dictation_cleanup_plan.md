# Plan: the new dictation clean-up, from evaluation to the app

Status: **draft, waiting for Khang's OK.** Nothing in the app has changed. Written 2026-11-02 by Claude Code from measurements on Khang's PC (RTX 5080). All numbers are on the 67 **dev** cases unless marked otherwise; the 33 holdout cases have not been run.

## 1. Where we are

| Dev, 67 cases, automatic checks | App today (Qwen2.5-3B) | 4B v1 full | 4B v2 full | 4B v2, no alternatives | **4B v3 full** |
| --- | --- | --- | --- | --- | --- |
| Pass | 46/67 | 63/67 | 60/67 | 55/67 | **62/67** |
| mispronounced / broken / mixed | 13 / 19 / 3 | 19 / 19 / 5 | 18 / 20 / 6 | 18 / 20 / 5 | 18 / 20 / 6 |
| ambiguous / clean controls | 1 / 10 | 10 / 10 | 6 / 10 | 2 / 10 | 8 / 10 |
| Over-edit (clean controls) | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| Silent guess (ambiguous) | 8/10 | 0/10 | 4/10 | 8/10 | 2/10 |
| Possible invented facts | 1 | 0 | 0 | 1 | 0 |
| Latency p50 / p90 / p99 (ms) | 435 / 573 / 748 | 1864 / 2205 / 2546 | 2017 / 2304 / 2548 | 1989 / 2296 / 2470 | 2122 / 2589 / 2878 |

All "4B" runs are Qwen3.5-4B, transformers, bf16, thinking off, seed 1234, one warm-up first. Run folders are in `eval/dictation_cleanup/runs/` (git-ignored); all were rescored with the same checks.

- **v3 is the candidate.** v1 scored one more automatic pass, but it left broken sentences broken, kept repeating the misheard original as an "alternative", and in A05 put "I can't go on Friday" first when the person said "I can go on Friday". v2 and v3 write real sentences ("The meeting is not on Tuesday; it is on Wednesday."), apply self-corrections, and keep the heard reading first when unsure. v3 fixed most of v2's silent guesses. Tuning stopped at v3 so the prompt doesn't get fitted to the 67 dev cases.
- **What the alternatives are worth:** without them (v2, no alternatives) the model still fixed 18 of 20 misheard words from the sentence and context alone, but it resolved 8 of 10 ambiguous cases silently, and once it read a negation into a misspelling: "Cansel doctor appoinment tomorow" became "I can't cancel the doctor appointment tomorrow." The invented-fact check caught it; the app's guard must too.
- **Still failing in v3:** M25 writes "eyeable" instead of "[eye]able"; M16 and X05 offer the right reading only as the second choice (safe, but not fixed); A04 and A08 kept the heard words without flagging the other reading.
- **Holdout honesty note:** during the first look at `testset.csv` (before any tuning), a printout showed a few holdout rows, among them a number ambiguity (A03). So no number example was added to the prompts.
- The automatic checks can't see meaning. The human review (section 3) decides.
- Where Qwen3.5-4B's time goes (12 cases): reading the prompt about 134 ms, then about 63 ms per output token. A 4B model on this card should manage roughly 10 ms per token, so the time is lost in the slow reference code transformers uses because Qwen3.5's fast kernels aren't installed. In v1, a mean answer was 26.6 tokens, of which the cleaned text itself was only 7.8.

## 2. Speed: how to get under the 1.5 s budget

The app waits at most `rewrite_wait_s` = 1.5 s for the quick clean-up after "yes", and `refine_wait_s` = 6 s for the whole-passage refine (`engine/config.py`).

| Option | What it takes | Expected gain | Risk |
| --- | --- | --- | --- |
| A. Fast kernels for transformers: `flash-linear-attention` (MIT) and `causal-conv1d` (BSD-3-Clause) | Two new packages; fla needs Triton, which has no official Windows build; causal-conv1d is a CUDA build | Large per-token gain, if they install | Neither page mentions Windows. Likely needs WSL2 (the known fallback in STATUS.md) |
| B. **llama.cpp (MIT) with a 4-bit GGUF** (unsloth, Apache 2.0: 4B Q4_K_M 2.74 GB, 2B 1.28 GB, 0.8B 0.53 GB) | One more backend in the evaluation (the interface was built for this); llama.cpp's own quick-start uses Qwen3.5 GGUFs | Native CUDA and CPU kernels; about a third of the GPU memory of bf16; the same runtime also serves low-end laptops | A prebuilt Windows CUDA build wasn't confirmed from the page; output must still be checked for quality loss from 4-bit |
| C. Smaller models in transformers (2B 4.55 GB, 0.8B 1.75 GB download) | Downloads only | Small: the slowness is per token, not model size | Lower quality, little speed gain |
| D. Fewer output tokens | The app shows `text` as soon as it is complete (streaming) instead of waiting for the whole JSON | Not measured. In v1, the text was about 8 of 27 tokens and comes first, so it would show after about a third of the time | None |

**Recommendation: B, plus D.** It fixes speed on the 5080, cuts GPU memory (bf16 4B needs about 8.7 GB, and the `full` profile already peaked at 15.1 GB with Qwen2.5-3B), and is the path to weak laptops anyway. A stays the fallback if 4-bit quality drops.

## 3. Human review and the pass bar

- Blind sheet: `eval/dictation_cleanup/reviews/2026-11-02_dev.csv`, 134 rows: today's app and Qwen3.5-4B v3, labelled A and B at random per case. Khang and Khoi fill in `y`/`n` for "meaning kept" and "nothing invented", and flag bad test cases under `case_looks_wrong`. Then `review_sheet score`. Limit: only the new model can show "[or: …]" and "[marked unsure]", so a reviewer can sometimes guess which output is which.
- **Proposed pass bar** (for the holdout run; agree it before anyone runs the holdout). On the 33 holdout cases, human-scored:
  1. Invented facts: **0**.
  2. Clean controls changed: **0 of 5**.
  3. Ambiguous cases resolved silently: **0 of 5** (marking unsure counts as safe).
  4. Main reading that flips a negation or changes a number, name or date against the heard words without evidence: **0**.
  5. Broken answers: **0**.
  6. Full pass (all four checks of the README): **at least 27 of 33 (80%)**, and at least 15 points above today's app on the same cases.
  7. Speed on the 5080, warm: p90 **at most 1.5 s** to use it for the quick clean-up; at most 6 s to use it for the refine only.
- With 33 cases one case is 3 points. This is a go/no-go for building, not a number to publish.

## 4. App change (after the bar is met)

Each step is its own pull request, tested with `scripts\run_demo.ps1` on the reference machine.

1. **Runtime and config.** A new clean-up backend behind the `Rewriter` interface (llama.cpp server on 127.0.0.1, or transformers). `config.py` gets the backend, model and prompt version; defaults must work on the 5080, overrides in `config.local.json`. `setup_profile.py` gets new GPU-memory needs per profile.
2. **New guard** (`engine/voice/rewrite.py`). Replace "reject if the text differs a lot / a word was swapped" with the evaluation's checks: reject a number, name or negation that isn't in what was heard, the alternatives, the context or the word list; reject dropped content (length floor); if the main reading flips a negation or a number against the heard words, swap it with the heard reading and mark it unsure. Broken JSON falls back to `light_clean`. Port the tests from `tests/test_cleanup_eval.py`.
3. **Use it for the refine first.** Qwen3.5-4B v3 already fits the 6 s refine budget today (p99 2.9 s). The quick clean-up moves over once it fits 1.5 s. Qwen2.5-3B (research licence) must be gone before anything is sold.
4. **Inputs.** The word list: start from `whisper_hotwords`, later its own list in settings. Context: the last text typed in this dictation. App: the foreground window's name. Alternatives: Whisper runs greedy (`BEAM_SIZE = 1`), so there's only one transcript today; options are a second Whisper pass, or the word confidences the refine already marks as `(?)word(?)`. The "no alternatives" run in section 1 shows what we get without them.
5. **Contracts (needs Khang's OK first).** `topics.py`: `REWRITE_REQUEST` gains optional `context`, `app`, `words`, `alternatives`; `REWRITE_RESULT` gains `alternatives` and `uncertain`. `messages.py`: `DICTATION_PREVIEW` gains optional `alternatives` and `uncertain`; the website's `js/engine.js` changes in the same PR.
6. **The card.** When unsure: show the heard reading first and the others under it; choose hands-free by voice ("option two") or gesture; "Use original" stays one step away. Never auto-type an unsure reading that changes a negation or a number.

## 5. Open questions

1. Is 1.5 s the right budget, or may the quick clean-up wait longer?
2. Option B (llama.cpp) for speed: OK to download a Windows build and the 4-bit files (4B 2.74 GB)?
3. Where do alternatives come from (section 4.4)? This decides how much of the gain the app keeps.
4. The pass bar in section 3: agree or change.
5. Test case M25 (the product name) after the rename.
