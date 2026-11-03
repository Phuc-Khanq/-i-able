# Status

Last updated: 2026-11-03 12:25 (wrap-up). Session-by-session record: `docs/SESSION_LOG.md`.

## Now

- **Dictation clean-up: our own trained model.** full1, a LoRA adapter on Qwen3.5-4B (Apache 2.0) trained on Claude-written examples, is the candidate. Dev full pass 62/67 by Khang's rule, but dev is tuned on, so a fresh hidden test is next. In llama.cpp it answers in 0.33 s median (Q8_0), under the 1.5 s quick clean-up budget.
- **Rename:** a new name is coming (close to the new repo name `-i-able`, not final). The old-name scan is done; changes wait for the exact name.
- **Head tracking:** paused until Khoi's open-source tracker research is in.

## Done recently

- 2026-11-03: full1 runs in llama.cpp (MIT, build b11960), 5080, dev: Q8_0 p50/p90/p99 328/378/412 ms (about 5.1 GB), Q4_K_M 264/292/332 (about 3.3 GB), bf16 488/565/603. With greedy decoding it gives the same answers as transformers on all 67 cases. Full pass by Khang's rule 59 to 62 across runs (sampling luck). `engine/voice/cleanup_train/export.py` makes the GGUF files; the eval runs them with `--backend llamacpp`.
- 2026-11-03: first trained adapter full1: 1,088 examples, 32.7 min on the 5080 (peak 9.5 GB). Dev: automatic 64/67, silent guess 1/10, over-edit 0/10, 0 invented, human full pass 62/67 (93%); p50 1770 ms in transformers.
- 2026-11-02: training pipeline `engine/voice/cleanup_train/`: Claude-written sentences, speech noise (stutters, fillers, restarts, misspellings), a leak check against the test set, LoRA trainer (peft). Pilot: 300 examples in 9.4 min, full pass 59/67. Training works on Windows with the slow reference kernels; WSL2 wasn't needed.
- 2026-11-02: holdout, run once with prompt v4: fails the pass bar (full pass 22/33, silent guesses 4/5; today's app 7/33). The holdout is spent.
- 2026-11-02: prompts v2 to v4 on dev (human full pass v3 53/67, v4 59/67; today's app 17/67); blind review sheets and `--rescore`; plan with the pass bar, agreed by Khang. Khang reviews alone; Claude fills the reviews with his "meaning kept" rule.
- 2026-11-02: Qwen3.5-4B prompt-only on dev: v1 full prompt 63/67 automatic, p50 1864 ms (today's app 46/67, p50 435 ms).
- 2026-11-02: `/start` and `/wrap-up` commands; scoring runner in `eval/dictation_cleanup/`; old-name scan (169 lines in about 60 files, about 60 file names); private repo `Phuc-Khanq/-i-able` (remote `origin`; the old public repo is remote `public`); Claude Code setup; test set of 100 cases (33 held out).
- Earlier: hackathon build (head pointer, long-blink click, voice commands, dictation with preview and clean-up, click by name, website live demo, Windows installer, about 900 tests). See README.

## How the clean-up guard works today (`engine/voice/rewrite.py`)

- Quick clean-up (`check_rewrite`) rejects the model's text if: length is outside 0.4x to 1.3x; a number goes missing; the count of negations changes ("not", "never", "...n't"); the point of view changes (I / we / you); a name goes missing; a content word is swapped in place for one the original never had ("photos" -> "videos"); under 60% of content words are kept; or over 25% of content words are new. Rejected: `light_clean` (fillers and stutters only) if it passes the same guard, else the user's own words.
- Whole-passage refine (`check_refine`) is looser: length 0.5x to 1.4x, at least 50% of content words kept, at most 45% new. It may drop numbers only if they were marked unsure, never invents one. Plain "no" isn't counted as a negation ("no wait" is a self-correction). Names and words Whisper was unsure of may change.
- Hackathon figure (5080, Oct 24, 40 cases): the model's rewrite was used in 17 cases, 23 fell back (`rewrite.py` lines 18-19).
- Known: the docstring and fallback default in `rewrite.py` still name Qwen2.5-1.5B; `config.py` uses 2.5-3B. Fix it as part of the model switch.
- `testset.csv` starts with a BOM on purpose (for Excel). Read it with `encoding="utf-8-sig"`.

## In progress

- Nothing running. Waiting for the hidden test phrases.

## Next

1. Hidden test: Khang or Hung writes about 35 new phrases (`messy → meant`), spread over misheard words, broken sentences, changes of mind, two possible meanings and already-correct ones. Claude turns them into cases and runs full1 once, as it would ship (llama.cpp Q8_0, greedy, if Khang agrees).
2. More training data for the cases that fail in every run (M16, M25, S28, X01), then a new adapter.
3. App change (plan, section 4): llama-server behind the `Rewriter` interface (127.0.0.1, with an API key), the new guard, the card's alternatives; the refine first.
4. Rename, once the exact name is given.
5. Decide whether the app's guard should reject invented numbers (M01: "to o'clock" became "1 o'clock" and was accepted).
6. Light models for weak laptops (Qwen3.5-2B, 0.8B; Q4_K_M or llama.cpp's CPU build), timed on a low-end laptop (8 GB RAM, no NVIDIA GPU).
7. Licence scan of all Python packages.

## Blockers and risks

- Dev is tuned on and the holdout is spent: only the fresh hidden set gives a number to trust.
- The test cases were invented, not taken from real users. Real users or a speech-language pathologist should review them later.
- For a lawyer: training on Claude-written sentences (Anthropic's terms forbid it only for competing products; Khang's reading: we don't compete); shipping NVIDIA's CUDA runtime DLLs with llama.cpp; the licence for new private code; IP agreements with everyone who wrote code.
- Website updates: the live site is served from the public repo, and pushing `main` there would publish the private code. Pick a way to update it (e.g. a separate public repo holding only `eyeable-frontend/`). The Pages workflow will also fail in the private repo until it's switched off there.
- Remove the two people who left from the old public repo's collaborators (Khang, on GitHub).

## Uncommitted work on Khang's PC

- `engine/capture/` and `tests/test_capture.py`: unreviewed draft, left untracked. Decide: delete, or keep on a separate branch.
- Git-ignored, only on this PC: training data (`data/cleanup_train/`), adapters and GGUF files (`checkpoints/cleanup/`, about 24 GB for full1; `merged/` and the bf16 GGUF can be rebuilt with export.py), llama.cpp (`downloads/llama.cpp/b11960/`).
- `.git/info/exclude` hides every `*.md` except README/CONTRIBUTING, so new docs need `git add -f <file>`.

## Ideas (not scheduled)

- Show the raw transcript in the preview while the clean-up runs.
- An "Exact" clean-up level that skips the model completely.

## Earlier

(Move old "Done" lines here, one dated line each.)
