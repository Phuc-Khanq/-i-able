# Status

Last updated: 2026-11-02 (first Claude Code session)

## Now

- **Dictation clean-up model switch.** Replace Qwen2.5-3B (research licence) with the Qwen3.5 family. First job: a script that runs a model over `eval/dictation_cleanup/testset.csv` and scores it, then a prompt-only baseline with Qwen3.5-4B (thinking off) next to the current Qwen2.5-3B clean-up.
- **Head tracking:** paused until Khoi's open-source tracker research is in.

## Done recently

- 2026-11-02: setup files and test set committed on branch `voice-cleanup-eval` (from `main`; not pushed). `origin/voice` no longer exists and local `voice` was 59 commits behind `main`.
- 2026-11-02: `eval/dictation_cleanup/` created: 100 original test cases (30 mispronounced, 30 broken structure, 10 mixed, 15 ambiguous, 15 clean controls; 33 held out) and a README with the scoring rules. Drafted by Claude, not yet reviewed by Khang or Khoi.
- 2026-11-02: Claude Code set up: `CLAUDE.md`, `.claude/rules/`, `.claude/settings.json`, this file, `docs/DECISIONS.md`.
- Earlier: hackathon build (head pointer, long-blink click, voice commands, dictation with preview and clean-up, click by name, website live demo, Windows installer, about 900 tests). See README.

## How the clean-up guard works today (`engine/voice/rewrite.py`)

- Quick clean-up (`check_rewrite`) rejects the model's text if: length is outside 0.4x to 1.3x; a number goes missing; the count of negations changes ("not", "never", "...n't"); the point of view changes (I / we / you); a name goes missing; a content word is swapped in place for one the original never had ("photos" -> "videos"); under 60% of content words are kept; or over 25% of content words are new. Rejected: `light_clean` (fillers and stutters only) if it passes the same guard, else the user's own words.
- Whole-passage refine (`check_refine`) is looser: length 0.5x to 1.4x, at least 50% of content words kept, at most 45% new. It may drop numbers only if they were marked unsure, never invents one. Plain "no" isn't counted as a negation ("no wait" is a self-correction). Names and words Whisper was unsure of may change.
- Hackathon figure (5080, Oct 24, 40 cases): the model's rewrite was used in 17 cases, 23 fell back (`rewrite.py` lines 18-19).
- Known: the docstring and fallback default in `rewrite.py` still name Qwen2.5-1.5B; `config.py` uses 2.5-3B. Fix it as part of the model switch.
- `testset.csv` starts with a BOM on purpose (for Excel). Read it with `encoding="utf-8-sig"`.

## In progress

- Plan for the scoring runner (waiting for Khang's OK).

## Next

1. Review the test set (Khang and Khoi): fix any case that doesn't sound like a real user or has the wrong intended meaning.
2. Build the scoring runner (`eval/dictation_cleanup/`): load a model by name from config, run every case, save outputs, score the automatic checks (must-keep items, unchanged controls), produce a sheet for the human checks (meaning kept, nothing invented).
3. Baseline: Qwen3.5-4B with thinking off vs the current Qwen2.5-3B clean-up, on dev cases. Measure latency p50/p90/p99.
4. Same set with Qwen3.5-2B and 0.8B.
5. Time the light sizes on a low-end laptop (8 GB RAM, no NVIDIA GPU) with the whole pipeline running.
6. Licence scan of all Python packages.

## Blockers and risks

- Qwen3.5 training uses custom kernels that may not work on Windows with the RTX 5080. Fallback: WSL2.
- The test cases were invented, not taken from real users. Real users or a speech-language pathologist should review them later.

## Uncommitted work on Khang's PC

- `engine/capture/` and `tests/test_capture.py`: unreviewed draft, left untracked. Decide: delete, or keep on a separate branch.
- `.git/info/exclude` hides every `*.md` except README/CONTRIBUTING, so new docs need `git add -f <file>`.

## Ideas (not scheduled)

- Show the raw transcript in the preview while the clean-up runs.
- An "Exact" clean-up level that skips the model completely.

## Earlier

(Move old "Done" lines here, one dated line each.)
