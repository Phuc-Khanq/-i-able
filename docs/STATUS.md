# Status

Last updated: 2026-11-02 18:31 (wrap-up). Session-by-session record: `docs/SESSION_LOG.md`.

## Now

- **Dictation clean-up model switch.** Replace Qwen2.5-3B (research licence) with the Qwen3.5 family. Prompt-only Qwen3.5-4B is measured: with the full prompt it beats today's clean-up on the automatic checks, but it is about 4x slower. Human review of the outputs is next.
- **Rename:** a new name is coming (close to the new repo name `-i-able`, not final). The old-name scan is done; changes wait for the exact name.
- **Head tracking:** paused until Khoi's open-source tracker research is in.

## Done recently

- 2026-11-02: Qwen3.5-4B (thinking off, prompt only) on dev, 5080, automatic checks. Plain prompt: 53/67 pass, silent guess 8/10, p50/p90/p99 1710/1976/2168 ms. Full prompt: 63/67 pass, ambiguous 10/10, silent guess 0/10, over-edit 0/10, 0 invented, 0 broken JSON, p50/p90/p99 1864/2205/2546 ms, 8.74 GB peak GPU. Baseline for comparison: 46/67, p50 435 ms. Speed caveat: Qwen3.5's fast kernels (`flash-linear-attention`, `causal-conv1d`) aren't installed, so transformers used its slower reference code.
- 2026-11-02: `/start` command (`.claude/skills/start/`): one session per job, started with `/start <task>`.

- 2026-11-02: scoring runner in `eval/dictation_cleanup/` (`python -m eval.dictation_cleanup.run --help`), 34 tests. Baseline measured for today's clean-up (Qwen2.5-3B, dev, 5080): 46/67 automatic-check pass, over-edit 0/10, silent guess 8/10, guard rejected 21/67 (16 of those would have passed), p50/p90/p99 435/573/748 ms, 6.34 GB peak GPU. Run folder: `eval/dictation_cleanup/runs/2026-11-02_Qwen2.5-3B-Instruct_app/`.
- 2026-11-02: old-name scan: 169 lines in about 60 files, plus about 60 file names (see SESSION_LOG.md, 18:31).

- 2026-11-02: `/wrap-up` command (`.claude/skills/wrap-up/`) and `docs/SESSION_LOG.md`. Commits carry only the developer's name (CLAUDE.md, Git workflow).
- 2026-11-02: moved to the private repo `Phuc-Khanq/-i-able` (remote `origin`): all 9 local branches and 5 tags pushed. The old public repo is now remote `public`. See DECISIONS.md.
- 2026-11-02: setup files and test set committed on branch `voice-cleanup-eval` (from `main`). Local `voice` was 59 commits behind `main`, so it was left as it was.
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

- Nothing running. Three run folders in `eval/dictation_cleanup/runs/` wait for human review.

## Next

1. Khang and Khoi fill in the review sheets (`review.csv` in each run folder), starting with the Qwen3.5-4B full run, and review the test set. The automatic checks can't see meaning.
2. Decide how the card shows the model's `alternatives`. In A05 the full prompt put "I can't go on Friday" first and the user's own words second, with `uncertain` false: the main reading flipped a negation.
3. Speed: check the licences of `flash-linear-attention` and `causal-conv1d` and whether they install on Windows; then re-time. Then Qwen3.5-2B and 0.8B with the full prompt.
3. Rename, once the exact name is given: display text first, then the voice command and file names, then the bigger renames (website folder, repo URL), each its own commit.
4. Decide whether the app's guard should reject invented numbers (M01: "to o'clock" became "1 o'clock" and was accepted).
5. Same set with Qwen3.5-2B and 0.8B; then time the light sizes on a low-end laptop (8 GB RAM, no NVIDIA GPU).
6. Licence scan of all Python packages.

## Blockers and risks

- Qwen3.5 training uses custom kernels that may not work on Windows with the RTX 5080. Fallback: WSL2.
- The test cases were invented, not taken from real users. Real users or a speech-language pathologist should review them later.
- Website updates: the live site is served from the public repo, and pushing `main` there would publish the private code. Pick a way to update it (e.g. a separate public repo holding only `eyeable-frontend/`). The Pages workflow will also fail in the private repo until it's switched off there.
- Open for a lawyer: the licence for new private code, and IP agreements with everyone who wrote code.
- Remove the two people who left from the old public repo's collaborators (Khang, on GitHub).

## Uncommitted work on Khang's PC

- `engine/capture/` and `tests/test_capture.py`: unreviewed draft, left untracked. Decide: delete, or keep on a separate branch.
- `.git/info/exclude` hides every `*.md` except README/CONTRIBUTING, so new docs need `git add -f <file>`.

## Ideas (not scheduled)

- Show the raw transcript in the preview while the clean-up runs.
- An "Exact" clean-up level that skips the model completely.

## Earlier

(Move old "Done" lines here, one dated line each.)
