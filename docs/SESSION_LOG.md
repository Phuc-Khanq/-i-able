# Session log (newest first)

## 2026-11-02 18:31: Clean-up scoring runner built; baseline measured; old-name scan
Branch: voice-cleanup-eval · Commits: d1cbb5b eval: clean-up prompts v1, plain and full; 9c29b5e eval: scoring runner for the dictation clean-up test set; 41ccec7 eval: a name from the user's word list isn't an invented fact · Uncommitted: engine/capture/, tests/test_capture.py (draft, untracked on purpose)
**Done:**
- Scoring runner in `eval/dictation_cleanup/` (`run.py`, `backends.py`, `scoring.py`, prompts `cleanup_v1_plain.txt` / `cleanup_v1_full.txt`), built to Khang's spec. 34 tests; full suite 982 tests and compile check pass.
- Baseline: today's app clean-up (Qwen2.5-3B + guard) run on the 67 dev cases. Results in `eval/dictation_cleanup/runs/2026-11-02_Qwen2.5-3B-Instruct_app/` (git-ignored; `review.csv` is the sheet for Khang and Khoi).
- Scanned the code for the old name: 169 lines in about 60 files, plus about 60 file names. Grouped into display text, voice command, file names, bigger renames and things to leave alone. Nothing changed yet.
- Test set checked (not edited): no wrong cases found.
**Measured:** all on Khang's PC (RTX 5080), dev split, 67 cases, one warm-up first, Qwen2.5-3B as the app runs it:
- automatic-check pass 46/67 (69%); by category: mispronounced 13/20, broken 19/20, mixed 3/7, ambiguous 1/10, clean 10/10
- over-edit 0/10; silent guess on ambiguous cases 8/10 (it can't offer alternatives); possible invented facts in 1 case
- guard rejected the model's text in 21/67 cases; in 16 of those, the rejected text would have passed the automatic checks ("word swapped" alone rejected 13, mostly good fixes like windo -> window)
- latency per case p50 435 ms, p90 573 ms, p99 748 ms; mean 8 tokens per case; peak GPU memory 6.34 GB (PyTorch)
**Decisions:** Qwen3.5-4B download approved (Apache 2.0, 9.34 GB). The invented-fact check also accepts names from the user's word list (case A08's own answer needs it); Khang to confirm. Presence penalty from the Qwen3.5 card skipped (not in transformers). All in DECISIONS.md.
**Still running:** Qwen3.5-4B download into the Hugging Face cache (`%USERPROFILE%\.cache\huggingface\hub`), 0.7 of 9.34 GB at 18:31.
**Problems / blockers:** none failing. Finding: the app's guard misses invented numbers. M01 "Thursday at to o'clock" became "Thursday at 1 o'clock" and was accepted, because `check_rewrite` only checks for numbers that go missing.
**Next:**
1. Run Qwen3.5-4B on dev with the plain prompt, then the full prompt.
2. Put the three summaries side by side and pick the 5 most telling failures.
3. Khang and Khoi fill in the review sheets.
**Khang must do by hand:** give the exact new name (display form, file-name form, how it's said) so the rename can start; confirm the word-list change to the invented-fact check; review `review.csv`; approve pushing `voice-cleanup-eval`. Still open from earlier: remove the two people who left from `-eye-able`; check Khoi's and Hung's invites to `-i-able`.
**For Cowork:** the new name (trademark and domain checks before it's final); should the app's guard reject invented numbers (M01)?; how the website gets updated now the code is private; the licence for new code and IP agreements (a lawyer's question).
**Notes from Khang:** none

## 2026-11-02 18:15: First Claude Code session: setup, private repo, wrap-up command, scoring runner started
Branch: voice-cleanup-eval · Commits: 0a26717 docs: Claude Code setup; 0116961 eval: dictation clean-up test set; 7b4e979 docs: development moves to a private repo; 45a58c0 docs: no Claude attribution in commits or PRs; 10b545c docs: /wrap-up command for the end of a session · Uncommitted: engine/capture/, tests/test_capture.py (unreviewed draft, left untracked on purpose)
**Done:**
- Moved the Claude Code settings and rules into `.claude/`. Read the setup docs and compared them with the code (mismatches are listed in STATUS.md, "How the clean-up guard works today").
- Committed the setup files and the 100-case test set on a new branch `voice-cleanup-eval`, made from `main`.
- Moved development to the private repo `Phuc-Khanq/-i-able` (remote `origin`): all 9 local branches and 5 tags pushed. The old public repo is now remote `public`.
- CLAUDE.md: commits carry only the developer's name (no Claude co-author line). New `/wrap-up` command and this log.
- Scoring runner: Khang's spec compared with Claude's plan; no big differences, build started (no code yet).
**Measured:** none
**Decisions:** private repo on Khang's account (new work shouldn't be public); no Claude attribution in commits (every commit under the developer's name); work branch `voice-cleanup-eval` (origin/voice was gone, local voice 59 commits behind main); exact pins only for new packages; testset.csv BOM is intentional (Excel); fix the 1.5B docstring in rewrite.py during the model switch; sessions end with /wrap-up; scoring runner spec as Khang wrote it. All in DECISIONS.md.
**Still running:** none
**Problems / blockers:** none blocking. The Pages workflow will fail in the private repo until it's switched off there. How the live website gets updated is still open (pushing `main` to `public` would publish private code).
**Next:**
1. Build the scoring runner in `eval/dictation_cleanup/`, tests first, then run the full test suite and the compile check.
2. Run the "current" clean-up (Qwen2.5-3B) on the dev split.
3. Download Qwen3.5-4B (after Khang's OK) and run it with the plain and full prompts.
**Khang must do by hand:** review the test set with Khoi; remove the two people who left from `-eye-able`'s collaborators; check that Khoi and Hung accepted the invites to `-i-able` and ran `git remote set-url origin https://github.com/Phuc-Khanq/-i-able.git`; approve the Qwen3.5-4B download; approve the next push.
**For Cowork:** how the website should be updated now that the code is private (e.g. a separate public repo with only `eyeable-frontend/`); the licence for new private code and IP agreements with everyone who wrote code (a lawyer's question); Silero VAD licence (MIT vs a CC BY-NC badge); a licence scan of all Python packages.
**Notes from Khang:** none
