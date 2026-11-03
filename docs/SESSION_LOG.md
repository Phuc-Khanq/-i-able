# Session log (newest first)

## 2026-11-03 12:25: Clean-up: prompt v4 and holdout, our own trained adapter (full1), llama.cpp speed
Branch: voice-cleanup-eval · Commits (all pushed to origin): 1236445 docs: /start command for the beginning of a session; a78ca0b eval: record the model's cached revision in settings.json; ed7d62c docs: Qwen3.5-4B baseline results in STATUS; c33731e eval: clean-up prompt v2, full and without alternatives; 611e3d6 eval: blind review sheets for the human checks; b4c3ce0 eval: titles aren't names; --rescore re-runs the checks on saved answers; 781f0d6 eval: clean-up prompt v3; 5c44f4b eval: blind review sheet for dev, today's app vs Qwen3.5-4B v3; 153bf52 docs: plan for the new dictation clean-up, proposed pass bar; 371374e docs: pass bar for the clean-up holdout, agreed by Khang; 70053bd docs: Khang reviews the clean-up outputs alone for now; 9d0b9a9 eval: dev review filled in and scored; 2eea154 eval: prompt v4 (complete, polite sentences), dev run and review; eb97393 eval: holdout run, Qwen3.5-4B v4 vs today's app, does not meet the bar; 465c877 voice: clean-up training data (seeds, noise, leak check) and LoRA trainer; e9a7876 voice: more clean-up training data, template cases, per-epoch adapters, merged adapter in eval; be33365 eval: first trained clean-up adapter (pilot and full1) on dev, reviewed; 27ace4b voice: export a trained clean-up adapter to GGUF for llama.cpp; b0e9657 eval: llama.cpp backend for the clean-up runner; dda5faf eval: full1 in llama.cpp on dev, about 5x faster, same answers with greedy · Uncommitted: engine/capture/, tests/test_capture.py (draft, untracked on purpose)
**Done:**
- Prompt-only Qwen3.5-4B pushed to its limit: prompts v2 to v4, blind review sheets, the pass bar (agreed by Khang). Khang reviews alone; Claude fills the reviews with his "meaning kept" rule.
- Holdout run once with v4: no-go against the bar. The holdout is spent.
- Training pipeline `engine/voice/cleanup_train/`: Claude-written everyday sentences, speech noise (stutters, fillers, restarts, misspellings), a leak check against the test set, LoRA trainer. Pilot and full1 trained overnight on the 5080.
- full1 reviewed on dev: the best result so far and the new candidate.
- llama.cpp b11960 downloaded (checksums verified); `export.py` makes GGUF files (merge, convert, Q8_0, Q4_K_M); the eval runs them with `--backend llamacpp`. 1,014 tests and the compile check pass.
**Measured:** all on Khang's PC (RTX 5080), engine off, dev 67 cases unless noted; "full pass" is the human check by Khang's rule, filled by Claude:
- Qwen3.5-4B prompt only: v1 full prompt 63/67 automatic, p50 1864 ms; full pass v3 53/67, v4 59/67 (p50/p90 2036/2449 ms); today's app 17/67.
- Holdout, 33 cases: v4 full pass 22/33, silent guesses 4/5, p90 2465 ms; today's app 7/33.
- Training: pilot 300 examples in 9.4 min; full1 1,088 examples in 32.7 min (387 steps, peak 9.5 GB).
- full1 in transformers (merged): automatic 64/67, silent guess 1/10, over-edit 0/10, 0 invented, full pass 62/67; p50/p90/p99 1770/2070/2237 ms.
- full1 in llama.cpp: Q8_0 328/378/412 ms (about 5.1 GB, nvidia-smi, whole card), Q4_K_M 264/292/332 (about 3.3 GB), bf16 488/565/603; full pass 59 (bf16, Q8_0) and 61 (Q4_K_M); greedy 61/67 in both engines.
- Greedy llama.cpp bf16 gave the same answers as transformers on 67/67 cases; llama.cpp's tokenizer matches Hugging Face's on 67/67 prompts.
**Decisions:** prompt v4 replaced v3; the holdout verdict (no-go); train our own adapter on Claude-written sentences (terms checked); full1 is the candidate; llama.cpp (MIT) runs it; one session per job with /start. All in DECISIONS.md. Proposed: Q8_0 with greedy decoding for the app.
**Still running:** none
**Problems / blockers:**
- Dev is tuned on (data written after seeing its failures, epoch picked on it) and the holdout is spent: only a fresh hidden set gives a fair number.
- Fail in every run: M16 (the right time only offered second), M25 ("eyeable" without brackets), S28 and X01 (broken word order; a bare order to a clinic).
- `/wrap-up` didn't load as a command in this session (probably because it was added after the session started); its steps were followed by hand.
**Next:**
1. Khang or Hung writes about 35 hidden test phrases; run full1 once as it would ship (llama.cpp Q8_0, greedy).
2. More training data for M16, M25, S28 and X01; train a new adapter.
3. App change, plan section 4: llama-server behind the `Rewriter` (127.0.0.1, API key), then the new guard.
**Khang must do by hand:** write the hidden test phrases (`messy → meant`); confirm full1 as the candidate and the Q8_0 + greedy proposal; spot-check the Claude-filled reviews; say whether "the right reading offered only second" passes when the context makes it clear (X08 was marked n, M16 and X05 y). Still open: the exact new name; remove the two people who left from `-eye-able`.
**For Cowork:** for the lawyer: Anthropic's terms and training on Claude-written data, NVIDIA's terms for shipping the CUDA runtime DLLs, the licence for new code, IP agreements. How the installer ships llama.cpp (the CUDA build with NVIDIA's DLLs is about 770 MB; CPU or Vulkan builds for laptops).
**Notes from Khang:** none

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
