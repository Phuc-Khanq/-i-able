# Session log (newest first)

## 2026-11-04 18:20: Clean-up model: big data set, the model in the app, big1/big2/big3 training, over- or underfitting
Branch: voice-cleanup-app · Commits: 3425396 big data set from 42 field banks; 1bf636b eval --cases/--greedy; 11e9e1a hand check of full1 (150); 3546167 llama-server behind the Rewriter, own guard; 48abc22 docs; 138c806 context/app in, other readings out (contracts); ac2d522 docs; 69687d2 card lists other readings, "option two"; ef79f47 docs; fb1c578 setup/preflight know the backend; bf72e3a licence scan; f2b79c2 app runs without PyAV; f67d8b2 big1 beats full1; 212ca23 guard: list words, dates, misspelled numbers; b6f3e0a big2 data, rank flag; 5179e52 big2 doesn't beat big1; 8f683a3 app model file is big1; ca754de big3 data and loss checks. All pushed to origin except ca754de · Uncommitted: engine/capture/, tests/test_capture.py (draft, untracked on purpose)
**Done:**
- Big clean-up data set (10,000, 2,500 of them a frozen test set, split by sentence); big1 trained on it and made the app's model file (`rewrite_gguf`; the default backend is still `transformers`).
- App plan steps 1 to 4 built: llama-server behind the Rewriter, the `check_cleanup` guard (then improved: list words must sound alike, dates and misspelled numbers count), the contract additions, the card's other readings.
- Licence scan of 91 Python packages; the app now runs without PyAV (GPL x264/x265).
- big2 overnight (rank 32, more data, 3 epochs): lost to big1. big3: twice the data, new kinds (noisy two readings, casual requests), and the trainer now measures over/underfitting (whole sentences held back, loss on 400 test cases never trained on). Explainer: `docs/cleanup_model_explained.md`.
**Measured:** Khang's PC, RTX 5080, llama.cpp b11960 Q8_0, greedy; "150" = the same random test cases judged by Khang's rule (Claude):
- big1 vs full1, 2,500 test cases: automatic 2,441 vs 2,389; silent guesses 0 vs 14 of 380; 150: 141 vs 132; dev full pass 63 vs 61. big1 trained 162.9 min (peak 9.65 GB).
- Rewriter with big1 on the 5080: p50/p90 280/309 ms, 4.8 GB, first start 49.8 s (repo on a USB disk), restart after a kill 2.9 s.
- big2 by epoch 1/2/3: automatic 2,362/2,409/2,411 (big1 2,441); clean sentences changed 29/15/14 of 378 (big1 4); 150: 139 (big1 141); dev automatic 64/62/62 (big1 65). 344 min, peak 10.19 GB.
- Fit: training loss vs held-back loss at the end: big1 0.0011 vs 0.0094, big2 0.0004 vs 0.0040 (memorizing; rank 32 gave nothing, so size isn't the limit). Different sentences: big1 2,778 of 7,500 examples, big2 4,323 of 12,000, big3 5,752 of 24,000.
- big3s first loss check: training 0.094, held-back sentences 0.095, test cases 0.106; 4.9 s per step.
**Decisions:** big1 is the candidate and the app's model file; big2 not used; big3 doubles the data and measures the fit; not built on purpose: flagging every fix made without a hint (dev expects sure fixes like "cur" -> car). All in DECISIONS.md (steps 1 to 4 recorded late).
**Still running:** big3s (8,000 random examples of big3's data, rank 32, 2 epochs), started 18:04, expected about 21:35 with scoring. Output: `checkpoints/cleanup/big3s/` (log.jsonl has the loss checks, results.json), runs `eval/dictation_cleanup/runs/2026-11-04_big3s-*`. big3 (24,000) waits: Khang starts it when he goes to sleep (about 9.5 h).
**Problems / blockers:** big3's first launch exited at once (the hidden bash had no tool path; relaunched through Git's bash); big3 stopped at step 33 on Khang's request. Dev is tuned on and the holdout is spent: only the hidden set gives a fair number.
**Next:**
1. big3s results (about 21:35), then big3 overnight when Khang goes to sleep, with the loss check for big1 and big2.
2. The hidden test phrases; run the candidate once as it ships.
3. A plan for many more different sentences (Khang's goal: 100,000 examples).
**Khang must do by hand:** say when to start big3; push ca754de and this wrap-up; try big1 in the demo with camera and mic (`"rewrite_backend": "llamacpp"` in config.local.json, then preflight and run_demo); write the hidden test phrases; decide whether llamacpp becomes the default backend and whether the guard rejects numbers from sound-alike words (M01, M02).
**For Cowork:** where 20,000+ new sentences come from (Claude-written batches through the API vs commercially licensed text: terms and licences); the lawyer list (training on Claude-written data, NVIDIA's CUDA DLLs, code licence, IP agreements); whether to use the model's own token probabilities to mark answers unsure.
**Notes from Khang:** none

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
