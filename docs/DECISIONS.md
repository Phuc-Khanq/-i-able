# Decisions

One entry per decision, newest at the bottom. Never edit an old entry: add a new one that says what it replaces.

Format: date, decision, why, evidence, who decided.

---

**2026-11-02. Only commercially usable licences.**
[eye]able will be sold, so every model, dataset and package must allow commercial use (Apache 2.0, MIT, BSD, CC BY or similar). Check the licence on the source's own page before adding anything.
Decided by: Khang.

**2026-11-02. Drop Qwen2.5-3B for the dictation clean-up; don't fine-tune it.**
Its licence is `qwen-research`, and fine-tuned copies keep that licence.
Evidence: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct (licence field). Decided by: Khang.

**2026-11-02. The clean-up uses the Qwen3.5 family.**
Test Qwen3.5-4B first, with thinking off. Qwen3.5-9B is a stretch goal. For weaker laptops try Qwen3.5-2B, then 0.8B, then Qwen3-1.7B. Stay inside the Qwen family so prompts, code and training stay consistent.
Why: Apache 2.0, one family with sizes from 0.8B to 9B, strongest small models in the comparison we made.
Evidence: model cards list Apache 2.0 for Qwen3.5-2B, 4B and 9B, and Qwen3-1.7B (huggingface.co/Qwen). A third-party test reports about 1.3 GB (2B) and 2.7 GB (4B) for 4-bit files, and 18 and 8.5 tokens/s on a 16-core server CPU (bkataru.github.io/igllama/showcase/qwen35-small). Not yet measured by us.
Decided by: Khang.

**2026-11-02. Rejected alternatives for the clean-up.**
Llama 3.2 3B: custom licence with naming and attribution rules. Phi-4-mini (MIT) and Gemma 4 (Apache 2.0 on its model cards): good, but no reason to leave the Qwen family. SmolLM3-3B: weaker on our comparison. Decided by: Khang.

**2026-11-02. Measure before training.**
Build the 100-case test set first, run prompt-only baselines, and fine-tune only the gap that shows. A third of the cases are held out and never used for tuning.
Evidence: `eval/dictation_cleanup/README.md`. Decided by: Khang.

**2026-11-02. Head-tracking training data: numbers only, five targets.**
Record tracker numbers over time, never video. Learn direction intent, gesture vs not, voluntary vs involuntary motion, personal sensitivity and gaze. Main concerns: direction intent and voluntary vs involuntary motion. Detailed design waits for Khoi's tracker research.
Decided by: Khang.

**2026-11-02. Claude Code for code, Cowork for coordination.**
Code, tests, evaluation runs and training happen in Claude Code on Khang's PC. Research, planning, docs and the weekly tracker happen in Cowork. `docs/STATUS.md` and this file connect the two.
Decided by: Khang.

**2026-11-02. Development moves to a private repo on Khang's account.**
Two of the original five left and the product is going a different way, so new work shouldn't be public. New private repo `Phuc-Khanq/-i-able` (remote `origin`) holds the full history: every local branch and tag was pushed on 2026-11-02. The old public repo `Phuc-Khanq/-eye-able` (remote `public`) stays as the hackathon archive and website host; nothing new is pushed to it. Replaces CLAUDE.md's old rule "the repo is public".
Known limits: the code already public stays usable by anyone under its MIT licence; going private only protects new work. The licence for new code and IP agreements with past and current contributors are still open (a lawyer's question).
Decided by: Khang.

**2026-11-02. Commits carry only the developer's name.**
No `Co-Authored-By` lines and no Claude or "Generated with" attribution in commits or pull requests. Khang wants every commit under his own name. Two commits pushed before this rule (0a26717, 0116961) keep the line, since pushed history isn't rewritten. Recorded in CLAUDE.md, Git workflow.
Decided by: Khang.

**2026-11-02. The clean-up evaluation works on branch `voice-cleanup-eval`.**
`origin/voice` no longer existed and local `voice` was 59 commits behind `main`, so the branch was made fresh from `main`. Local `voice` was left at fd85b60 (already in `main`).
Decided by: Khang.

**2026-11-02. Small findings from the first code review.**
- The exact-version rule in CLAUDE.md applies to new packages only. Existing lines in `requirements.txt` (e.g. `transformers>=4.57`) stay as they are.
- `eval/dictation_cleanup/testset.csv` starts with a BOM on purpose, so Excel opens it correctly. Code reads it with `encoding="utf-8-sig"`.
- The docstring and fallback default in `engine/voice/rewrite.py` still name Qwen2.5-1.5B (config uses 2.5-3B). Fix it as part of the model switch, not before.
Decided by: Khang.

**2026-11-02. Sessions end with /wrap-up.**
Khang runs `/wrap-up` (`.claude/skills/wrap-up/SKILL.md`). It writes a new entry at the top of `docs/SESSION_LOG.md` and updates STATUS.md and this file, so the Cowork chat knows what happened. Claude doesn't wrap up on its own. Replaces the CLAUDE.md rule "update STATUS.md at the end of every session".
Decided by: Khang.

**2026-11-02. Scoring runner spec for the clean-up evaluation.**
Measure only, no app changes. Runner in `eval/dictation_cleanup/`, results in `eval/dictation_cleanup/runs/` (git-ignored). Backends "current" (today's clean-up exactly as the app runs it, the baseline), "transformers" (any model on the GPU; Qwen3.5 with thinking off and the card's non-thinking sampling with a fixed seed) and "fake" (tests). Prompts are versioned text files: "plain" (asr_best only) and "full" (plus alternatives, app, context, personal words). New models answer in JSON with `text`, `alternatives` and `uncertain`; broken JSON counts as a failure, with no silent retry. Dev split by default; holdout only with an explicit flag, after the pass bar is set.
Decided by: Khang.

**2026-11-02. Qwen3.5-4B downloaded for the evaluation.**
Licence: Apache 2.0 (the model page's licence field, read through the Hugging Face API on 2026-11-02, revision 851bf6e). 9.34 GB, in the Hugging Face cache on Khang's PC (`%USERPROFILE%\.cache\huggingface\hub`). Loaded as text only with `AutoModelForCausalLM` (`Qwen3_5ForCausalLM`), thinking off. Sampling from the card's non-thinking "general tasks" line: temperature 0.7, top_p 0.8, top_k 20, min_p 0, seed 1234 per case; the card's presence_penalty 1.5 isn't available in transformers and is not applied.
Evidence: https://huggingface.co/Qwen/Qwen3.5-4B. Decided by: Khang (download approved).

**2026-11-02. The invented-fact check also accepts names from the user's word list.**
The spec said input, alternatives and context. Case A08's own intended answer ("Tell Alice I am coming") uses a name that is only on the word list, so a name from the user's own list isn't treated as invented. Commit 41ccec7.
Decided by: Claude while building; Khang to confirm.

**2026-11-02. The pass bar for the clean-up holdout run.**
Agreed before anyone has run the holdout. It replaces the draft under "Proposed" (2026-11-02), with Khang's changes: a softer rule on clean controls and invented facts, a stricter full-pass rate. On the 33 holdout cases, scored by Khang and Khoi, who agree one answer per case (Khang has the final say):
1. Serious invented facts: 0. Serious means numbers, dates and times, names, places, amounts and negations; small connecting words (my, please, the) don't count.
2. Clean controls changed: at most 1 of 5, and the change must keep the meaning. Punctuation and capitals don't count as changes.
3. Ambiguous cases resolved silently: at most 1 of 5.
4. A main reading that flips a negation or changes a number, name or date against the heard words without evidence: 0.
5. Broken answers: 0.
6. Full pass (README's four checks): at least 28 of 33 (85%), and at least 15 points above today's app on the same cases.
7. Speed on the 5080, warm: p90 at most 6 s to use it for the whole-passage refine; at most 1.5 s to also use it for the quick clean-up.
The holdout is run once, with the final dev-tuned candidate, after the dev review. With 33 cases one case is 3 points: this is a go/no-go for building, not a number to publish.
Decided by: Khang (Q4: A, after comparing "never" with "allowed if marked unsure").

**2026-11-02. Khang reviews the clean-up outputs alone, for now.**
Khoi is on other work. Khang fills in the `_Khang` columns of the review sheets for dev and holdout, and the `_Khoi` columns stay empty. This replaces "scored by Khang and Khoi, who agree one answer per case" in the pass bar above; everything else in the bar stays. The README's two-reviewer rule is paused, so there is no check on one person's judgement: the sheets stay blind (A/B in random order) to limit bias. A second reviewer can join later without changing the sheets.
Decided by: Khang.

**2026-11-02. Khang's rule for "meaning kept", and Claude does the review work from now on.**
Rule, in Khang's words, roughly: filler, repeats and changes of mind can go; the output passes if it carries the main ideas of what the person wanted to say, as complete enough sentences, polite and meaningful. The exact order and structure don't matter. Khang filled in "meaning kept" for the dev sheet himself (2026-11-02_dev). Claude filled in "nothing invented" for that sheet: every row's notes say so. From now on Claude fills in both checks with this rule, and Khang spot-checks and has the final say. Risk: an AI is then judging an AI. Keep the sheets blind, and Khang should spot-check the "n" answers and the cases where Claude and the automatic checks disagree.
Decided by: Khang.

**2026-11-02. Prompt v4 replaces v3 as the clean-up candidate.**
v4 (`prompts/cleanup_v4_full.txt`) is v3 plus: never leave a fragment or broken word order as it is, write complete, polite sentences; a bare instruction becomes a polite request; "even a real word" can be a mishearing; two made-up examples (laundry, umbrella). Dev only, Qwen3.5-4B, same settings. Dev review (Claude, Khang's rule; unchanged outputs keep Khang's v3 answers): full pass 59/67 (88%) vs v3 53/67 (79%) and the app 17/67 (25%); automatic 63/67; silent guesses 2/10; p90 2.45 s. New weak spots: X02 invented "tomatoes", M08 (search query made into an odd question), A07 ("Please" added to a clean sentence). Holdout note: while picking example words, a grep of `testset.csv` printed four holdout rows (M30, S06, S24, X06: keys, a movie time, keys, a window); the examples avoid those topics. Tuning stops at v4; next is the holdout run, once.
Evidence: `eval/dictation_cleanup/reviews/2026-11-02_dev_v4_score.md`, run folder `runs/2026-11-02_Qwen3.5-4B_cleanup_v4_full`. Decided by: Claude (Khang asked for v4); Khang to confirm before the holdout.

**2026-11-02. Holdout result: Qwen3.5-4B with prompt v4 does not meet the pass bar.**
Run once, as agreed (33 holdout cases, 5080, same settings as dev). Review filled in by Claude with Khang's rule; not truly blind, since only the new model shows "[or: ...]". Against the bar:
1. Serious invented facts: 0 (pass). Today's app invented one (M27, "three minutes").
2. Clean controls changed: 0 of 5 (pass).
3. Ambiguous cases resolved silently: 4 of 5 (FAIL, bar is at most 1). A06, A09, A12, A15.
4. Main reading flips a negation or changes a number, name or date without evidence: 0 (pass). Borderline: A12 wrote "Jim" (word list and other hearing) instead of the heard "him", without showing both.
5. Broken answers: 0 (pass).
6. Full pass: 22 of 33 (67%) (FAIL, bar is 28). Today's app: 7 of 33 (21%), so +46 points (pass).
7. Speed, warm: p50/p90/p99 2041/2465/2981 ms. Fits the 6 s refine; not the 1.5 s quick clean-up.
Verdict: no-go for building on this prompt as it stands. Dev said 88% and the holdout says 67%: the prompt was fitted to the dev cases more than it looked. Main misses: wrong word first with the right one only as an option even when the context points to it (M12, M21, M30), broken sentences still curt (X03, S30, S18 wrong meaning), and ambiguous cases with no other hearing given (A06, A09) or a negation pair (A15) not shown as two readings.
The holdout is now spent: tuning against what it showed and running it again would not be a fair test. Any next candidate needs fresh held-out cases.
Evidence: `eval/dictation_cleanup/reviews/2026-11-02_holdout_score.md`, run folders `runs/2026-11-02_Qwen3.5-4B_cleanup_v4_full_2` and `runs/2026-11-02_Qwen2.5-3B-Instruct_app_2`. Decided by: the agreed bar; Khang to choose what comes next.

**2026-11-02. Train our own clean-up adapter on Qwen3.5-4B; Claude writes the training sentences.**
Prompting has hit its limit (dev 88%, holdout 67%), so the next step is a LoRA adapter on Qwen3.5-4B (Apache 2.0), trained on our own examples: everyday dictation with real speech habits (stutters, fillers, restarts, changes of mind, misheard words). Start with a pilot of about 200 examples to prove training works on the 5080, then scale up. Claude writes the clean base sentences and the hard cases; a script adds the speech noise. No test-set case or word pair may appear in the training data (a leak check enforces it), and a fresh hidden test set, written by Khang or Hung, replaces the spent holdout.
Terms checked on 2026-11-02: Anthropic's Consumer Terms (effective Oct 8, 2025, section 3) forbid using the Services "to develop any products or services that compete with our Services, including to develop or train any artificial intelligence or machine learning algorithms or models"; the Commercial Terms (June 17, 2025, D.4) say the same for "competing AI models"; both give the user ownership of outputs. Training is only forbidden for competing products. Khang's reading: an accessibility dictation clean-up does not compete with Claude. Not legal advice; added to the open questions for a lawyer. The alternative, if a lawyer disagrees: regenerate the base sentences with the local Qwen3.5 (Apache 2.0) and retrain.
Decided by: Khang.

**2026-11-03. First trained clean-up adapter ("full1"): best dev result so far; it is the new candidate.**
LoRA on Qwen3.5-4B (r=16, alpha=32, all linear layers except in_proj_a/b, lr 2e-4, 3 epochs, batch 8, bf16, peft 0.21.2 already in requirements-train.txt). Data: 1,088 examples written by Claude (seeds.py, seeds_extra.py, templates.py) with noise.py adding stutters, fillers, restarts, dropped small words and misspellings; the leak check dropped 37. Measured on the 5080 with the engine off:
- Training time: 32.7 min for 387 steps (5.2 s per step), peak GPU 9.5 GB. The pilot (300 examples) took 9.4 min.
- Dev, automatic checks: full1 64/67, over-edit 0/10, silent guess 1/10, 0 invented, 0 broken JSON. Epoch 2 also 64/67; epoch 3 kept because it fixed X08. Pilot 64/67, silent 0/10. The untrained model with the same short prompt: 60/67, silent 4/10, over-edit 1/10.
- Dev, human review (Claude with Khang's rule; outputs seen before keep their earlier answers): full1 full pass 62/67 (93%), meaning 65/67, nothing invented 67/67. Pilot 59/67; prompt v4 59/67; today's app 17/67.
- Speed, adapter merged into the weights: p50/p90/p99 1770/2070/2237 ms (the pilot unmerged: 2662/3041/3700). Still the slow reference kernels; not yet under the 1.5 s quick clean-up budget.
Choices made: training uses the short prompt `prompts/cleanup_t1_short.txt` (about 125 tokens per example, against about 1,185 with v4_full: the smoke test with v4_full needed about 250 s per step and filled the 16 GB card); the eval merges the adapter (merge_and_unload) as the app would ship it; the leak check no longer blocks negation and number words (they are categories the model must learn), but still blocks test sentences and the test set's own misheard word pairs.
Honest limits: the dev cases are no longer a fair test. The training data was written after seeing dev's failures, and the epoch was picked on dev. Only a fresh hidden test set gives a number to trust. The adapter (about 120 MB, `checkpoints/cleanup/full1/adapter`) and the data (`data/cleanup_train/`) are git-ignored; the code to rebuild both is committed. Built on Claude-written data: see the 2026-11-02 terms entry.
Evidence: `eval/dictation_cleanup/reviews/2026-11-03_dev_full1_score.md`, `..._dev_pilot_score.md`, run folders `runs/2026-11-03_Qwen3.5-4B-full1*` and `runs/2026-11-03_Qwen3.5-4B_cleanup_t1_short`. Decided by: Claude, inside the training run Khang approved; Khang to confirm.

**2026-11-03. llama.cpp (MIT) runs the trained clean-up model; build b11960 downloaded.**
llama.cpp v0.8.0, its stable release of 2026-10-28 (build b11960, commit 8642541b). Downloaded with Khang's OK for the speed check: the Windows CUDA 13.4 build (165 MB), its CUDA runtime zip (424 MB) and the source of the same tag (41 MB; its converter turns a Hugging Face model into a GGUF file and reads Qwen3.5). The SHA-256 of both binary zips matched the digests GitHub publishes. Kept in `downloads/llama.cpp/b11960/` (git-ignored). No new Python package: the converter's needs were already in `.venv`, and gguf-py comes with the source.
Licence: MIT (`LICENSE` at the v0.8.0 tag, github.com/ggml-org/llama.cpp, checked 2026-11-03). Not open source: the runtime zip holds NVIDIA's `cudart64_13.dll`, `cublas64_13.dll` and `cublasLt64_13.dll`, under NVIDIA's CUDA licence (PyTorch already ships the same kind of files). Their terms for shipping them inside our installer are not checked yet: one more question for the lawyer before they ship.
Decided by: Khang (approved the speed check).

**2026-11-03. full1 in llama.cpp: the same model, about 5x faster, under the 1.5 s budget.**
How: `engine/voice/cleanup_train/export.py` merges the adapter into Qwen3.5-4B on the CPU (bf16, as the eval did), converts it with llama.cpp's converter (`--no-mtp`: the text-only model has no multi-token-prediction head) and quantizes it: bf16 8.42 GB, Q8_0 4.48 GB, Q4_K_M 2.71 GB. The eval's new `llamacpp` backend runs llama-server on 127.0.0.1 only, offline, every layer on the GPU, 4096-token context, one slot. It sends the prompt as Hugging Face token ids and uses the same sampling as the transformers runs, in transformers' order. Measured on the 5080 with the engine off, dev (67 cases), warm:

| full1, dev | p50 / p90 / p99 (ms) | GPU (GB) | Automatic | Silent guess | Over-edit | Full pass (Khang's rule) |
| --- | --- | --- | --- | --- | --- | --- |
| transformers bf16 (reviewed earlier today) | 1770 / 2070 / 2237 | 8.61 | 64/67 | 1/10 | 0/10 | 62/67 |
| llama.cpp bf16 | 488 / 565 / 603 | 9.15 | 63/67 | 0/10 | 0/10 | 59/67 |
| llama.cpp Q8_0 | 328 / 378 / 412 | 5.06 | 63/67 | 0/10 | 0/10 | 59/67 |
| llama.cpp Q4_K_M | 264 / 292 / 332 | 3.25 | 64/67 | 0/10 | 0/10 | 61/67 |
| greedy, transformers bf16 | 2236 / 2643 / 2961 | 8.61 | 64/67 | 0/10 | 0/10 | 61/67 |
| greedy, llama.cpp Q8_0 | 336 / 398 / 461 | 5.04 | 64/67 | 0/10 | 0/10 | 61/67 |

0 invented and 0 broken answers in every run. GPU: for transformers, torch's peak; for llama.cpp, nvidia-smi for the whole card at the end of the run minus before the server started (Windows gives no per-process number), so it includes the context cache and is approximate. llama-server loads the Q8_0 file in about 3 s.
Same model, different luck: with greedy decoding (always the most likely token, no sampling), llama.cpp bf16 gave exactly the same answer as transformers on all 67 cases, and Q8_0 differed on 1 (S19, same meaning). With sampling (temperature 0.7) 15 answers differ between the two engines, because their random draws differ; bf16 and Q8_0 inside llama.cpp differ on only 1. So 62 against 59 is sampling luck, not a loss. M16, M25, S28 and X01 fail in every run; S11, S19, S29, X08 and A08 come and go with the draw.
Also checked: llama.cpp's own tokenizer gives the same ids as Hugging Face's on all 67 dev prompts, so an app can send plain text. The server log holds timings only, no dictated text. llama-server warns that it has no API key and accepts requests from any web page: an app must give it an API key and keep it on 127.0.0.1.
Review: Claude with Khang's rule; outputs judged before keep their answers.
Evidence: `eval/dictation_cleanup/reviews/2026-11-03_dev_llamacpp_score.md` and `..._dev_greedy_score.md`, run folders `runs/2026-11-03_full1-*` and `runs/2026-11-03_Qwen3.5-4B-full1-greedy_cleanup_t1_short`. Decided by: measured in the speed check Khang approved; which file and decoding the app uses is Khang's call (Proposed, below).

**2026-11-02. One Claude Code session per job, started with /start.**
Khang wants one chat per job. `/start <task>` (`.claude/skills/start/`) begins a new session: it reads STATUS.md, the latest SESSION_LOG.md entries and git, then gives a short briefing and a plan. `/wrap-up` ends it. Recorded late, at the 2026-11-03 wrap-up.
Decided by: Khang.

**2026-11-03. The big clean-up data set: 10,000 examples, 2,500 of them a test set, split by sentence.**
Khang asked for 10,000 examples with 2,500 for testing, more fields than daily life, and the categories as in today's test set. `engine/voice/cleanup_train/bigset.py` builds them from 42 field banks (`fields_*.py`: 1,353 sentences written by Claude, 300+ field terms with how a recognizer mishears them, pairs with two meanings) plus the older seeds (training only: full1 saw them). Example kinds: already correct, messy, broken, a misheard term (the right word from the word list, the line before, another hearing, or only the sentence), a change of mind, cut off, two meanings (negation, number and name pairs), and two- or three-sentence passages.
The split is by sentence: a sentence, and every misheard term, goes to one side only (a term goes to the same side in every field), template frames go by frame, and training examples within 0.6 token overlap of a test text are dropped. Checked: no test sentence or test mishearing is in training. Test mix: clean 15%, two meanings 15%, misheard 24%, broken 30%, mixed 15%. The test file uses the eval's layout (`run.py --cases`); the eval also gained `--greedy`.
Baseline, full1 (llama.cpp Q8_0, greedy, 5080): 2,389/2,500 automatic (95.6%), p50/p90 294/345 ms. Hand check of 150 random cases by Khang's rule (Claude): 132/150 (88%, about ±5), so the automatic checks run about 7 points high. Failures: word-list words offered as alternatives or the wrong one picked (7), a misheard term guessed wrong or kept with only a weak hint (5), small words dropped and the meaning changed (6; two of these are unfair cases where the noise dropped "will" or "can"; fix in the next test set). Evidence: `eval/dictation_cleanup/reviews/2026-11-03_big1_test_sample_full1.md`.
Limit: the test sentences are Claude-written, like the training ones; the human-written hidden set stays the real check.
Decided by: Khang (size, split, fields); the build details by Claude.

**2026-11-03. A guard of its own for the trained clean-up model (check_cleanup).**
The app's guard (check_rewrite) rejects any reworded word, which would throw away most of what the trained model is for (fixing misheard words and broken sentences). `check_cleanup` (`engine/voice/rewrite.py`) is used only with `rewrite_backend = "llamacpp"`. It blocks a number, name or negation that nobody said (a name may come from the user's word list, or sound like what was heard: "medi cure" -> Medicare, but not Ann -> Dan), one that was said and is gone without a change of mind ("no", "sorry", "I mean", "wait" ...), or most of the said words gone (a dropped sentence, or a different one).
Measured on full1's 2,500 test answers: it blocks 13. At least 6 are real failures (a different sentence written, two dropped sentences, an invented "until Monday", essay -> thesis), about 6 were good answers (W-2, EpiPen, nonprofit: 0.25% of the good answers), and those fall back to the user's own words without fillers. It catches none of the 18 failures of the hand-checked sample: those are wrong meanings, which no word rule sees. The preview and confirm step stays the safety net.
Without other hearings (the app has none until the contract change), a homophone number fix ("to o'clock" -> "two o'clock") is blocked: numbers never change without evidence.
Decided by: Claude, in step 2 of the app plan that Khang approved; Khang to confirm.

**2026-11-03. Licence scan of the Python packages: one risk for the lawyer, the rest usable.**
91 packages in `.venv` (88 used at run time, 2 training only), from each package's own licence files; bundled native libraries opened for `av`, `torch`, `opencv-contrib-python`, `soundfile` and `llvmlite`. Report: `docs/licences/2026-11-03_python_packages.md`.
- Risk: PyAV (`av` 18.1.0), loaded whenever faster-whisper is imported, ships libx264 and libx265 (GPL-2.0-or-later) with its FFmpeg (FFmpeg itself reports LGPL-3.0-or-later). Our code doesn't use `av`. Options: a PyAV without x264/x265, not shipping `av` (only faster-whisper's file decoding needs it), or complying with the GPL. For Khang and the lawyer.
- Conditions, normal for sold software: LGPL (`pynput`, `soxr`, libsndfile in `soundfile`, FFmpeg in `av` and OpenCV): ship them unchanged as separate libraries with their licence texts. MPL-2.0 (`certifi`, `tqdm`). NVIDIA runtime files inside PyTorch's CUDA build: the same question as llama.cpp's (lawyer).
- The other 83 are permissive (MIT, BSD, Apache-2.0, ISC, PSF ...); `llvmlite` declares nothing in metadata but its LICENSE file is BSD-2-Clause.
- Closed: Silero VAD is MIT (LICENSE file and README on GitHub, checked 2026-11-03); the "CC BY-NC 4.0" was the hover text of a badge that reads MIT.
Not legal advice. Decided by: scan by Claude at Khang's request; what to do about PyAV is Khang's call with the lawyer.

**2026-11-03. The app runs without PyAV.**
Replaces the open PyAV point above for the app. `engine/voice/whisper_import.py` gives faster-whisper a stand-in for `av` (it only needs the real one to decode audio files; the engine always passes samples), `engine/main.py` sets it up before anything heavy loads (torchvision imports `av` too), the installers (`scripts/setup.ps1`, `scripts/install.ps1`) add faster-whisper 1.2.1 with `--no-deps`, and `av` and faster-whisper are out of `requirements.lock.txt` (its other dependencies, ctranslate2 and onnxruntime, are listed in `requirements.txt`). Training keeps PyAV (`requirements-train.txt`, TORGO files) and is never shipped.
Evidence (Khang's PC, 2026-11-03): with the real PyAV made impossible to import, the engine's dictation code transcribed a synthetic clip correctly ("Please send the report to my manager before Friday.", Whisper large-v3-turbo, CPU int8, 10.8 s) and no FFmpeg, x264 or x265 library was loaded; with PyAV installed the engine still doesn't load it; a pip dry run of a fresh install from the lock file fetches 85 packages, none of them PyAV; `tests/test_whisper_import.py`. Not yet done: a real fresh install on a clean PC.
Decided by: Khang ("try it"), built by Claude.

**2026-11-03. big1, trained on the big data set, is the clean-up candidate in place of full1.**
big1: LoRA on Qwen3.5-4B, 7,125 training examples from `bigset.py`, 2 epochs, 162.9 min on the 5080 (peak 9.65 GB). Both models run as the app would (llama.cpp Q8_0, greedy, 5080):
- big1 test, 2,500 cases, automatic: big1 2,441 (97.6%), full1 2,389 (95.6%). Silent guesses 0/380 vs 14/380, clean sentences changed 4/378 vs 6/378, broken answers 0 vs 1, p90 338 vs 345 ms.
- The same 150 random cases, full pass by Khang's rule (Claude): big1 141 (94%), full1 132 (88%).
- Dev, 67 hand-written cases, full pass by Khang's rule (Claude): big1 63 (94%), full1 61 (91%). Invented 0 in both.
- New weak spots: "medic care" -> Medicaid in all 3 Medicare cases (the guard blocks them), and "fifteend of March" -> "end of March" (the guard misses it).
Limit: the 2,500 come from the same generator as big1's training data, and dev is small and tuned on, so the human-written hidden set decides. After one epoch big1 was already ahead (2,430 automatic); the 2-epoch adapter is the candidate.
The real app backend on the 5080, through the Rewriter: p50/p90 280/309 ms, 4.8 GB, the server restarts after a crash in 2.9 s and is gone after close() or a hard kill. First start 50 s, because the repo is on a USB hard disk (D:).
Evidence: `eval/dictation_cleanup/reviews/2026-11-03_big1_vs_full1.md`, `..._dev_big1_score.md`, `docs/plans/2026-11-03_cleanup_app_integration.md` ("Measured on the reference machine"); run folders `runs/2026-11-03_big1-*`.
Decided by: measured by Claude in the run Khang approved; switching the app's default model file is Khang's call (Proposed, below).

**2026-11-03. The clean-up guard: list words must sound like what they replace, and dates and misspelled numbers count as numbers.**
Built from the proposal below (it replaces that item). In `check_cleanup` (`engine/voice/rewrite.py`):
- A word from the user's word list that wasn't said must sound like the heard words it replaced (1 to 4 in a row, all gone from the clean text): spelled close (0.6), or close by a rough sound key (0.8; qu/hard c as k, soft c as s, vowels dropped: "keen wa" -> quinoa, "raise you may" -> résumé, "precious" -> pressure, "Coa" -> Khoa). Otherwise "word-list word added". Before, a list word always passed. What was typed earlier still counts without sounding alike.
- Ordinals are numbers ("15th" / "fifteenth" is fifteen; "second" isn't, it's mostly time), and a misspelled number word is the number ("thowsand", "fifteend", "fourty": 5+ letters, same first two letters, 0.85 close; "there", "thirsty", "forth" aren't).
Measured on the saved answers (no new model runs): big1's 2,500 go from 12 blocks to 18. Newly caught: 3 real mistakes ("fifteend of March" -> "end of March"; "deduction" and "invoice" from the list in place of W-2 and deliverables). Newly blocked good answers: 4, all a number nobody said ("the tend" -> "the tenth" 3 times, "dirtieth" -> "thirtieth"), the same rule as "to o'clock" -> "two o'clock". No longer blocked: "five thowsand dollers". full1's 2,500 go from 13 to 22: 7 more list words added or swapped in (barbecue, consignment, memoir, colonoscopy, Minecraft, Leo for Theo, Beth for Bess), the same 3 date fixes lost. Dev unchanged (big1 2 blocks, full1 3). The GPU test's "of sertraline" is blocked. Tests: `tests/test_cleanup_model.py` (1,071 in all pass).
Known gap: letters spelled out ("double you two" -> W-2) don't sound alike by the key and stay blocked, as before.
Decided by: Khang (go-ahead to fix), built and measured by Claude.

**2026-11-04. big2 (twice the LoRA rank, more data, 3 epochs) does not beat big1; big1 stays the candidate.**
big2: LoRA rank 32 / alpha 64 on Qwen3.5-4B (big1: 16 / 32, every layer in both), 11,400 examples (careful noise, targeted examples for big1's misses, 16 new field banks, test-side terms filtered out), 3 epochs, 344.1 min on the 5080 (peak 10.19 GB). Scored as the app runs it (llama.cpp Q8_0, greedy, 5080) on the unchanged big1_test.csv:
- Automatic, 2,500: big1 2,441; big2 epoch 1 2,362, epoch 2 2,409, final 2,411.
- Clean sentences changed (of 378): big1 4; big2 29, 15, 14 (it adds "the", "please", "to"; once "building" -> "billing").
- The same 150, full pass by Khang's rule (Claude): big1 141, big2 final 139.
- Dev automatic: big1 65; big2 64, 62, 62. Latency the same (p50 about 289 ms).
Why: big2's data drops every test-side term, which explains about 9 of the 30 cases (on the 2,307 cases where neither run had the advantage, big1 still wins 2,281 to 2,260); the over-editing and worse broken questions are a real step back, cause unmeasured (rank, length or the new data). Lesson: on this test set more capacity and more generated data no longer help; the next gain has to be measured on the human-written hidden set.
Evidence: `eval/dictation_cleanup/reviews/2026-11-04_big2_vs_big1.md`; run folders `runs/2026-11-04_big2-*`; `checkpoints/cleanup/big2/results.json`.
Decided by: measured by Claude in the overnight run Khang approved ("Yes, start prep"); the app default stays Khang's call (Proposed, below, still names big1).

---

## Proposed, not yet agreed

- Clean-up levels the user chooses: Exact (as said), Tidy (fix grammar), Full help (work out what was meant).
- The clean-up gets Whisper's top candidates, the last text typed, the app and the user's word list as input.
- The guard changes from "reject if the text differs a lot" to "reject if it adds facts nobody said".
- Tiers: light model by default on weak laptops, heavier model where the GPU allows.
- (2026-11-02) The pass bar for the clean-up holdout run: 0 invented facts, 0 of 5 clean controls changed, 0 of 5 ambiguous cases resolved silently, 0 main readings that flip a negation or change a number, name or date, 0 broken answers, full human pass at least 27 of 33 and at least 15 points above today's app, p90 at most 1.5 s on the 5080 for the quick clean-up (6 s for the refine only). Details: `docs/plans/2026-11-02_dictation_cleanup_plan.md`, section 3. The holdout is not run until this is agreed.
- (2026-11-02) Run the new clean-up on llama.cpp with 4-bit GGUF files for speed and GPU memory (plan, section 2).
- (2026-11-03) The clean-up runs in llama.cpp with the Q8_0 file on NVIDIA GPUs (no measurable loss against bf16, 4.5 GB, p90 0.38 s on the 5080) and greedy decoding (the same answer every time for the same words; 61/67 on dev, inside the 59 to 62 of the sampled runs). Q4_K_M (2.7 GB, 61/67 sampled) where memory is tight, once timed on a low-end laptop.
- (2026-11-03) The app's default model file (`rewrite_gguf` in `engine/config.py`) changes from full1 to big1 Q8_0 (`checkpoints/cleanup/big1/gguf/big1-q8_0.gguf`), until the hidden test says otherwise.
- (2026-11-03) Two guard fixes found in the GPU test: a word-list word must sound like something that was heard (full1 added "of sertraline" from the list), and a misspelled number word counts as a number ("thowsand", "fifteend"), so it's neither blocked nor lost. Measured on the 2,500 before it goes in.
