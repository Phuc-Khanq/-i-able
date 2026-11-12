# Status

Last updated: 2026-11-11 21:20 (clean-up model chat: hearings plan approved, rewrite.py passes them; draft data task for Hung; GPU free). Session-by-session record: `docs/SESSION_LOG.md`.

## Now

- **Dictation clean-up: our own trained model.** big1, a LoRA adapter on Qwen3.5-4B (Apache 2.0) trained on Claude-written examples, is the candidate and the app's model file (`rewrite_gguf`, big1 Q8_0). It runs in llama-server behind the Rewriter (0.28 s median on the 5080), but only with `rewrite_backend: "llamacpp"`; the default backend is still the old Qwen2.5-3B. Dev full pass 63/67 by Khang's rule; dev is tuned on, so the hidden test decides.
- **Training experiments: the settings are settled, the data is the limit.** Rank 16 and learning rate 1e-4 fixed what was wrong: big3s16lr is the first run to beat big1 (test 2,446 vs 2,441, fair cases 2,281 vs 2,275), by a small margin. After that nothing moves the score: one epoch, 5e-5 and three times the data all tie it (2,439 to 2,447, no difference beyond chance). The last points turn on rare sound-alikes (heartworm, prednisone, AFO) that each run gets a different few of. Proposed: big3s16lr epoch 1 as the new candidate (needs Khang's OK). More data must be new sentences from new sources (Khang's goal: 100,000 examples); one-epoch runs (half the time) are enough to test it. Explainer: `docs/cleanup_model_explained.md`. For slides: `docs/cleanup_progress_for_slides.md`.
- **Rename:** a new name is coming (close to the new repo name `-i-able`, not final). The old-name scan is done; changes wait for the exact name.
- **Head tracking:** paused until Khoi's open-source tracker research is in.

## Code chats now (rules: CLAUDE.md, "Two Code chats at once")

- **Clean-up model chat** ("Clean-up model" in `ListAgents`): trains and scores the clean-up model. Owns `engine/voice/cleanup_train/`, `eval/dictation_cleanup/`, `data/cleanup_train/`, `checkpoints/cleanup/`, `docs/cleanup_*.md`. GPU: none (the confidence test ended 2026-11-10 22:20). Another project on this PC (DigiErhu, its own folder) sometimes trains on the GPU: check `nvidia-smi` before any GPU job here.
- **Dictation chat** ("Dictation" in `ListAgents`, since 2026-11-06): the dictation path from mic to typed text, except the clean-up model: mic and voice detection, Whisper and how well it hears our users, commands vs dictation, the dictation card, confirm/drop. Owns `engine/voice/` audio_in, mic, dictation, commands, command_list, personal_commands, launch, voice_data, live_check, whisper_bench, whisper_import, command_bench, `finetune/`; `engine/eyes/dictation_card.py`; `engine/core/mode_manager.py` (message the clean-up chat before changing how it calls the Rewriter); their tests. GPU: none so far; Whisper runs only after checking "In progress" and `nvidia-smi`.
- Shared, message first: `engine/voice/rewrite.py` (the trained model needs its exact prompt), `engine/voice/voice_check.py`, `engine/config.py`, the contracts. Both chats work on `voice-cleanup-app`. A training or scoring chain imports these and `eval/` at every step: no changes to them while one runs.

## Done recently

- 2026-11-11: the unsure flag is in the app. `CleanupModel.clean()` asks llama-server for word probabilities and marks the answer unsure when a content word is under 0.8 (llamacpp backend only, no contract change). The app's code path on dev gives the same answers and flags as the eval (67 of 67; median request 313 ms with probabilities). 1,178 tests pass. Found on the way: the app never sends Whisper's other hearings to the model (see Next 5b).
- 2026-11-10: the model's own word confidence flags most of its wrong guesses. The model gives all its wrong answers as sure (54 of 54 on the test). A cut-off of 0.8 on content words catches 33 of them, 18 of the 20 that change the meaning (the guard blocks one more), and flags 5.3% of right answers (big1: 40 of 59, 5.0%). On dev it flags 16 to 21% of right answers (phrasing choices in broken sentences). It costs about 23 ms per sentence. The runner-up word as a second reading is right for only 8 of 36. `eval/dictation_cleanup/reviews/2026-11-10_confidence_flag.md`, tool `eval/dictation_cleanup/confidence.py`.
- 2026-11-06 night: big3s16lr1ep (one epoch, 84 min) and big3s16lr5 (learning rate 5e-5, 168 min) both tie big3s16lr: test 2,445 and 2,447 (both epochs) vs 2,446, fair 2,301 cases 2,280 and 2,283 vs 2,281, head to head 4-5 and 4-2 (p 1.00, 0.69). Lowest losses on new data of any run (test cases 0.0170, 0.0169 vs 0.0192), but no higher score. Dev 62 and 63 vs 65 (a dropped "not", one more silent guess, "Hawthorne"; the guard blocks them). Settings settled; candidate unchanged. `eval/dictation_cleanup/reviews/2026-11-06_big3s16lr1ep_and_lr5.md`.
- 2026-11-05 night: big3lr epoch 1 (big3's 24,000 at rank 16 and 1e-4; only epoch 1 exists, because of the restart) ties big1: test 2,439 vs 2,441, fair 2,301 cases 2,275 (7 wins, 7 losses), and stays a little behind big3s16lr epoch 1 (2,446; fair 3 wins, 9 losses). Far ahead of big3, the same data at the old settings (2,395; 28 to 3). Clean changed 7 vs 4, dev 65. `eval/dictation_cleanup/reviews/2026-11-05_big3lr.md`.
- 2026-11-05 evening: two Code chats can share this folder (rules in CLAUDE.md, "Code chats now" above, /start and /wrap-up know about it). The first night chain was stopped at 22:27 when the Kaspersky installer restarted Windows: big3lr died at step 3,163 of 5,698, its epoch-1 adapter survived. Its test-case loss at steps 1,500 and 2,500 (0.0173, 0.0176) was the lowest of any run, but swings up to 0.006 between checks. Slides brief: `docs/cleanup_progress_for_slides.md`.
- 2026-11-05 afternoon: big3s16lr (big3s16 with learning rate 1e-4, 150 min) is the first run to beat big1: test epoch 1 2,446 / final 2,444 vs 2,441, fair 2,301 cases 2,281 vs 2,275 (10 wins, 4 losses), terms no run trained on 89 vs 84, invented flagged 7 vs 8, clean changed 4 and dev 65 (equal). Lowest test-case loss measured, 0.0187 at the end of epoch 1, rising slightly in epoch 2 (0.0199). At step 500 all losses beat 2e-4's, so the old steps overshot. `eval/dictation_cleanup/reviews/2026-11-05_big3s16lr.md`.
- 2026-11-05 midday: big3s16 (big3s's data at rank 16 instead of 32, 182 min) beats big3s: test 2,432 vs 2,418, fair 2,301 cases 2,277 vs 2,262 (17 wins, 2 losses), clean changed 4 vs 7, dev 65 vs 64, digits for spoken numbers 0 vs 4; final loss on held-back sentences 0.0133 vs 0.0170, on test cases 0.0200 vs 0.0208. Ties big1 on the fair cases (2,277 vs 2,275), clean (4) and dev (65), but makes more confident wrong guesses on medical words (HIV, AFO, PRP). `eval/dictation_cleanup/reviews/2026-11-05_big3s16.md`.
- 2026-11-05: big3 (24,000 examples, rank 32, 2 epochs, 428 min) loses to big3s, its own 8,000-example part: test 2,395 vs 2,418, fair 2,301 cases 2,250 vs 2,262, clean changed 15 vs 7, dev 63 vs 64 (big1: 2,441, 2,275, 4, 65). Held-back sentences 0.0067 (big3s 0.0170), but test cases flat after step 500 (0.0279 at the end; big3s 0.0208, big1 0.0190). big2's test-case loss rises every epoch (0.0324, 0.0339, 0.0361). Cause of the clean-sentence changes: `drop_small_words` in broken examples (43% of them add "the"). `eval/dictation_cleanup/reviews/2026-11-05_big3.md`.
- 2026-11-04 evening: big3s (8,000 of big3's examples, rank 32, 2 epochs, 175 min) doesn't beat big1: test 2,418 vs 2,441, fair 2,301 cases 2,262 vs 2,275, clean changed 7 vs 4, dev 64 vs 65. Loss on new sentences never rises but ends 19x the training loss (0.0170 vs 0.0009); 93% of its fall is in the first 500 of 1,898 steps. `eval/dictation_cleanup/reviews/2026-11-04_big3s.md`.
- 2026-11-04: trainer measures over/underfitting (`--held-back-by sentence`, `--watch`, `--check-every`); big3 data (24,000 examples, 5,752 different sentences; noisy two readings, casual requests). Training loss ends about 10x below held-back loss (big1 0.0011 vs 0.0094).
- 2026-11-04: big2 (rank 32, 11,400 examples, 3 epochs, 344 min) loses to big1 on every measure: test automatic 2,411 vs 2,441, clean sentences changed 14 vs 4, the 150 by Khang's rule 139 vs 141, dev automatic 62 vs 65. `eval/dictation_cleanup/reviews/2026-11-04_big2_vs_big1.md`.
- 2026-11-04: the app's model file is big1 Q8_0 (was full1).
- 2026-11-03: big1 (big data set, 2 epochs, 163 min) beats full1: test automatic 2,441 vs 2,389, silent guesses 0 vs 14, the 150: 141 vs 132, dev full pass 63 vs 61.
- 2026-11-03: app plan steps 1 to 4 built: llama-server behind the Rewriter (127.0.0.1, API key, dies with the engine), the `check_cleanup` guard, contract additions (`context`, `app`, `alternatives`, `uncertain`), the card's other readings ("option two"). Not yet tried with camera and mic.
- 2026-11-03: big data set (10,000; frozen test set of 2,500 in `data/cleanup_train/big1_test.csv`); licence scan of 91 packages (`docs/licences/`); the app runs without PyAV.
- 2026-11-03: full1 in llama.cpp (MIT, b11960): Q8_0 p50/p90/p99 328/378/412 ms on dev; greedy gives the same answers as transformers.
- 2026-11-03: first trained adapter full1: 1,088 examples, 32.7 min; dev full pass 62/67.
- 2026-11-02: training pipeline `engine/voice/cleanup_train/`; holdout run once with prompt v4 (fails the bar, 22/33; the holdout is spent); prompts v2 to v4; scoring runner; `/start` and `/wrap-up`; private repo `Phuc-Khanq/-i-able` (remote `origin`; old public repo is remote `public`).
- Earlier: hackathon build (head pointer, long-blink click, voice commands, dictation with preview and clean-up, click by name, website live demo, Windows installer, about 900 tests). See README.

## How the clean-up guard works today (`engine/voice/rewrite.py`)

- Default backend (Qwen2.5-3B): `check_rewrite` rejects the model's text if length is outside 0.4x to 1.3x, a number, name or negation changes, the point of view changes, a content word is swapped, under 60% of content words are kept, or over 25% are new. Then `light_clean`, else the user's own words. The refine (`check_refine`) is looser.
- Trained model (`llamacpp`): `check_cleanup` blocks a number, name or negation nobody said, one said and gone without a change of mind, or most words gone. Word-list words must sound like what they replace; ordinals and misspelled number words count as numbers. A number from a sound-alike word ("the tend" -> tenth) is blocked.
- Known: the docstring and fallback default in `rewrite.py` still name Qwen2.5-1.5B. Fix with the backend switch.
- `testset.csv` starts with a BOM on purpose (for Excel). Read it with `encoding="utf-8-sig"`.

## In progress

- Nothing from the clean-up model chat (the confidence test ended 2026-11-10 22:20; GPU free).

## Next

1. Khang picks the candidate (big3s16lr epoch 1 is proposed; the 2026-11-06 runs didn't beat it), before the hidden test. Optional, low value: big3lr1ep (big3's 24,000 in one epoch, about 4 h overnight).
2. Hidden test: Khang or Hung writes about 35 new phrases (`messy → meant`). Claude turns them into cases and runs the candidate once as it ships (llama.cpp Q8_0, greedy).
3. Try big1 in the demo with camera and mic (`"rewrite_backend": "llamacpp"` in config.local.json, preflight, run_demo); then decide whether llamacpp becomes the default.
4. A plan for many more different sentences (toward 100,000 examples): source, licence, cost. Draft task for Hung (real words and sentences from our users, then what the app heard them say; consent form first), waiting for Khang's review: Claude Docs "Real-world data task for Hung", https://claude.ai/artifact/LsBqWiHwbnyAPRqLgkYatN
5. Decide whether the guard rejects numbers from sound-alike words (dev M01 and M02 expect them fixed).
5b. Send Whisper's other hearings to the clean-up. The model was trained and scored with "Other hearings", but the Rewriter always sends "(none)". On dev without them (big1, 2026-11-11), 7 cases that come out unsure with them come out sure ("bank" or "bench"), and 6 answers change. Plan approved by Khang 2026-11-11: `docs/cleanup_hearings_plan.md`. Done: topics.py's optional "hearings" field (inside the engine only) and rewrite.py passing it to the model. **For the Dictation chat:** step 0 (measure Whisper's top 3 hearings and their cost first), then dictation.py and mode_manager.py. Whisper runs greedy today, so there are no other hearings yet. Later: send the weak word so the card highlights it.
6. Rename, once the exact name is given.
7. Light models for weak laptops (Qwen3.5-2B, 0.8B; Q4_K_M or llama.cpp's CPU build), timed on a low-end laptop.

## Blockers and risks

- Dev is tuned on and the holdout is spent: only the fresh hidden set gives a number to trust. The 2,500-case test comes from the same generator as the training data.
- D: (this repo, the model files) is a USB drive (Transcend StoreJet). On 2026-11-11 the first llama-server start timed out at the app's 120 s limit; the next one took a few seconds. Cause not measured (likely the first read of the 4.5 GB file over USB, maybe the antivirus scan). Before a demo, start the engine once early, or put the GGUF on the internal SSD.
- A restart kills a training run, and the trainer can't resume (only per-epoch adapters are saved). No installers or Windows restarts while a chain runs; Windows Update's active hours end at 02:00.
- The test cases were invented, not taken from real users. Real users or a speech-language pathologist should review them later.
- For a lawyer: training on Claude-written sentences (Anthropic's terms forbid it only for competing products; Khang's reading: we don't compete); shipping NVIDIA's CUDA runtime DLLs with llama.cpp; PyAV stays in training only; the licence for new private code; IP agreements with everyone who wrote code.
- Website updates: the live site is served from the public repo, and pushing `main` there would publish the private code. Pick a way to update it. The Pages workflow will also fail in the private repo until it's switched off there.
- Remove the two people who left from the old public repo's collaborators (Khang, on GitHub).

## Uncommitted work on Khang's PC

- `engine/capture/` and `tests/test_capture.py`: unreviewed draft, left untracked. Decide: delete, or keep on a separate branch.
- Not pushed: everything after 4577341 (the 2026-11-06 wrap-up, the Dictation chat's first two commits, and the clean-up chat's confidence test and unsure flag).
- Disk: each training run keeps about 20 GB (one epoch) or 40 GB (two) in `checkpoints/cleanup/<name>/`; its `merged*/` folders and bf16 GGUFs can be rebuilt with export.py and deleted. D: had 771 GB free on 2026-11-06.
- Git-ignored, only on this PC: training data (`data/cleanup_train/`: big1, big2, big3, big3s, the frozen test and its watch file), adapters and GGUF files (`checkpoints/cleanup/`; `merged/` and bf16 GGUFs can be rebuilt with export.py; `smoke3/` is a 3-step test and can go), llama.cpp (`downloads/llama.cpp/b11960/`).
- `.git/info/exclude` hides every `*.md` except README/CONTRIBUTING, so new docs need `git add -f <file>`.

## Ideas (not scheduled)

- Cheap training tests: more LoRA dropout; a heavier loss weight on the words that change.
- Let the trainer resume after a restart (save the optimizer state with each check).
- Show the raw transcript in the preview while the clean-up runs.
- An "Exact" clean-up level that skips the model completely.

## Earlier

(Move old "Done" lines here, one dated line each.)
