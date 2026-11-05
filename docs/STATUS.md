# Status

Last updated: 2026-11-05 12:45 (big3s16 results). Session-by-session record: `docs/SESSION_LOG.md`.

## Now

- **Dictation clean-up: our own trained model.** big1, a LoRA adapter on Qwen3.5-4B (Apache 2.0) trained on Claude-written examples, is the candidate and the app's model file (`rewrite_gguf`, big1 Q8_0). It runs in llama-server behind the Rewriter (0.28 s median on the 5080), but only with `rewrite_backend: "llamacpp"`; the default backend is still the old Qwen2.5-3B. Dev full pass 63/67 by Khang's rule; dev is tuned on, so the hidden test decides.
- **Training experiments:** rank 32 was costing us. big3s16 (big3s at rank 16, nothing else changed) beats big3s on every measure and ties big1 on the fair cases, so rank 16 from now on. big2, big3s and big3 were all rank 32; "three times the data made it worse" is untested at rank 16. The model still overfits to our data generator: loss on new data stops improving while training loss keeps falling. More data must be new sentences from new sources (Khang's goal: 100,000 examples). Explainer: `docs/cleanup_model_explained.md`.
- **Rename:** a new name is coming (close to the new repo name `-i-able`, not final). The old-name scan is done; changes wait for the exact name.
- **Head tracking:** paused until Khoi's open-source tracker research is in.

## Done recently

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

- Nothing running.

## Next

1. Training tests at rank 16, one change per run, judged by the test-case loss and the fair split: learning rate 1e-4 on big3s's data (about 3 h); big3's 24,000 at rank 16 (about 7 h, overnight: does more data help once the rank is right?); the "add the" fix in the data (less urgent: no "the" added to clean sentences at rank 16). Then choose big1 or big3s16 before the hidden test.
2. Hidden test: Khang or Hung writes about 35 new phrases (`messy → meant`). Claude turns them into cases and runs the candidate once as it ships (llama.cpp Q8_0, greedy).
3. Try big1 in the demo with camera and mic (`"rewrite_backend": "llamacpp"` in config.local.json, preflight, run_demo); then decide whether llamacpp becomes the default.
4. A plan for many more different sentences (toward 100,000 examples): source, licence, cost.
5. Decide whether the guard rejects numbers from sound-alike words (dev M01 and M02 expect them fixed).
6. Rename, once the exact name is given.
7. Light models for weak laptops (Qwen3.5-2B, 0.8B; Q4_K_M or llama.cpp's CPU build), timed on a low-end laptop.

## Blockers and risks

- Dev is tuned on and the holdout is spent: only the fresh hidden set gives a number to trust. The 2,500-case test comes from the same generator as the training data.
- The test cases were invented, not taken from real users. Real users or a speech-language pathologist should review them later.
- For a lawyer: training on Claude-written sentences (Anthropic's terms forbid it only for competing products; Khang's reading: we don't compete); shipping NVIDIA's CUDA runtime DLLs with llama.cpp; PyAV stays in training only; the licence for new private code; IP agreements with everyone who wrote code.
- Website updates: the live site is served from the public repo, and pushing `main` there would publish the private code. Pick a way to update it. The Pages workflow will also fail in the private repo until it's switched off there.
- Remove the two people who left from the old public repo's collaborators (Khang, on GitHub).

## Uncommitted work on Khang's PC

- `engine/capture/` and `tests/test_capture.py`: unreviewed draft, left untracked. Decide: delete, or keep on a separate branch.
- Not pushed: c4ab513 (big3s review), 6c94b8b (big3 review) and the big3s16 review.
- Git-ignored, only on this PC: training data (`data/cleanup_train/`: big1, big2, big3, big3s, the frozen test and its watch file), adapters and GGUF files (`checkpoints/cleanup/`; `merged/` and bf16 GGUFs can be rebuilt with export.py; `smoke3/` is a 3-step test and can go), llama.cpp (`downloads/llama.cpp/b11960/`).
- `.git/info/exclude` hides every `*.md` except README/CONTRIBUTING, so new docs need `git add -f <file>`.

## Ideas (not scheduled)

- Use the model's own token probabilities (llama.cpp can report them) to mark an answer unsure.
- Cheap training tests: learning rate 1e-4; more LoRA dropout; a heavier loss weight on the words that change.
- Show the raw transcript in the preview while the clean-up runs.
- An "Exact" clean-up level that skips the model completely.

## Earlier

(Move old "Done" lines here, one dated line each.)
