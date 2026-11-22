# Session log (newest first)

## 2026-11-22 [Face-gesture]: Design agreed; task list for the face-gesture layer (deliverable 3)
Branch: `eyes-face-gestures` (worktree `D:\CODING PROJ\Diseyeble-face`); records on `voice-cleanup-app`.
**Done:**
- Khang accepted D1 to D9 ("Accept all, do tasks"); recorded in DECISIONS 2026-11-22.
- Wrote `docs/plans/face_behaviours_tasks.md`: 25 tasks in phases 0 to 5, each with an owner, a "done when" test and a guessed effort, and the order they depend on.
**Measured:** nothing.
**Decisions:** the face-gesture layer as designed (D1 to D9).
**Still running:** nothing.
**Problems / blockers:** Khoi's `iable_eye` still unknown (task 0.1); phase 3 needs the Dictation chat for `mode_manager.py`.
**Next:** Khang's OK, then phase 1 (signals, strength, sensitivity, trigger, modes, guards, simulator, profile file), all camera-free.
**Khang must do by hand:** OK the task list; ask Khoi about `iable_eye` (0.1).
**For Cowork:** none new.

## 2026-11-22 [Face-gesture]: Design of the face-gesture layer (deliverable 2)
Branch: `eyes-face-gestures` (worktree `D:\CODING PROJ\Diseyeble-face`); records on `voice-cleanup-app`.
**Done:**
- Khang OK'd the survey ("Yes, write the design").
- Read how `engine/eyes` wires gestures today: `service.py` freezes the cursor during a brow raise, a held blink or an open mouth (100 ms lookback, 2 s limit); `mode_manager.py` maps `Gesture.type` to intents per mode; `os_input.py` has whole clicks and key presses but no button-down/up.
- Wrote `docs/plans/face_behaviours_design.md`: signals per behaviour, strength normalised to the user's calibrated range, one trigger state machine with release level, hold, cooldown and gap rules, sensitivity 1 to 10, pulse/hold/toggle with safe release, actions and bindings, conflict rules (cursor freeze, gesture vs gesture, talking, laughing, eating), global pause, the simulator, replay and recording plan with metrics and a proposed bar, all settings with defaults and reasons, and the files it touches.
**Measured:** nothing.
**Decisions:** none yet; D1 to D9 in the design wait for Khang (DECISIONS, Proposed).
**Still running:** nothing.
**Problems / blockers:** Khoi's `iable_eye` still not found; `mode_manager.py` is the Dictation chat's and `types.py` is a contract, so both need agreement before code.
**Next:** Khang's answers to D1 to D9, then deliverable 3 (task list with owners and tests).
**Khang must do by hand:** read the design and answer D1 to D9.
**For Cowork:** none new.

## 2026-11-22 [Face-gesture]: Survey of face-gesture variables (deliverable 1 of the face-gesture layer)
Branch: `eyes-face-gestures` in the worktree `D:\CODING PROJ\Diseyeble-face` (Khang chose a new branch in this repo); the records are on `voice-cleanup-app` in the main folder.
**Done:**
- Wrote `docs/plans/face_behaviours_survey.md`: SensePilot's variables (name, what it does, range, step, default, what it interacts with, where seen), Project Gameface's Windows and Android settings with their code defaults, ChromeOS Face control's settings, MediaPipe Face Landmarker's outputs and weak blendshapes, and what `engine/eyes` already does.
- SensePilot only from public material: its setup manual PDF (2026-04-15), two of its YouTube videos (transcripts), FAQ, the Can I Play That? preview and a reseller page. Nothing installed, unpacked or copied.
- Gameface cloned shallowly into the scratchpad to read its code (not in the repo).
- Licences recorded in DECISIONS (2026-11-22).
**Measured:** nothing. All numbers are from the sources.
**Decisions:** licences for the face-gesture layer (DECISIONS 2026-11-22).
**Still running:** nothing.
**Problems / blockers:** Khoi's `iable_eye` isn't on this PC or in any branch. MediaPipe's tongueOut is missing from its output, and cheekPuff, jaw left/right and dimples are reported near 0 by others: tongue out and cheek puff need our own measurement before they're offered.
**Next:** Khang's OK on the survey, then deliverable 2 (design doc: parameters, defaults, reasons).
**Khang must do by hand:** read the survey; say where `iable_eye` is; decide whether to run the SensePilot trial (checklist at the end of the survey) after Cowork checks its terms.
**For Cowork:** SensePilot's terms for a trial used to study a competitor; patents on gesture-to-key mapping (COWORK_NOTES).

## 2026-11-22 11:50 [Dictation]: The whole app on Khang's PC; blink only clicks, text only into a text box, patient pause
Branch: voice-cleanup-app · Commits: this entry's commit and the ones before it today
**Done:**
- Ran the whole app on Khang's PC as the desktop program (`scripts\run_demo.ps1`): head pointer, gestures, voice, the new dictation and the trained clean-up (big1, llama-server). Switched the PC to stock Whisper (removed `whisper_model` from config.local.json): the old hackathon model garbled clear speech.
- Pushed everything to origin (`voice-cleanup-app`, 50 commits, all under Khang's name).
- Khang's three changes (DECISIONS 2026-11-22): a blink never types the card, only clicks; "yes" types only into a text box, else the card asks for a click on one and types after it; pause 2.5 s by default and a one-time "still on the card" reminder after 60 s of quiet. New `engine/eyes/text_focus.py`, changes in `mode_manager.py`, `config.py` (shared; the clean-up chat was told), `main.py`, the card hint, the help list.
**Measured:** none new. Unit tests: 1,269 pass (10 new in `tests/test_card_placement.py`; one old test that expected a blink to type the card now expects a click). The text-box check answered True with the focus in a terminal on this PC.
**Decisions:** DECISIONS 2026-11-22 (the card: blink, text box, patience).
**Still running:** the app is not running now (restart it with `.\scripts\run_demo.ps1`).
**Problems / blockers:** first run showed the mouth gate getting no readings (every word "-") and one 24 s Whisper call; not looked into yet. The website still says a blink types the card and "pause up to 4 s" (website copy, `ui` route).
**Next:** Khang tries the three changes with mic and camera: dictate, say "yes" with no text box picked, blink on a text box; check a blink no longer types; speak slowly with long pauses.
**Khang must do by hand:** the try above.
**For Cowork:** none.
**Notes from Khang:** "I don't want the fact that blinking is going to drop the line ... ask them to use the cursor to move to the text box ... between noises, as long as it's not more than 1 minute, then continue to listen."

## 2026-11-22 11:00 [Clean-up model]: New record-keeping rule taken in; the two over-editing findings checked
Branch: voice-cleanup-app · Commits: see `git log` (this entry's files are committed with it)
**Done:**
- Re-read CLAUDE.md, STATUS.md and COWORK_NOTES.md (no /wrap-up now; records updated after every finished piece of work). Nothing in COWORK_NOTES is addressed to this chat.
- Brought my STATUS lines up to date: the hearings are done (by the Dictation chat, 2026-11-19; DECISIONS 2026-11-19), GPU line, and a new Next 5c.
- Checked the Dictation chat's two findings (STATUS Next 5b) against its row files, no GPU. Both hold. The model is sure about most of the changes, so the unsure flag doesn't cover them.
**Measured** (Khang's PC; big1 + other hearings, greedy; rows `checkpoints/hearings/libri-dev.cleanup-greedy.jsonl` and `personal-test.cleanup-greedy.jsonl`; my counts from a scratch script that compares words ignoring case and punctuation):
- LibriSpeech dev, 298 clear sentences, 6,191 words: words wrong 1.50% before the clean-up, 3.28% after (3.21% with hearings). 83 sentences changed: 63 got more word errors, 7 fewer. 44 of the 83 were marked unsure, 39 were not.
- Khang's 29 recorded phrases, 240 words: 10.0% before, 13.33% after (11.67% with hearings). 8 changed: 6 worse, 2 better; 6 of the 8 not marked unsure. "Refresh." -> "Please refresh the page." is not marked unsure.
- The Dictation chat counted 76 changed sentences, not 83: a slightly different word comparison. The direction is the same.
- Then ran Next 5c's offline guard test (Khang said yes, 11:00): `eval/dictation_cleanup/guard_replay.py` (+ `tests/test_guard_replay.py`, 12 tests), review `eval/dictation_cleanup/reviews/2026-11-22_guard_replay.md`. It replays saved big1 answers through guard variants: dev 67, test 2,500, and Whisper rows (LibriSpeech dev, Khang's 29 phrases, TORGO dev).
**Measured (the guard test; Khang's PC, no GPU; big1 answers already collected):**
- "No new content word, input up to 6 words": Khang's phrases 11.67% -> 10.00% words wrong (Whisper alone 10.00%): 2 phrases fixed, none worse. Dev, TORGO and LibriSpeech unchanged. Test: 6 more answers blocked, 1 more equal to what was meant, 8 right answers lost either way, clean controls changed 4 -> 3.
- Same rule on any length: blocks 104 test answers, loses 53 right ones. Dropped-run of 3 / 2 content words: loses 44 / 268 right ones, for 1 / 8 better LibriSpeech rows.
- LibriSpeech with hearings: 114 word edits, 112 in sentences with no filler or repeat; 54 swap a content word (many spelling variants the reference counts wrong), 18 add and 16 drop a small word, 14 swap a small word, 12 drop content words. Not guardable from the text alone.
- 1,252 tests pass, compile check clean.
- Khang OK'd the rule (13:00); added it to `check_cleanup` in the shared `rewrite.py` (`gained_content_words`, `CLEANUP_SHORT_INPUT = 6`, optional `hearings` argument passed by `Rewriter._check` and `_readings`). A first version also blocked swapped words and broke the real-word fixes in the unit tests ("pitcher" -> "picture"), so it now counts only a net gain of content words. Replay with the shipped rule: Khang's 29 phrases 11.67% -> 10.00%; on the 2,500 test 1 more answer blocked, no score change; dev, TORGO, LibriSpeech unchanged. 7 new tests in `tests/test_cleanup_model.py`; 1,259 tests pass, compile check clean. I messaged the Dictation chat before editing (delivery not confirmed).
**Decisions:** the short-phrase rule in `check_cleanup`, agreed by Khang 2026-11-22 (DECISIONS 2026-11-22).
**Still running:** nothing. At 10:53 the GPU had 2.9 GB in use at 10%, not from this chat.
**Problems / blockers:** the clean-up changes clear and short text, and the guard lets the added words through. This is the same kind of over-editing as the `drop_small_words` finding of 2026-11-05.
**Next:**
1. Recheck the rule on Hung's hidden phrases (short ones especially).
2. If the over-editing of clear text matters: more clear and short phrases in the model's data (Hung's phrases will help), then a one-epoch run; or Whisper's word confidence in the rewrite request (contract change).
3. Hung's 35 hidden-test phrases, when they arrive.
**Khang must do by hand:** try short phrases with the mic ("refresh", "close tab") and see that nothing is added; the other items from 2026-11-12 are still open (share Hung's doc, candidate yes or no, mic try-out of "Check this", push).
**For Cowork:** none.

## 2026-11-22 10:45 [Dictation]: Own-voice adaptation measured; command rescue and end-of-turn probes; records now kept as we go
Branch: voice-cleanup-app · Commits: see `git log` (this entry's files are committed with it)
**Done:**
- Ran the personal voice experiment once the GPU was free (2026-11-19 night, again 2026-11-20 with 10 and 20 passes). The first two runs were identical: the trainer stops at 5 passes (55 steps for 132 clips).
- Closed-set listening for commands: a tool in `engine/voice/finetune/hearing_eval.py` (`--closed-test`, `--closed-summary`) scores each allowed word against the audio with Whisper's decoder (CTranslate2 `align()`). Not in the app.
- `engine/voice/finetune/turn_eval.py`: asks Smart Turn v3 (Pipecat, BSD-2-Clause, 8 MB ONNX, CPU) whether the speaker has finished, at each pause. Only a 60-clip probe so far; model in `downloads/smart-turn/` (git-ignored).
- New post box `docs/COWORK_NOTES.md`; CLAUDE.md now says to update the records after every finished piece of work (no /wrap-up).
**Measured:** Khang's PC, RTX 5080, stock large-v3-turbo, shared with DigiErhu during the command run:
- Own voice (30 test clips, 241 words), words wrong: stock 25 (10.37%); 5 passes 21 (8.71%); 10 passes 19 (7.88%); 20 passes 22 (9.13%). LibriSpeech dev 1.45% -> 1.36 to 1.55%. Half the gain is names and number spelling; "Open Chrome" -> "Open crawl" in every adapted run.
- Command rescue, TORGO's 44 command-like words (240 clips from 6 tuning speakers, 86 from the test speakers F04 and M01, not used to choose the rule), right commands: app today 45.0% / 60.5%; the app plus a rescue when it would do nothing (best word within 0 of Whisper's own reading, 2 ahead of the runner-up; `closed_pick(r, 0, 2)`) 60.0% / 68.6%; wrong command picked 0 -> 1 and 1 -> 2. Other single words trigger a command 2.8% -> 6.0% (tune) and 3.2% -> 6.1% (test); sentences never. Scoring 44 words takes p50 290 to 350 ms. The free reading alone gets 58.8% / 70.9%, so part of the gain is just keeping what the app's noise filter drops.
- Smart Turn, 60-clip probe (10 sentences): "finished" median 0.127 at pauses inside speech and 0.242 at the true end, so it separates them poorly on this speech. Too few clips to conclude; 50 ms per question on the CPU.
**Decisions:** no onboarding step for adapting Whisper to one voice yet (DECISIONS 2026-11-22).
**Still running:** none (GPU free 2026-11-20 09:56).
**Problems / blockers:** none. Not tried with mic and camera: everything from 2026-11-19 (checklist in `docs/dictation_learning_from_wispr.md`).
**Next:**
1. Command rescue, if Khang wants it: wire it for yes or no while a card waits, and "resume" while paused (small word sets), shown as "Heard: ...".
2. Smart Turn on the full TORGO dev set before judging it.
3. Khang: mic and camera checklist; the pause setting (2.5 s proposed); push approval (about 40 commits unpushed); the Speech Accessibility Project proposal.
**Khang must do by hand:** the mic and camera checklist; decide the 2.5 s pause; approve the push.
**For Cowork:** see `docs/COWORK_NOTES.md`, "Code to Cowork".
**Notes from Khang:** "let's not use the wrap up. You just lock everything, update everything every single time we finish something" (2026-11-22).

## 2026-11-12 15:01 [Dictation]: Voice fixes, spell-it, the word list, and the dysarthria simulator (stopped: it fails)
Branch: voice-cleanup-app · Commits: 8fd59f3 voice: fix words on the dictation card by voice; 7ac3a8f docs: add the Dictation chat to Code chats now; b71333c voice: spell a word letter by letter, and a personal word list Whisper leans toward; c33c11b voice: dysarthria simulator experiment (kept as a record; it failed the bar); 7cde0e0 docs: stop the dysarthria simulator; plus this wrap-up. All but the wrap-up pushed to origin (Khang's OK) · Uncommitted: engine/capture/, tests/test_capture.py (draft, untracked on purpose)
**Done:**
- Fix words on the card by voice ("change X to Y", "delete X", "scratch that"); spell a word ("spell H A T H A W A Y", "capital", "double", "all caps"); a personal word list ("add <word> to my words") that Whisper uses as hotwords from the next sentence.
- Searched for dysarthric speech data we may sell with: none open. Drafted the Speech Accessibility Project proposal (Claude Doc https://claude.ai/artifact/SdxuomVgQfoY3paCui9Bzv, blanks for Khang).
- Built a dysarthria simulator on LibriSpeech (CC BY 4.0), made it harder (v2), and ran the overnight experiment: simulated vs clean training, scored on TORGO test speakers only.
**Measured:** Khang's PC, RTX 5080, stock large-v3-turbo / LoRA r32 2,000 steps:
- Simulator v2, stock Whisper, % words wrong on 120 dev-clean clips: clean 2.2, mild 4.5, moderate 20.0, severe 48.4 (v1: 2.3, 2.7, 6.7). 200 other clips: 4.8, 17.2, 50.1.
- TORGO F04+M01 (801 clips), % words wrong: stock 41.8; clean control 39.3; simulated 44.5 (best-on-dev step 44.0). M01: 71.9 stock, 79.9 simulated. LibriSpeech test-clean: 1.92 stock, 1.81 simulated. Training 2.26 s a step.
- Unit tests: 1,190 pass.
**Decisions:** simulator v2 targets (2026-11-11); simulator stopped, get real dysarthric speech instead (2026-11-12). Both in DECISIONS.md.
**Still running:** nothing. GPU free.
**Problems / blockers:** none open. On the way: faster-whisper needed torch loaded first for cuBLAS (fixed in evaluate.py).
**Next:**
1. Khang sends the Speech Accessibility Project proposal; plan consented recordings of users.
2. Next 5b: measure Whisper's top-3 hearings (how often they hold the right words, and the cost) before wiring them in.
**Khang must do by hand:** try with the mic: "change X to Y", "spell H A T H A W A Y", "add that to my words" then say the name again; fill in and send the proposal; OK to delete the old simulator clips and converted test models (disk only); push this wrap-up.
**For Cowork:** how to record consented users legally (consent form, storage, what a tester agrees to); whether scoring on TORGO as a company fits its licence (for the lawyer).
**Notes from Khang:** none

## 2026-11-12 15:00 [Clean-up model]: Unsure flag in the app, Whisper's other hearings planned, Hung's data task
Branch: voice-cleanup-app · Commits: c8990c1 eval: the model's own word confidence flags most of its wrong guesses; d115eb9 voice: the trained clean-up marks its answer unsure when a content word is under 0.8; e78a89a voice: the clean-up passes Whisper's other hearings from the rewrite request to the model; f26eafe docs: Hung's task is now writing training phrases himself; 2c19793 docs: Hung writes the hidden test, 500 training phrases and word lists by hand; plus this wrap-up. All but this wrap-up are on origin · Uncommitted: engine/capture/, tests/test_capture.py (draft, untracked on purpose)
**Done:**
- Gave Khang the `/start` prompt for the new Dictation chat (it exists now).
- Measured the model's own word confidence as an unsure flag, then built it into the app: `CleanupModel.clean()` asks llama-server for word probabilities, and the card says "Check this" when a content word is under 0.8.
- Found that the app never sent Whisper's other hearings to the model. Wrote the plan with Khang's OK (`docs/cleanup_hearings_plan.md`) and did the clean-up side: an optional "hearings" field in `topics.py` (inside the engine only) and `rewrite.py` passing it to the model.
- Wrote Hung's task as a Claude Docs doc and revised it twice with Khang: https://claude.ai/artifact/LsBqWiHwbnyAPRqLgkYatN
- Pushed the 6 commits after 4577341 with Khang's OK.
**Measured (Khang's PC, RTX 5080, llama.cpp b11960 Q8_0, greedy):**
- All 54 of big3s16lr epoch 1's wrong answers on the 2,500 test cases come out as sure.
- A 0.8 cut-off on content words catches 33 of them, including 18 of the 20 that change the meaning. It flags 5.3% of right answers (big1: 5.0%), and 16 to 21% on dev.
- The probabilities cost about 23 ms a sentence. The runner-up word is right for only 8 of 36.
- The app's own code path on the 67 dev cases gives the same answers and flags as the eval, 67 of 67 (median request 313 ms).
- Without other hearings (big1, dev): 7 cases that come out unsure with them come out sure, and 6 answers change.
- 1,185 tests pass.
**Decisions:** all recorded in DECISIONS.md, 2026-11-11.
- The unsure flag, at 0.8.
- Other hearings go to the clean-up only after a measurement says they help.
- Hung writes the hidden test (35), 500 training phrases and a word list per group, by hand, with no AI tools and no outreach.
**Still running:** nothing (GPU 656 MiB, 6% at 15:01).
**Problems / blockers:**
- Messages to the Dictation chat were held and expired undelivered twice, probably because the two chats run in different permission modes. The hand-over is in `docs/cleanup_hearings_plan.md` instead.
- D: is a USB drive. The first llama-server start once timed out at the app's 120 s limit.
**Next:**
1. Dictation chat: step 0 of the hearings plan (Whisper's top 3 hearings and their cost). Then the clean-up chat compares the clean-up with and without them.
2. When Hung's 35 hidden-test phrases come in: turn them into cases and run the candidate once.
3. Turn Hung's phrases into training data, and test them with a one-epoch run.
**Khang must do by hand:**
- Share Hung's doc with him.
- Start the Dictation chat on the hearings plan.
- Yes or no to big3s16lr epoch 1 as the candidate.
- Try the trained model with camera and mic, and check that "Check this" shows.
- Push this wrap-up.
**For Cowork:** none
**Notes from Khang:** none

## 2026-11-06 09:10 [Clean-up model]: Pushed the night's results; the GPU is now busy with another project
Branch: voice-cleanup-app · Commits: none besides this wrap-up; pushed 343d50a, 72268c2, 7a7c5dc and 4577341 to origin (Khang's OK) · Uncommitted: engine/capture/, tests/test_capture.py (draft, untracked on purpose)
**Done:** pushed the night's four commits to origin `voice-cleanup-app` (616427f..4577341): the big3lr review, the review of the one-epoch and 5e-5 runs, the slides brief with all results, and the 05:22 wrap-up. Nothing went to `public`.
**Measured:** none new (the night's numbers are in the 05:22 entry).
**Decisions:** none
**Still running:** nothing from this repo. At 09:08 the GPU had 9.9 GB in use at 75%: a training run from another project (DigiErhu, its own folder, started 07:23, 150,000 steps). Not ours; check `nvidia-smi` before any GPU job here.
**Problems / blockers:** while DigiErhu trains, the 16 GB card has no room for a clean-up run (ours peaks at 9.7 GB).
**Next:**
1. Khang: yes or no to big3s16lr epoch 1 as the candidate; then the hidden test (about 35 new phrases).
2. A plan for new sentences toward 100,000 (source, licence, cost), tested with one-epoch runs.
3. Try the trained model in the demo with camera and mic.
**Khang must do by hand:** push this wrap-up; yes or no to the candidate; the hidden-test phrases.
**For Cowork:** build the presentation from `docs/cleanup_progress_for_slides.md` (all results in); research commercially usable sources of new, varied dictation-style sentences.
**Notes from Khang:** none

## 2026-11-06 05:22 [Clean-up model]: The night's three runs scored; the training settings are settled
Branch: voice-cleanup-app · Commits: 72268c2 eval: big3lr epoch 1 ties big1 but doesn't beat big3s16lr; 7a7c5dc eval: one epoch and learning rate 5e-5 both tie big3s16lr; plus this wrap-up. None pushed (616427f and earlier are on origin; 343d50a isn't either) · Uncommitted: engine/capture/, tests/test_capture.py (draft, untracked on purpose)
**Done:**
- Scored big3lr epoch 1 and reviewed it (`eval/dictation_cleanup/reviews/2026-11-05_big3lr.md`).
- Night chain 2's other two runs refused to start at 23:41: a game held about 5 to 7 GB of the GPU. Chain 3 (23:46) waited for the GPU to be free 3 minutes in a row; it was free at 00:03, and the chain ended at 05:13.
- Reviewed big3s16lr1ep and big3s16lr5 (`reviews/2026-11-06_big3s16lr1ep_and_lr5.md`).
- Updated the slides brief (all three night runs in), the explainer, STATUS, DECISIONS.
**Measured:** Khang's PC, RTX 5080, llama.cpp b11960 Q8_0 greedy, prompt t1_short, big1_test (2,500) and dev (67):
- big3lr epoch 1 (24,000 examples): test 2,439, fair 2,301 cases 2,275, clean changed 7, dev 65. Fair head to head: 3-9 against big3s16lr epoch 1, 28-3 against big3.
- big3s16lr1ep (one epoch, 83.8 min): test 2,445, fair 2,280, clean changed 3, silent guesses 1, dev 62. Loss at the end: training 0.0034, held-back 0.0120, test cases 0.0170. Fair head to head: 4-5 against big3s16lr epoch 1.
- big3s16lr5 (5e-5, 168.0 min): epoch 1 test 2,447, fair 2,281, dev 64, test-case loss 0.0169. Final: 2,447, fair 2,283, dev 63, invented flagged 9, one number as digits, test-case loss 0.0188. Fair head to head: 4-2 against big3s16lr final (p 0.69).
- Latency p50 about 285 ms on every run.
- The app's guard (`check_cleanup`) blocks the dropped "not" (S29), "seventeen" for "seventy", "Alex" and "Hawthorne". It doesn't block "2.1" (same meaning).
**Decisions:**
- The training settings are settled (rank 16, 1e-4; one epoch is enough for experiments); the candidate proposal is unchanged (in DECISIONS.md).
- More copies of the same data don't help even at the new settings (DECISIONS.md, 2026-11-05).
**Still running:** nothing. The GPU is free.
**Problems / blockers:**
- A game blocked the GPU from 23:41 to 00:03. The chain now waits, so nothing was lost.
- big3s16lr5 took 168 min vs big3s16lr's 150 for the same steps (cause not checked).
- The last few test points are a coin flip on rare words (heartworm, prednisone, AFO). Settings can't fix that.
**Next:**
1. Khang: yes or no to big3s16lr epoch 1 as the candidate; then the hidden test (about 35 new phrases from Khang or Hung).
2. Plan new sentences toward 100,000 (source, licence, cost), and test them with one-epoch runs.
3. Optional, low value: big3lr1ep (24,000 in one epoch, about 4 h overnight).
**Khang must do by hand:** push (343d50a, 72268c2, 7a7c5dc and this wrap-up); yes or no to the candidate; the hidden-test phrases.
**For Cowork:**
- Build the presentation from `docs/cleanup_progress_for_slides.md` (all results are in).
- Research sources of new, varied sentences we can train on commercially (dictation-style, with medical and disability words).
**Notes from Khang:** asked for this wrap-up to run after the night's runs finished.

## 2026-11-05 23:30 [Clean-up model]: Pushed the two-chat rules and the last wrap-up; night chain 2 running
Branch: voice-cleanup-app · Commits: none besides this wrap-up; pushed 5fe1afa and 616427f to origin (Khang's OK) · Uncommitted: engine/capture/, tests/test_capture.py (draft, untracked on purpose)
**Done:** pushed 5fe1afa (two-chat rules) and 616427f (the 23:20 wrap-up and the slides brief) to origin. Fixed the times in the 23:20 entry, STATUS and the slides brief (they said 23:35 and 23:30; the commit was at 23:20).
**Measured:** none new (the big3lr loss checks are in the 23:20 entry).
**Decisions:** none
**Still running:** night chain 2 (started 23:16, expected done about 05:00 on 11-06). At 23:30 it was scoring big3lr epoch 1 on the 2,500 test cases (1,000 done, about 0.35 s each); then big3s16lr1ep, then big3s16lr5. Output and step log as in the 23:20 entry. A watcher wakes this chat when the chain ends; it then writes the reviews and fills in the slides brief.
**Problems / blockers:** none new. Another Claude session ("DigiErhu physics synthesizer setup") started on this PC at 23:29; by its name it looks like another project. If it uses the GPU before 05:00, training slows down or runs out of memory.
**Next:**
1. Read tonight's results; write the reviews; update the slides brief, STATUS and DECISIONS.
2. Khang picks the candidate (big3s16lr epoch 1 is proposed); then the hidden test runs once on it.
3. Proposed next run: big3lr1ep (big3's 24,000, one epoch, about 4 h).
**Khang must do by hand:** keep the PC on and the GPU free until about 05:00 (no installers or restarts, nothing GPU-heavy in the other session); push this wrap-up; yes or no to big3s16lr epoch 1 as the candidate.
**For Cowork:** build the presentation from `docs/cleanup_progress_for_slides.md`; its last section says whether tonight's results are in.
**Notes from Khang:** none

## 2026-11-05 23:20 [Clean-up model]: Two Code chats set up; a PC restart stopped the night chain; brief for slides
Branch: voice-cleanup-app · Commits: 5fe1afa docs: rules for two Code chats; plus this wrap-up. Neither pushed (e8f4804 and earlier are on origin) · Uncommitted: engine/capture/, tests/test_capture.py (draft, untracked on purpose)
**Done:**
- Pushed the five commits up to e8f4804 to origin (Khang's OK).
- Two Code chats in one folder: rules in CLAUDE.md ("Two Code chats at once"), "Code chats now" in STATUS.md; /start checks for the other chat (`ListAgents`) and adds its own line; /wrap-up edits only its own lines and puts the chat's name in the title. This chat is renamed "Clean-up model". The dictation chat isn't started yet.
- Found why the night chain stopped: at 22:27 the Kaspersky installer in Downloads restarted Windows (System log event 1074, "restart", from Khang's account). big3lr died at step 3,163 of 5,698; its epoch-1 adapter (saved 21:56) survived. Not a fault in our code.
- Restarted the chain at 23:16 (see Still running).
- `docs/cleanup_progress_for_slides.md`: the clean-up story with measured numbers, for Cowork's presentation.
**Measured:** big3lr loss checks (big3's 24,000, rank 16, lr 1e-4; training / held-back sentences / 400 test cases), Khang's PC, RTX 5080: step 500 0.0117/0.0125/0.0212; 1,000 0.0095/0.0090/0.0204; 1,500 0.0046/0.0094/0.0173; 2,000 0.0066/0.0130/0.0235; 2,500 0.0036/0.0083/0.0176; 2,849 (end of epoch 1) 0.0022/0.0109/0.0213; 3,000 0.0011/0.0076/0.0218. At 1,500 and 2,500 the test-case loss is the lowest of any run (big3s16lr's best 0.0187), but it swings by up to 0.006 between checks, and the epoch-1 save landed on a high point. About 4.7 s per step.
**Decisions:** two Code chats may work in this folder at once (one branch, own files, one GPU job, live messages by `SendMessage`); in DECISIONS.md. Tonight scores big3lr's saved epoch 1 instead of re-training it: the trainer can't resume, and a full re-run (about 7.5 h) would leave no time for the other two runs.
**Still running:** night chain 2, started 23:16, expected done about 05:00 on 11-06: A big3lr epoch 1 (export and score only), B big3s16lr1ep (8,000, one epoch, about 1.7 h with scoring), C big3s16lr5 (8,000, learning rate 5e-5, 2 epochs, about 3.3 h with scoring). Output: `checkpoints/cleanup/<name>/`, scores in `eval/dictation_cleanup/runs/<date>_<name>-*`; step log `chain_night2/steps.txt` in this chat's scratchpad. A watcher wakes this chat when it ends; it then writes the reviews and fills in the slides brief.
**Problems / blockers:** big3lr's second epoch never ran (4.2 h of GPU gave only its epoch-1 adapter). Windows Update's active hours end at 02:00, so Windows could restart for updates after that (no restart is pending now). Results come about 05:00: a presentation made before then won't have them.
**Next:**
1. Read tonight's results; write the reviews; update the slides brief, STATUS and DECISIONS.
2. Khang picks the candidate (big3s16lr epoch 1 is proposed); then the hidden test runs once on it.
3. Proposed next run: big3lr1ep (big3's 24,000, one epoch, about 4 h).
**Khang must do by hand:** no installers or restarts while a chain runs (pause Windows updates for the night if you can); push 5fe1afa and this wrap-up; yes or no to big3s16lr epoch 1 as the candidate; start the dictation chat in this same folder (no worktree) and run /start there.
**For Cowork:** build the presentation from `docs/cleanup_progress_for_slides.md` (and `docs/cleanup_model_explained.md` for how the network works); its last section says whether tonight's results are in. Still open: where many new sentences come from, the lawyer list.
**Notes from Khang:** none

## 2026-11-05 18:25: Clean-up training: why the newer runs lost (rank 32, learning rate); first run to beat big1
Branch: voice-cleanup-app · Commits: c4ab513 big3s review; 6c94b8b big3 review; ed90cef big3s16 review; 3b6750f big3s16lr review; plus this wrap-up. None pushed · Uncommitted: engine/capture/, tests/test_capture.py (draft, untracked on purpose)
**Done:**
- Four training runs, each one change from an earlier one, each with loss checks every 500 steps on held-back whole sentences and on 400 test cases never trained on: big3s (8,000 of big3's data, rank 32), big3 (all 24,000, rank 32), big3s16 (big3s at rank 16), big3s16lr (big3s16 at learning rate 1e-4). Reviews: `eval/dictation_cleanup/reviews/2026-11-04_big3s.md`, `2026-11-05_big3.md`, `2026-11-05_big3s16.md`, `2026-11-05_big3s16lr.md`.
- Found why clean sentences get a "the" added: broken examples delete small words (`noise.drop_small_words`, chance 0.6) and the answer puts them back. It shows at rank 32, hardly at rank 16.
- Explainer updated: `docs/cleanup_model_explained.md`.
**Measured:** Khang's PC, RTX 5080, llama.cpp b11960 Q8_0, greedy, automatic scoring (not hand-checked). Test = big1_test.csv (2,500); fair = the 2,301 without a term only big1 trained on; loss = dropout off, at the end:
- big1 (reference; test and dev scored on 11-03, fair split and loss measured now): test 2,441, fair 2,275, clean changed 4 of 378, dev 65 of 67, test-case loss 0.0190.
- big3s (rank 32, 2e-4, 175 min): 2,418, 2,262, 7, 64; test-case loss 0.0208.
- big3 (rank 32, 3x the data, 428 min): 2,395, 2,250, 15, 63; held-back sentences 0.0067, but test cases flat after step 500 (0.0279).
- big3s16 (rank 16, 182 min): 2,432, 2,277, 4, 65; beats big3s 17 to 2 on the fair cases; held-back 0.0133 (big3s 0.0170).
- big3s16lr (rank 16, 1e-4, 150 min): epoch 1 2,446, final 2,444; fair 2,281 (beats big1 10 to 4); terms no run trained on 89 of 94 (big1 84); invented flagged 7 (big1 8); clean 4; dev 65. Test-case loss 0.0187 at step 1,000 (lowest measured), 0.0199 at the end.
- big2's test-case loss rose every epoch (0.0324, 0.0339, 0.0361): textbook overfitting.
**Decisions:** big1 stays the app's model; rank 16 from now on; learning rate 1e-4 from now on (passed as `--lr 1e-4`; the trainer default is still 2e-4). Proposed, needs Khang: big3s16lr epoch 1 as the new candidate and app model file. All in DECISIONS.md.
**Still running:** a night chain started 18:12, one run after another, each one change from big3s16lr: A big3lr (big3's 24,000, about 8 h), B big3s16lr1ep (one epoch, about 1.6 h), C big3s16lr5 (learning rate 5e-5, about 3 h). Expected done about 07:00 on 2026-11-06. A failed run is logged and the next one starts. Output: `checkpoints/cleanup/<name>/` (log.jsonl has the loss checks), scores in `eval/dictation_cleanup/runs/<date>_<name>-*`; the step log is in this session's scratchpad (`chain_night/steps.txt`).
**Problems / blockers:** "three times the data made it worse" was measured at the old settings only; run A re-tests it. big3s16lr still makes confident wrong guesses on two medical words ("afo" -> foam, "pred knee zone" -> physical therapy), not marked unsure. Each run keeps about 40 GB on disk (merged weights and bf16 GGUFs can be rebuilt and deleted). Dev is tuned on and the holdout is spent: only the hidden set gives a fair number.
**Next:**
1. Read the night chain: does more data help at the new settings (A), is one epoch enough (B), is 5e-5 better (C).
2. Khang picks the candidate; then the hidden test runs once on it.
3. A plan for many more different sentences toward 100,000.
**Khang must do by hand:** push the five commits; say yes or no to big3s16lr epoch 1 as the candidate; write the hidden test phrases; try the trained model in the demo with camera and mic.
**For Cowork:** where many new, different sentences come from (now clearly the limit, not the settings), with licences; whether the guard should catch confident medical-word swaps, or the model's token probabilities should mark them unsure; the lawyer list (unchanged).
**Notes from Khang:** none

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
