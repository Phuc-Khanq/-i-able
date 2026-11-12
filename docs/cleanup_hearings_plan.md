# Sending Whisper's other hearings to the clean-up model

Plan approved by Khang on 2026-11-11. Written by the Clean-up model chat for the Dictation chat; messages between the
two chats aren't getting through, so this file is the hand-over.

## Why

The trained clean-up model learned, and is scored, with a prompt line "Other hearings" (for example "can you meet me
at the bench" when Whisper heard "bank"). The app always sends "(none)". On the 67 dev cases (big1, 2026-11-11),
without hearings 7 cases that come out unsure with them come out sure, and 6 answers change. Whisper runs greedy
(`BEAM_SIZE = 1` in `engine/voice/dictation.py`), so it has no other hearings today.

## Done (Clean-up model chat)

- `engine/core/topics.py`: TRANSCRIPT and REWRITE_REQUEST have an optional `"hearings"` field (list of str, at most
  2). Inside the engine only: `messages.py`, `types.py` and the website don't change.
- `engine/voice/rewrite.py`: the RewriteWorker passes a request's `"hearings"` to `rewrite_full` / `refine_full`, and
  the trained model gets them as "Other hearings". It drops blank ones, ones equal to the text, and any past 2. The
  old Qwen2.5-3B backend ignores them. Tests: `tests/test_cleanup_model.py` (`TestHearings`).

## Steps for the Dictation chat

**0. Measure first.** Khang's stop rule: if hearings rarely hold the right words, or cost too much time, we don't
wire them in.
- Use clips with known right text already on this PC (the sim set or TORGO dev). Write text only, to a git-ignored
  folder. No audio is written.
- Keep the greedy pass as today, then run a second pass that gives the top 3 hypotheses. faster-whisper 1.2.1
  doesn't return them from `transcribe()`, but its internals can, in this order:
  1. `model.feature_extractor(audio)`
  2. `model.encode(features)`
  3. `model.get_prompt(tokenizer, [], without_timestamps=True, hotwords=<same as the main pass>)`
  4. `model.model.generate(encoder_output, [prompt], beam_size=5, num_hypotheses=3, return_scores=True)`
  5. `tokenizer.decode` on each of `sequences_ids`
- Report:
  - how often greedy is wrong and a hearing has the right words;
  - the extra time per clip (median and p90) on the 5080;
  - one row per clip: reference, greedy text, hearings, word confidences.
- The Clean-up model chat then runs the clean-up on those rows with and without hearings, and compares each to the
  reference.

**1. `dictation.py`.**
- Run the second pass only when some word is under a cut-off chosen in step 0.
- Drop hearings that equal the best after lowercasing and removing punctuation. Keep at most 2.
- Add them to TRANSCRIPT as `"hearings"`, each a full other version of the same utterance.
- Send none when the mouth gate dropped some words (they no longer line up), and none for clips over 30 s.

**2. `mode_manager.py`.**
- When a whole transcript goes onto the preview, remember where it sits and its hearings.
- Any other change forgets them: a voice fix, spell, scratch that, or clauses taken out as commands.
- `_hints()` adds `"hearings"`: up to 2 full versions of the request's `"text"` (the whole preview), each with one
  place swapped, newest first. This shape matches the training data, and it serves both the rewrite and the refine.

## Open

- Should the guard (`check_cleanup`) accept a number that appears only in another hearing ("fifteen" vs "fifty")?
  Khang's call. The plan's default is no; look again after step 0.
- Later: send the model's weak word to the card so it can highlight it (that one does change `messages.py`).
