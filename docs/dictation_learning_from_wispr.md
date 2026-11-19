# What we can learn from Wispr Flow (Dictation chat, 2026-11-19)

Khang asked: why is Wispr Flow so good at dictation, and what can we copy? This is the research, what it means for
[eye]able, and what was built. Sources are Wispr's own pages (read 2026-11-19), listed at the end. We copy ideas, not
code: Wispr is closed source and runs in the cloud; we stay on the user's PC.

## Why Wispr Flow is good, in their own words

1. **It isn't one model, it's a pipeline with a tight time budget.** Speech recognition (ASR), then a fine-tuned LLM
   that cleans and formats, all within 700 ms of the user stopping: ASR under 200 ms, LLM under 200 ms, network under
   200 ms. Their LLM is a fine-tuned Llama on dedicated GPUs (Baseten, TensorRT-LLM). Since September 2026 the ASR is
   their own model, "Canto" (about 2 billion parameters), built for real conditions (quiet speech, noise, accents).
2. **Context does the heavy lifting.** Their CTO: improving English WER from 3% to 2% on hard test sets "hardly made a
   dent", because short phrases and names ("Lauren" or "Laurent") can't be told apart from audio alone. So both the ASR
   and the LLM get context: up to 500 characters before and after the cursor, the app and its category (email, work
   chat, personal chat, document), names on screen, the user's dictation history, their dictionary, and an embedding of
   the user's voice. When audio is too hard, the product should ask the user rather than guess.
3. **It learns from corrections.** "When you correct a spelling, Flow adds it automatically to your personal
   dictionary." A dictionary entry helps recognition and can also be a replacement rule (wrong spelling -> right
   spelling, whole words, longer phrases first). Entries are ranked by how often they're used. Their stated goal: "a
   product that never makes the same mistake twice."
4. **It does the editing the user would do.** Fillers removed; "Backtrack": "Let's do coffee at 2 actually 3" becomes
   "Let's do coffee at 3" (only clear self-corrections); spoken punctuation ("comma", "new line", "new paragraph");
   lists from "one ... two ..."; a style per app category (formal email, casual chat, no final period in messages).
5. **Personal style.** They learn token-level style (dashes or commas, capitals) per user, on the device where possible.
6. **Small things that matter:** whisper mode (quiet speech works), hands-free mode, snippets (a spoken cue expands to
   saved text), a bad or Bluetooth microphone named as the main cause when quality "drops".

## What it means for [eye]able

Our users' speech is much harder than Wispr's (stock Whisper gets 42% of TORGO's dysarthric words wrong, against about
2% on clear speech), so Wispr's main lesson applies even more: **audio alone isn't enough; context and the user's own
history must fill the gaps, and the user must be able to fix things cheaply and never twice.**

| Wispr Flow | [eye]able today | Gap / plan |
| --- | --- | --- |
| ASR + fine-tuned LLM clean-up, < 700 ms | Whisper large-v3-turbo + our trained Qwen3.5-4B clean-up, about 0.13 s + 0.3 s | Similar shape. Fine. |
| Personal dictionary helps recognition | Word list as Whisper hotwords (2026-11-12) | Done. |
| Auto-add words from corrections | none | **Built 2026-11-19** (below). |
| Replacement rules ("never the same mistake twice") | none | **Built 2026-11-19**: a fix made twice is made automatically. |
| ASR gets text before the cursor | Clean-up gets it (`context`); Whisper doesn't | Measured below. |
| Several readings when unsure / ask the user | Card: unsure flag, other readings from the clean-up | Whisper's own other readings: measured below (hearings plan step 0). |
| Names on screen | none | Later: read the focused window's names with UI Automation (in memory only). |
| Backtrack, fillers, per-app style | Clean-up model (the clean-up chat's) | For the clean-up chat: add "actually" self-corrections and per-app style to its training data. |
| Spoken punctuation, new line | none found in commands | Candidate. |
| Whisper mode | AutoGain in `audio_in.py` | Have it. |
| Voice embedding / personalized ASR | none (TORGO can't ship; no commercial dysarthric data) | Real recordings first (SAP proposal, consented users). |

## Built and measured

Numbers from Khang's PC (RTX 5080, stock large-v3-turbo float16), with DigiErhu training on the same GPU at the time,
so the times are higher than on an idle card. Tool: `engine/voice/finetune/hearing_eval.py`; rows (text only) in
`checkpoints/hearings/`. TORGO is only scored here, never trained on.

**1. Learning from fixes (Wispr's auto-add and correction rules).** Commit 3e18697. A fix made by voice and typed is
counted in `data/voice/personal/fixes.txt`; the same fix twice is made automatically from then on, visibly on the card
(changing it back forgets it). Spelled words, and names or terms put in by a fix, join the word list. Tests only; not
yet tried with the mic.

**2. Snippets (Wispr's snippets, made by voice).** Commit d2f96b6. "save this as a snippet called my address", then
"insert my address" (or the cue alone while dictating). Tests only.

**3. Spoken punctuation (Wispr's smart formatting).** Commit 7d10bd9. "comma", "period" / "full stop", "question
mark", "exclamation point", only in clear cases. Tests only.

**4. What Whisper can give the clean-up: beam search and other readings (hearings plan step 0).**

TORGO dev speaker M05 (470 clips: 355 single words, 108 sentences seen in training lists, 7 new sentences), % of words
wrong:

| Whisper pass | All | Sentences | Single words |
| --- | --- | --- | --- |
| The app today (greedy, beam 1) | 53.1 | 26.4 | 113.2 |
| Beam 5, best reading (no timestamps, no fallback, no noise filter) | 45.0 | 23.0 | 95.2 |
| Best of greedy and the top 3 readings (oracle) | 36.2 | 19.1 | 75.0 |
| Greedy with the previous clip's text as a prompt (unrelated text) | 70.5 | 25.5 | 173.6 |

- Single words get WER over 100 because Whisper adds words. The app's pass returned nothing at all for 146 of the
  first 300 single-word clips (its noise filters drop short unsure results); beam search got 79 of 300 single words
  exactly right, against 46 for the app.
- When greedy is wrong, one of the 3 readings is exactly right on 22% of clips and has fewer errors on 38%.
- The extra pass for 3 readings: p50 115 ms, p90 145 ms (GPU shared). Running it only when a word is under 0.5
  confidence runs it on 70% of these clips and catches 88% of the clips a reading improves.

LibriSpeech dev (298 clips of clear speech): greedy 1.50%, beam 5 1.55%, oracle 0.94%, previous-sentence prompt 1.58%.
So beam search doesn't help clear speech, and **the text before the cursor as a Whisper prompt helps neither clear
speech (1.50 -> 1.58) nor dysarthric speech, and badly hurts dysarthric speech when the text is unrelated** (53 -> 71).
Wispr's own ASR is trained to use context; stock Whisper isn't. Not wired in.

**5. Other hearings into the clean-up (built: commit 8a29693).** The trained clean-up (big1 Q8_0, llama-server) on
TORGO dev's 115 sentences, without and with Whisper's other readings as "Other hearings":

| Whisper text given to the clean-up | Whisper alone | Clean-up | Clean-up + hearings | Wrong answers marked unsure |
| --- | --- | --- | --- | --- |
| The app's pass today | 27.0% | 26.9% | 25.3% | 20 of 71 -> 30 of 70 |
| Beam 5 | 24.2% | 24.4% | 23.9% | 21 of 69 -> 30 of 68 |

So the hearings help a little with today's pass and mostly make the card's "Check this" honest: it now shows on 43%
of the clean-up's wrong answers instead of 28% (with 8 right answers marked instead of 2).

**For the clean-up chat:** on clear LibriSpeech book sentences (298) the clean-up takes WER from 1.50% to 3.28%. It
changes the words of 76 sentences (25%) and makes 7 better. Only 8 of the 76 are its job (fillers like "You know",
"Huh", a repeated "enough, enough"); the rest add small words ("the" 8 times, "is" 4), drop them ("and" 8 times) or
change words in 37 sentences ("wants" -> "want", "pravity" -> "depravity", "signalizes" -> "signals", "persisted like
his predecessor" -> "persisted in his predecessor Chrysostom's"). Book prose isn't dictation, but it is the same
over-editing as the 2026-11-05 "drop_small_words" finding. Rows: `checkpoints/hearings/libri-dev.cleanup-greedy.jsonl`.

On Khang's own recordings (the personal pack's 29 test phrases, his mic): Whisper 10.0% wrong, the clean-up 13.3%,
the clean-up with other hearings 11.7% (better on 4, worse on 2). Most of the clean-up's damage is on command phrases
that in the app never reach it, but it shows the clean-up inventing words in short phrases: "refresh" -> "Please
refresh the page.", "paste down" -> "Please paste the document down." (with the hearings: "Page down.", right), "Close
tab." -> "Close the tab.". The guard lets these through. Rows: `checkpoints/hearings/personal-test.cleanup-greedy.jsonl`.

**6. Short unsure words kept when the mouth moved (built: commit 8a29693).** The app's noise filter dropped every
1-to-3-word result under 0.35 confidence. On TORGO dev's 355 single words it left 165 results empty; without it 99
single words come out exactly right instead of 60. Now such a result goes to a dictation card (marked unsure) when the
camera saw the user's mouth moving, never to a command. Without a camera it is dropped as before. On TORGO's held-out
test speakers (F04 and M01, 801 clips) keeping them takes WER from 41.5% to 39.9% and exactly right single words from
221 to 270 (of 612).

**7. Whisper's decoding settings (built: commit b5dc89e).** All with word timestamps, as the mouth gate needs them.
% of words wrong, TORGO dev (M05, 470 clips; chosen here) and LibriSpeech dev (298 clips of clear speech):

| Setting | TORGO dev all | Sentences | Single words | LibriSpeech | Time p50 |
| --- | --- | --- | --- | --- | --- |
| The app before (beam 1, temperature fallback on) | 55.4 | 26.0 | 121.6 | 1.50 | 150 ms |
| Fallback only for repetition loops (`log_prob_threshold` None): **now the app's** | 51.5 | 26.0 | 108.7 | 1.50 | 128 ms |
| No fallback at all (temperature 0) | 51.5 | 26.0 | 108.7 | | 129 ms |
| Beam 5, fallback only for loops | 52.7 | 23.2 | 119.7 | 1.57 | 157 ms |
| Beam 5, no fallback at all | 61.4 | 23.2 | 148.6 | | 158 ms |
| No timestamp tokens (inside transcribe) | 168.7 | 63.8 | 411.8 | | 426 ms |

- When Whisper is unsure (a low average log probability, which is most of the time for a hard voice), faster-whisper's
  default decodes again with random sampling. For dysarthric single words that mostly adds made-up words. Keeping
  Whisper's first answer, while still re-decoding repetition loops, is free on clear speech.
- Beam 5 looked good on the dev speaker's sentences (26.0% -> 23.2%) but not on the held-out speakers (TORGO test
  sentences 27.5% -> 28.4%), so the app stays at beam 1.
- Without the loop check, beam 5 repeats words ("BIT BIT BIT ..."); without timestamp tokens, transcribe() repeats
  whole sentences. Neither is used.
- On Khang's own recordings through his mic (the personal pack's 30 test clips) every setting gives 10.37%: the new
  decoding costs a typical voice nothing and is a little faster (p50 158 ms against 174 ms).

**8. Is the word list safe? (measured; nothing to change now).** A 30-word list the speaker isn't saying (names,
medications, apps), through the app's path. With the old fallback, Whisper wrote list words into TORGO dev's single
words 112 times (WER 118.5% -> 141.0%): the random re-decoding filled short unclear answers from the list. With the new
setting it wrote them 6 times and the list helped (108.7% -> 102.0%; sentences 26.0% -> 25.9%; LibriSpeech 1.50% ->
1.62%).

**9. The front door: what `audio_in.py` loses before Whisper hears anything (measured).** Each TORGO dev clip was
fed through AudioIn as if from the mic (auto gain, Silero VAD, pre-roll, pads; no camera), then the app's pass ran on
what it cut. This first run used the pause setting on Khang's PC, **1.0 s** (`config.local.json`; the default is
1.5 s); at 1.5 s, 4 dev sentences were cut (28.4% wrong) instead of 22:

| | Whole clip | What audio_in cut | No utterance at all | Cut into pieces |
| --- | --- | --- | --- | --- |
| 108 sentences | 26.0% | 32.7% | 0 | 22 |
| 355 single words | 110.1% | 111.0% | 48 (13.5%) | 3 |

So for dysarthric speech the front door costs as much as the decoding fix gains: sentences with long pauses get cut
into pieces that Whisper hears without their context, and some short, weak words never become an utterance.

With the pause setting at 2.5 s no sentence was cut (27.4% on what was cut, against 32.7% at 1.0 s and 28.4% at
1.5 s); 3 s was no better (27.2%). On the held-out test speakers (F04, M01, 801 clips): at 1.5 s, 8 of 184 sentences were cut and 43.9%
of words came out wrong on what audio_in cut (41.5% on whole clips); at 2.5 s none were cut and 41.7%. Proposed in
DECISIONS (not decided): a 2.5 s pause for users who pause mid-sentence.

The missed short words are this one speaker's quiet voice: Silero's threshold 0.35 recovered 5 of them and 0.25
recovered 13 (43 -> 30 never heard; the words-wrong rate on what was cut then matches the whole clip), while the held-out
speakers lost only 2 words at any threshold. The shortest-utterance limit isn't the cause (0.1 s: still 43), nor is
the auto gain's ceiling (32x instead of 8x: still 43). A lower
threshold may let in coughs and room noise and keep utterances from ending in a noisy room, which can't be measured
offline, so the default stays 0.5 and a new setting, `vad_threshold`, lets a quiet speaker try 0.3 with the camera on
(commit 68f8adc).

**10. Learning from fixes, replayed on TORGO dev.** `hearing_eval.py --fix-sim` replays a speaker's clips in recording
order as if the user fixed every wrong word by voice. Learned fixes rarely fired, and at first were wrong more often
than right (2 right, 4 wrong: "that" -> "the", and minimal pairs like "shoot" -> "suit"), because TORGO's word lists
don't repeat a speaker's own vocabulary the way daily use does. Two safety rules came from it (commit 2028fcf): a pair
with a single common word on either side is never learned, and typing a fix's heard word as it was counts against the
fix. After them: 0 right, 1 wrong, WER unchanged. The gain is meant for names and terms people say every day, which
only real use can show.

**11. Whisper's other hearings as card options (commit 431e63d).** When the clean-up gives no other readings, or is
off, the card numbers Whisper's other hearings, so "option two" can fix a misheard word. Counted on what the card would
show (TORGO dev, short unsure words kept, options only when a word is under 0.5): one of the two options is exactly
right on 32 of 328 wrong answers (10%; single words 12%, sentences 3%, where a whole sentence rarely comes out exactly
right). The 22% in step 0 above also counts clips where the app showed nothing; the commit message's 22% should be
read that way. **12. The word list ranked by use (commit 88e173a):** words typed move to the end Whisper gets first.

## Held-out check and what the app does now

TORGO's test speakers (F04, M01, 801 clips) were never used to choose anything above:

| | % of words wrong |
| --- | --- |
| The app's pass before today (fallback on) | 43.1 |
| With the new decoding (no random re-decoding) | 41.5 |
| ...and short unsure words kept (as for a dictation card when the mouth moved) | 39.9 |
| Beam 5 instead (not used) | 42.6 |

Before/after in one line: on held-out dysarthric speech the app's Whisper pass goes from 43.1% to 41.5% of words
wrong, and to 39.9% for what reaches a dictation card, with clear speech unchanged (1.50%), plus the clean-up now gets
Whisper's other hearings and says "Check this" on 43% of its wrong answers instead of 28%.

## How to try it (Khang, with mic and camera)

Start the demo as usual (`.\scripts\run_demo.ps1`), say "start dictation", then:

1. **Short words:** say one quiet word ("water"). It should appear on the card marked unsure, instead of nothing.
2. **Learning from fixes:** dictate "bring me a waiter", say "change waiter to water", then "yes". Do it again; the
   status says "waiter → water from now on". The third time, "waiter" comes out as "water" on its own. "change water
   to waiter" undoes it.
3. **Spelled words join your list:** "call doctor", "spell capital H A T H A W A Y", "yes". The status says
   "Hathaway" is in your words; dictate "Hathaway" in a sentence and it should be spelled right.
4. **Snippets:** dictate your address, fix it, say "save this as a snippet called my address", "yes". Later say
   "insert my address" (or "my address" alone) and "yes".
5. **Punctuation:** "are you coming comma mom question mark" should show "are you coming, mom?".
6. **Options:** when a word is unsure, the card may list other readings; "option two" takes one.
7. **Pause setting:** this PC's `config.local.json` sets 1.0 s, which cut 22 of TORGO dev's 115 dysarthric
   sentences in pieces. If your sentences get cut in two, try the pause slider at 2.5 s.

Switches in `config.local.json` if something misbehaves in a demo: `"whisper_hearings": false` (no extra pass, no
options), `"show_unsure_short": false` (short unsure words dropped as before), `"vad_threshold": 0.3` (to try hearing a
quiet voice). The learned files are `data/voice/personal/words.txt`, `fixes.txt` and `snippets.txt` (plain text, git
ignores them); "delete my voice data" removes them.

## Not done, for later

- Re-hear the whole passage: when a sentence came in pieces, transcribe the pieces' audio again as one clip after the
  user goes quiet. **Measured, not worth building:** at 1.5 s it recovers only part of the loss (held-out sentences
  29.0% -> 28.2%; whole clips 27.5%), while a 2.5 s pause recovers all of it (27.4%).
- Names on screen as Whisper hotwords (Wispr's context awareness): **measured, not worth it with stock Whisper.**
  Proxy: each clip got the rare words (4+ letters, more than one Whisper token, up to 40) of the other clips of its
  chapter or session as hotwords, like names read off the screen (`hearing_eval.py --context-test`). LibriSpeech dev:
  WER 1.50% -> 1.60%; the clip's own listed words came out right 190 times instead of 188 (of 195). TORGO dev:
  sentences 26.0% -> 41.8%, single words 108.7% -> 121.6%: Whisper puts listed words that sound a bit alike into
  unclear speech. So a long list of ordinary words is harmful for our users; the word list must stay short and hold
  names and terms (which is what auto-add keeps to).
- For the clean-up chat: Wispr's "Backtrack" ("at 2 actually 3" -> "at 3") and per-app style in its training data;
  the LibriSpeech over-editing above.
- Real dysarthric recordings (the Speech Accessibility Project proposal) stay the biggest lever: every number here is
  stock Whisper on someone else's voice.
- **Your own voice (Wispr's "personalized ASR"), ready to run when the GPU is free:** adapt Whisper to one person
  with the personal pack already on this PC (132 clips, 11.9 min of Khang's voice; stock Whisper gets 10.4% of the 30
  test clips' words wrong). If a short adapter run brings that down clearly, a "read 50 phrases" step in onboarding is
  worth building (`engine/voice/finetune/record_pack.py` already records the phrases). About 15 min of GPU, not
  started because DigiErhu was training all day:

  `.\.venv\Scripts\python.exe -m engine.voice.finetune.train --name personal-only --main-sources personal --typical-sources librispeech --lr 1e-4 --max-steps 400 --patience 1000 --typical-mix 0.25`

  then export `checkpoints\personal-only\last` and score `--groups personal` with `evaluate.py`, as for the sim runs.

## For the Speech Accessibility Project proposal (paste if useful)

Measured on Khang's PC with stock Whisper large-v3-turbo, the speech model most dictation tools build on: on two
dysarthric speakers it has never been tuned on (TORGO test, F04 and M01, 801 clips) it gets 41.5% of words wrong,
against 1.5% on clear read speech (LibriSpeech). Everything we could do in software around it (decoding settings,
keeping unsure words the camera saw the user say, several readings for the user to pick from) brings that to 39.9%. The
remaining gap is the model's hearing of dysarthric speech, which only training on consented dysarthric recordings can
close; no openly licensed set allows commercial use (checked 2026-11-10).

## Sources

- Wispr Flow, "Technical challenges and breakthroughs behind Flow" (Sep 2025): https://wisprflow.ai/post/technical-challenges
- Wispr Flow, "The challenges and directions in speech recognition research": https://wisprflow.ai/post/speech-recognition-challenges
- Wispr Flow, Features: https://wisprflow.ai/features
- Wispr Flow Help Center, Context Awareness: https://docs.wisprflow.ai/articles/4678293671-feature-context-awareness
- Wispr Flow Help Center, Dictionary: https://docs.wisprflow.ai/articles/4052411709-teach-flow-your-words-with-the-dictionary
- Wispr Flow Help Center, Smart Formatting and Backtrack: https://docs.wisprflow.ai/articles/5373093536-how-do-i-use-smart-formatting-and-backtrack
- Wispr Flow, What's new (release notes, 2026): https://wisprflow.ai/whats-new
- Baseten customer story (fine-tuned Llama, TensorRT-LLM): https://www.baseten.co/resources/customers/wispr-flow/
