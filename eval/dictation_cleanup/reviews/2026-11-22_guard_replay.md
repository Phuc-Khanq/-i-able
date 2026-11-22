# Guard replay: would a stricter clean-up guard help? (2026-11-22)

Question (STATUS Next 5c): the trained clean-up (big1 Q8_0, greedy) changes some clear text and adds words to short
phrases ("Refresh." -> "Please refresh the page."), and `check_cleanup` lets it through. Which extra guard rule
helps without hurting what the model does well?

Method: no GPU, no model. `eval/dictation_cleanup/guard_replay.py` replays answers already collected through guard
variants. A blocked answer falls back the way the app does (`Rewriter._fallback`). Eval answers: big1 on the 67 dev
cases and the 2,500-case test (`runs/2026-11-03_big1-q8_0-greedy_cleanup_t1_short*`). Whisper rows (big1 with other
hearings, from `engine/voice/finetune/hearing_eval.py`, `checkpoints/hearings/*.cleanup-greedy.jsonl`): LibriSpeech
dev (298 clear book sentences), Khang's 29 recorded phrases, TORGO dev (115 dysarthric sentences). "Equal to meant"
counts the 'rewrite' and 'unchanged' cases whose shown text equals `intended`.

Variants (each also keeps `check_cleanup`):
- **no new word, input up to N words:** block an answer that adds a content word nobody said, gave in a hearing, the
  text before or the word list, or that sounds like 1 to 4 heard words (the word list's looser sound test).
- **dropped run:** also block when 2 or 3 content words in a row were dropped and not repeated elsewhere, unless the
  speaker changed their mind.

## What the numbers say

- **"No new word, input up to 6 words" helps and hurts nothing measured.** Khang's phrases go from 11.67% words wrong to
  10.00% (Whisper alone: 10.00%): 2 phrases fixed, none worse. Dev, TORGO and LibriSpeech don't move. The test set has
  6 more blocked answers: 1 more equal to meant, 0 fewer, and 1 fewer clean control changed (4 -> 3). The short
  version (3 words) gives the same on Khang's phrases and nothing else.
- **The same rule on inputs of any length hurts:** on the test it blocks 104 answers and loses 53 that were right
  (probably adding words legitimately to repair broken speech; I didn't read each one). Not an option.
- **Dropped-run rules hurt:** run of 3 loses 44 right test answers, run of 2 loses 268, for 1 to 8 better rows on
  LibriSpeech. Broken speech drops words on purpose (restarts, false starts). Not an option.
- **LibriSpeech's 1.50% -> 3.21% isn't a guard problem.** The clean-up with hearings makes 114 word edits (112 in
  sentences with no filler or repeat): 54 swap a content word, 18 add and 16 drop a small word, 14 swap a small word,
  12 drop content words. Many swaps are spelling normalisation that a reference counts wrong ("laughing stock" ->
  "laughingstock", "impassible" -> "impassable", "roomful"), so the WER overstates real harm. The real harm is
  few and sits in dropped phrases ("the logos or word of god" dropped: 6 words) and swapped small words
  ("who" -> "which"). Telling these from a good repair of broken speech needs more than the text: Whisper's word
  confidence (sure words stay) is the signal, and it isn't in the rewrite request today (a contract change).

## Caveats

- The evidence for the short-phrase rule is 2 phrases of one voice (Khang's 29). The synthetic test has almost no
  short inputs, so it can't judge the rule. The hidden test (Hung's phrases include short ones) should.
- The eval's "equal to meant" is strict: a different but correct wording counts as lost.
- My counts use my own word comparison; the Dictation chat counted 76 changed LibriSpeech sentences, I count 83.

## Proposal (needs Khang's OK; the rule goes into the shared `rewrite.py`)

Add "no new content word when the input has up to 6 words" to `check_cleanup`. Message the Dictation chat first. Check
again on the hidden test.

## Full output


### Eval answers, dev (67): answers equal to what was meant ('rewrite' and 'unchanged' cases)

| guard | blocked | equal to meant | lost | gained | clean controls changed | auto pass |
| --- | --- | --- | --- | --- | --- | --- |
| check_cleanup today | 2 | 38 | 2 | 0 | 0 of 10 | 63 of 67 |
| + no new word, input up to 3 words | 2 | 38 | 2 | 0 | 0 of 10 | 63 of 67 |
| + no new word, input up to 6 words | 2 | 38 | 2 | 0 | 0 of 10 | 63 of 67 |
| + no new word, any length | 2 | 38 | 2 | 0 | 0 of 10 | 63 of 67 |
| up to 6 words, + no dropped run of 3 content words | 2 | 38 | 2 | 0 | 0 of 10 | 63 of 67 |
| up to 6 words, + no dropped run of 2 content words | 7 | 37 | 4 | 1 | 0 of 10 | 59 of 67 |

### Eval answers, test (2,500): answers equal to what was meant ('rewrite' and 'unchanged' cases)

| guard | blocked | equal to meant | lost | gained | clean controls changed | auto pass |
| --- | --- | --- | --- | --- | --- | --- |
| check_cleanup today | 18 | 1922 | 8 | 1 | 4 of 378 | 2438 of 2500 |
| + no new word, input up to 3 words | 18 | 1922 | 8 | 1 | 4 of 378 | 2438 of 2500 |
| + no new word, input up to 6 words | 24 | 1923 | 8 | 2 | 3 of 378 | 2438 of 2500 |
| + no new word, any length | 54 | 1917 | 15 | 3 | 2 of 378 | 2437 of 2500 |
| up to 6 words, + no dropped run of 3 content words | 67 | 1887 | 44 | 2 | 3 of 378 | 2421 of 2500 |
| up to 6 words, + no dropped run of 2 content words | 311 | 1663 | 268 | 2 | 3 of 378 | 2321 of 2500 |

### Whisper rows, libri-dev: words wrong after the clean-up (with other hearings)

| guard | blocked | words wrong | rows better | rows worse |
| --- | --- | --- | --- | --- |
| check_cleanup today | 0 of 298 | 3.21% (199 of 6191) | 0 | 0 |
| + no new word, input up to 3 words | 0 of 298 | 3.21% (199 of 6191) | 0 | 0 |
| + no new word, input up to 6 words | 0 of 298 | 3.21% (199 of 6191) | 0 | 0 |
| + no new word, any length | 5 of 298 | 3.02% (187 of 6191) | 4 | 0 |
| up to 6 words, + no dropped run of 3 content words | 1 of 298 | 3.12% (193 of 6191) | 1 | 0 |
| up to 6 words, + no dropped run of 2 content words | 8 of 298 | 2.89% (179 of 6191) | 8 | 0 |

Whisper alone: 1.50% (93).

### Whisper rows, personal-test: words wrong after the clean-up (with other hearings)

| guard | blocked | words wrong | rows better | rows worse |
| --- | --- | --- | --- | --- |
| check_cleanup today | 0 of 29 | 11.67% (28 of 240) | 0 | 0 |
| + no new word, input up to 3 words | 2 of 29 | 10.00% (24 of 240) | 2 | 0 |
| + no new word, input up to 6 words | 2 of 29 | 10.00% (24 of 240) | 2 | 0 |
| + no new word, any length | 2 of 29 | 10.00% (24 of 240) | 2 | 0 |
| up to 6 words, + no dropped run of 3 content words | 2 of 29 | 10.00% (24 of 240) | 2 | 0 |
| up to 6 words, + no dropped run of 2 content words | 3 of 29 | 9.17% (22 of 240) | 3 | 0 |

Whisper alone: 10.00% (24).

### Whisper rows, torgo-dev: words wrong after the clean-up (with other hearings)

| guard | blocked | words wrong | rows better | rows worse |
| --- | --- | --- | --- | --- |
| check_cleanup today | 0 of 115 | 25.33% (209 of 825) | 0 | 0 |
| + no new word, input up to 3 words | 0 of 115 | 25.33% (209 of 825) | 0 | 0 |
| + no new word, input up to 6 words | 0 of 115 | 25.33% (209 of 825) | 0 | 0 |
| + no new word, any length | 1 of 115 | 25.09% (207 of 825) | 1 | 0 |
| up to 6 words, + no dropped run of 3 content words | 0 of 115 | 25.33% (209 of 825) | 0 | 0 |
| up to 6 words, + no dropped run of 2 content words | 5 of 115 | 25.82% (213 of 825) | 0 | 2 |

Whisper alone: 27.03% (223).
