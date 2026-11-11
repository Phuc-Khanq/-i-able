# The model's own word confidence flags most of its wrong guesses

**Question:** the clean-up model gives every wrong answer as sure: of big3s16lr epoch 1's 54 wrong answers on the 2,500
test cases, all 54 had `"uncertain": false` and no other reading. Can the probability the model gives each word it
writes tell us which answers to mark unsure?

**How:** llama-server (b11960) reports each generated token's probability (`n_probs`, from the full distribution,
before sampling). The same greedy Q8_0 runs the scorer scored were run again with it: big3s16lr epoch 1 (the proposed
candidate) and big1 (the app's model today), on big1_test.csv (2,500) and dev (67), prompt `cleanup_t1_short`,
Khang's PC (RTX 5080), 2026-11-10. All 5,134 answers came out identical to the scored runs, so each probability
belongs to the answer the scorer judged. Tool: `eval/dictation_cleanup/confidence.py` (collect, second, report).

**The flag:** take the least-sure token inside the answer's text, counting only content words. Skip small words
("the", "my", "please"...; negations and numbers are *not* skipped) and punctuation. If its probability is under a
cut-off, mark the answer unsure.

## Result

Answers the model gave as sure (no "uncertain", one reading), automatic scoring:

| Flag if under | Candidate: wrong caught (of 54) | ...with a new wrong word (of 24) | Right answers flagged (of 2,065) | Clean sentences flagged (of 374) | big1: wrong caught (of 59) | ...new wrong word (of 31) | Right flagged (of 2,061) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5 | 20 | 14 | 21 (1.0%) | 0 | 16 | 11 | 15 (0.7%) |
| 0.7 | 29 | 18 | 77 (3.7%) | 4 | 34 | 19 | 64 (3.1%) |
| **0.8** | **33** | **20** | **109 (5.3%)** | **7** | **40** | **24** | **103 (5.0%)** |
| 0.9 | 38 | 20 | 190 (9.2%) | 16 | 46 | 29 | 150 (7.3%) |

- **The meaning-changing errors are what it catches best.** "New wrong word" means the answer has a content word that
  neither Whisper heard nor the user meant. Read by hand, 20 of the candidate's 24 change the meaning ("hardware" for
  heartworm, "script" for sertraline, "kidney" for quinoa, "brake controls" for hand controls). The other 4 are an accent
  (resume), a plural or a spelling. At 0.8 the flag catches **18 of those 20**. The two it misses: "1099" for W-2 (the
  app's guard already blocks it: "number added") and "prayer list" for playlist (the model was 99% sure).
- **The rest of the wrong answers** are mostly misheard words left as heard ("curve side pickup", "fish soap",
  "jewelry duty"). It catches 13 of the candidate's 30. These are less dangerous: the text shows what was said.
- **The cut-off holds on a second model.** 0.8 was looked at on the candidate's answers, and it catches about the same
  share of big1's (40 of 59, false alarms 5.0%). Both are the same test set, though, so the hidden test has to confirm it.
- **The model's own "uncertain" is no help here:** its probability of writing `true` was over 1% for 0 of the
  candidate's 54 wrong answers (big1: 1 of 59).

Other ways of counting, at 0.8 on the candidate: every token including punctuation catches 36 of 54 but flags twice as
many right answers (10.7%); only words Whisper didn't hear flags no clean sentence but catches 20 of 54.

## Cautions

- **On dev the flag fires far more often on right answers:** 12 of 56 (21%) for the candidate, 9 of 55 for big1, at 0.8.
  Dev is hand-written and its broken sentences are harder. The flagged ones are right in meaning, but the model was
  choosing between phrasings ("cancelled"/"canceled", where "Mom" goes, "two times a day"). Real speech may look more
  like dev than like the generated test, so expect more than 5%. Dev has 1 wrong answer per model ("eyeable"); big1's is
  caught, the candidate's (0.87) isn't.
- **The automatic score is strict:** a few "wrong" answers are fine ("power chair" for power wheelchair).
- **The cut-off was chosen on this test set.** The fresh hidden test should confirm it before it ships.

## A second reading from the runner-up word: not good enough on its own

At the least-sure token, putting the model's second choice there and letting it finish gives the right reading for 8 of
the 36 wrong answers the any-token flag catches at 0.8 (candidate; big1: 18 of 45). It works for "hard worm" →
"heartworm", "hen controls" → "hand controls", "guest limit" → "guest list". Most often, though, the runner-up is
another wrong word ("afro" → "axe", "script" → "refill"). So the flag is worth more as a warning, pointing at the weak
word, than as an automatic option two.

## Cost

Asking for one probability per token adds about 23 ms per sentence: median request 341 vs 318 ms, p90 402 vs 366 ms.
That's 201 requests each (the 67 dev cases × 3, alternating, no prompt cache), candidate, RTX 5080.

## What it means

1. The trained model is overconfident: it never doubts itself on its wrong answers. Its word probabilities do carry
   the doubt. A cut-off of 0.8 on content words catches 18 of the 20 meaning-changing errors on the test, for about
   5% extra "unsure" marks on right answers (21% on the harder dev cases).
2. **Proposed for the app (needs Khang's OK):** in `CleanupModel.clean()` (`engine/voice/cleanup_model.py`), ask for
   `n_probs: 1` and set `uncertain` when a content word is under 0.8. The card already shows "unsure". No contract or
   shared-file change, and about 23 ms per sentence.
3. Later, worth more: send *which* word is weak so the card can highlight it, and the user fixes it by voice (the
   Dictation chat's word fix). That adds a field to the contract (Khang's OK, and the website's copy changes too).
4. Don't offer the runner-up reading as option two by itself: it is right in only a quarter to two-fifths of cases.
5. Check the flag on the hidden test. It may need a different cut-off for heavily broken sentences, where phrasing
   doubt is normal.

## Files

- `eval/dictation_cleanup/runs/confidence/` (on this PC only): `<label>_<cases>.jsonl` (answers and token
  probabilities) and `_second.jsonl` (second readings), for `lr-e1` (big3s16lr epoch 1) and `big1`.
- Rerun the report: `python -m eval.dictation_cleanup.confidence report --label lr-e1 --cases test --run 2026-11-05_big3s16lr-epoch1-q8_0-greedy_cleanup_t1_short_big1_test --show 0.8`
