# Dictation cleanup test set

100 original test cases for the step that turns a raw transcript into what the person meant. The text here was written for this test set. It is not copied from anywhere and holds no real person's words.

The question this set answers: **did the cleanup keep the meaning, without inventing anything?** We run every candidate model on the same cases (Qwen3.5 0.8B, 2B, 4B, later 9B) and compare. Nothing is trained until we have the baseline.

## The cases

| Category | Count | What it checks |
| --- | --- | --- |
| mispronounced (M) | 30 | Wrong or garbled words. Includes real-word mix-ups (recipe/receipt, pitcher/picture) that only context fixes. |
| broken_structure (S) | 30 | Fragments, missing words, repeats, restarts, scrambled order. |
| mixed (X) | 10 | Both problems in the same sentence. |
| ambiguous (A) | 15 | Two readings are plausible. The right move is to show both, or leave the text alone. |
| clean_control (C) | 15 | Already correct. Must come back unchanged. Catches a model that over-edits. |

About a third (33 cases) are marked `holdout`. Never use them to tune prompts or to train. Use `dev` cases for that. Run the holdout only to report results.

## Columns (`testset.csv`)

- `id`, `split` (dev or holdout), `category`
- `app` and `context`: where the text is going and what came before. Empty means no context.
- `asr_best`: what the speech step produced (the model's input).
- `asr_alt1`, `asr_alt2`: other candidate transcripts, when we have them. This mimics Whisper's n-best list.
- `personal_words`: names or terms from the user's own word list, separated by `|`.
- `intended`: what the person meant. For ambiguous cases the readings are separated by ` || `.
- `must_keep`: names, numbers, dates and negations that must appear in the output exactly, separated by `|`.
- `expected_behavior`: one of
  - `rewrite`: output should carry the `intended` meaning.
  - `unchanged`: output should equal the input (ignoring capitals and punctuation).
  - `show_both`: show the readings, or keep the original and mark it uncertain. Silently picking one fails.
  - `keep_original_and_flag`: do not change it, and mark it uncertain (for example a cut-off sentence).
- `notes`: why the case is hard.

## How to score a case

Score each case on these checks. Each is pass or fail.

1. **Meaning kept.** A person reading the output would take the same message as `intended`. Different wording is fine.
2. **Nothing invented.** No new people, places, times, numbers, reasons or promises.
3. **Must-keep intact.** Every `must_keep` item is present and unchanged.
4. **Right behavior.** `show_both` cases surface both readings. `unchanged` cases stay unchanged. Ambiguous cases are not silently resolved.

A case passes only if all four checks pass. Two reviewers should score the same outputs at first, to catch disagreement.

Numbers to report per model, per category: pass rate, plus three safety rates that matter more than the average: **invented-fact rate**, **over-edit rate** (clean controls that changed), and **silent-guess rate** (ambiguous cases resolved without showing both).

The bar for "good enough" is not set yet. Set it after we see the baseline, and write it down before looking at holdout results.

## Limits (be honest about these)

- Claude drafted these cases. The mishearings are plausible Whisper errors, not recorded ones, and the broken sentences are our guesses about how real users write. They have not been checked by people with speech or language differences.
- It is 100 cases. That is enough to compare models, not to claim a percentage in public.
- It measures text in, text out. It does not measure the speech recognition step.

## Next improvements

1. Khang and Khoi correct any case that sounds wrong or has the wrong `intended`.
2. Replace invented mishearings with real Whisper outputs from recordings (with consent).
3. Have real users, or a speech-language pathologist, review a sample.
4. Add non-English cases if the product will support other languages.

## Rules

- Keep it original text. Do not paste in copyrighted text or real people's messages.
- Use made-up names only, or team names already on the personal list.
- Do not add recordings or personal data to this folder.
