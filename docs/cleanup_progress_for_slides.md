# Dictation clean-up model: progress so far (material for slides)

Written 2026-11-05 23:20 by the Claude Code "Clean-up model" chat, for Cowork to build a presentation from; big3lr added
at 23:49, the last two night runs on 2026-11-06 at 05:17.
Every number below was measured on Khang's PC (RTX 5080) and comes from the reviews in `eval/dictation_cleanup/reviews/`
and `docs/SESSION_LOG.md`. All of the night's runs are in: see the last section.

## The story in one paragraph

People with speech differences dictate, Whisper mishears, and a clean-up model must fix what was meant without
inventing anything. The app's old clean-up (Qwen2.5-3B with a prompt) kept the meaning in 17 of 67 dev cases. A bigger
model with only a prompt reached 59 of 67 but took about 2 s and failed the pass bar on the holdout. So we trained our
own model: a small add-on (LoRA, 0.75% of the weights) to Qwen3.5-4B (Apache 2.0), trained on Claude-written examples.
It reached 63 of 67, runs in 0.28 s, and is in the app behind a guard that blocks invented numbers, names and negations.
Since then we have been learning how to train it well: the settings mattered more than the amount of data, and the
settings are now as good as they get (five runs tie). The next gain has to come from new sentences.

## Slide: where we started and where we are (dev, 67 cases, "meaning kept" by Khang's rule)

| Clean-up | Meaning kept (dev 67) | Time per sentence |
| --- | --- | --- |
| The app before (Qwen2.5-3B + prompt + old guard) | 17 | 0.44 s median |
| Qwen3.5-4B, prompt only (best prompt, v4) | 59 | 2.0 s median |
| full1, our first trained adapter (1,088 examples) | 62 | 1.8 s (transformers), 0.33 s (llama.cpp) |
| big1, trained on 7,500 examples (the app's model today) | 63 | 0.28 s median in the app |

Holdout (33 cases, run once): the app before 7, prompt only 22; the bar was 27. That is why we trained our own.
Caveat for the slide: dev was used while building, so these dev numbers are optimistic. A fresh hidden test decides.

## Slide: the training runs (test of 2,500 cases, automatic scoring)

Each run changes one thing from an earlier run. "Fair 2,301" leaves out 199 cases whose terms only big1 trained on. A dash: not compared on that measure.

| Run | What changed | Examples | Test (2,500) | Fair (2,301) | Clean sentences changed (of 378) | Training time |
| --- | --- | --- | --- | --- | --- | --- |
| full1 | first adapter | 1,088 | 2,389 | – | – | 33 min |
| big1 | big data set | 7,500 | 2,441 | 2,275 | 4 | 163 min |
| big2 | rank 32, more data, 3 epochs | 11,400 | 2,411 | – | 14 | 344 min |
| big3s | new data kinds, rank 32 | 8,000 | 2,418 | 2,262 | 7 | 175 min |
| big3 | 3x big3s's data, rank 32 | 24,000 | 2,395 | 2,250 | 15 | 428 min |
| big3s16 | big3s at rank 16 | 8,000 | 2,432 | 2,277 | 4 | 182 min |
| big3s16lr | big3s16 at half the learning rate | 8,000 | 2,446 (epoch 1) | 2,281 | 4 | 150 min |
| big3lr | big3s16lr's settings, 3x the data | 24,000 | 2,439 (epoch 1) | 2,275 | 7 | 223 min to epoch 1 (then a restart) |
| big3s16lr1ep | big3s16lr in one epoch | 8,000 | 2,445 | 2,280 | 3 | 84 min |
| big3s16lr5 | big3s16lr at half the learning rate again (5e-5) | 8,000 | 2,447 | 2,283 | 4 | 168 min |

Headline: big3s16lr is the first run to beat big1, by a small margin (on the fair cases, 10 wins and 4 losses).
Three times the data at the same settings (big3lr) ties big1 and doesn't beat it (fair cases: 3 wins, 9 losses against
big3s16lr; 7 and 7 against big1). One epoch, or an even smaller learning rate, ties big3s16lr too (2,445 to 2,447; no
difference beyond chance): the settings have reached a plateau.

## Slide: what we learned about training

1. **The network is big enough.** Training loss falls to 0.0004 (the model is right on nearly every token of the
   examples it has seen), so it is not underfitting. A bigger adapter (rank 32) was worse, not better.
2. **It memorizes our data generator.** By the end of training, the loss on sentences it never saw is 30 to 50 times higher than on its
   training data. At the old settings, more copies of the same sentence banks made it worse (big3 vs big3s); at the new
   settings they no longer hurt much, but they don't help either (big3lr vs big3s16lr), and they take 3x the time.
3. **The settings were costing us, and now they are done.** Rank 16 beat rank 32 on every measure; learning rate 1e-4
   beat 2e-4 and even learned faster. After that, five runs (one or two epochs, 1e-4 or 5e-5) all score 2,444 to 2,447.
   One epoch is enough: the same score in half the time (84 vs 150 min), which makes future tests cheaper.
4. **What decides the last few points is rare words.** The same few hard sound-alikes (heartworm, prednisone, AFO,
   wheelchair van) flip between runs: each run gets a different few right. That is noise, and only more training
   sentences with those kinds of words can fix it.
5. **So the next gain is new sentences, not more copies.** Khang's goal is 100,000 examples, which needs new sources.

Chart data, big3s16lr loss checks (lower is better; "new sentences" and "test cases" were never trained on):

| Step | Training data | New sentences | Test cases |
| --- | --- | --- | --- |
| 500 | 0.0111 | 0.0163 | 0.0243 |
| 1,000 (end of epoch 1) | 0.0021 | 0.0126 | 0.0187 |
| 1,500 | 0.0010 | 0.0139 | 0.0197 |
| 1,898 (end) | 0.0004 | 0.0135 | 0.0199 |

## Slide: what still goes wrong

- Confident wrong guesses on rare medical words, not marked unsure: "afo" became "foam", "pred knee zone" (prednisone)
  became "physical therapy".
- The test cases were invented by us (Claude), not said by real users; a speech-language pathologist should review them.

## Slide: next steps and decisions for the team

1. Khang: approve big3s16lr epoch 1 as the new candidate (proposed).
2. Khang or Hung: write about 35 fresh test phrases (the hidden test); the candidate runs on them once.
3. Try the trained model live in the demo with camera and mic, then decide whether it becomes the default.
4. Find many new, different sentences toward 100,000 examples (source, licence, cost): the real limit now.
5. Lawyer questions: training on Claude-written sentences, shipping NVIDIA's CUDA DLLs, the code licence, IP agreements.

## Other work in this period (one line each)

- Private repo `-i-able` for all new work; the old public repo stays as the hackathon archive and website host.
- Scoring runner for the clean-up test set; licence scan of 91 packages; the app runs without PyAV (GPL parts).
- llama.cpp (MIT) runs the model: about 5x faster than transformers, the same answers.
- Two Claude Code chats can now work in the repo at once (clean-up model and dictation), with rules for sharing it.

## The night's runs (2026-11-05/06): all three in

- **In: does more data help at the new settings? Not really.** big3lr (big3's 24,000 at rank 16, learning rate 1e-4) was
  stopped at 22:27 when the PC restarted (the Kaspersky installer restarted Windows), 3,163 of 5,698 steps in. Its saved
  epoch-1 adapter scores 2,439 of 2,500: level with big1 (2,441), a little behind big3s16lr epoch 1 (2,446), far ahead of
  big3 at the old settings (2,395). So the old "more data made it worse" was mostly the settings; but more copies of the
  same sentences still don't help. Review: `eval/dictation_cleanup/reviews/2026-11-05_big3lr.md`.
- **In: is one pass enough? Yes.** big3s16lr1ep (one epoch, 84 min) scores 2,445 vs big3s16lr epoch 1's 2,446 (fair
  cases: 4 wins, 5 losses), with the lowest loss on new data of any run. But on dev it dropped a "not" once and made one
  silent guess more; the app's guard blocks both.
- **In: is an even lower learning rate better? No.** big3s16lr5 (5e-5, 168 min) scores 2,447 at both epochs, again
  within chance of big3s16lr (4 wins, 2 losses). It invents one name ("Hawthorne", blocked by the guard).
- **So:** the settings are done; big3s16lr epoch 1 stays the proposed candidate (best dev, 65, and no new safety
  slips). Review: `eval/dictation_cleanup/reviews/2026-11-06_big3s16lr1ep_and_lr5.md`.
- They started at 00:03, once a game freed the GPU, and the chain ended at 05:13.
