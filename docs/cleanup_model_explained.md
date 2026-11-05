# The clean-up model: what we train, and how

Plain-language notes for the team (2026-11-04). Numbers are measured on Khang's PC (RTX 5080) unless marked.

## The network

The clean-up runs on **Qwen3.5-4B** (Alibaba's Qwen team, Apache 2.0), a language model with about 4 billion
numbers ("weights"). It reads text as tokens (pieces of words; its vocabulary has 248,320 of them) and predicts
the next token, over and over, until the answer is written. From its `config.json`:

- **32 layers**, stacked. Each token is a list of 2,560 numbers that every layer updates.
- **Two kinds of layers.** Every 4th layer (8 of them) is full attention: each word looks at every earlier word
  (16 query heads, 4 key/value heads, 256 numbers per head). The other 24 are "linear attention" layers (Gated DeltaNet:
  transformers runs them with its `chunk_gated_delta_rule`): a running memory that is cheaper on long text. Their
  parts are `in_proj_qkv`, `in_proj_z`, `out_proj`, with a short convolution of 4 tokens. On Windows we run them with
  transformers' slow reference code (the fast kernels, `flash-linear-attention` and `causal_conv1d`, aren't
  installed); that's a speed matter only, not accuracy.
- **An MLP in every layer** (`gate_proj`, `up_proj`, `down_proj`): widens the 2,560 numbers to 9,216, applies a
  non-linearity, and narrows back. Most of the model's knowledge sits here.
- **Thinking is off.** Qwen3.5-4B can "think" before answering, which is far too slow for dictation.

## What training changes: LoRA

We don't change the 4 billion weights. They stay frozen, and we train **LoRA adapters**: for each big weight
matrix W in the layers above, two thin matrices A and B are added, and the layer computes `W·x + (alpha/r)·B·A·x`.
`r` (the rank) sets how thin they are.

- big1: r = 16, alpha = 32, on every linear layer of all 32 layers: **30.5 million trainable numbers** (0.75% of
  the model). The adapter file is 122 MB. big2 and big3 use r = 32 (about 61 million).
- For the app, the adapter is merged into the weights and saved as one GGUF file (Q8_0, about 4.5 GB) that
  llama.cpp runs: about 0.29 s per sentence (median) on the 5080.
- Why not train all 4 billion: it needs several times the GPU memory (it doesn't fit on 16 GB), and it's much
  easier to wipe out what the model already knows about English.

## How one training step works (`engine/voice/cleanup_train/train.py`)

1. One example is the exact prompt the app sends (system text; the heard words, other hearings, app, the line
   before, the word list) followed by the right answer as JSON: `{"text": ..., "alternatives": [...],
   "uncertain": ...}`.
2. The model predicts the answer one token at a time. The **loss** is how surprised it was by the right tokens
   (cross-entropy). Only the answer counts; the prompt is masked out. A loss of 0.01 means it gave the right
   token about 99% probability on average.
3. The gradient says how to nudge A and B to lower that loss. We add up 8 examples' gradients, then take one
   step (AdamW, learning rate 2e-4, warm-up 10 steps, cosine decay to 0, gradient clipping 1.0, bf16).
4. An **epoch** is one pass over every training example. 24,000 examples (with 5% held back) is 2,850 steps per
   epoch; the adapter is saved after each epoch, so each can be scored.

## Over- or underfitting: what we know

- **Underfitting** (the model can't even learn the training data) would show as a training loss that stays
  high. It doesn't: big1's training loss ends at about 0.001, big2's at about 0.0004. And doubling the rank (big2)
  didn't help. **Network size is not the limit.**
- **Overfitting** (memorizing the training examples instead of the skill) shows as a gap between the loss on
  training examples and on examples it never saw, and a test score that stops improving or falls.
  - big1, end of epoch 2: training loss 0.0011, held-back loss 0.0094 (8.5 times higher).
  - big2: training loss fell from 0.0015 (epoch 2) to 0.0004 (epoch 3), but the held-back loss only from 0.0047 to
    0.0040 and the test from 2,409 to 2,411 of 2,500. The third epoch mostly memorized.
  - Those held-back examples share sentences with the training examples (noisy copies of the same sentence land
    on both sides), so the real gap on new sentences is larger than this. big3 measures it properly.
- So the model is **memorizing more than it is generalizing**: the limit is what the data covers, not the network.
  More data helps only if it is **new** data. Each sentence appears as about 3 to 4 noisy copies: big1 7,500
  examples from 2,778 different sentences, big2 12,000 from 4,323, big3 24,000 from 5,752.

## What big3s and big3 showed (2026-11-04 and 05)

big3s (8,000 examples) and big3 (24,000; big3s is a random part of it) have the same recipe, so the pair shows what
three times the data does. Every 500 steps the trainer measures the loss on training examples, on held-back
**whole sentences** and on 400 test cases it never trains on.

- **Three times the data made it worse:** test 2,395 vs 2,418 of 2,500, clean sentences changed 15 vs 7.
- **It overfits to our data generator.** The loss on held-back sentences (from the same sentence banks) kept falling,
  to 0.0067. The loss on test cases (written by an earlier generator, other terms) stopped improving at step 500 and
  ended at 0.0279, worse than big3s (0.0208) and big1 (0.0190). big2's test-case loss rose at every epoch.
- **One habit we found the cause of:** the "broken" examples delete small words and the answer puts them back, so the
  model learns to add "the" wherever one would fit, also to clean sentences ("for kids" -> "for the kids").
- Details: `eval/dictation_cleanup/reviews/2026-11-04_big3s.md` and `2026-11-05_big3.md`.

## What big3s16 showed (2026-11-05)

big3s16 is big3s with only the rank changed, 16 instead of 32. Rank 16 wins on the test, the fair cases, clean sentences,
dev and both new-data losses, and it stops writing spoken numbers as digits. It even ends lower on its own training
data. So the bigger adapter wasn't "more room to learn". A guess, not tested: with alpha at twice the rank, rank 32 moves
the model further per step, like a higher learning rate. Details: `eval/dictation_cleanup/reviews/2026-11-05_big3s16.md`.

## Knobs we can turn, and what we know about each

| Knob | Today | What we know |
| --- | --- | --- |
| Data: different sentences | 5,752 (big3) | The real limit (above). Growing it needs new sentences from new sources, not more template fills or noise passes. |
| Data: examples | 24,000 (big3) | Three times the same data hurt (big3 vs big3s). |
| Data: how noise is made | small words dropped with chance 0.6 in broken examples | Teaches adding "the" to clean sentences. Not yet changed. |
| Rank r | 16 | 32 costs: on the same data, rank 16 beats rank 32 on every measure, also on both new-data losses (big3s16 vs big3s, 2026-11-05). Use 16. |
| Layers trained | all 32, every linear layer | Nothing left to add. |
| Epochs and steps | 2 | Test-case loss stops improving after about 500 steps (big3); big2's rose every epoch. |
| Learning rate | 2e-4, cosine | Not tested. A lower rate (1e-4) is a cheap next test. |
| LoRA dropout | 0.05 | Not tested. More dropout is a standard brake on memorizing. |
| Loss weighting | every answer token equal | Not tested. Could weigh the tokens that matter (a fixed word, the "uncertain" flag) more. |
| The model's own confidence | not used | llama.cpp can report each token's probability; a low one on a changed word could set "unsure" by itself. Likely the biggest step for reliability, and it needs no training. |
