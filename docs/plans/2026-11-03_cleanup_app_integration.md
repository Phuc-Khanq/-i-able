# Plan: the trained clean-up model in the app

Status (2026-11-03): steps 1 and 2 are built on branch `voice-cleanup-app` (Khang: "proceed"), tested with
fakes; the run on the real GPU waits until big1's training ends. The default backend stays "transformers" until
Khang has run the demo. Steps 3 and 4 change contract files and need his OK first (CLAUDE.md).

## Goal

Replace the app's clean-up model, Qwen2.5-3B (research licence, can't be sold), with our own trained model:
Qwen3.5-4B plus our adapter (Apache 2.0 base, our data), merged into one GGUF file and run by llama.cpp (MIT).
Measured on the 5080: 0.33 s median and 0.38 s p90 with the Q8_0 file and greedy decoding, so it fits the 1.5 s
quick clean-up and the 6 s refine. full1 is the candidate today; big1 replaces it if its scores are better.

What stays the same: the preview and confirm step, the worker thread, "never stop dictation because of the
clean-up" (any failure gives the user's own words), and the two bus topics (until step 3).

## How the app does it today

- `engine/voice/rewrite.py`: `Rewriter.rewrite()` (quick clean-up of the preview) and `refine()` (whole passage after
  3 s of quiet), both through transformers on the GPU; `check_rewrite` / `check_refine` decide if the text is used,
  else `light_clean` or the original.
- `RewriteWorker` runs it on its own thread; the mode manager sends `REWRITE_REQUEST {id, text}` and gets
  `REWRITE_RESULT {id, clean, used, ms}` (`engine/core/topics.py`).
- Settings: `rewrite_model`, `rewrite_device`, `rewrite_wait_s` 1.5, `rewrite_max_s` 3.0, `refine_*`
  (`engine/config.py`).

## Steps (one pull request each, tested with `scripts\run_demo.ps1` on the reference machine)

**1. Runtime: llama-server behind the Rewriter (no contract change).**
- New `engine/voice/llama_server.py`: starts llama-server as a child process on 127.0.0.1 with a free port, a random
  API key per run (so a web page on the PC can't call it), `--offline`, `--no-webui`, all layers on the GPU, 4096
  context, one slot. Waits for `/health`, restarts it once if it dies, stops it with the engine. On Windows a Job
  Object makes sure it dies with the engine even after a crash.
- `Rewriter` gets a second backend: it sends the training prompt (`cleanup_t1_short`) as text (llama.cpp's
  tokenizer matched Hugging Face's on 67/67 prompts, so the runtime needs no torch or transformers for the clean-up),
  greedy, and reads the JSON answer `{text, alternatives, uncertain}`. For now it uses `text` only.
- Settings: `rewrite_backend` ("transformers" | "llamacpp"), `rewrite_gguf` (a file in `models/`, git-ignored),
  `rewrite_prompt` ("t1_short"). The default switches to llamacpp in its own commit, after the demo works.
- The eval's `LlamaCppBackend` then uses this same client, so the eval measures exactly what the app runs.

**2. The new guard (`engine/voice/rewrite.py`).** Replace "reject if the text differs a lot" with the eval's checks:
reject a number, name or negation that isn't in what was heard, the context or the word list; reject an answer
that drops most of the words; broken JSON goes to `light_clean`. If the main reading flips a negation or a number
against the heard words, show the heard words instead. Port the tests from `tests/test_cleanup_eval.py`; keep
`tests/rewrite_cases.csv` passing.

**3. Contracts (Khang's OK first; the website's copy changes in the same PR).**
- `REWRITE_REQUEST` gains optional `context` (text already typed in this dictation), `app` (the foreground window's
  name) and `words` (the word list: `whisper_hotwords` for now).
- `REWRITE_RESULT` and `DICTATION_PREVIEW` gain optional `alternatives` (list of strings) and `uncertain` (bool).

**4. The card.** When the model is unsure: the heard reading first, the others below, chosen by voice ("option two")
or a gesture; "use original" stays one step away. Never type an unsure reading that changes a negation or a number
without the user picking it. (`engine/eyes/dictation_card.py` and the website.)

Step 3 is built (2026-11-03, branch `voice-cleanup-app`): the request carries `context` and `app`, the result and
`DICTATION_PREVIEW` carry `alternatives` and `uncertain`, and the website stores them. Nothing shows them yet.

Step 4, **built 2026-11-03 with Khang's OK on the look** (commit 69687d2). Two changes from the proposal: the
card is click-through, so instead of "Option 2" labels in its action row each reading carries its spoken words
("option two") on its own line; the website panel has real Option buttons. Readings are numbered from 1 (the
model's), and the preview's optional `choice` says which one shows. The proposal as agreed:
- Card, "Will type" state: under the text, a small "Or:" list, numbered from 2 ("2 · Send the pitcher to Mom."),
  at most two lines. The action row gains "Option 2" (and "Option 3") next to "Use original"; the hint field says
  "say option two". With `uncertain` the state label reads "Check this" instead of "Will type".
- Voice: "option two", "option three" (exact-only commands, like "use original") put that reading in place of the
  text; "use original" still gives the user's own words. A dwell or blink on the button does the same.
- Website demo panel: the same list under the preview, same commands.
- Safety: the main reading already can't add or drop a number, name or negation without a change of mind
  (check_cleanup), so "yes" stays one step. Readings are only shown, never typed, until picked.

**5. Shipping.** The installer or first run needs llama-server (the CUDA build is about 770 MB with NVIDIA's DLLs;
CPU or Vulkan builds for laptops) and the model file (Q8_0 4.5 GB, Q4_K_M 2.7 GB).

## Trying it in the demo (Khang, on the reference machine)

1. Close anything else using the GPU (no training running).
2. Add one line to `config.local.json` (keep what's there): `"rewrite_backend": "llamacpp"`. Optional: your own
   words, `"rewrite_words": "Priya, Dr. Okafor, baclofen"`.
3. `.\scripts\preflight.ps1`: it now checks llama-server, the model file and the GPU memory for this backend.
4. `.\scripts\run_demo.ps1`, then dictate and look at the card:
   - a messy sentence ("um can you you send the pitcher to mom") comes back clean within about half a second;
   - an unsure one shows "Check this" and "option two" lines; say "option two", then "yes";
   - "use original" still gives your own words; a dictation mentioning a name you typed earlier keeps it;
   - quit the demo and check Task Manager: no `llama-server.exe` left running.
5. To go back: remove the line (or set it to "transformers").

## GPU memory

Q8_0 takes about 5 GB (nvidia-smi, whole card) against 6.3 GB for today's Qwen2.5-3B, so the `full` profile
(15.1 GB peak today) gets a little more room. To check in step 1 with Whisper and the face tracker running together.

## Decisions for Khang

1. OK to start steps 1 and 2 (no contract change)?
2. OK in principle for the contract additions in step 3?
3. Q8_0 with greedy decoding as the default (DECISIONS.md, Proposed)?
4. The refine: the trained model never saw the `(?)word(?)` marks for unsure words. Send the plain passage first,
   and train on marked text later?
5. Where the model file comes from for users (a private download we host, or in the installer); this also waits on
   the lawyer question about Claude-written training data.

## Risks

- A child process on Windows: antivirus may flag it, and a port may be taken (a free port is picked each run).
- The clean-up model still decides alone on unsure words until step 4: steps 1 and 2 use only `text`.
- full1 and big1 are measured on generated tests; the human-written hidden set is still to come.
