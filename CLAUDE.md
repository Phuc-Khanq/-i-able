# [eye]able: instructions for Claude Code

[eye]able lets people control a Windows computer with their head, blinks and voice, using only a webcam and a microphone. The main users are people who can't use their hands (ALS, spinal cord injury, cerebral palsy, muscular dystrophy, severe tremor). It started at Gator Hack V and is now being built into a product that will be sold.

Read `docs/STATUS.md` (imported below) at the start of every session. Read `docs/DECISIONS.md` before changing anything a decision covers.

## Reference machine

Khang's PC is the machine that counts: Windows, Python 3.11 in the repo's `.venv`, NVIDIA RTX 5080 (16 GB), Intel Core Ultra 7 265KF, 64 GB RAM, PyTorch built for CUDA 12.8. Code that doesn't run there doesn't count.

- Always use `.\.venv\Scripts\python.exe`, never a system Python.
- Write commands for PowerShell.
- Only Python 3.11 features.

## Commands

| What | Command |
| --- | --- |
| All tests | `.\.venv\Scripts\python.exe -m unittest discover -s tests -t .` |
| One test file | `.\.venv\Scripts\python.exe -m unittest tests.test_rewriter` |
| Compile check | `.\.venv\Scripts\python.exe -m compileall -q engine tests` |
| Start the engine | `.\.venv\Scripts\python.exe -m engine.main` (`--help`; useful: `--no-eyes`, `--stdin`, `--wav clip.wav`, `--no-rewrite`, `--gpu-log`) |
| The demo app | `.\scripts\run_demo.ps1` |
| Checks before a demo | `.\scripts\preflight.ps1` |
| Website tests | `node --test tests/lite_core.test.js tests/lite_voice.test.js` |
| Website without a camera | `.\.venv\Scripts\python.exe -m tests.fake_engine` |

## Layout

```text
engine/core/     event bus, mode manager, message contracts (types.py, messages.py), WebSocket relay
engine/eyes/     camera, face landmarks, head pointer, blink and head gestures, Windows mouse and keyboard
engine/voice/    mic + Silero VAD, faster-whisper, commands, dictation, clean-up (rewrite.py)
engine/voice/finetune/   Whisper LoRA training scripts (TORGO, see the licence rule below)
engine/capture/  DRAFT data-capture tool, not reviewed, design paused. Do not build on it
eval/            evaluation sets and their READMEs (eval/dictation_cleanup/ is the clean-up test set)
eyeable-frontend/eyeable-frontend/   website and live demo (plain HTML, CSS, JS on GitHub Pages)
scripts/         installer, launcher, preflight, release zip (PowerShell)
tests/           unit tests and browser checks
docs/            STATUS.md, DECISIONS.md, brand/, media/
```

Path-specific rules load automatically from `.claude/rules/` when you work in `engine/voice/`, `eval/`, `engine/eyes/`, `engine/capture/` or the website.

## Hard rules: never break these

1. **Privacy.** Camera frames, audio and screenshots stay in memory and are never written to disk. Nothing leaves the machine at runtime except first-run model downloads. No API keys or tokens in code. Never read `.env`.
2. **Never commit:** `models/`, `data/`, `checkpoints/`, `downloads/`, recordings, `calibration/`, `config.local.json`, `.env`, `.venv`, `node_modules`, files over 50 MB, or private notes (`*Handoff*.md`, `*Build Plan.md`, `*_Project_Description.md`, `CLAUDE.local.md`). Stage named folders (`git add engine/voice`), never `git add .` or `git add -A`.
3. **TORGO is academic and non-profit only.** Never publish, upload, ship or commit the TORGO data or any model or adapter trained on it.
4. **Licences must allow commercial use.** Any new model, dataset or package needs Apache 2.0, MIT, BSD, CC BY or similar. Check the licence on the source's own page before adding it, and record it in `docs/DECISIONS.md`. Qwen2.5-3B (research licence) and UI-Venus 1.5 (conflicting licence statements) are being replaced: build nothing new on them.
5. **Two remotes.** `origin` is the private repo (`Phuc-Khanq/-i-able`, on Khang's account): all work goes there. `public` is the old public repo (`Phuc-Khanq/-eye-able`), kept as the hackathon archive and website host. Never push `main` or any working branch to `public`: that publishes the private code to the world. Even in the private repo: no secrets and no personal details of users or testers.
6. **Never commit to `main`, never force-push, never rewrite pushed history.** Ask before every `git push`.

## Code conventions (full list in CONTRIBUTING.md)

- Open text files with `encoding="utf-8"`. Build paths with `pathlib` from the repo root; no drive letters or user paths.
- Import heavy libraries (torch, transformers, faster-whisper) inside the function that uses them, so the engine starts without CUDA.
- Parts talk only through `engine/core/types.py`, `engine/core/messages.py` and the event bus. A change to either contract file needs Khang's OK first, and the website's copy changes in the same PR.
- Never block the camera or audio thread; models run in worker threads.
- Defaults in `engine/config.py` must work on the reference machine. Machine-specific overrides go in `config.local.json`.
- A new package goes into `requirements.txt` with an exact version (`name==1.2.3`) in the same commit as the code that needs it. Training-only packages go in `requirements-train.txt`.

## Git workflow

- Work on the route branch: `voice` (dictation, speech, `eval/`), `eyes` (pointer, gestures, capture), `ui` (website). Start with `git switch <route>` then `git pull origin main`.
- Commit messages start with the route: `voice:`, `eyes:`, `ui:`, `core:`, `eval:`, `docs:`. Small commits, one idea each.
- Every commit is under the developer's own git name. No `Co-Authored-By` lines and no Claude or "Generated with" attribution in commit messages or pull requests.
- Merge into `main` only through a pull request (template in `.github/`). Khang merges after `scripts\run_demo.ps1` works on the reference machine.

## How to work here

- **Plan first** when a change touches more than three files, a contract, `config.py` defaults, or a model. Give a short plan and wait for Khang's OK.
- **Verify before saying done.** Run the tests for what you changed plus the compile check. If something needs the camera, mic or screen and you couldn't run it, say exactly what Khang should check by hand.
- **Measure, don't guess.** Report only numbers you measured, with how and on what machine. Say plainly when something is unmeasured.
- **Ask before** downloading anything over 1 GB, starting a run that takes more than 10 minutes, or using the GPU while the engine might be running. Give the expected size and time. The 16 GB card fits one heavy job at a time.
- **Stay in scope.** Do the task asked. Put other ideas in STATUS.md under "Ideas" instead of building them.
- **Simple and reliable beats clever.** This has to hold up in a live demo and for users who can't use a mouse to recover from errors.
- **Explain in plain language.** Khang reads the summary; lead with the result, then what changed, then what he needs to do.

## Keeping the record (Claude Code and Cowork share these files)

Khang uses two Claude sessions: this one for code, and a Cowork chat for research, planning, docs and a weekly tracker. Cowork reads this folder, so these files are how the two stay in sync.

- End of session: Khang runs /wrap-up. Don't wrap up on your own. docs/SESSION_LOG.md is the session-by-session record Cowork reads first.
- **Every decision:** add a dated entry to `docs/DECISIONS.md` with the reason and the evidence (link or measured number). Never edit an old entry; add a new one that replaces it.
- **"Weekly summary":** when Khang asks, write it from STATUS.md, DECISIONS.md and `git log --since="last friday" --all --oneline`, in this shape: Done, In progress, Decisions, Measured numbers, Blockers, Questions for the team. Plain text Khang can paste into Cowork.

@docs/STATUS.md
