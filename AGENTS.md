# AGENTS.md (for Codex and other coding agents)

[eye]able lets people control a Windows computer with their head, blinks and voice using a webcam and a microphone. It
is a product we will sell. The main users can't use their hands. Khang is the only person who decides, merges and pushes.

Before any work, read, in this order:
1. `docs/AGENT_RULES.md` (how several agents share this project, and how to take over from another agent)
2. `CLAUDE.md` (the project rules; it is written for Claude, but the rules apply to you too; ignore its `@` import
   lines and the Claude-only commands)
3. `docs/STATUS.md`, the newest entries of `docs/SESSION_LOG.md`, `docs/COWORK_NOTES.md`

Hard rules, if you read nothing else:
- Windows and PowerShell. Python 3.11 in the repo's `.venv`: `.\.venv\Scripts\python.exe`.
- Camera frames, audio and screenshots stay in memory and are never written to disk. No network calls in product
  code. No keys in code. Never read `.env`.
- Never open, upload or send the git-ignored folders `data/`, `calibration/`, `checkpoints/`, `models/`, `downloads/`.
  TORGO data and anything trained on it is academic-only and never leaves the machine.
- New models, datasets and packages need a licence that allows commercial use; record it in `docs/DECISIONS.md`.
- No `Co-Authored-By` lines and no AI attribution in commits. Commit messages start with the route (`voice:`, `eyes:`,
  `ui:`, `core:`, `eval:`, `docs:`). Never push. Never switch branch, pull, merge, stash or reset while another agent
  is active.
- Ask Khang before downloading over 1 GB, running anything longer than 10 minutes, or using the GPU.
- Update `docs/STATUS.md`, `docs/SESSION_LOG.md`, `docs/DECISIONS.md` and `docs/COWORK_NOTES.md` after every finished
  piece of work, then commit your own files.
- Report only numbers you measured, and say what needs the camera, mic or screen and must be checked by hand.
