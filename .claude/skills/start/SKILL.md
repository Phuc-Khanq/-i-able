---
name: start
description: Start-of-session briefing. Reads STATUS, the latest SESSION_LOG entries and git, then gives Khang a short picture and a plan for the task he names. Run only when Khang types /start.
disable-model-invocation: true
---

Start a new session on [eye]able. Earlier sessions and the Cowork chat left their state in the files, not in this conversation, so read them before doing anything.

The task for this session (may be empty): $ARGUMENTS

1. Catch up from the files, don't guess:
   - `docs/STATUS.md` (already loaded through CLAUDE.md) and the newest two entries of `docs/SESSION_LOG.md`.
   - `docs/DECISIONS.md` if the task touches anything a decision covers (models, licences, the test set, repos, names).
   - If Khang mentions something "decided in Cowork" that isn't in DECISIONS.md, ask him for it and add it.

2. Check the machine:
   - `git branch --show-current`, `git status --short`, `git log --oneline -5`, and whether the current branch is ahead of `origin`.
   - Whether the engine, a training run or an evaluation is running (Python processes running `engine.main`, `finetune`, `cleanup_train` or `eval.`), and whether the GPU is busy (`nvidia-smi`).
   - Other Code chats: read "Code chats now" in STATUS.md and run `ListAgents`. Say which other chat is active and what it owns. If your chat has no line there yet, add one (name, job, the files you'll edit, GPU use) and agree it with Khang. Follow CLAUDE.md, "Two Code chats at once".

3. Pick the branch the task belongs on (CLAUDE.md, "Git workflow"). If another chat is active in this folder, stay on the current branch: switching would change its files. Otherwise, if the current branch is `main` or doesn't fit the task, say which branch you would use and why, and switch only after Khang says yes.

4. If the task is empty, list the top three items from "Next" in STATUS.md and ask which one to do. Stop there.

5. Reply in at most 12 lines, plain language, in this shape:

   **Where things stand:** 1 or 2 lines from the latest SESSION_LOG entry and STATUS.md
   **Machine:** branch, uncommitted files, anything running or unpushed, the other chat and what it owns
   **Task:** the task in your own words, plus anything unclear about it
   **Plan:** 2 to 5 short steps
   **Needs Khang:** approvals this task will need (downloads over 1 GB, runs over 10 minutes, GPU use, contract or config changes, pushes), or none

6. Then:
   - If the task touches more than three files, a contract, `config.py` defaults or a model, wait for Khang's OK on the plan (CLAUDE.md, "Plan first").
   - Otherwise start on the first step straight away.

7. Don't wrap up on your own: Khang ends the session with `/wrap-up`.
