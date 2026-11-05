---
name: wrap-up
description: End-of-session wrap-up. Updates docs/SESSION_LOG.md, docs/STATUS.md and docs/DECISIONS.md so the Cowork chat knows what happened. Run only when Khang types /wrap-up.
disable-model-invocation: true
---

Wrap up this session so the Cowork chat can pick up from here. Cowork reads these files from this folder. It can't see this conversation or run git, so everything it needs must be in the files.

Khang's notes for this session (may be empty): $ARGUMENTS

1. Gather facts, don't guess: run `git branch --show-current` and `git status --short`, list the commits made in this session, and check what was changed, run and measured in this conversation. Also check whether anything is still running (training, an evaluation, the engine) and where its output goes.

2. Add a new entry at the TOP of docs/SESSION_LOG.md (create the file with a one-line header if it's missing). Use exactly this shape, write "none" for empty lines, keep it under 35 lines, plain language:

## YYYY-MM-DD HH:MM [<chat name>]: <one-line title of what this session was about>
Branch: <branch> · Commits: <short hash + message, or none> · Uncommitted: <files, or none>
**Done:** what was finished
**Measured:** number, what it measures, how, on which machine (only numbers actually measured this session)
**Decisions:** decision + reason (also add them to DECISIONS.md)
**Still running:** jobs left running and where their output goes
**Problems / blockers:** what failed or is stuck, with the key error line
**Next:** the next 1 to 3 steps, in order
**Khang must do by hand:** reviews, approvals, pushes, camera or mic tests
**For Cowork:** questions to research, decisions that need the team, docs to write
**Notes from Khang:** the notes above, or none

3. Update docs/STATUS.md to match reality: the date, Now, In progress, Next, Blockers and Uncommitted work. Move finished items to "Done recently". Move "Done" items older than two weeks to "Earlier" as one dated line each. Keep it under about 80 lines. If another Code chat is active ("Code chats now"), edit only your own lines and your own chat's line there, and leave the other chat's lines as they are.

4. Add a dated entry to docs/DECISIONS.md for every decision made this session. Never edit old entries. If a "Proposed, not yet agreed" item was agreed or dropped, add an entry saying so.

5. If SESSION_LOG.md is over 300 lines, move everything except the newest 15 entries into docs/session-log-archive/<YYYY-MM>.md.

6. Check that these files contain no secrets, keys, tokens or personal details of users or testers, and that every number under "Measured" was measured this session.

7. Commit only these docs files, using `git add -f` (.git/info/exclude hides .md files), with the message "docs: session wrap-up YYYY-MM-DD". If the branch is main, don't commit: say so instead. Don't push. If the other chat has uncommitted edits in these same docs, they go into this commit too: say so in the reply, and message the other chat.

8. Reply with the new SESSION_LOG entry, then one line: "Ready for Cowork."
