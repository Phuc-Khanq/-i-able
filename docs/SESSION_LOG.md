# Session log (newest first)

## 2026-11-02 18:15: First Claude Code session: setup, private repo, wrap-up command, scoring runner started
Branch: voice-cleanup-eval · Commits: 0a26717 docs: Claude Code setup; 0116961 eval: dictation clean-up test set; 7b4e979 docs: development moves to a private repo; 45a58c0 docs: no Claude attribution in commits or PRs; 10b545c docs: /wrap-up command for the end of a session · Uncommitted: engine/capture/, tests/test_capture.py (unreviewed draft, left untracked on purpose)
**Done:**
- Moved the Claude Code settings and rules into `.claude/`. Read the setup docs and compared them with the code (mismatches are listed in STATUS.md, "How the clean-up guard works today").
- Committed the setup files and the 100-case test set on a new branch `voice-cleanup-eval`, made from `main`.
- Moved development to the private repo `Phuc-Khanq/-i-able` (remote `origin`): all 9 local branches and 5 tags pushed. The old public repo is now remote `public`.
- CLAUDE.md: commits carry only the developer's name (no Claude co-author line). New `/wrap-up` command and this log.
- Scoring runner: Khang's spec compared with Claude's plan; no big differences, build started (no code yet).
**Measured:** none
**Decisions:** private repo on Khang's account (new work shouldn't be public); no Claude attribution in commits (every commit under the developer's name); work branch `voice-cleanup-eval` (origin/voice was gone, local voice 59 commits behind main); exact pins only for new packages; testset.csv BOM is intentional (Excel); fix the 1.5B docstring in rewrite.py during the model switch; sessions end with /wrap-up; scoring runner spec as Khang wrote it. All in DECISIONS.md.
**Still running:** none
**Problems / blockers:** none blocking. The Pages workflow will fail in the private repo until it's switched off there. How the live website gets updated is still open (pushing `main` to `public` would publish private code).
**Next:**
1. Build the scoring runner in `eval/dictation_cleanup/`, tests first, then run the full test suite and the compile check.
2. Run the "current" clean-up (Qwen2.5-3B) on the dev split.
3. Download Qwen3.5-4B (after Khang's OK) and run it with the plain and full prompts.
**Khang must do by hand:** review the test set with Khoi; remove the two people who left from `-eye-able`'s collaborators; check that Khoi and Hung accepted the invites to `-i-able` and ran `git remote set-url origin https://github.com/Phuc-Khanq/-i-able.git`; approve the Qwen3.5-4B download; approve the next push.
**For Cowork:** how the website should be updated now that the code is private (e.g. a separate public repo with only `eyeable-frontend/`); the licence for new private code and IP agreements with everyone who wrote code (a lawyer's question); Silero VAD licence (MIT vs a CC BY-NC badge); a licence scan of all Python packages.
**Notes from Khang:** none
