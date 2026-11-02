# Decisions

One entry per decision, newest at the bottom. Never edit an old entry: add a new one that says what it replaces.

Format: date, decision, why, evidence, who decided.

---

**2026-11-02. Only commercially usable licences.**
[eye]able will be sold, so every model, dataset and package must allow commercial use (Apache 2.0, MIT, BSD, CC BY or similar). Check the licence on the source's own page before adding anything.
Decided by: Khang.

**2026-11-02. Drop Qwen2.5-3B for the dictation clean-up; don't fine-tune it.**
Its licence is `qwen-research`, and fine-tuned copies keep that licence.
Evidence: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct (licence field). Decided by: Khang.

**2026-11-02. The clean-up uses the Qwen3.5 family.**
Test Qwen3.5-4B first, with thinking off. Qwen3.5-9B is a stretch goal. For weaker laptops try Qwen3.5-2B, then 0.8B, then Qwen3-1.7B. Stay inside the Qwen family so prompts, code and training stay consistent.
Why: Apache 2.0, one family with sizes from 0.8B to 9B, strongest small models in the comparison we made.
Evidence: model cards list Apache 2.0 for Qwen3.5-2B, 4B and 9B, and Qwen3-1.7B (huggingface.co/Qwen). A third-party test reports about 1.3 GB (2B) and 2.7 GB (4B) for 4-bit files, and 18 and 8.5 tokens/s on a 16-core server CPU (bkataru.github.io/igllama/showcase/qwen35-small). Not yet measured by us.
Decided by: Khang.

**2026-11-02. Rejected alternatives for the clean-up.**
Llama 3.2 3B: custom licence with naming and attribution rules. Phi-4-mini (MIT) and Gemma 4 (Apache 2.0 on its model cards): good, but no reason to leave the Qwen family. SmolLM3-3B: weaker on our comparison. Decided by: Khang.

**2026-11-02. Measure before training.**
Build the 100-case test set first, run prompt-only baselines, and fine-tune only the gap that shows. A third of the cases are held out and never used for tuning.
Evidence: `eval/dictation_cleanup/README.md`. Decided by: Khang.

**2026-11-02. Head-tracking training data: numbers only, five targets.**
Record tracker numbers over time, never video. Learn direction intent, gesture vs not, voluntary vs involuntary motion, personal sensitivity and gaze. Main concerns: direction intent and voluntary vs involuntary motion. Detailed design waits for Khoi's tracker research.
Decided by: Khang.

**2026-11-02. Claude Code for code, Cowork for coordination.**
Code, tests, evaluation runs and training happen in Claude Code on Khang's PC. Research, planning, docs and the weekly tracker happen in Cowork. `docs/STATUS.md` and this file connect the two.
Decided by: Khang.

**2026-11-02. Development moves to a private repo on Khang's account.**
Two of the original five left and the product is going a different way, so new work shouldn't be public. New private repo `Phuc-Khanq/-i-able` (remote `origin`) holds the full history: every local branch and tag was pushed on 2026-11-02. The old public repo `Phuc-Khanq/-eye-able` (remote `public`) stays as the hackathon archive and website host; nothing new is pushed to it. Replaces CLAUDE.md's old rule "the repo is public".
Known limits: the code already public stays usable by anyone under its MIT licence; going private only protects new work. The licence for new code and IP agreements with past and current contributors are still open (a lawyer's question).
Decided by: Khang.

**2026-11-02. Commits carry only the developer's name.**
No `Co-Authored-By` lines and no Claude or "Generated with" attribution in commits or pull requests. Khang wants every commit under his own name. Two commits pushed before this rule (0a26717, 0116961) keep the line, since pushed history isn't rewritten. Recorded in CLAUDE.md, Git workflow.
Decided by: Khang.

**2026-11-02. The clean-up evaluation works on branch `voice-cleanup-eval`.**
`origin/voice` no longer existed and local `voice` was 59 commits behind `main`, so the branch was made fresh from `main`. Local `voice` was left at fd85b60 (already in `main`).
Decided by: Khang.

**2026-11-02. Small findings from the first code review.**
- The exact-version rule in CLAUDE.md applies to new packages only. Existing lines in `requirements.txt` (e.g. `transformers>=4.57`) stay as they are.
- `eval/dictation_cleanup/testset.csv` starts with a BOM on purpose, so Excel opens it correctly. Code reads it with `encoding="utf-8-sig"`.
- The docstring and fallback default in `engine/voice/rewrite.py` still name Qwen2.5-1.5B (config uses 2.5-3B). Fix it as part of the model switch, not before.
Decided by: Khang.

**2026-11-02. Sessions end with /wrap-up.**
Khang runs `/wrap-up` (`.claude/skills/wrap-up/SKILL.md`). It writes a new entry at the top of `docs/SESSION_LOG.md` and updates STATUS.md and this file, so the Cowork chat knows what happened. Claude doesn't wrap up on its own. Replaces the CLAUDE.md rule "update STATUS.md at the end of every session".
Decided by: Khang.

**2026-11-02. Scoring runner spec for the clean-up evaluation.**
Measure only, no app changes. Runner in `eval/dictation_cleanup/`, results in `eval/dictation_cleanup/runs/` (git-ignored). Backends "current" (today's clean-up exactly as the app runs it, the baseline), "transformers" (any model on the GPU; Qwen3.5 with thinking off and the card's non-thinking sampling with a fixed seed) and "fake" (tests). Prompts are versioned text files: "plain" (asr_best only) and "full" (plus alternatives, app, context, personal words). New models answer in JSON with `text`, `alternatives` and `uncertain`; broken JSON counts as a failure, with no silent retry. Dev split by default; holdout only with an explicit flag, after the pass bar is set.
Decided by: Khang.

---

## Proposed, not yet agreed

- Clean-up levels the user chooses: Exact (as said), Tidy (fix grammar), Full help (work out what was meant).
- The clean-up gets Whisper's top candidates, the last text typed, the app and the user's word list as input.
- The guard changes from "reject if the text differs a lot" to "reject if it adds facts nobody said".
- Tiers: light model by default on weak laptops, heavier model where the GPU allows.
