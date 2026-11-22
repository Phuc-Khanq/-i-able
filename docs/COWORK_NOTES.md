# Notes between Cowork and the Code chats

The two sides can't message each other live, so this file is the post box. Cowork and the Code chats both read it at
the start of every session. Keep it short: one entry per item, newest first, never delete an entry (mark it Done).
Facts and numbers live in `docs/STATUS.md`, `docs/DECISIONS.md` and `docs/SESSION_LOG.md`; this file only carries
questions, requests and answers.

**How to use it**
- **Cowork writes in "Cowork to Code"**: a request, a decision made in Cowork, a finding the code should react to.
  Start with the date and a status: `Open`. Khang can also paste anything here.
- **Code chats write in "Code to Cowork"**: what Cowork should research, check or decide, and a pointer to where the
  numbers are. Start with the date, the chat's name and `Open`.
- **Whoever answers** adds `Answer (date, who):` under the entry and changes `Open` to `Done`.
- A decision made in Cowork also gets a dated entry in `docs/DECISIONS.md` (the Code chat adds it when it reads it).
- Code chats update STATUS, SESSION_LOG, DECISIONS and this file every time they finish something, not at the end of
  a session (CLAUDE.md, "Keeping the record").

## Cowork to Code (read by the Code chats)

(nothing yet)

## Code to Cowork (read by Cowork)

- **2026-11-22, Dictation chat, Open: where things stand.** Since 2026-11-19 the dictation path was reworked after
  studying Wispr Flow; every number is in `docs/dictation_learning_from_wispr.md`, the short version in STATUS.md
  ("Now" and "Done recently"). Nothing is tried with mic and camera yet (Khang's checklist is in that doc).
- **2026-11-22, Dictation chat, Open: Speech Accessibility Project proposal.** The only route found to consented
  dysarthric recordings we may sell with; stock Whisper gets about 41% of words wrong on dysarthric speakers, and
  software around it only brings that to about 40%. Needs Khang's blanks filled and sent. Claude Doc
  https://claude.ai/artifact/SdxuomVgQfoY3paCui9Bzv. Cowork: anything else worth contacting (hospitals, ALS or
  cerebral palsy groups) for consented recordings, with the licence terms each allows?
- **2026-11-22, Dictation chat, Open: Whisper adapted to one voice is not worth an onboarding step yet.** On
  Khang's own voice: 25 wrong words of 241 stock, 19 to 22 adapted; half of that is names and number spelling the
  word list handles anyway. Cowork: did Wispr or others publish how much a few minutes of a user's own speech
  helps? Not needed for a decision now.
- **2026-11-22, Dictation chat, Open: rename.** Needs the exact name (display form, file form, how it is said), plus
  trademark and domain checks before it's final.
- **2026-11-22, Dictation chat, Open: for a lawyer.** Training on Claude-written sentences; shipping NVIDIA's CUDA DLLs
  with llama.cpp; the licence for new private code; IP agreements with everyone who wrote code (details in STATUS.md,
  "Blockers and risks").
