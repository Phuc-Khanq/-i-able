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

- **2026-11-22, Cowork, Done: new file `docs/AGENT_RULES.md` (and `AGENTS.md` for Codex).** Khang wants us to keep
  working when Claude hits its usage limit, with Codex and ChatGPT chats standing in. The rules cover handoffs (a fixed
  HANDOFF entry in this file), taking over from another agent, one writer at a time, and what must never be sent to
  another AI company (TORGO, personal voice data, camera data, git-ignored folders). It follows `CLAUDE.md` and does not
  repeat it; on a conflict `CLAUDE.md` wins. Both new files are uncommitted. Code: read `docs/AGENT_RULES.md`; the first
  Code chat that reads this commits `docs/AGENT_RULES.md` and `AGENTS.md` (`git add -f` for docs), and adds one line to
  `CLAUDE.md` under "Keeping the record" pointing to `docs/AGENT_RULES.md` (no `@` import, to keep CLAUDE.md short).
  Then mark this Done.
  Answer (2026-11-22, Dictation chat): read; committed both files, added the line to CLAUDE.md. Not pushed.

- **2026-11-22, Cowork, Open: the SAP data can be used commercially, under a data use agreement (DUA).** The SAP FAQ
  says the data "can be used for commercial development", but it cannot go to any entity that has not signed the DUA.
  The DUA template (University of Illinois form, approved Nov 2025) sets the Permitted Purpose to include training
  machine learning and AI models "for commercial purposes". Limits: no sharing with third parties; only the recipient's
  staff and contractors may use it; delete a contributor's data when they withdraw; only contributors whose metadata
  allows presentations or demos may appear in a demo ("Authorized Samples"); at most 999 people's biometric identifiers;
  data originated in the US. Sources: `speechaccessibilityproject.beckman.illinois.edu` FAQ and the DUA PDF. Code: add
  to DECISIONS.md once Khang confirms.
- **2026-11-22, Cowork, Open: question for the lawyer on the DUA.** A trained model counts as a "Result" (not Data) only if
  it includes no more than a de minimis part of the Data. The DUA also says "more than twenty Authorized Samples on which
  the use is based", which reads unclearly for a model trained on thousands of clips. Ask the lawyer how that clause
  applies to our model.
- **2026-11-22, Cowork, Open: Khang fills in the SAP proposal.** Only Khang's blanks are left (recipient, scientist,
  end date, purpose text). Claude Doc https://claude.ai/artifact/SdxuomVgQfoY3paCui9Bzv.
- **2026-11-22, Cowork, Open: other consented sources to check.** My Voice Library (protocol at
  researchprotocols.org, 2026/1/e97614) is a consented dysarthria collection built with a commercial company. Its
  access and licence terms are not checked yet. UA-Speech says "commercial uses are not permitted", and TORGO is only
  under the LDC non-member agreement (academic). Neither should be used for the product model.

## Code to Cowork (read by Cowork)

- **2026-11-22, Face-gesture chat, Open: SensePilot's terms before Khang runs its trial.** We study SensePilot, a
  competing head-mouse app, to learn which settings a mature product exposes (survey in
  `docs/plans/face_behaviours_survey.md`, branch `eyes-face-gestures`; so far from public pages only). Before Khang
  installs its 30-day trial to note its default settings by eye: do SensePilot's licence or trial terms forbid
  using the trial to evaluate or benchmark it, or for competitive analysis? Its website shows no terms page we could
  find; the installer may show an EULA. Also for the lawyer later: any patents (SensePilot's or Google's) on mapping
  facial gestures to keys at staged strength levels, or on toggle/hold gesture modes.
- **2026-11-22, Face-gesture chat, Open: where is Khoi's `iable_eye`?** The face-gesture design should build on his
  package (features, swappable landmark adapter, simulator), but it is not on Khang's PC or in the repo. A path or
  repo, and whether it may live in this repo, would help.

- **2026-11-22, Dictation chat, Open: where things stand.** Since 2026-11-19 the dictation path was reworked after
  studying Wispr Flow; every number is in `docs/dictation_learning_from_wispr.md`, the short version in STATUS.md
  ("Now" and "Done recently"). Nothing is tried with mic and camera yet (Khang's checklist is in that doc).
- **2026-11-22, Dictation chat, Open: Speech Accessibility Project proposal.** The only route found to consented
  dysarthric recordings we may sell with; stock Whisper gets about 41% of words wrong on dysarthric speakers, and
  software around it only brings that to about 40%. Needs Khang's blanks filled and sent. Claude Doc
  https://claude.ai/artifact/SdxuomVgQfoY3paCui9Bzv. Cowork: anything else worth contacting (hospitals, ALS or
  cerebral palsy groups) for consented recordings, with the licence terms each allows?
  Answer (2026-11-22, Cowork): the DUA allows commercial model training (see the first Cowork to Code entry). Khang's
  form is still open (second Cowork to Code entry). Hospitals and ALS or CP groups were not searched yet; My Voice
  Library is the one new lead.
- **2026-11-22, Dictation chat, Open: Whisper adapted to one voice is not worth an onboarding step yet.** On
  Khang's own voice: 25 wrong words of 241 stock, 19 to 22 adapted; half of that is names and number spelling the
  word list handles anyway. Cowork: did Wispr or others publish how much a few minutes of a user's own speech
  helps? Not needed for a decision now.
- **2026-11-22, Dictation chat, Open: rename.** Needs the exact name (display form, file form, how it is said), plus
  trademark and domain checks before it's final.
- **2026-11-22, Dictation chat, Open: for a lawyer.** Training on Claude-written sentences; shipping NVIDIA's CUDA DLLs
  with llama.cpp; the licence for new private code; IP agreements with everyone who wrote code (details in STATUS.md,
  "Blockers and risks").
