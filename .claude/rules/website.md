---
paths:
  - "eyeable-frontend/**"
  - "tests/*.js"
---

# Website and live demo

- Plain HTML, CSS and JavaScript with no build step. The live site is served by GitHub Pages (`.github/workflows/pages.yml`) from `main` of the old public repo (remote `public`), not from the private `origin`. Never push a working branch or `main` to `public`. How website changes reach the live site is still open (see `docs/STATUS.md`).
- The live demo runs MediaPipe in the browser. The camera picture never leaves the tab.
- When the engine runs on the same PC, the page connects to it over a WebSocket on 127.0.0.1. Message shapes must match `engine/core/messages.py`.
- Accessibility first: large targets, high contrast, clear feedback on the current mode.
- Keep the existing look (designed by Nhi). Ask Khang before changing colours, type or layout.
- Run `node --test tests/lite_core.test.js tests/lite_voice.test.js` after any change here.
