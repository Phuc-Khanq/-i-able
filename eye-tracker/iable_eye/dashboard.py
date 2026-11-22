"""S018 dashboard: one self-contained HTML page of V2 reports (SIM and REAL apart).

The page has no script, no link, no image and no external resource: only HTML text,
one small inline style block and one inline <svg> bar chart per entry.
Simulated entries live only in <section id="sim">, real entries only in <section id="real">.
People appear as codes (P001, S001), never names.

Bar height for a measured cell: h = 100 * m / m_max (m = cell median, m_max = the largest
measured median of that entry), rounded to 1 decimal.
"""
from __future__ import annotations

import html
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from iable_eye.metrics import CELLS, MIN_DOTS, V2Report
from iable_eye.paths import DATA_ROOT, ROOT, assert_data_path
from iable_eye.session_store import DISTANCES, LIGHTS

DASH_VERSION = 1
FORBIDDEN = ("<script", "<img", "<link", "<iframe", "<object", "src=", "href=", "url(", "@import", "http", "data:")
HONEST_LINES = ("Three or four team members is not a fairness test.",
                "Eliminates environmental bias: not yet measured (needs PLAN targets V1 and V2 on real recordings).")

_PERSON_RE = re.compile(r"^[PS][0-9]{3}$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_TIME_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}$")
_NAME_RE = re.compile(r"^[a-z0-9_]+\.html$")

SIM_HEADING = "SIMULATED (not a product result)"
REAL_HEADING = "REAL"
NO_REAL = "no real sessions yet: not yet measured"

_STYLE = ("body{font-family:sans-serif;margin:1em;max-width:60em}"
          "section{border:2px solid #444;padding:0.5em 1em;margin:1em 0}"
          "#sim{background:#fff4e0}#real{background:#e8f4ff}"
          "table{border-collapse:collapse}td,th{border:1px solid #888;padding:0.2em 0.5em}"
          "rect{fill:#557}")

_HELP = (
    "How to read this page.",
    "Each block is one person code and one V2 report.",
    "A cell is one light (bright, dim, backlit) and one distance (near, normal, far).",
    f"A cell is measured only if it has at least {MIN_DOTS} valid dots.",
    "The number in a cell is the median dot error in screen pixels, then the dot count and the lost rate.",
    "V2 is the worst measured cell median divided by the best one. V2 = 1.00 means every cell is equally good.",
    "V2 is shown only when all 9 cells are measured; otherwise it says UNMEASURED.",
    "The partial V2 uses only the measured cells; it is not a product number.",
    "The interval is the bootstrap interval for V2 (none if V2 is unmeasured).",
    "Reasons list every gate that failed (empty means the gates passed).",
    "Gates sha is the first 12 hex of the gates file hash used for this report.",
    "The bars show each measured cell median relative to the largest one (tallest bar = worst cell).",
    "Simulated numbers come from the simulator. They are never product results.",
)


class DashboardError(ValueError):
    """A dashboard request was refused. `code` says why (one short string)."""

    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code


@dataclass(frozen=True)
class DashEntry:
    person: str
    report: V2Report
    interval: tuple | None
    gates_sha: str


def _e(x) -> str:
    return html.escape(str(x), quote=True)


def _check_entry(en) -> None:
    if not isinstance(en, DashEntry):
        raise DashboardError("entry", f"need a DashEntry, got {type(en).__name__}")
    if not isinstance(en.person, str) or not _PERSON_RE.match(en.person):
        raise DashboardError("person", f"person must be a code like P001 or S001, got {en.person!r}")
    if not isinstance(en.report, V2Report):
        raise DashboardError("report", f"report must be a V2Report, got {type(en.report).__name__}")
    want = "real" if en.person[0] == "P" else "sim"
    if en.report.source != want:
        raise DashboardError("source", f"{en.person} needs a {want} report, got {en.report.source!r}")
    if not isinstance(en.gates_sha, str) or not _SHA_RE.match(en.gates_sha):
        raise DashboardError("sha", "gates_sha must be 64 lowercase hex")
    iv = en.interval
    if iv is not None:
        ok = isinstance(iv, tuple) and len(iv) == 2 and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in iv)
        if not ok:
            raise DashboardError("interval", f"interval must be None or (lo, hi) finite, got {iv!r}")
    if tuple(c.cell for c in en.report.cells) != CELLS:
        raise DashboardError("cells", "report cells must be the 9 cells in CELLS order")


def _check_time(t) -> None:
    if not isinstance(t, str) or not _TIME_RE.match(t):
        raise DashboardError("time", f"generated_at must look like 2026-11-18T23:00:00, got {t!r}")
    try:
        datetime.strptime(t, "%Y-%m-%dT%H:%M:%S")
    except ValueError as exc:
        raise DashboardError("time", f"generated_at is not a real time: {t!r}") from exc


def _cell_text(c) -> str:
    if c.measured:
        return f"{c.median_px:.1f} px (n {c.n}, loss {c.loss * 100:.1f} %)"
    if c.n == 0 and c.lost == 0:
        return "no data"
    return f"UNMEASURED (n={c.n} < {MIN_DOTS})"


def _bar_heights(report: V2Report) -> list[tuple[str, float]]:
    measured = [c for c in report.cells if c.measured]
    if not measured:
        return []
    m_max = max(c.median_px for c in measured)
    out = []
    for c in measured:
        h = 0.0 if m_max <= 0 else round(100.0 * c.median_px / m_max, 1)
        out.append((c.cell, h))
    return out


def _svg(report: V2Report) -> list[str]:
    bars = _bar_heights(report)
    lines = ['<svg width="300" height="120" viewBox="0 0 300 120" role="img" '
             'aria-label="bar per measured cell, height relative to the largest median">']
    for i, (cell, h) in enumerate(bars):
        x = 5 + 32 * i
        y = round(100.0 - h, 1)
        lines.append(f'<rect data-cell="{_e(cell)}" x="{x}" y="{y:.1f}" width="26" height="{h:.1f}"></rect>')
    if not bars:
        lines.append('<text x="5" y="60">no measured cell</text>')
    lines.append("</svg>")
    return lines


def _fmt_num(x, digits: int) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "no data"
    return f"{x:.{digits}f}"


def _entry(en: DashEntry) -> list[str]:
    r = en.report
    v2 = f"{r.v2:.2f}" if r.v2 is not None else f"UNMEASURED ({r.n_measured} of 9 cells measured)"
    iv = "none" if en.interval is None else f"[{en.interval[0]:.2f}, {en.interval[1]:.2f}]"
    rows = [
        ("person", en.person),
        ("label", r.label),
        ("status", r.status),
        (r.label, v2),
    ]
    if r.v2_partial is not None:
        rows.append(("partial V2 (measured cells only)", f"{r.v2_partial:.2f}"))
    rows += [
        ("interval", iv),
        ("worst cell", r.worst_cell if r.worst_cell is not None else "none"),
        ("best cell", r.best_cell if r.best_cell is not None else "none"),
        ("overall median px", _fmt_num(r.overall_median_px, 1)),
        ("reasons", ", ".join(r.reasons) if r.reasons else "none"),
        ("gates sha", en.gates_sha[:12]),
    ]
    out = [f'<div class="entry" id="entry-{_e(en.person)}">', f"<h3>{_e(en.person)}</h3>", "<dl>"]
    for k, v in rows:
        out.append(f"<dt>{_e(k)}</dt><dd>{_e(v)}</dd>")
    out.append("</dl>")
    by_cell = {c.cell: c for c in r.cells}
    out.append("<table>")
    out.append("<tr><th>light / distance</th>" + "".join(f"<th>{_e(d)}</th>" for d in DISTANCES) + "</tr>")
    for light in LIGHTS:
        tds = "".join(f'<td data-cell="{_e(f"{light}/{d}")}">{_e(_cell_text(by_cell[f"{light}/{d}"]))}</td>'
                      for d in DISTANCES)
        out.append(f"<tr><th>{_e(light)}</th>{tds}</tr>")
    out.append("</table>")
    out += _svg(r)
    out.append("</div>")
    return out


def render_dashboard(entries, *, generated_at: str) -> str:
    """Render the entries into one HTML string. Same input gives the same bytes."""
    _check_time(generated_at)
    entries = list(entries)
    for en in entries:
        _check_entry(en)
    sims = [en for en in entries if en.report.source == "sim"]
    reals = [en for en in entries if en.report.source == "real"]

    out = ["<!DOCTYPE html>", '<html lang="en">', "<head>", '<meta charset="utf-8">',
           f"<title>Iable eye tracker dashboard v{DASH_VERSION}</title>",
           f"<style>{_STYLE}</style>", "</head>", "<body>",
           f"<h1>Iable eye tracker dashboard v{DASH_VERSION}</h1>",
           f'<p id="generated">generated at {_e(generated_at)}</p>']
    out.append('<section id="sim">')
    out.append(f"<h2>{_e(SIM_HEADING)}</h2>")
    if not sims:
        out.append("<p>no simulated entries</p>")
    for en in sims:
        out += _entry(en)
    out.append("</section>")
    out.append('<section id="real">')
    out.append(f"<h2>{_e(REAL_HEADING)}</h2>")
    if not reals:
        out.append(f"<p>{_e(NO_REAL)}</p>")
    for en in reals:
        out += _entry(en)
    out.append("</section>")
    out.append('<div id="honest">')
    for line in HONEST_LINES:
        out.append(f"<p>{_e(line)}</p>")
    out.append("</div>")
    out.append('<div id="help">')
    for line in _HELP:
        out.append(f"<p>{_e(line)}</p>")
    out.append("</div>")
    out += ["</body>", "</html>", ""]
    text = "\n".join(out)
    _scan(text)
    return text


def _scan(text: str) -> None:
    low = text.lower()
    bad = [f for f in FORBIDDEN if f in low]
    if bad:
        raise DashboardError("forbidden", f"page contains forbidden items {bad}")


def write_dashboard(html_text: str, data_root=DATA_ROOT, root=ROOT, name: str = "dashboard.html") -> Path:
    """Write the page to <data_root>/dashboard/<name> (tmp file, then os.replace)."""
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise DashboardError("name", f"name must match ^[a-z0-9_]+\\.html$, got {name!r}")
    if not isinstance(html_text, str):
        raise DashboardError("html", "html_text must be a str")
    _scan(html_text)
    path = assert_data_path(Path(data_root) / "dashboard" / name, data_root=data_root, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{name}.tmp{os.getpid()}")
    try:
        tmp.write_bytes(html_text.encode("utf-8"))
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path
