"""The list of everything you can say, built from the code so it can never fall behind it.

Route: voice (Khang)
Publishes: nothing; prints Markdown
APIs: engine.voice.commands (FIXED_COMMANDS, COMMAND_INFO), engine.config (apps and sites), the mode manager's gestures

    python -m engine.voice.command_list                 # print the list
    python -m engine.voice.command_list > COMMANDS.md   # save it

To add a command: put its phrases in FIXED_COMMANDS and a row in COMMAND_INFO (engine/voice/commands.py), make the
mode manager do something for the name (a key combo goes in KEY_SHORTCUTS and nothing else), and run this again.
tests/test_command_list.py fails when a command has no row, so the list can't be forgotten.
"""
from __future__ import annotations

from collections import OrderedDict

from engine.voice import commands

CATEGORY_ORDER = ["Dictation", "Answers", "Click and mouse", "Scroll", "Modes", "Browser and tabs", "Editing",
                  "Windows", "Keys and moving around", "Volume and screen", "Setup"]

TRIGGER_NOTE = ('Say "Venus" or "Computer" first ("Venus, scroll down") to make sure a phrase is read as a command '
                'in any mode, even in the middle of dictated text.')

GESTURES = [
    ("Head", "Move the cursor"),
    ("Long blink (about 0.4 to 1.6 s, both eyes)", "Left click, snapped onto the nearest button; also says yes"),
    ("Eyebrow raise", "Says yes to a waiting question, preview or search; leaves Scroll mode. Does not click"),
    ("Head shake", "Says no; closes the menu"),
    ("Open the mouth and close it (off by default)", "Right click"),
    ("Tilt the head toward a shoulder (off in Pointer mode by default)", "Scroll up or down"),
]


def phrases_by_command() -> "OrderedDict[str, list[str]]":
    out: "OrderedDict[str, list[str]]" = OrderedDict()
    for phrase, name in commands.FIXED_COMMANDS.items():
        out.setdefault(name, []).append(phrase)
    return out


def rows(category: str) -> list[tuple[str, str]]:
    """(phrases, what it does) for a category, in the order they are written in FIXED_COMMANDS."""
    by = phrases_by_command()
    result = []
    for name, (cat, text) in commands.COMMAND_INFO.items():
        if cat == category and name in by:
            result.append((" · ".join(f'"{p}"' for p in by[name]), text))
    return result


def render(settings=None) -> str:
    if settings is None:
        from engine.config import settings
    lines = ["# [eye]able: what you can say", "",
             "The microphone is always on. In Pointer mode every sentence is checked for a command and anything else "
             "is ignored. Say \"dictate\" to start dictating; the card only shows what it hears while dictating.", "",
             TRIGGER_NOTE, ""]
    seen = set()
    for cat in CATEGORY_ORDER + [c for c, _ in commands.COMMAND_INFO.values() if c not in CATEGORY_ORDER]:
        if cat in seen:
            continue
        seen.add(cat)
        table = rows(cat)
        if not table:
            continue
        lines += [f"## {cat}", "", "| Say | What it does |", "| --- | --- |"]
        lines += [f"| {say} | {does} |" for say, does in table]
        lines.append("")
    lines += ["## Fixing words on the card", "",
              "While a preview waits. It only counts as a fix when those words are on the card; otherwise it is "
              "dictated like any other sentence. Say \"Venus\" first to make sure it is read as a fix.", "",
              "| Say | What it does |", "| --- | --- |",
              '| "change <words> to <new words>", "replace <words> with <new words>", "correct <words> to <new words>" '
              '| Swaps those words on the card (the last place they appear) |',
              '| "delete <words>", "remove <words>" | Takes those words off the card |',
              '| "spell H A T H A W A Y" | Adds the spelled word to the card (in Pointer mode it starts a preview). '
              'Say "capital H" for a capital letter, "double L" for two, "all caps" first for capitals |',
              '| "change <words> to T H E R E", "change <words> to spell T H E R E" | Swaps those words for the '
              'spelled one |',
              '| "scratch that" | Takes back the last thing you said or your last fix, one step at a time |', ""]
    lines += ["## Your words", "",
              "Names and terms the speech recogniser should lean toward. They help from the next sentence on. "
              "A word you spell, or a name you put in with \"change\", joins the list by itself when the card is "
              "typed. The same fix made twice (\"change waiter to water\") is made for you after that; change it "
              "back to undo it. \"Delete my voice data\" deletes the list and the fixes.", "",
              "| Say | What it does |", "| --- | --- |",
              '| "add <word> to my words", "add H A T H A W A Y to my words" | Adds a word or name to your list |',
              '| "add that to my words" | Adds the word you just spelled or changed |',
              '| "remove <word> from my words" | Takes it off your list |', "",
              "## Snippets", "",
              "Text you would otherwise dictate again and again (an address, an email, a sign-off). Dictate it once "
              "and fix it on the card, then save it.", "",
              "| Say | What it does |", "| --- | --- |",
              '| "save this as a snippet called my address" | Saves the card\'s text as the snippet "my address" |',
              '| "insert my address" (or "my address" said alone while dictating) | Puts the saved text on the card; '
              '"yes" types it |',
              '| "delete the snippet my address" | Deletes it |', ""]
    apps = ", ".join(sorted(settings.launch_apps))
    sites = ", ".join(sorted(settings.launch_sites))
    lines += ["## Apps and sites", "", "| Say | What it does |", "| --- | --- |",
              f'| "open <app>" | Opens an app from this list: {apps} |',
              f'| "go to <site>" | Opens a site: {sites}, or any name.com |',
              '| "search for <words>" | Searches the web |',
              '| "click <name>", "select <name>", "open <name>" | Looks for a button, link or box with that name and '
              'asks before clicking |', ""]
    lines += ["## Gestures", "", "| Do | What it does |", "| --- | --- |"]
    lines += [f"| {do} | {does} |" for do, does in GESTURES]
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    print(render())
