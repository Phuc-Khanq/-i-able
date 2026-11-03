"""Sentence banks by field for the big clean-up data set (bigset.py), written by Claude.

Each field is a kind of thing people dictate (doctor's messages, work email, cooking, gaming ...): the apps it
happens in, lines that come before it ("before" in the prompt), its own words and how a recognizer mishears them,
clean sentences, and pairs where both hearings make sense. bigset.py turns them into examples and splits them into
training and test by sentence, so no test sentence is ever trained on.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Field:
    name: str
    apps: list[str]
    befores: list[str]
    terms: dict[str, list[str]]          # the right word -> how a recognizer mishears it
    sentences: list[str]
    ambiguous: list[tuple[str, str]] = field(default_factory=list)   # (heard, other hearing): both make sense


def all_fields() -> list[Field]:
    from engine.voice.cleanup_train import fields_health, fields_life, fields_more, fields_social, fields_work

    return [*fields_health.FIELDS, *fields_work.FIELDS, *fields_life.FIELDS, *fields_social.FIELDS,
            *fields_more.FIELDS]
