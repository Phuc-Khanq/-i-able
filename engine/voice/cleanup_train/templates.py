"""Generated hard cases: changes of mind, and two readings that differ in a negation or a number.

These behaviours need many examples with different words, so the model learns the rule and not the sentence.
Every function takes a random.Random, so the same seed gives the same examples.
"""

from __future__ import annotations

import random

MARKERS = ["no", "no wait", "sorry", "I mean", "actually", "wait, no", "scratch that", "oops, no", "no, sorry"]

# (heard frame, answer frame, choices): the speaker says {a}, then corrects it to {b}.
CORRECTIONS = [
    ("Can you come at {a}, {m}, at {b}", "Can you come at {b}?",
     ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "noon"]),
    ("Let's meet on {a}, {m}, {b}", "Let's meet on {b}.",
     ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]),
    ("I need {a}, {m}, {b} bags of rice", "I need {b} bags of rice.", ["two", "three", "four", "five", "six"]),
    ("Order {a}, {m}, {b} pizzas for the party", "Order {b} pizzas for the party.",
     ["two", "three", "four", "five", "six", "eight"]),
    ("Tell {a}, {m}, tell {b} I'm on my way", "Tell {b} I'm on my way.",
     ["Mom", "Dad", "Grandma", "Grandpa", "my sister", "my brother", "my aunt", "my uncle"]),
    ("Meet me at the {a}, {m}, the {b}", "Meet me at the {b}.",
     ["park", "station", "front gate", "coffee shop", "parking lot", "main entrance", "pharmacy", "bakery"]),
    ("Put it in the {a}, {m}, in the {b}", "Put it in the {b}.",
     ["kitchen", "bedroom", "garage", "closet", "hallway", "bathroom", "car"]),
    ("I'd like the {a}, {m}, the {b} please", "I'd like the {b}, please.",
     ["soup", "salad", "pasta", "chicken", "fish", "burger", "omelette"]),
    ("The appointment is in {a}, {m}, in {b}", "The appointment is in {b}.",
     ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October"]),
    ("Call me in {a}, {m}, {b} minutes", "Call me in {b} minutes.", ["five", "ten", "twenty", "thirty"]),
    ("Turn the volume {a}, {m}, {b}", "Turn the volume {b}.", ["up", "down"]),
    ("Grab the {a} one, {m}, the {b} one", "Grab the {b} one.",
     ["black", "white", "red", "green", "yellow", "big", "small"]),
]

# Negation pairs: (with, without). The heard reading is kept first, the other one offered, marked unsure.
NEGATION_FRAMES = [
    ("I can {v}", "I can't {v}"), ("I will {v}", "I won't {v}"), ("I did {v}", "I didn't {v}"),
    ("We should {v}", "We shouldn't {v}"), ("Please {v}", "Please don't {v}"), ("You can {v}", "You can't {v}"),
    ("I do want to {v}", "I don't want to {v}"), ("We could {v}", "We couldn't {v}"),
]
VERB_PHRASES = ["come tonight", "lock the door", "call him back", "take the bus", "wait for you", "sign the form",
                "go to work tomorrow", "drive today", "turn it off", "eat that", "open the window", "pay today",
                "join the call", "bring the dog", "stay late", "tell her", "use the car", "move the meeting",
                "buy the tickets", "fix it myself", "go to the party", "answer the door", "cook dinner",
                "leave the light on"]

# Numbers a recognizer mixes up: -teen against -ty (fifteen/fifty is left out: a test case uses it).
TEEN_TY = [("thirteen", "thirty"), ("fourteen", "forty"), ("sixteen", "sixty"), ("seventeen", "seventy"),
           ("eighteen", "eighty"), ("nineteen", "ninety")]
NUMBER_FRAMES = ["It costs {n} dollars", "We need {n} chairs", "The meeting is in room {n}", "She is turning {n}",
                 "Order {n} cupcakes", "I read {n} pages today", "We sold {n} tickets", "The bus number is {n}",
                 "Add {n} dollars to my card", "There are {n} people coming"]


def _end(text: str) -> str:
    q = text.split()[0].lower() in {"can", "could", "is", "are", "do", "did", "will", "would", "what", "where", "when"}
    return text if text[-1] in ".?!" else text + ("?" if q else ".")


def corrections(rng: random.Random, n: int) -> list[tuple[str, str, str, str]]:
    """(app, before, heard, intended) changes of mind."""
    out, seen = [], set()
    while len(out) < n:
        heard_f, answer_f, choices = rng.choice(CORRECTIONS)
        a, b = rng.sample(choices, 2)
        heard = heard_f.format(a=a, b=b, m=rng.choice(MARKERS))
        if heard not in seen:
            seen.add(heard)
            out.append(("chat", "", heard, answer_f.format(b=b)))
    return out


def negation_pairs(rng: random.Random, n: int) -> list[tuple[str, str, str, str, list]]:
    """(app, before, heard, other hearing, word list): either reading may be the heard one."""
    out, seen = [], set()
    while len(out) < n:
        pos, neg = rng.choice(NEGATION_FRAMES)
        v = rng.choice(VERB_PHRASES)
        if pos.startswith("I did") and any(w in v for w in ("tomorrow", "tonight")):
            continue                                # "I did go to work tomorrow" is not a sentence anyone says
        if pos.startswith("You") and "you" in v:
            continue                                # "You can wait for you"
        a, b = pos.format(v=v), neg.format(v=v)
        heard, other = (a, b) if rng.random() < 0.5 else (b, a)
        if heard not in seen:
            seen.add(heard)
            out.append(("chat", "", heard, other, []))
    return out


def number_pairs(rng: random.Random, n: int) -> list[tuple[str, str, str, str, list]]:
    out, seen = [], set()
    while len(out) < n:
        frame = rng.choice(NUMBER_FRAMES)
        a, b = rng.choice(TEEN_TY)
        heard, other = (frame.format(n=a), frame.format(n=b)) if rng.random() < 0.5 else (frame.format(n=b), frame.format(n=a))
        if heard not in seen:
            seen.add(heard)
            out.append(("chat", "", heard, other, []))
    return out
