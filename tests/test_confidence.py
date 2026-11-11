"""The confidence signals of eval/dictation_cleanup/confidence.py, on hand-made token lists (no model)."""

import math
import unittest

from eval.dictation_cleanup import confidence as c


def toks(*parts):
    """parts: (text, probability) or (text, probability, [(other text, probability), ...])."""
    out = []
    for n, part in enumerate(parts):
        text, p = part[0], part[1]
        others = part[2] if len(part) > 2 else []
        top = [[n, list(text.encode("utf-8")), math.log(p)]] + \
              [[1000 + n + k, list(o.encode("utf-8")), math.log(q)] for k, (o, q) in enumerate(others)]
        out.append({"id": n, "b": list(text.encode("utf-8")), "lp": math.log(p), "top": top})
    return out


def answer(*text_parts, unsure=(" false", 0.99, [(" true", 0.01)])):
    return toks(('{"', 1.0), ('text', 1.0), ('": "', 1.0), *text_parts, ('", "', 1.0), ('alternatives', 1.0),
                ('": [], "', 1.0), ('uncertain', 1.0), ('":', 1.0), unsure, ('}', 1.0))


class TextSpanTest(unittest.TestCase):
    def test_finds_the_value_past_an_escaped_quote(self):
        raw = b'{"text": "He said \\"hi\\".", "alternatives": [], "uncertain": false}'
        s, e = c.text_span(raw)
        self.assertEqual(raw[s:e], b'He said \\"hi\\".')

    def test_no_text_field(self):
        self.assertIsNone(c.text_span(b'{"oops": 1}'))


class SignalsTest(unittest.TestCase):
    def test_low_changed_word_is_found(self):
        t = answer(("The", 1.0), (" hardware", 0.16, [(" heart", 0.1)]), (" test", 1.0), (".", 1.0))
        s = c.signals(t, "The hard worm test")
        self.assertAlmostEqual(math.exp(s["changed"]), 0.16, places=6)
        self.assertAlmostEqual(math.exp(s["content"]), 0.16, places=6)
        self.assertEqual(s["weak"], "hardware")
        self.assertEqual(s["cweak"], "hardware")

    def test_punctuation_and_small_words_do_not_count_for_content(self):
        t = answer(("My", 0.4, [("The", 0.6)]), (" dog", 1.0), (" ate", 1.0), ("!", 0.3, [(".", 0.7)]))
        s = c.signals(t, "my dog ate")
        self.assertAlmostEqual(math.exp(s["any"]), 0.3, places=6)      # "any" sees the "!"
        self.assertEqual(s["content"], 0.0)                              # "My" is small, "!" is punctuation
        self.assertEqual(s["changed"], 0.0)                              # every word was heard

    def test_negation_counts_as_content(self):
        t = answer(("I", 1.0), (" can't", 0.5), (" go", 1.0))
        s = c.signals(t, "I can go")
        self.assertAlmostEqual(math.exp(s["content"]), 0.5, places=6)
        self.assertEqual(s["cweak"], "can't")

    def test_uncertain_probability_is_read_from_the_runner_up(self):
        t = answer(("Fine", 1.0), unsure=(" false", 0.8, [(" true", 0.2)]))
        self.assertAlmostEqual(c.signals(t, "fine")["p_unsure"], 0.2, places=6)

    def test_rebuilds_the_raw_answer(self):
        t = answer(("Fine", 1.0))
        self.assertEqual(c.signals(t, "fine")["raw"],
                         '{"text": "Fine", "alternatives": [], "uncertain": false}')

    def test_weakest_text_token(self):
        t = answer(("The", 1.0), (" hard", 0.51, [(" heart", 0.47)]), (" worm", 0.9))
        self.assertEqual(bytes(t[c.weakest_text_token(t)]["b"]), b" hard")


class NewWrongWordsTest(unittest.TestCase):
    def test_a_word_nobody_said_or_meant(self):
        row = {"asr_best": "The hard worm test came back negative.",
               "intended": "The heartworm test came back negative.", "text": "The hardware test came back negative."}
        self.assertEqual(c.new_wrong_words(row), ["hardware"])

    def test_a_misheard_word_left_as_heard_is_not_new(self):
        row = {"asr_best": "Add fish soap to the list.", "intended": "Add dish soap to the list.",
               "text": "Add fish soap to the list."}
        self.assertEqual(c.new_wrong_words(row), [])


if __name__ == "__main__":
    unittest.main()
