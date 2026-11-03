"""Tests for the clean-up training data: the noise, the leak check and the training file."""

import json
import random
import tempfile
import unittest
from pathlib import Path

from engine.voice.cleanup_train import build, noise, seeds


class NoiseTest(unittest.TestCase):
    def test_protected_words(self):
        self.assertTrue(noise.protected("Friday", 3))
        self.assertTrue(noise.protected("seven", 2))
        self.assertTrue(noise.protected("can't", 1))
        self.assertTrue(noise.protected("not", 1))
        self.assertTrue(noise.protected("12:30", 1))
        self.assertFalse(noise.protected("kitchen", 1))
        self.assertFalse(noise.protected("Please", 0))   # first word: a capital is not a name

    def test_misspell_changes_the_word_and_skips_short_or_blocked(self):
        rng = random.Random(0)
        self.assertNotEqual(noise.misspell("coffee", rng), "coffee")
        self.assertIsNone(noise.misspell("cat", rng))
        self.assertIsNone(noise.misspell("library", rng, blocked={"library"}))

    def test_messy_keeps_names_numbers_and_negations(self):
        text = "I can't meet Priya at seven on Friday, not today."
        for s in range(200):
            heard, _ = noise.messy(text, random.Random(s))
            low = heard.lower()
            for keep in ("can't", "priya", "seven", "friday", "not"):
                self.assertIn(keep, low, heard)

    def test_messy_never_returns_the_clean_text(self):
        for s in range(100):
            heard, _ = noise.messy("Thanks.", random.Random(s))
            self.assertNotEqual(heard, "Thanks.")

    def test_same_seed_same_output(self):
        a = noise.messy("Can you water the plants while I'm away?", random.Random(3))
        b = noise.messy("Can you water the plants while I'm away?", random.Random(3))
        self.assertEqual(a, b)


class LeakCheckTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.leak = build.LeakCheck()

    def test_a_test_sentence_is_too_close(self):
        self.assertTrue(self.leak.too_close("I am feeling much better today"))

    def test_an_unrelated_sentence_is_fine(self):
        self.assertFalse(self.leak.too_close("The plumber is coming between ten and noon."))

    def test_test_case_word_pairs_are_blocked(self):
        self.assertIn("pitcher", self.leak.swap_words)
        self.assertTrue(self.leak.uses_test_pair("Send the pitcher to Grandpa", "Send the picture to Grandpa"))

    def test_no_seed_example_leaks(self):
        kept, _ = build.examples(random.Random(7), self.leak)
        for c in kept:
            self.assertFalse(self.leak.too_close(c["asr_best"], c["target"]["text"]), c["asr_best"])


class BuildTest(unittest.TestCase):
    def test_answers_follow_the_prompt_rules(self):
        kept, _ = build.examples(random.Random(7), build.LeakCheck())
        for c in kept:
            t = c["target"]
            self.assertEqual(set(t), {"text", "alternatives", "uncertain"})
            if not t["uncertain"]:
                self.assertEqual(t["alternatives"], [], c)     # alternatives only when unsure
            if c["kind"] == "clean":
                self.assertEqual(c["asr_best"], t["text"])     # already fine: keep every word
            if c["kind"] in ("ambiguous", "context_ambiguous"):
                self.assertTrue(t["uncertain"] and t["alternatives"], c)
        kinds = {c["kind"] for c in kept}
        for kind in ("messy", "broken", "clean", "sentence_fix", "context_fix", "name_fix", "self_correct",
                     "ambiguous", "context_ambiguous", "cut_off", "not_for_you"):
            self.assertIn(kind, kinds)

    def test_build_writes_chat_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, counts, _ = build.build("t", out_dir=Path(tmp))
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), sum(counts.values()))
            first = json.loads(lines[0])
            self.assertEqual([m["role"] for m in first["messages"]], ["system", "user", "assistant"])
            json.loads(first["messages"][2]["content"])            # the answer is valid JSON
            self.assertTrue((Path(tmp) / "t_preview.csv").exists())

    def test_seed_shapes(self):
        for row in seeds.NAME_FIX:
            self.assertIsInstance(row[4], list)
        for row in seeds.CONTEXT_AMBIGUOUS:
            self.assertIsInstance(row[4], list)


class AdapterLabelTest(unittest.TestCase):
    def test_label_names_the_adapter_run(self):
        from eval.dictation_cleanup.backends import TransformersBackend

        b = TransformersBackend("Qwen/Qwen3.5-4B", adapter="checkpoints/cleanup/pilot/adapter")
        self.assertEqual(b.model_label, "Qwen/Qwen3.5-4B+pilot")
        self.assertEqual(b.settings()["adapter"], "checkpoints/cleanup/pilot/adapter")
        self.assertEqual(TransformersBackend("Qwen/Qwen3.5-4B").model_label, "Qwen/Qwen3.5-4B")

    def test_short_prompt_loads(self):
        from eval.dictation_cleanup.prompt_files import load_prompt

        msgs = load_prompt("t1_short").messages({"asr_best": "um hi"})
        self.assertIn("Heard: um hi", msgs[1]["content"])
        self.assertLess(len(msgs[0]["content"]), 400)

if __name__ == "__main__":
    unittest.main()
