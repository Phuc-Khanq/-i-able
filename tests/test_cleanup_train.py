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

    def test_careful_noise_keeps_a_question_and_a_noun_will(self):
        question = "Can you send the report to my boss?".split()
        will = "I need to update my will this week.".split()
        for s in range(100):
            self.assertEqual(noise.drop_small_words(question, random.Random(s), careful=True)[0], "Can")
            self.assertIn("will", noise.drop_small_words(will, random.Random(s), careful=True))
        dropped = [noise.drop_small_words(question, random.Random(s))[0] for s in range(100)]
        self.assertIn("you", dropped)                    # big1's noise did drop it

    def test_careful_noise_makes_the_same_random_draws(self):
        toks = "Will you take the dog to the park for me?".split()
        for s in range(50):
            a, b = random.Random(s), random.Random(s)
            noise.drop_small_words(toks, a)
            noise.drop_small_words(toks, b, careful=True)
            self.assertEqual(a.random(), b.random())     # so big1's data still rebuilds byte for byte


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


class FullBuildTest(unittest.TestCase):
    def test_templates_make_distinct_valid_cases(self):
        from engine.voice.cleanup_train import templates

        rng = random.Random(1)
        fixes = templates.corrections(rng, 40)
        self.assertEqual(len({f[2] for f in fixes}), 40)
        for _, _, heard, meant in fixes:
            self.assertLess(len(meant), len(heard))            # the correction is shorter than what was said
        for _, _, heard, other, _ in templates.negation_pairs(rng, 30) + templates.number_pairs(rng, 12):
            self.assertNotEqual(heard, other)
            self.assertNotIn("I did go to work tomorrow", heard)

    def test_full_build_has_more_and_keeps_the_rules(self):
        leak = build.LeakCheck()
        pilot, _ = build.examples(random.Random(7), leak)
        full, _ = build.examples(random.Random(7), leak, extra=True, generated=build.FULL_GENERATED)
        self.assertGreater(len(full), 3 * len(pilot))
        kinds = {c["kind"] for c in full}
        self.assertIn("context_keep", kinds)
        for c in full:
            t = c["target"]
            if not t["uncertain"]:
                self.assertEqual(t["alternatives"], [], c)
            self.assertFalse(leak.too_close(c["asr_best"], t["text"]), c["asr_best"])
        lines = [(c["asr_best"], c["asr_alt1"], c["context"]) for c in full]
        self.assertEqual(len(lines), len(set(lines)))           # no line twice

    def test_negation_words_are_not_blocked_but_test_pairs_are(self):
        leak = build.LeakCheck()
        self.assertFalse(any(w.endswith("n't") for w in leak.swap_words))
        self.assertNotIn("not", leak.swap_words)
        self.assertIn("pitcher", leak.swap_words)

class AdapterLabelTest(unittest.TestCase):
    def test_label_names_the_adapter_run(self):
        from eval.dictation_cleanup.backends import TransformersBackend

        b = TransformersBackend("Qwen/Qwen3.5-4B", adapter="checkpoints/cleanup/pilot/adapter")
        self.assertEqual(b.model_label, "Qwen/Qwen3.5-4B+pilot")
        self.assertEqual(b.settings()["adapter"], "checkpoints/cleanup/pilot/adapter")
        self.assertEqual(TransformersBackend("Qwen/Qwen3.5-4B").model_label, "Qwen/Qwen3.5-4B")
        self.assertEqual(TransformersBackend("Qwen/Qwen3.5-4B", adapter="checkpoints/cleanup/full1/epoch2").model_label,
                         "Qwen/Qwen3.5-4B+full1-epoch2")

    def test_short_prompt_loads(self):
        from eval.dictation_cleanup.prompt_files import load_prompt

        msgs = load_prompt("t1_short").messages({"asr_best": "um hi"})
        self.assertIn("Heard: um hi", msgs[1]["content"])
        self.assertLess(len(msgs[0]["content"]), 400)

class BigSetTest(unittest.TestCase):
    """The big data set (bigset.py): sizes, the split by sentence, and the test cases' fields."""

    @classmethod
    def setUpClass(cls):
        from engine.voice.cleanup_train import bigset

        cls.bigset = bigset
        cls.train, cls.test, cls.notes = bigset.make(seed=11)

    def test_sizes_and_mix(self):
        self.assertEqual((len(self.train), len(self.test)), (7500, 2500))
        cats = {c["category"] for c in self.test}
        self.assertEqual(cats, set(self.bigset.MIX))
        for cat, share in self.bigset.MIX.items():
            n = sum(1 for c in self.test if c["category"] == cat)
            self.assertLess(abs(n / 2500 - share), 0.03, cat)

    def test_no_test_sentence_is_trained_on(self):
        meant = lambda c: c["target"]["text"].lower()
        self.assertFalse({meant(c) for c in self.test} & {meant(c) for c in self.train})
        self.assertFalse({c["asr_best"] for c in self.test} & {c["asr_best"] for c in self.train})
        self.assertTrue(all(c["side"] == "test" for c in self.test))
        self.assertTrue(all(c["side"] == "train" for c in self.train))

    def test_a_term_has_one_side_in_every_field(self):
        b = self.bigset
        both = [f for f in b.all_fields() if "potluck" in f.terms]
        self.assertEqual(len(both), 2)                          # family_friends and community
        sides = {b.sentence_side(f, "Bring a dish to the potluck.") for f in both}
        self.assertEqual(sides, {b.side("term:potluck")})

    def test_test_cases_are_scorable(self):
        for c in self.test:
            for item in filter(None, c["must_keep"].split("|")):
                self.assertIn(item.lower(), c["intended"].lower(), c)
            if c["expected"] == "show_both":
                self.assertIn(" || ", c["intended"])
                self.assertTrue(c["target"]["uncertain"])
            if c["kind"] == "cut_off":
                last = c["asr_best"].split()[-1].lower()
                self.assertIn(last, self.bigset.DANGLING, c["asr_best"])

    def test_spacing_is_not_a_mishearing(self):
        self.assertEqual(self.bigset.mishearings("carpool", ["car pool", "car pole"]), ["car pole"])

    def test_must_keep_finds_names_and_numbers(self):
        keep = self.bigset.keep_items("Ask Dr. Lee about the two pills on Friday.", ["pills"])
        self.assertEqual(keep.split("|"), ["pills", "Lee", "two", "Friday"])


class MoreDataTest(unittest.TestCase):
    """big2's training data (bigset.make_more): training only, scored on a test set already written."""

    @classmethod
    def setUpClass(cls):
        from engine.voice.cleanup_train import bigset, fields_big2

        cls.bigset = bigset
        train, test, notes = bigset.make(seed=11)
        cls.tmp = tempfile.TemporaryDirectory()
        bigset.write("t", train[:10], test, notes, 11, out_dir=Path(cls.tmp.name))
        cls.test_path = Path(cls.tmp.name) / "t_test.csv"
        cls.test = test
        cls.more, cls.notes = bigset.make_more(cls.test_path, seed=21, train_size=6000, passes=1,
                                               extra_fields=fields_big2.FIELDS[:4])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_nothing_from_the_test_set(self):
        meant = lambda c: c["target"]["text"].lower()
        self.assertEqual(len(self.more), 6000)
        self.assertFalse({meant(c) for c in self.test} & {meant(c) for c in self.more})
        self.assertFalse({c["asr_best"] for c in self.test} & {c["asr_best"] for c in self.more})
        forms = self.bigset.test_term_forms()
        self.assertIn("Medicare", forms)
        self.assertFalse([c for c in self.more if self.bigset.mentions_any(c, forms)])

    def test_targeted_kinds_are_in(self):
        kinds = {c["kind"].split(":")[0] for c in self.more}
        for k in ("unused_words", "broken_question", "misspelled_date", "misspelled_amount", "term_fix"):
            self.assertIn(k, kinds)
        for c in self.more:
            if c["kind"] == "unused_words":
                for w in c["personal_words"].split("|"):
                    self.assertNotIn(w.lower(), c["target"]["text"].lower())

    def test_misspelled_numbers_read_as_the_same_number_in_the_guard(self):
        from engine.voice import rewrite

        same = lambda a, b: rewrite._number_stems(a.split()) == rewrite._number_stems(b.split())
        for ordinal, forms in self.bigset.ORDINAL_FORMS.items():
            for form in forms:
                self.assertTrue(same(ordinal.replace("-", " "), form), form)
        for number, form in self.bigset.AMOUNT_FORMS:
            self.assertTrue(same(number, form), form)

    def test_new_banks_avoid_test_terms(self):
        from engine.voice.cleanup_train import fields_big2

        forms = self.bigset.test_term_forms()
        for f in fields_big2.FIELDS:
            for text in [*f.sentences, *f.befores, *f.apps, *f.terms]:
                self.assertFalse([x for x in forms if self.bigset._term_in(x, text)], (f.name, text))


class TrainArgsTest(unittest.TestCase):
    def test_rank_sets_alpha_to_twice_the_rank(self):
        from unittest import mock

        from engine.voice.cleanup_train import train

        with mock.patch.object(train, "train", return_value={}) as fake:
            train.main(["--name", "x", "--rank", "32"])
            train.main(["--name", "y", "--rank", "32", "--alpha", "32"])
            train.main(["--name", "z"])
        cfgs = [call.args[0] for call in fake.call_args_list]
        self.assertEqual([(c.lora_r, c.lora_alpha) for c in cfgs], [(32, 64), (32, 32), (16, 32)])


class ExportTest(unittest.TestCase):
    """The llama.cpp export's commands and file names (llama.cpp itself isn't run)."""

    def test_labels(self):
        from engine.voice.cleanup_train import export

        self.assertEqual(export.label("full1"), "full1")
        self.assertEqual(export.label("full1", "epoch2"), "full1-epoch2")

    def test_conversion_leaves_out_the_mtp_head(self):
        from unittest import mock

        from engine.voice.cleanup_train import export

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(export.subprocess, "run") as fake:
            out = export.to_gguf(Path(tmp) / "full1" / "merged", "full1")
            cmd = fake.call_args.args[0]
            self.assertEqual(out, Path(tmp) / "full1" / "gguf" / "full1-bf16.gguf")
            self.assertTrue(out.parent.is_dir())
        self.assertEqual(cmd[1], str(export.CONVERTER))
        self.assertIn("--no-mtp", cmd)
        self.assertEqual(cmd[cmd.index("--outtype") + 1], "bf16")

    def test_quantized_file_is_named_by_type(self):
        from unittest import mock

        from engine.voice.cleanup_train import export

        src = Path("x") / "gguf" / "full1-bf16.gguf"
        with mock.patch.object(export.subprocess, "run") as fake:
            out = export.quantize(src, "Q4_K_M")
        self.assertEqual(out, Path("x") / "gguf" / "full1-q4_k_m.gguf")
        self.assertEqual(fake.call_args.args[0], [str(export.QUANTIZE), str(src), str(out), "Q4_K_M"])

if __name__ == "__main__":
    unittest.main()
