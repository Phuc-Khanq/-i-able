"""Run with: python -m unittest tests.test_cleanup_eval  (the clean-up evaluation's checks and runner; no GPU, no downloads)"""
import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

from eval.dictation_cleanup import backends, run, scoring
from eval.dictation_cleanup.prompt_files import load_prompt


def case(**kw) -> dict:
    base = {"id": "T01", "split": "dev", "category": "mispronounced", "app": "", "context": "", "asr_best": "",
            "asr_alt1": "", "asr_alt2": "", "personal_words": "", "intended": "", "must_keep": "",
            "expected_behavior": "rewrite", "notes": ""}
    base.update(kw)
    return base


def answer(text: str, alternatives=(), uncertain=False) -> scoring.Answer:
    raw = json.dumps({"text": text, "alternatives": list(alternatives), "uncertain": uncertain})
    return scoring.parse_answer(raw)


class TestTestSet(unittest.TestCase):
    def test_dev_is_the_default_and_holdout_stays_out(self):
        dev = scoring.load_cases()
        self.assertEqual(len(dev), 67)
        self.assertEqual({c["split"] for c in dev}, {"dev"})
        self.assertEqual(len(scoring.load_cases(split="holdout")), 33)

    def test_the_bom_is_read_as_part_of_the_encoding(self):
        self.assertIn("id", scoring.load_cases()[0])

    def test_unknown_split_is_refused(self):
        with self.assertRaises(ValueError):
            scoring.load_cases(split="all")


class TestParseAnswer(unittest.TestCase):
    def test_valid_json(self):
        a = scoring.parse_answer('{"text": "Hi.", "alternatives": ["Hey."], "uncertain": true}')
        self.assertTrue(a.json_ok)
        self.assertEqual((a.text, a.alternatives, a.uncertain), ("Hi.", ["Hey."], True))

    def test_a_code_fence_around_the_json_is_allowed(self):
        a = scoring.parse_answer('```json\n{"text": "Hi.", "alternatives": [], "uncertain": false}\n```')
        self.assertTrue(a.json_ok)
        self.assertEqual(a.text, "Hi.")

    def test_broken_json_is_a_failure_and_keeps_the_raw_text(self):
        a = scoring.parse_answer('{"text": "Hi.", "alternatives": [')
        self.assertFalse(a.json_ok)
        self.assertEqual(a.raw, '{"text": "Hi.", "alternatives": [')
        self.assertEqual(a.text, "")

    def test_missing_keys_or_wrong_types_are_failures(self):
        for raw in ['{"text": "Hi."}', '{"text": 3, "alternatives": [], "uncertain": false}',
                    '{"text": "Hi.", "alternatives": "Hey.", "uncertain": false}',
                    '{"text": "Hi.", "alternatives": [], "uncertain": "no"}', '["Hi."]', "Hi."]:
            self.assertFalse(scoring.parse_answer(raw).json_ok, raw)

    def test_thinking_output_is_a_failure(self):
        a = scoring.parse_answer('<think>hmm</think>{"text": "Hi.", "alternatives": [], "uncertain": false}')
        self.assertFalse(a.json_ok)
        self.assertIn("thinking", a.error)


class TestMustKeep(unittest.TestCase):
    def test_case_insensitive_whole_words(self):
        c = case(must_keep="Thursday|two")
        self.assertEqual(scoring.must_keep_missing(c, "Can we meet thursday at TWO?"), [])
        self.assertEqual(scoring.must_keep_missing(c, "Can we meet Thursday at twofold?"), ["two"])

    def test_phrases_and_brackets(self):
        self.assertEqual(scoring.must_keep_missing(case(must_keep="six thirty"), "Dinner is at six  thirty."), [])
        self.assertEqual(scoring.must_keep_missing(case(must_keep="[eye]able"), "Our product is [eye]able."), [])
        self.assertEqual(scoring.must_keep_missing(case(must_keep="[eye]able"), "Our product is eyeable."),
                         ["[eye]able"])

    def test_digits_are_not_number_words(self):
        # the rule is literal: "4" doesn't count for "four" (the review sheet shows these)
        self.assertEqual(scoring.must_keep_missing(case(must_keep="four"), "See you at 4."), ["four"])


class TestBehaviour(unittest.TestCase):
    def test_unchanged_ignores_case_punctuation_and_spaces(self):
        c = case(category="clean_control", expected_behavior="unchanged", asr_best="see you at the station")
        self.assertTrue(scoring.check_case(c, answer("See you at the station.  "))["auto_pass"])
        r = scoring.check_case(c, answer("See you at the train station."))
        self.assertFalse(r["auto_pass"])
        self.assertIn("changed", r["problems"])

    def test_show_both_needs_two_readings_or_uncertain(self):
        c = case(category="ambiguous", expected_behavior="show_both", asr_best="Meet me at the bank",
                 asr_alt1="Meet me at the bench")
        self.assertTrue(scoring.check_case(c, answer("Meet me at the bank.", ["Meet me at the bench."]))["auto_pass"])
        self.assertTrue(scoring.check_case(c, answer("Meet me at the bank.", uncertain=True))["auto_pass"])
        r = scoring.check_case(c, answer("Meet me at the bench."))
        self.assertFalse(r["auto_pass"])
        self.assertTrue(r["silent_guess"])

    def test_a_repeated_reading_is_not_two_readings(self):
        c = case(category="ambiguous", expected_behavior="show_both", asr_best="Meet me at the bank")
        self.assertFalse(scoring.check_case(c, answer("Meet me at the bank.", ["meet me at the bank"]))["auto_pass"])

    def test_keep_original_and_flag(self):
        c = case(category="ambiguous", expected_behavior="keep_original_and_flag", asr_best="Because of the")
        self.assertTrue(scoring.check_case(c, answer("Because of the", uncertain=True))["auto_pass"])
        self.assertIn("not marked uncertain", scoring.check_case(c, answer("Because of the"))["problems"])
        self.assertIn("changed", scoring.check_case(c, answer("Because of the rain.", uncertain=True))["problems"])

    def test_broken_json_fails_with_one_reason(self):
        c = case(category="clean_control", expected_behavior="unchanged", asr_best="Hi there")
        r = scoring.check_case(c, scoring.parse_answer("Hi there"))
        self.assertEqual(r["problems"], ["broken JSON"])
        self.assertFalse(r["auto_pass"])

    def test_answers_without_json_are_scored_on_their_text(self):
        # the "current" backend answers in plain text: no JSON check, never uncertain
        c = case(category="clean_control", expected_behavior="unchanged", asr_best="Hi there")
        self.assertTrue(scoring.check_case(c, scoring.Answer("Hi there.", [], False, None, "Hi there."))["auto_pass"])


class TestInventedFacts(unittest.TestCase):
    def test_new_number_is_flagged(self):
        c = case(asr_best="Dinner is at sex thirty")
        self.assertEqual(scoring.invented_facts(c, answer("Dinner is at seven thirty.")), ["number: seven"])

    def test_number_from_an_alternative_or_digits_is_fine(self):
        c = case(asr_best="Dinner is at sex thirty", asr_alt1="Dinner is at six thirty")
        self.assertEqual(scoring.invented_facts(c, answer("Dinner is at 6:30.")), [])

    def test_new_name_is_flagged_but_one_from_the_context_is_not(self):
        c = case(asr_best="please call doctor about it", context="Replying to Dr. Alvarez")
        self.assertEqual(scoring.invented_facts(c, answer("Please call Dr. Alvarez about it.")), [])
        self.assertEqual(scoring.invented_facts(c, answer("Please call Dr. Moreno about it.")), ["name: moreno"])

    def test_new_negation_is_flagged_and_contractions_count_as_not(self):
        c = case(asr_best="I do not want the blue one")
        self.assertEqual(scoring.invented_facts(c, answer("I don't want the blue one.")), [])
        c = case(asr_best="I want the blue one")
        self.assertEqual(scoring.invented_facts(c, answer("I don't want the blue one.")), ["negation: not"])

    def test_alternatives_are_checked_too(self):
        c = case(asr_best="Send it to Sam")
        self.assertEqual(scoring.invented_facts(c, answer("Send it to Sam.", ["Send it to Sandy."])), ["name: sandy"])


class TestSummary(unittest.TestCase):
    def test_rates(self):
        cases = [case(id="C1", category="clean_control", expected_behavior="unchanged", asr_best="Hi there"),
                 case(id="C2", category="clean_control", expected_behavior="unchanged", asr_best="Bye now"),
                 case(id="A1", category="ambiguous", expected_behavior="show_both", asr_best="At the bank"),
                 case(id="M1", category="mispronounced", asr_best="at sex", must_keep="six", asr_alt1="at six")]
        answers = [answer("Hi there."), answer("Goodbye now."), answer("At the bench."),
                   scoring.parse_answer("not json")]
        results = [scoring.result_row(c, a, ms, tokens=10) for c, a, ms in zip(cases, answers, [10, 20, 20, 40])]
        s = scoring.summarize(results)
        self.assertEqual(s["n"], 4)
        self.assertEqual(s["auto_pass"], 1)
        self.assertEqual(s["over_edit"], [1, 2])
        self.assertEqual(s["silent_guess"], [1, 1])
        self.assertEqual(s["json_failures"], 1)
        self.assertEqual(s["by_category"]["clean_control"], [1, 2])
        self.assertEqual(s["latency_ms"]["p50"], 20)
        self.assertEqual(s["latency_ms"]["p99"], 40)
        self.assertEqual(s["tokens"]["total"], 40)


class TestPrompts(unittest.TestCase):
    def test_plain_gets_only_the_transcript(self):
        p = load_prompt("v1_plain")
        c = case(asr_best="I need to by milk", asr_alt1="I need to buy milk", personal_words="Khoa", app="notes")
        msgs = p.messages(c)
        self.assertEqual([m["role"] for m in msgs], ["system", "user"])
        self.assertEqual(msgs[1]["content"], "I need to by milk")

    def test_full_gets_every_input(self):
        c = case(asr_best="I need to by milk", asr_alt1="I need to buy milk", asr_alt2="I need to bye milk",
                 personal_words="Khoa|Gainesville", app="notes", context="Shopping list")
        user = load_prompt("v1_full").messages(c)[1]["content"]
        for part in ["I need to by milk", "I need to buy milk", "I need to bye milk", "Khoa", "Gainesville", "notes",
                     "Shopping list"]:
            self.assertIn(part, user)
        self.assertNotIn("$", user)

    def test_empty_fields_say_none(self):
        user = load_prompt("cleanup_v1_full").messages(case(asr_best="Hi"))[1]["content"]
        self.assertIn("(none)", user)

    def test_prompts_ask_for_the_json_answer(self):
        for name in ["v1_plain", "v1_full"]:
            system = load_prompt(name).system
            for key in ['"text"', '"alternatives"', '"uncertain"']:
                self.assertIn(key, system)

    def test_unknown_prompt(self):
        with self.assertRaises(FileNotFoundError):
            load_prompt("v9_nothing")


class FakeRewriter:
    """Stands in for engine.voice.rewrite.Rewriter: answers from a script."""

    def __init__(self, outcomes):
        self.outcomes = outcomes
        self.model_id = "fake/rewriter"

    def load(self):
        pass

    def warm_up(self):
        pass

    def rewrite_full(self, text):
        return self.outcomes[text]


class TestCurrentBackend(unittest.TestCase):
    def test_records_which_path_the_app_took(self):
        from engine.voice.rewrite import RewriteOutcome

        rw = FakeRewriter({
            "um hi there": RewriteOutcome("um hi there", "Hi there.", True, 5.0, None, "Hi there."),
            "the the bus": RewriteOutcome("the the bus", "the bus", True, 5.0, "light clean: model word swapped",
                                          "The coach."),
            "call Sam": RewriteOutcome("call Sam", "call Sam", False, 5.0, "name missing: sam", "Call him."),
        })
        b = backends.CurrentBackend(rewriter=rw)
        r1 = b.clean(case(asr_best="um hi there"), None)
        self.assertEqual((r1.answer.text, r1.extra["path"], r1.raw), ("Hi there.", "model", "Hi there."))
        self.assertIsNone(r1.answer.json_ok)
        r2 = b.clean(case(asr_best="the the bus"), None)
        self.assertEqual((r2.answer.text, r2.extra["path"], r2.raw), ("the bus", "light_clean", "The coach."))
        r3 = b.clean(case(asr_best="call Sam"), None)
        self.assertEqual((r3.answer.text, r3.extra["path"]), ("call Sam", "original"))
        self.assertEqual(r3.extra["guard_reason"], "name missing: sam")


class TestRunner(unittest.TestCase):
    def test_fake_run_writes_every_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = run.run(backends.FakeBackend(), scoring.load_cases(), load_prompt("v1_full"), Path(tmp),
                          model_label="fake", log=lambda *a: None)
            names = {p.name for p in out.iterdir()}
            self.assertEqual(names, {"results.jsonl", "review.csv", "summary.md", "summary.json", "settings.json"})
            rows = [json.loads(line) for line in (out / "results.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 67)
            raw = (out / "review.csv").read_bytes()
            self.assertTrue(raw.startswith(b"\xef\xbb\xbf"))           # BOM, so Excel opens it right
            with (out / "review.csv").open(encoding="utf-8-sig", newline="") as f:
                review = list(csv.DictReader(f))
            self.assertEqual(len(review), 67)
            for col in ["meaning_kept_Khang", "nothing_invented_Khang", "meaning_kept_Khoi", "nothing_invented_Khoi"]:
                self.assertEqual({r[col] for r in review}, {""})
            s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(s["over_edit"], [0, 10])                 # echo never edits a clean control...
            self.assertEqual(s["silent_guess"][0], 8)                 # ...and never shows both readings
            self.assertIn("Latency", (out / "summary.md").read_text(encoding="utf-8"))
            self.assertTrue(out.name.endswith("_fake_cleanup_v1_full"))

    def test_a_crash_on_one_case_is_recorded_and_the_run_goes_on(self):
        class Crashy(backends.FakeBackend):
            def clean(self, c, prompt):
                if c["id"] == "M01":
                    raise RuntimeError("boom")
                return super().clean(c, prompt)

        with tempfile.TemporaryDirectory() as tmp:
            out = run.run(Crashy(), scoring.load_cases()[:3], load_prompt("v1_plain"), Path(tmp), model_label="fake",
                          log=lambda *a: None)
            rows = [json.loads(line) for line in (out / "results.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 3)
            m01 = next(r for r in rows if r["id"] == "M01")
            self.assertFalse(m01["auto_pass"])
            self.assertIn("error: boom", m01["problems"])

    def test_holdout_needs_the_flag_and_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = run.main(["--backend", "fake", "--split", "holdout", "--out", tmp, "--limit", "2"])
            self.assertEqual(code, 0)
            self.assertIn("WARNING", buf.getvalue())
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                run.main(["--backend", "fake", "--out", tmp, "--limit", "2"])
            self.assertNotIn("WARNING", buf.getvalue())

    def test_run_folders_never_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = run.run(backends.FakeBackend(), scoring.load_cases()[:1], load_prompt("v1_plain"), Path(tmp),
                        model_label="fake", log=lambda *a: None)
            b = run.run(backends.FakeBackend(), scoring.load_cases()[:1], load_prompt("v1_plain"), Path(tmp),
                        model_label="fake", log=lambda *a: None)
            self.assertNotEqual(a, b)


if __name__ == "__main__":
    unittest.main()
