"""Run with: python -m unittest tests.test_guard_replay  (eval/dictation_cleanup/guard_replay.py: the guard variants
tried offline on saved clean-up answers; no model, no GPU)"""
import unittest

from eval.dictation_cleanup.guard_replay import VARIANTS, Variant, blocked, dropped_runs, new_words, shown

SHORT = Variant("short", 3, 1)
SIX = Variant("six + drops", 6, 1, 3)


class TestNewWords(unittest.TestCase):
    def test_an_added_content_word(self):
        self.assertEqual(new_words("refresh", "Please refresh the page."), ["page", "please"])
        self.assertEqual(new_words("paste down", "Please paste the document down."), ["document"])

    def test_small_words_and_the_same_word_spelled_close_are_not_new(self):
        self.assertEqual(new_words("close tab", "Close the tab."), [])
        self.assertEqual(new_words("he wants it", "He want it."), [])

    def test_a_sound_alike_of_heard_words_is_not_new(self):
        self.assertEqual(new_words("pred a zone at noon", "Prednisone at noon."), [])

    def test_what_a_hearing_the_context_or_the_word_list_gave_is_not_new(self):
        self.assertEqual(new_words("meet me at the bank", "Meet me at the bench.", hearings=["meet me at the bench"]), [])
        self.assertEqual(new_words("send it", "Send the photos.", context="The photos are ready."), [])
        self.assertEqual(new_words("call okafor", "Call Dr. Okafor.", words=["Dr. Okafor"]), [])


class TestBlocked(unittest.TestCase):
    def test_a_short_phrase_that_gains_a_word_is_blocked_and_falls_back_to_what_was_said(self):
        self.assertIsNone(blocked(VARIANTS[0], "refresh", "Please refresh the page."))
        self.assertTrue(blocked(SHORT, "refresh", "Please refresh the page.").startswith("new words"))
        self.assertEqual(shown(SHORT, "refresh", "Please refresh the page."), "refresh")

    def test_a_longer_input_is_left_alone_by_the_short_rule(self):
        heard = "send her the the file by friday"
        self.assertIsNone(blocked(SHORT, heard, "Send her the file by Friday, please, with the invoice attached."))

    def test_check_cleanup_still_comes_first(self):
        why = blocked(SHORT, "meet at four", "Meet at five.")
        self.assertIn("number", why)

    def test_a_good_short_fix_passes(self):
        self.assertIsNone(blocked(SHORT, "close tab", "Close the tab."))


class TestDroppedRuns(unittest.TestCase):
    def test_a_dropped_phrase_is_a_run(self):
        runs = dropped_runs("he said the logos or word of god was near", "He said it was near.")
        self.assertTrue(any(len(r) >= 3 for r in runs))

    def test_a_repeat_removed_is_not(self):
        self.assertEqual(dropped_runs("we go to the store we go to the store later", "We go to the store later."), [])

    def test_fillers_are_not_content(self):
        self.assertEqual(dropped_runs("um uh so we go", "So we go."), [])

    def test_the_drop_variant_blocks_it_unless_the_mind_changed(self):
        heard = "the old farmer walked slowly along the quiet river bank beside the tall green hedge before sunrise"
        clean = "The old farmer walked slowly along the quiet river bank before sunrise."
        self.assertIsNone(blocked(VARIANTS[0], heard, clean))              # check_cleanup lets it through
        self.assertTrue(blocked(SIX, heard, clean).startswith("dropped"))
        changed = "turn left sorry I mean right at the lights"
        self.assertIsNone(blocked(SIX, changed, "Turn right at the lights."))


if __name__ == "__main__":
    unittest.main()
