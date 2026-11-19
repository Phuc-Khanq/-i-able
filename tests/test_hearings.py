"""Run with: python -m unittest tests.test_hearings

Whisper's other hearings for the clean-up model (docs/cleanup_hearings_plan.md, steps 1 and 2): the DictationWorker
adds them to a transcript when a word is unsure, and the mode manager sends them with the rewrite request as whole
versions of the card's text.
"""
import unittest

from engine.core import topics
from engine.core.mode_manager import ModeManager
from engine.voice import dictation as dictation_module
from engine.voice.dictation import Dictation, DictationWorker
from tests.helpers import FakeBus, FakeWhisperModel


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class HearingDictation(Dictation):
    """The fake Whisper plus fixed other readings ("audio" is the text, as in FakeWhisperModel)."""

    def __init__(self, readings=None, fail=False):
        super().__init__(model=FakeWhisperModel())
        self.readings, self.fail, self.calls = readings or {}, fail, 0

    def hearings(self, audio, **kwargs):
        self.calls += 1
        if self.fail:
            raise RuntimeError("no GPU")
        return [(t, -0.5) for t in self.readings.get(str(audio), [])]


class TestWorker(unittest.TestCase):
    def make(self, readings=None, fail=False):
        self.bus = FakeBus()
        self.d = HearingDictation(readings, fail)
        w = DictationWorker(self.bus, self.d, settings=None, log=lambda *_: None)
        w.send_hearings = True
        return w

    def transcript(self):
        return self.bus.of(topics.TRANSCRIPT)[-1]

    def test_unsure_words_get_other_hearings(self):
        w = self.make({"meet at the bank? today": ["Meet at the bench today.", "meet at the bank today",
                                                   "Meet at the bend today.", "Meet at the band today."]})
        w.process("meet at the bank? today")
        self.assertEqual(self.transcript()["hearings"], ["Meet at the bench today.", "Meet at the bend today."])

    def test_sure_words_skip_the_extra_pass(self):
        w = self.make({"meet at the bank today": ["Meet at the bench today."]})
        w.process("meet at the bank today")
        self.assertNotIn("hearings", self.transcript())
        self.assertEqual(self.d.calls, 0)

    def test_a_failing_extra_pass_never_stops_dictation(self):
        w = self.make(fail=True)
        w.process("meet at the bank? today")
        self.assertEqual(self.transcript()["text"], "meet at the bank today")
        self.assertNotIn("hearings", self.transcript())

    def test_off_for_tests_that_pass_their_own_model(self):
        self.assertFalse(DictationWorker(FakeBus(), Dictation(model=FakeWhisperModel()), settings=None).send_hearings)

    def test_gpu_only(self):
        from types import SimpleNamespace
        cpu = SimpleNamespace(whisper_device="cpu", whisper_hearings=True)
        gpu = SimpleNamespace(whisper_device="cuda", whisper_hearings=True)
        self.assertFalse(DictationWorker(FakeBus(), Dictation(cpu), settings=cpu).send_hearings)
        self.assertTrue(DictationWorker(FakeBus(), Dictation(gpu), settings=gpu).send_hearings)

    def test_constants(self):
        self.assertEqual(dictation_module.MAX_HEARINGS, 2)


class TestDecode(unittest.TestCase):
    def test_no_random_redecoding_when_unsure(self):
        seen = {}

        class Recording(FakeWhisperModel):
            def transcribe(self, audio, **kwargs):
                seen.update(kwargs)
                return super().transcribe(audio, **kwargs)

        Dictation(model=Recording()).transcribe("hello there")
        self.assertIsNone(seen["log_prob_threshold"])        # the loop check (compression ratio) stays on
        self.assertNotIn("temperature", seen)
        self.assertTrue(seen["word_timestamps"])


class TestVadThreshold(unittest.TestCase):
    def test_the_setting_reaches_silero(self):
        from engine.config import settings
        from engine.voice import audio_in

        self.assertEqual(audio_in.load_vad(1.0).threshold, settings.vad_threshold)
        self.assertEqual(audio_in.load_vad(1.0, threshold=0.3).threshold, 0.3)

    def test_the_default_is_unchanged(self):
        from engine.config import Settings
        from engine.voice import audio_in

        self.assertEqual(Settings().vad_threshold, audio_in.VAD_THRESHOLD)    # 0.5 unless config.local.json says


class ConstMouth:
    def __init__(self, f):
        self.f = f

    def moving_fraction(self, start_ms, end_ms):
        return self.f


class TestOptionsWithoutCleanUp(unittest.TestCase):
    def test_option_two_takes_whispers_other_hearing(self):
        bus = FakeBus()
        mm = ModeManager(bus, log=lambda *_: None)                     # no rewrite_wait_s: the clean-up is off
        mm.on_transcript({"text": "start dictation", "words": [{"text": "start", "conf": 1.0},
                                                              {"text": "dictation", "conf": 1.0}]})
        mm.on_transcript({"text": "meet at the bank", "hearings": ["meet at the bench", "Meet at the bank."],
                          "words": [{"text": t, "conf": 0.4} for t in "meet at the bank".split()]})
        self.assertEqual(bus.ui("dictation_preview")[-1]["alternatives"], ["meet at the bench"])   # no same-says
        mm.on_transcript({"text": "option two", "words": [{"text": "option", "conf": 1.0},
                                                         {"text": "two", "conf": 1.0}]})
        card = bus.ui("dictation_preview")[-1]
        self.assertEqual(" ".join(w["text"] for w in card["words"]), "meet at the bench")
        self.assertNotIn("alternatives", card)                          # picked: the card is fixed now
        mm.on_transcript({"text": "option three", "words": [{"text": "option", "conf": 1.0},
                                                           {"text": "three", "conf": 1.0}]})
        self.assertTrue(any("no option three" in m["text"] for m in bus.ui("status")))


class TestUnsureShort(unittest.TestCase):
    """ "up?" comes back at 0.3 (FakeWhisperModel), under the noise filter's 0.35."""

    def make(self, mouth):
        self.bus = FakeBus()
        w = DictationWorker(self.bus, Dictation(model=FakeWhisperModel()), mouth=mouth, settings=None,
                            log=lambda *_: None)
        w.show_unsure_short = True
        self.mm = ModeManager(self.bus, log=lambda *_: None)
        return w

    def test_kept_for_a_dictation_card_when_the_mouth_moved(self):
        w = self.make(ConstMouth(1.0))
        w.process("start dictation", {"start_ms": 0, "end_ms": 800})
        w.process("up?", {"start_ms": 1000, "end_ms": 1400})
        t = self.bus.of(topics.TRANSCRIPT)[-1]
        self.assertTrue(t["unsure"])
        preview = self.bus.ui("dictation_preview")[-1]["words"]
        self.assertEqual([(x["text"], x["conf"]) for x in preview], [("up", 0.3)])

    def test_never_a_command(self):
        w = self.make(ConstMouth(1.0))
        w.process("click?", {"start_ms": 0, "end_ms": 400})            # Pointer mode: "click" would click
        self.assertTrue(self.bus.of(topics.TRANSCRIPT)[-1]["unsure"])
        self.assertEqual(self.bus.of(topics.OS_CLICK), [])

    def test_an_unsure_yes_or_command_never_lands_on_the_card(self):
        w = self.make(ConstMouth(1.0))
        w.process("start dictation", {"start_ms": 0, "end_ms": 800})
        w.process("see you at noon", {"start_ms": 1000, "end_ms": 2600})
        for said in ("yes?", "stop?", "stop? dictation?"):
            w.process(said, {"start_ms": 3000, "end_ms": 3800})
            card = " ".join(x["text"] for x in self.bus.ui("dictation_preview")[-1]["words"])
            self.assertEqual(card, "see you at noon", said)
        self.assertEqual(self.bus.of(topics.OS_TYPE), [])                      # and nothing was typed
        self.assertTrue(any("Please say it again" in m["text"] for m in self.bus.ui("status")))

    def test_dropped_when_the_mouth_didnt_move(self):
        w = self.make(ConstMouth(0.0))
        w.process("start dictation", {"start_ms": 0, "end_ms": 800})
        before = len(self.bus.of(topics.TRANSCRIPT))
        w.process("up?", {"start_ms": 1000, "end_ms": 1400})
        self.assertEqual(len(self.bus.of(topics.TRANSCRIPT)), before)

    def test_dropped_without_a_camera(self):
        w = self.make(None)
        w.process("up?")
        self.assertEqual(self.bus.of(topics.TRANSCRIPT), [])


class TestModeManager(unittest.TestCase):
    def setUp(self):
        self.bus = FakeBus()
        self.mm = ModeManager(self.bus, clock=Clock(), rewrite_wait_s=1.5, log=lambda *_: None)
        self.mm.on_transcript({"text": "start dictation", "words": [{"text": "start", "conf": 1.0},
                                                                   {"text": "dictation", "conf": 1.0}]})

    def say(self, text, hearings=None):
        words = [{"text": t, "conf": 0.4} for t in text.split()]
        t = {"text": text, "words": words}
        if hearings:
            t["hearings"] = hearings
        self.mm.on_transcript(t)

    def request(self):
        return self.bus.of(topics.REWRITE_REQUEST)[-1]

    def test_whole_versions_of_the_card_newest_first(self):
        self.say("meet at the bank", ["meet at the bench"])
        self.assertEqual(self.request()["hearings"], ["meet at the bench"])
        self.say("at noon", ["at new", "at moon"])
        self.assertEqual(self.request()["text"], "meet at the bank at noon")
        self.assertEqual(self.request()["hearings"], ["meet at the bank at new", "meet at the bank at moon"])

    def test_none_without_hearings(self):
        self.say("meet at the bank")
        self.assertNotIn("hearings", self.request())

    def test_the_card_offers_them_when_the_clean_up_has_no_other_readings(self):
        self.say("meet at the bank", ["meet at the bench"])
        req = self.request()
        self.bus.publish(topics.REWRITE_RESULT, {"id": req["id"], "clean": "Meet at the bank.", "used": True,
                                                 "ms": 300.0})
        card = self.bus.ui("dictation_preview")[-1]
        self.assertEqual(card["alternatives"], ["meet at the bench"])
        self.mm.on_transcript({"text": "option two", "words": [{"text": "option", "conf": 1.0},
                                                               {"text": "two", "conf": 1.0}]})
        self.assertEqual(" ".join(w["text"] for w in self.bus.ui("dictation_preview")[-1]["words"]),
                         "meet at the bench")

    def test_a_fix_forgets_them(self):
        self.say("meet at the bank", ["meet at the bench"])
        self.say("change bank to bench")
        self.say("today")
        self.assertNotIn("hearings", self.request())


if __name__ == "__main__":
    unittest.main()
