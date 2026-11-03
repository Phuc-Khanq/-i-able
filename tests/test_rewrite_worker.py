"""Run with: python -m unittest tests.test_rewrite_worker  (real threads, fake rewriter, no model)"""
import threading
import time
import unittest

from engine.core import messages, topics
from engine.core.events import EventBus
from engine.voice.rewrite import RewriteWorker
from tests.helpers import FakeBus


class FakeRewriter:
    """Upper-cases the text. hold=True makes each rewrite wait until release() (to test the queue)."""

    def __init__(self, hold=False, fail_warm_up=None, raises=False):
        self.calls = []
        self.started = threading.Event()
        self.gate = threading.Event()
        if not hold:
            self.gate.set()
        self.fail_warm_up, self.raises = fail_warm_up, raises

    def warm_up(self):
        if self.fail_warm_up:
            raise self.fail_warm_up

    def rewrite(self, text):
        self.calls.append(text)
        self.started.set()
        self.gate.wait(5)
        if self.raises:
            raise RuntimeError("boom")
        return text.upper(), True, 12.5


def results_when(bus, n, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        got = bus.of(topics.REWRITE_RESULT)
        if len(got) >= n:
            return got
        time.sleep(0.01)
    return bus.of(topics.REWRITE_RESULT)


class TestRewriteWorker(unittest.TestCase):
    def setUp(self):
        self.workers = []

    def tearDown(self):
        for w in self.workers:
            w.stop()

    def start(self, rewriter, warm_up=False):
        bus = FakeBus()
        w = RewriteWorker(bus, rewriter, log=lambda s: None)
        w.start(warm_up=warm_up)
        self.workers.append(w)
        return w, bus

    def test_result_carries_the_id_and_the_rewrite(self):
        w, bus = self.start(FakeRewriter())
        bus.publish(topics.REWRITE_REQUEST, {"id": 7, "text": "hi mom"})
        self.assertEqual(results_when(bus, 1), [{"id": 7, "clean": "HI MOM", "used": True, "ms": 12.5}])

    def test_newest_wins_while_busy(self):
        rw = FakeRewriter(hold=True)
        w, bus = self.start(rw)
        bus.publish(topics.REWRITE_REQUEST, {"id": 1, "text": "one"})
        self.assertTrue(rw.started.wait(2))                    # "one" is running
        for i, text in ((2, "two"), (3, "three"), (4, "four"), (5, "five")):
            bus.publish(topics.REWRITE_REQUEST, {"id": i, "text": text})
        rw.gate.set()                                         # let it finish
        got = results_when(bus, 2)
        time.sleep(0.1)                                       # nothing else may follow
        self.assertEqual(rw.calls, ["one", "five"])
        self.assertEqual([r["id"] for r in bus.of(topics.REWRITE_RESULT)], [1, 5])
        self.assertEqual(got[-1]["clean"], "FIVE")

    def test_request_handler_only_queues(self):
        rw = FakeRewriter(hold=True)
        w, bus = self.start(rw)
        bus.publish(topics.REWRITE_REQUEST, {"id": 1, "text": "one"})
        rw.started.wait(2)
        t0 = time.perf_counter()
        bus.publish(topics.REWRITE_REQUEST, {"id": 2, "text": "two"})    # e.g. from the camera thread
        self.assertLess(time.perf_counter() - t0, 0.05)
        rw.gate.set()

    def test_stop_ends_the_thread(self):
        w, bus = self.start(FakeRewriter())
        w.stop(timeout=2)
        self.assertFalse(w._thread.is_alive())

    def test_a_raising_rewriter_still_answers_with_the_original(self):
        w, bus = self.start(FakeRewriter(raises=True))
        bus.publish(topics.REWRITE_REQUEST, {"id": 3, "text": "hi mom"})
        self.assertEqual(results_when(bus, 1), [{"id": 3, "clean": "hi mom", "used": False, "ms": 0.0}])

    def test_model_that_cant_load_turns_clean_up_off_but_still_answers(self):
        rw = FakeRewriter(fail_warm_up=OSError("Qwen not downloaded"))
        w, bus = self.start(rw, warm_up=True)
        bus.publish(topics.REWRITE_REQUEST, {"id": 1, "text": "hi mom"})
        bus.publish(topics.REWRITE_REQUEST, {"id": 2, "text": "see you"})
        got = results_when(bus, 1)
        self.assertEqual(got[-1]["used"], False)
        self.assertEqual(rw.calls, [])                         # never asks the dead model
        [status] = bus.ui(messages.STATUS)
        self.assertEqual(status["level"], "warn")
        self.assertEqual(status["text"], "Text clean-up is off: Qwen not downloaded")

    def test_works_on_the_real_bus(self):
        bus = EventBus()
        got = []
        bus.subscribe(topics.REWRITE_RESULT, got.append)
        w = RewriteWorker(bus, FakeRewriter(), log=lambda s: None)
        w.start(warm_up=False)
        self.workers.append(w)
        bus.publish(topics.REWRITE_REQUEST, {"id": 1, "text": "ok"})
        deadline = time.monotonic() + 5
        while not got and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(got, [{"id": 1, "clean": "OK", "used": True, "ms": 12.5}])


class ContextRewriter:
    """Like the Rewriter: takes the request's context and app, answers with other readings."""

    takes_context = True

    def __init__(self):
        self.calls = []

    def warm_up(self):
        pass

    def rewrite_full(self, text, context="", app=""):
        from engine.voice.rewrite import RewriteOutcome

        self.calls.append(("rewrite", text, context, app))
        return RewriteOutcome(text, "Send the picture.", True, 9.0, None, "", ["Send the pitcher."], True)

    def refine_full(self, marked, context="", app=""):
        from engine.voice.rewrite import RewriteOutcome

        self.calls.append(("refine", marked, context, app))
        return RewriteOutcome(marked, marked, False, 9.0, "rejected")


class TestHintsAndReadings(unittest.TestCase):
    def test_context_and_app_go_in_and_other_readings_come_out(self):
        bus, rw = FakeBus(), ContextRewriter()
        w = RewriteWorker(bus, rw, log=lambda s: None)
        w.process({"id": 3, "text": "send the pitcher", "context": "Here are the photos.", "app": "chat"})
        self.assertEqual(rw.calls, [("rewrite", "send the pitcher", "Here are the photos.", "chat")])
        self.assertEqual(bus.of(topics.REWRITE_RESULT)[-1],
                         {"id": 3, "clean": "Send the picture.", "used": True, "ms": 9.0,
                          "alternatives": ["Send the pitcher."], "uncertain": True})

    def test_no_hints_no_new_fields_when_not_used(self):
        bus, rw = FakeBus(), ContextRewriter()
        w = RewriteWorker(bus, rw, log=lambda s: None)
        w.process({"id": 4, "mode": "refine", "text": "a b", "marked": "a (?)b(?)"})
        self.assertEqual(rw.calls, [("refine", "a (?)b(?)", "", "")])
        self.assertEqual(bus.of(topics.REWRITE_RESULT)[-1],
                         {"id": 4, "clean": "a (?)b(?)", "used": False, "ms": 9.0, "mode": "refine"})


if __name__ == "__main__":
    unittest.main()
