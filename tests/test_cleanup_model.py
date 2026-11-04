"""Run with: python -m unittest tests.test_cleanup_model  (the trained clean-up model in the app: the prompt,
the answer, llama-server's handling, the Rewriter's llamacpp backend and its guard; no GPU, no server)"""
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from engine.voice import cleanup_model, rewrite
from engine.voice.cleanup_model import Answer, CleanupModel, LlamaServer, build_prompt, chat_text, parse_answer
from engine.voice.rewrite import Rewriter, check_cleanup

ROOT = Path(__file__).resolve().parents[1]


class TestPrompt(unittest.TestCase):
    def test_same_prompt_file_as_the_eval_and_the_training(self):
        eval_copy = ROOT / "eval" / "dictation_cleanup" / "prompts" / "cleanup_t1_short.txt"
        self.assertEqual(cleanup_model.PROMPT_FILE.read_bytes(), eval_copy.read_bytes())

    def test_fields_as_the_training_data_fills_them(self):
        p = build_prompt("um hi there", ["Priya", "Dr. Okafor"])
        self.assertIn("Heard: um hi there\nOther hearings: (none)\nApp: (none)\nBefore: (none)\n"
                      "Word list: Priya, Dr. Okafor<|im_end|>", p)
        self.assertTrue(p.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"))

    def test_chat_text_matches_the_hugging_face_template(self):
        try:
            from eval.dictation_cleanup.backends import _chat_template
            from eval.dictation_cleanup.prompt_files import load_prompt

            _, tpl = _chat_template("Qwen/Qwen3.5-4B")
        except Exception as exc:                             # no cached tokenizer on this machine
            self.skipTest(f"Qwen3.5 tokenizer not available: {exc}")
        msgs = load_prompt("t1_short").messages({"asr_best": "um hi there", "personal_words": "Priya|Dr. Okafor"})
        hf = tpl.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        self.assertEqual(build_prompt("um hi there", ["Priya", "Dr. Okafor"]), hf)
        self.assertEqual(chat_text(msgs[0]["content"], msgs[1]["content"]), hf)


class TestAnswer(unittest.TestCase):
    def test_json_answers(self):
        self.assertEqual(parse_answer('{"text": "Hi there.", "alternatives": [], "uncertain": false}'),
                         Answer("Hi there."))
        self.assertEqual(parse_answer('```json\n{"text": "A.", "alternatives": ["B."], "uncertain": true}\n```'),
                         Answer("A.", ["B."], True))

    def test_anything_else_is_none(self):
        for raw in ['{"text": "Hi', "Hi there.", '{"text": ""}', '{"text": 5}', "[]"]:
            self.assertIsNone(parse_answer(raw), raw)
        self.assertEqual(parse_answer('{"text": "A.", "alternatives": "B."}').alternatives, [])


class FakeResponse:
    def __init__(self, status=200, body=b"{}"):
        self.status, self._body = status, body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeHTTP:
    """Stands in for urllib: /health answers 503 `loading` times, then 200; /completion answers `reply`."""

    def __init__(self, reply: dict, loading: int = 1):
        self.reply, self.loading, self.requests = reply, loading, []

    def open(self, req, timeout=None):
        if isinstance(req, str):                             # GET /health
            if self.loading > 0:
                self.loading -= 1
                raise OSError("503 loading")
            return FakeResponse(200)
        self.requests.append(req)
        return FakeResponse(200, json.dumps(self.reply).encode())


class FakeProc:
    def __init__(self, cmd):
        self.cmd, self.returncode, self.stdout, self._handle = cmd, None, [b"model loaded\n"], 0

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = 0

    def wait(self, timeout=None):
        return self.returncode


class TestServer(unittest.TestCase):
    def server(self, tmp: Path, reply: dict | None = None) -> LlamaServer:
        exe, model = tmp / "llama-server.exe", tmp / "m.gguf"
        exe.write_bytes(b"")
        model.write_bytes(b"")
        self.procs = []

        def popen(cmd, **kw):
            self.procs.append(FakeProc(cmd))
            return self.procs[-1]

        s = LlamaServer(exe, model, log=lambda *a: None, popen=popen)
        s._http = FakeHTTP(reply or {"content": '{"text": "Hi.", "alternatives": [], "uncertain": false}',
                                     "stop_type": "eos"})
        return s

    def test_this_pc_only_offline_with_a_key(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            s = self.server(Path(tmp))
            s.start()
            cmd = self.procs[0].cmd
            self.assertEqual(cmd[cmd.index("--host") + 1], "127.0.0.1")
            for flag in ("--offline", "--no-webui", "--no-slots", "--api-key"):
                self.assertIn(flag, cmd)
            self.assertGreaterEqual(len(cmd[cmd.index("--api-key") + 1]), 24)
            s.complete("prompt", 64, 3.0)
            req = s._http.requests[0]
            self.assertEqual(req.get_header("Authorization"), f"Bearer {cmd[cmd.index('--api-key') + 1]}")
            body = json.loads(req.data)
            self.assertEqual((body["top_k"], body["samplers"], body["n_predict"]), (1, ["top_k"], 64))

    def test_a_server_that_died_starts_again_once(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            s = self.server(Path(tmp))
            s.start()
            self.procs[0].returncode = 1                     # crashed
            s._http.loading = 0
            s.complete("prompt", 64, 3.0)
            self.assertEqual(len(self.procs), 2)
            s.stop()
            self.assertEqual(self.procs[1].returncode, 0)

    def test_missing_files_say_which(self):
        s = LlamaServer(ROOT / "no" / "llama-server.exe", ROOT / "no" / "m.gguf", popen=None)
        with self.assertRaisesRegex(FileNotFoundError, "llama-server"):
            s.start()


class FakeCleanup:
    """Stands in for CleanupModel: answers from a script ({text sent: raw JSON or None})."""

    def __init__(self, replies: dict):
        self.replies, self.sent, self.stopped = replies, [], False

    def start(self):
        pass

    def clean(self, text, max_s, max_tokens=None, before="", app=""):
        self.sent.append(text)
        self.hints = {"before": before, "app": app}
        raw = self.replies.get(text)
        return (parse_answer(raw) if raw else None), False

    def stop(self):
        self.stopped = True


def llamacpp_rewriter(replies: dict, words: str = "") -> tuple[Rewriter, FakeCleanup]:
    fake = FakeCleanup(replies)
    settings = SimpleNamespace(rewrite_backend="llamacpp", rewrite_gguf="m.gguf", rewrite_words=words,
                               rewrite_max_s=3.0, refine_max_s=15.0)
    return Rewriter(settings, cleanup=fake, log=lambda *a: None), fake


def reply(text, alts=(), unsure=False):
    return json.dumps({"text": text, "alternatives": list(alts), "uncertain": unsure})


class TestLlamaCppBackend(unittest.TestCase):
    def test_the_models_text_is_used_when_the_guard_agrees(self):
        rw, fake = llamacpp_rewriter({"send the pitcher to mom now": reply("Send the picture to Mom now.")})
        r = rw.rewrite_full("send the pitcher to mom now")
        self.assertEqual((r.clean, r.used, r.reason), ("Send the picture to Mom now.", True, None))
        self.assertEqual(rw.last_answer.text, "Send the picture to Mom now.")

    def test_broken_json_falls_back_to_the_rule_based_clean_up(self):
        rw, _ = llamacpp_rewriter({})
        r = rw.rewrite_full("um so I'll call you later")
        self.assertTrue(r.used)
        self.assertEqual(r.clean, "so I'll call you later")             # light_clean: the filler goes, nothing else
        self.assertIn("valid JSON", r.reason)

    def test_an_invented_number_is_not_used(self):
        rw, _ = llamacpp_rewriter({"meet me at to o'clock": reply("Meet me at one o'clock.")})
        r = rw.rewrite_full("meet me at to o'clock")
        self.assertEqual(r.clean, "meet me at to o'clock")
        self.assertIn("number added", r.reason)

    def test_refine_sends_the_plain_words(self):
        rw, fake = llamacpp_rewriter({"the meeting is on tuesday": reply("The meeting is on Tuesday.")})
        r = rw.refine_full("the meeting is on (?)tuesday(?)")
        self.assertEqual(fake.sent, ["the meeting is on tuesday"])
        self.assertTrue(r.used)

    def test_the_word_list_reaches_the_guard(self):
        rw, _ = llamacpp_rewriter({"tell pre a I'll be late": reply("Tell Priya I'll be late.")}, words="Priya, Dr. Lee")
        self.assertEqual(rw.words, ["Priya", "Dr. Lee"])
        self.assertTrue(rw.rewrite_full("tell pre a I'll be late").used)

    def test_context_and_app_reach_the_model_and_the_guard(self):
        rw, fake = llamacpp_rewriter({"forward it to doctor alvarez": reply("Forward it to Dr. Alvarez.")})
        r = rw.rewrite_full("forward it to doctor alvarez", context="Thanks, Dr. Alvarez.", app="email")
        self.assertEqual(fake.hints, {"before": "Thanks, Dr. Alvarez.", "app": "email"})
        self.assertTrue(r.used)
        # a name only the earlier text holds is not invented
        rw, _ = llamacpp_rewriter({"tell him I'm late": reply("Tell Marcus I'm late.")})
        self.assertFalse(rw.rewrite_full("tell him I'm late").used)
        rw, _ = llamacpp_rewriter({"tell him I'm late": reply("Tell Marcus I'm late.")})
        self.assertTrue(rw.rewrite_full("tell him I'm late", context="Marcus, are you coming?").used)

    def test_other_readings_pass_the_guard_too(self):
        rw, _ = llamacpp_rewriter({"order tick out from the Italian place": reply(
            "Order takeout from the Italian place.",
            ["Order takeout from the Italian place.", "Order gluten-free takeout from the Italian place.",
             "Order takeaway from the Italian place."], True)})
        r = rw.rewrite_full("order tick out from the Italian place")
        self.assertTrue(r.uncertain)
        self.assertEqual(r.alternatives, ["Order takeaway from the Italian place."])   # same as text: dropped;
                                                                                          # "gluten-free" adds content
    def test_close_stops_the_server(self):
        rw, fake = llamacpp_rewriter({})
        rw.close()
        self.assertTrue(fake.stopped)

    def test_settings_build_the_server_from_repo_paths(self):
        s = SimpleNamespace(llama_server_path="downloads/x/llama-server.exe", rewrite_gguf="models/cleanup/m.gguf",
                            rewrite_words="Priya, , Sam")
        cm = CleanupModel.from_settings(s, log=lambda *a: None)
        self.assertEqual(cm.server.server, ROOT / "downloads" / "x" / "llama-server.exe")
        self.assertEqual(cm.words, ["Priya", "Sam"])


class TestCleanupGuard(unittest.TestCase):
    def test_fixes_and_full_sentences_pass(self):
        self.assertIsNone(check_cleanup("hungry now food please", "I'm hungry now. Can I have some food, please?"))
        self.assertIsNone(check_cleanup("send the pitcher to mom", "Send the picture to Mom."))
        self.assertIsNone(check_cleanup("Can you come at six, no, at seven", "Can you come at seven?"))
        self.assertIsNone(check_cleanup("Does medi cure cover the chair", "Does Medicare cover the chair?"))
        self.assertIsNone(check_cleanup("The Di- Discord server is down", "The Discord server is down."))
        self.assertIsNone(check_cleanup("The venue needs it by Tuesday, no Friday", "The venue needs it by Tuesday, not Friday."))

    def test_numbers(self):
        self.assertIn("number added", check_cleanup("meet at to o'clock", "Meet at two o'clock."))
        self.assertIn("number missing", check_cleanup("we need six chairs", "We need chairs."))

    def test_dates_and_misspelled_numbers(self):
        # an ordinal is a number: "the fifteenth" -> "the end" loses it, spelled right or not (GPU test, B1232)
        self.assertIn("number missing", check_cleanup("the deadline is the fifteenth of March",
                                                      "The deadline is the end of March."))
        self.assertIn("number missing", check_cleanup("the grant deadline is the fifteend of march",
                                                      "The grant deadline is the end of March."))
        self.assertIn("number added", check_cleanup("meet me on the fifth", "Meet me on the sixth."))
        # a misspelled number is still the number it was meant to be
        self.assertIsNone(check_cleanup("the grant deadline is the fifteend of march",
                                        "The grant deadline is the fifteenth of March."))
        self.assertIsNone(check_cleanup("we raised five thowsand dollers for the walk",
                                        "We raised five thousand dollars for the walk."))
        self.assertIsNone(check_cleanup("the party is on the 15th", "The party is on the fifteenth."))
        self.assertIsNone(check_cleanup("we need fourty chairs", "We need forty chairs."))

    def test_words_close_to_numbers_are_not_numbers(self):
        self.assertEqual(rewrite._number_stems("there these then tent tree thirsty forth seconds".split()), set())
        self.assertEqual(rewrite._number_stems(["fifteend", "thowsand", "fifteenth", "fourty"]),
                         {"fifteen", "thousand", "forty"})
        self.assertIsNone(check_cleanup("give me a second to find it", "Give me a moment to find it."))

    def test_a_word_list_word_must_sound_like_what_was_heard(self):
        words = ["Dr. Okafor", "sertraline"]
        heard = "please let doctor oh car for know the new dose makes me dizzy"
        self.assertIsNone(check_cleanup(heard, "Please let Dr. Okafor know the new dose makes me dizzy.", words))
        self.assertIn("word-list word added: sertraline", check_cleanup(
            heard, "Please let Dr. Okafor know the new dose of sertraline makes me dizzy.", words))
        self.assertIsNone(check_cleanup("can I switch the sir tra lean to the evening",
                                        "Can I switch the sertraline to the evening?", words))
        self.assertIsNone(check_cleanup("I need my power wheel chair charged", "I need my power wheelchair charged.",
                                        ["power wheelchair"]))
        self.assertIn("word-list word added", check_cleanup(
            "the s- score server needs a new moderator", "The Minecraft speedrun server needs a new moderator.",
            ["Minecraft", "speedrun", "Discord"]))
        # sounds alike though spelled apart (big1 test B1529, B0332)
        self.assertIsNone(check_cleanup("Rinse the kin wa before you cook it", "Rinse the quinoa before you cook it.",
                                        ["quinoa", "marinade"]))
        self.assertIsNone(check_cleanup("I've attached my raise you may", "I've attached my résumé.", ["résumé"]))
        self.assertIsNone(check_cleanup("Tell Coa I will be late", "Tell Khoa I will be late.", ["Khoa"]))   # dev M10
        # a different name from the list in place of one that was heard (B0217)
        self.assertIn("word-list word added: leo", check_cleanup("Ask Theo if the store is open",
                                                                 "Ask Leo if the store is open.", ["Leo", "Theo"]))
        # what was typed earlier is evidence too, and isn't held to sounding alike
        self.assertIsNone(check_cleanup("and tell her thanks", "And tell Priya thanks.", context="Hi Priya,"))

    def test_negations(self):
        self.assertIn("negation added", check_cleanup("I can come tonight", "I can't come tonight."))
        self.assertIn("negation missing", check_cleanup("Don't add the salt", "Add the salt."))

    def test_names(self):
        self.assertIn("name added", check_cleanup("send it to Ann", "Send it to Dan."))
        self.assertIsNone(check_cleanup("tell pre a I'll be late", "Tell Priya I'll be late.", ["Priya"]))
        self.assertIn("name missing", check_cleanup("Tell Marcus the van is late", "Tell him the van is late."))
        self.assertIsNone(check_cleanup("forward this to doctor Lee's office", "Forward this to Dr. Lee's office."))

    def test_a_dropped_sentence_or_a_different_one(self):
        heard = "how about a picnic at the park if it's sunny Thanks for being so patient with my typing"
        self.assertIn("content dropped", check_cleanup(heard, "How about a picnic at the park if it's sunny?"))
        self.assertIn("content dropped", check_cleanup("my insurance will cover the repair after the deductible",
                                                       "Can you check the tire pressure on the van?"))


if __name__ == "__main__":
    unittest.main()
