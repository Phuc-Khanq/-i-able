"""Run with: python -m unittest tests.test_foreground_app  (the kind of app in front, for the clean-up; no Windows calls)"""
import unittest

from engine.eyes.os_input import app_kind, foreground_app


class TestAppKind(unittest.TestCase):
    def test_programs(self):
        self.assertEqual(app_kind(r"C:\Program Files\Microsoft Office\OUTLOOK.EXE"), "email")
        self.assertEqual(app_kind("slack.exe"), "Slack")
        self.assertEqual(app_kind("WINWORD.EXE"), "document")
        self.assertEqual(app_kind("game.exe"), "")

    def test_a_browser_by_its_title(self):
        self.assertEqual(app_kind("chrome.exe", "Inbox (3) - me@example.com - Gmail"), "email")
        self.assertEqual(app_kind("msedge.exe", "WhatsApp"), "WhatsApp")
        self.assertEqual(app_kind("firefox.exe", "Weather today"), "browser")

    def test_never_raises(self):
        self.assertIsInstance(foreground_app(), str)


if __name__ == "__main__":
    unittest.main()
