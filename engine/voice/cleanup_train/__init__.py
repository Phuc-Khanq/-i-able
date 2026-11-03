"""Training data and LoRA training for our own dictation clean-up adapter on Qwen3.5-4B.

seeds.py holds the hand-written base sentences and hard cases, noise.py turns a clean sentence into what a
speech recognizer might hear from a speaker who stutters or slurs, and build.py writes the training file.
"""
