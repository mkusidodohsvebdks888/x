"""Autonomous practice, data isolation, resource limits and persistence tests."""
import json
from pathlib import Path
import shutil
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import torch

from srt_m15.autolearn import AutoLearningController
from srt_m15.chat import ChatReply, SRTChatEngine
from srt_m15.practice import AutonomousPracticeRunner, PracticeConfig
from srt_m15.tasks import ArithmeticTokenizer


class PracticeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        cp = self.root / "checkpoints"
        cp.mkdir()
        source = Path(__file__).resolve().parents[1] / "checkpoints/srt_m15_toy.pt"
        if not source.is_file():
            self.skipTest("Model checkpoint unavailable")
        shutil.copyfile(source, cp / source.name)
        self.engine = SRTChatEngine(self.root)
        self.lock = threading.RLock()
        # High threshold disables training for basic practice mechanics tests.
        self.learning = AutoLearningController(self.engine, self.lock, threshold=1000,
                                               steps=2, cooldown_seconds=0)
        self.practice = AutonomousPracticeRunner(self.engine, self.lock, self.learning, seed=17)
        self.addCleanup(self.practice.pause)

    def test_config_limits(self):
        for values in ({"interval_seconds": 0}, {"questions_per_round": 0},
                       {"max_training_attempts": 9}, {"max_runtime_seconds": -1},
                       {"max_questions": 501}, {"max_questions": 1.5},
                       {"interval_seconds": float("nan")},
                       {"max_questions": True}, {"surprise": 1}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                PracticeConfig.from_request(values)
        good = PracticeConfig.from_request({"interval_seconds": 0.2,
                                            "max_runtime_seconds": 1})
        self.assertEqual(good.max_runtime_seconds, 1)

    def test_generator_only_uses_training_pool_and_covers_allowed_pairs(self):
        got = [self.practice._next_pair() for _ in range(2 * len(self.learning._train_set))]
        self.assertEqual(set(got), self.learning._train_set)
        self.assertTrue(set(got).isdisjoint(self.learning._test_set))
        self.assertTrue(set(got).isdisjoint(self.learning._guard_set))
        self.assertEqual(len(self.learning._test_set), 20)
        self.assertTrue(all(0 <= a <= 9 and 0 <= b <= 9 for a, b in got))

    def test_error_verification_queue_and_stats(self):
        self.learning.set_enabled(True)
        wrong_pair = next((p for p in sorted(self.learning._train_set)
                           if (r := self.engine.reply(f"{p[0]}+{p[1]}")).prediction != r.expected))
        # Because .pop() returns the final list entry, force one eligible wrong sample.
        self.practice._deck = [wrong_pair]
        self.practice._run_round(1)
        snapshot = self.practice.status()
        self.assertEqual(snapshot["total_questions"], 1)
        self.assertEqual(snapshot["total_errors"], 1)
        self.assertEqual(snapshot["pending"], 1)
        self.assertEqual(snapshot["last_problem"]["verified_answer"], sum(wrong_pair))
        self.assertEqual(snapshot["last_problem"]["learning_event"], "queued")
        self.assertEqual(json.loads(self.practice.file_path.read_text())["total_questions"], 1)

    def test_start_pause_and_restart_keeps_counters_without_resuming(self):
        self.practice.start({"interval_seconds": 0.2, "questions_per_round": 2,
                             "max_training_attempts": 2, "max_runtime_seconds": 20,
                             "max_questions": 70})
        deadline = time.monotonic() + 5
        while self.practice.status()["total_questions"] < 2 and time.monotonic() < deadline:
            time.sleep(0.04)
        self.assertGreaterEqual(self.practice.status()["total_questions"], 1)
        stopped = self.practice.pause()
        self.assertFalse(stopped["running"])
        self.assertEqual(stopped["last_stop_reason"], "manual_pause")
        count = stopped["total_questions"]
        time.sleep(0.35)
        self.assertEqual(self.practice.status()["total_questions"], count)
        saved = AutonomousPracticeRunner(self.engine, self.lock, self.learning, seed=4)
        self.assertFalse(saved.status()["running"])
        self.assertEqual(saved.status()["total_questions"], count)
        self.assertFalse(self.learning.status()["enabled"])

    def test_max_questions_auto_stops(self):
        self.practice.start({"interval_seconds": 0.2, "questions_per_round": 4,
                             "max_training_attempts": 2, "max_runtime_seconds": 20,
                             "max_questions": 5})
        deadline = time.monotonic() + 7
        while self.practice.status()["running"] and time.monotonic() < deadline:
            time.sleep(0.05)
        snapshot = self.practice.status()
        self.assertFalse(snapshot["running"])
        self.assertEqual(snapshot["last_stop_reason"], "max_questions")
        self.assertEqual(snapshot["session_questions"], 5)
        self.assertFalse(self.learning.status()["enabled"])

    def test_max_runtime_auto_stops(self):
        self.practice.start({"interval_seconds": 0.2, "questions_per_round": 1,
                             "max_training_attempts": 2, "max_runtime_seconds": 1,
                             "max_questions": 500})
        deadline = time.monotonic() + 5
        while self.practice.status()["running"] and time.monotonic() < deadline:
            time.sleep(0.05)
        snapshot = self.practice.status()
        self.assertFalse(snapshot["running"])
        self.assertEqual(snapshot["last_stop_reason"], "max_runtime")
        self.assertLessEqual(snapshot["session_questions"], 30)

    def test_runtime_limit_even_when_interval_exceeds_budget(self):
        start = time.monotonic()
        self.practice.start({"interval_seconds": 60, "questions_per_round": 1,
                             "max_training_attempts": 2, "max_runtime_seconds": 1,
                             "max_questions": 500})
        deadline = start + 4
        while self.practice.status()["running"] and time.monotonic() < deadline:
            time.sleep(0.05)
        snapshot = self.practice.status()
        self.assertFalse(snapshot["running"])
        self.assertEqual(snapshot["last_stop_reason"], "max_runtime")
        self.assertLess(time.monotonic() - start, 3)

    def test_training_limit_stops_practice_after_one_attempt(self):
        self.learning.threshold = 1
        self.practice.start({"interval_seconds": 0.2, "questions_per_round": 5,
                             "max_training_attempts": 1, "max_runtime_seconds": 15,
                             "max_questions": 100})
        deadline = time.monotonic() + 12
        while self.practice.status()["running"] and time.monotonic() < deadline:
            time.sleep(0.05)
        result = self.practice.status()
        self.assertFalse(result["running"])
        self.assertEqual(result["last_stop_reason"], "max_training_attempts")
        self.assertEqual(result["training_attempts"], 1)
        self.assertFalse(self.learning.status()["enabled"])
        if self.learning._worker:
            self.learning._worker.join(timeout=15)
        self.assertTrue((self.root / "checkpoints/srt_m15_toy.pt").exists())


class CandidateAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "checkpoints").mkdir()
        original = Path(__file__).resolve().parents[1] / "checkpoints/srt_m15_toy.pt"
        if not original.is_file():
            self.skipTest("Model checkpoint unavailable")
        shutil.copyfile(original, self.root / "checkpoints" / original.name)
        self.engine = SRTChatEngine(self.root)
        self.controller = AutoLearningController(self.engine, threading.RLock(), threshold=1,
                                                 max_attempts=1, steps=1, cooldown_seconds=0)
        self.addCleanup(lambda: self.controller.set_enabled(False))
        self.sample = next(iter(sorted(self.controller._train_set)))
        self.original_bytes = self.controller.original_file.read_bytes()

    def _fire(self, scoring, modify_fast=True, change_backbone=False):
        def controlled_adapt(learner, _batch, _guard, **_kwargs):
            with torch.no_grad():
                if modify_fast:
                    next(learner.model.fast_parameters()).add_(0.001)
                if change_backbone:
                    next(learner.model.backbone.parameters()).add_(0.001)
            return {"accepted": True, "cancelled": False, "updated_parameters": 100}

        with patch("srt_m15.autolearn.accuracy", side_effect=scoring), \
             patch("srt_m15.autolearn.OnlineLearner.adapt", new=controlled_adapt):
            self.controller.set_enabled(True)
            result = self.controller.observe(ChatReply(
                "math", "", observed_pair=self.sample, prediction=-1, expected=sum(self.sample)))
            self.assertEqual(result, "queued")
            self.controller._worker.join(timeout=15)
        self.assertFalse(self.controller._worker.is_alive())
        return self.controller.status()

    def test_accepted_saves_and_restart_can_load_then_roll_back(self):
        status = self._fire([0.2, 0.8, 0.7, 0.8])
        self.assertEqual(status["last_result"]["state"], "accepted")
        self.assertEqual(status["accepted_updates"], 1)
        self.assertTrue(self.controller.adapted_file.exists())
        loaded = torch.load(self.controller.adapted_file, weights_only=True)
        self.assertTrue(loaded["online_learning"]["fast_parameters_only"])
        restarted = SRTChatEngine(self.root)
        self.assertTrue(restarted._load_math())
        self.assertEqual(restarted.math_checkpoint_path, self.controller.adapted_file)
        self.assertEqual(self.controller.original_file.read_bytes(), self.original_bytes)
        self.controller.rollback()
        self.assertFalse(self.controller.adapted_file.exists())
        self.assertEqual(self.engine.math_checkpoint_path, self.controller.original_file)

    def test_rejected_for_guard_regression(self):
        status = self._fire([0.2, 0.8, 0.7, 0.5])
        self.assertEqual(status["last_result"]["state"], "rejected")
        self.assertFalse(self.controller.adapted_file.exists())
        self.assertEqual(self.controller.original_file.read_bytes(), self.original_bytes)

    def test_rejected_if_protected_main_network_modified(self):
        status = self._fire([0.2, 0.8, 0.7, 0.8], change_backbone=True)
        self.assertEqual(status["last_result"]["state"], "rejected")
        self.assertEqual(status["last_result"]["basic_check"], "protected_parameter_changed")
        self.assertFalse(self.controller.adapted_file.exists())

    def test_rejected_if_candidate_contains_nonfinite_fast_weight(self):
        def broken_update(learner, _batch, _guard, **_kw):
            with torch.no_grad():
                next(learner.model.fast_parameters()).fill_(float("nan"))
            return {"accepted": True, "cancelled": False, "updated_parameters": 100}
        with patch("srt_m15.autolearn.accuracy", side_effect=[0.2, 0.8, 0.9, 0.8]), \
             patch("srt_m15.autolearn.OnlineLearner.adapt", new=broken_update):
            self.controller.set_enabled(True)
            self.controller.observe(ChatReply("math", "", observed_pair=self.sample,
                                              prediction=-1, expected=sum(self.sample)))
            self.controller._worker.join(timeout=15)
        self.assertEqual(self.controller.status()["last_result"]["state"], "rejected")
        self.assertEqual(self.controller.status()["last_result"]["basic_check"], "nonfinite_parameters")
        self.assertFalse(self.controller.adapted_file.exists())

    def test_pause_cancels_candidate_and_keeps_queue(self):
        started = threading.Event()
        resume = threading.Event()
        def controlled_adapt(learner, batch, guard, **kw):
            started.set()
            resume.wait(timeout=5)
            return {"accepted": False, "cancelled": bool(kw["cancel_check"]()),
                    "updated_parameters": 100}
        with patch("srt_m15.autolearn.OnlineLearner.adapt", new=controlled_adapt):
            self.controller.set_enabled(True)
            self.controller.observe(ChatReply("math", "", observed_pair=self.sample,
                                              prediction=-1, expected=sum(self.sample)))
            self.assertTrue(started.wait(timeout=8))
            self.controller.set_enabled(False)
            self.controller.cancel_running()
            resume.set()
            self.controller._worker.join(timeout=10)
        self.assertEqual(self.controller.status()["last_result"]["state"], "cancelled")
        self.assertEqual(self.controller.status()["pending"], 1)
        self.assertFalse(self.controller.adapted_file.exists())


if __name__ == "__main__":
    unittest.main()
