import json
import shutil
import tempfile
import threading
import unittest
from pathlib import Path

from srt_m15.autolearn import AutoLearningController
from srt_m15.chat import ChatReply, SRTChatEngine


class AutonomousLearningTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "checkpoints").mkdir()
        source = Path(__file__).resolve().parents[1] / "checkpoints" / "srt_m15_toy.pt"
        if not source.is_file():
            self.skipTest("test weight is not installed")
        shutil.copy2(source, self.root / "checkpoints" / source.name)
        self.engine = SRTChatEngine(self.root)
        self.guard = threading.RLock()
        self.control = AutoLearningController(self.engine, self.guard, threshold=3,
                                              steps=8, cooldown_seconds=0)

    def test_default_disabled_and_persistence(self):
        status = self.control.status()
        self.assertFalse(status["enabled"])
        self.assertEqual(status["reserved_test_pairs"], 20)
        self.assertGreater(status["eligible_pairs"], 0)
        self.assertEqual(self.control.observe(self.engine.reply("0+9")), "disabled")
        self.control.set_enabled(True)
        self.assertTrue(AutoLearningController(self.engine, self.guard).status()["enabled"])
        self.control.set_enabled(False)
        self.assertFalse(json.loads(self.control.state_file.read_text())["enabled"])

    def test_verified_errors_only_and_holdout_protected(self):
        self.control.set_enabled(True)
        guard = next(iter(self.control._guard_set))
        holdout = next(iter(self.control._test_set))
        eligible = next(iter(self.control._train_set))
        for pair, expected in [(guard, "guard_only"), (holdout, "reserved_test")]:
            resp = ChatReply("math", "", observed_pair=pair, prediction=0,
                             expected=sum(pair))
            self.assertEqual(self.control.observe(resp), expected)
        self.assertEqual(self.control.observe(ChatReply("text", "unverified")), "not_trainable")
        self.assertEqual(self.control.observe(ChatReply("math", "", observed_pair=eligible,
                                                            prediction=-1, expected=999)), "unverified")
        self.assertEqual(self.control.observe(ChatReply("math", "", observed_pair=eligible,
                                                            prediction=sum(eligible),
                                                            expected=sum(eligible))), "correct")
        self.assertEqual(self.control.observe(ChatReply("math", "", observed_pair=eligible,
                                                            prediction=-1,
                                                            expected=sum(eligible))), "queued")
        self.assertEqual(self.control.status()["pending"], 1)
        self.assertEqual(self.control.observe(ChatReply("math", "", observed_pair=eligible,
                                                            prediction=-1,
                                                            expected=sum(eligible))), "already_queued")

    def test_actual_worker_saves_or_reverts_and_does_not_change_base(self):
        self.control.set_enabled(True)
        first_bytes = self.control.original_file.read_bytes()
        errors = []
        for a, b in sorted(self.control._train_set):
            res = self.engine.reply(f"{a}+{b}")
            if res.prediction != res.expected:
                errors.append(res)
            if len(errors) == 3:
                break
        self.assertGreaterEqual(len(errors), 3, "fixture unexpectedly lacks enough errors")
        for res in errors:
            self.assertIn(self.control.observe(res), ("queued", "already_queued"))
        self.assertIsNotNone(self.control._worker)
        self.control._worker.join(timeout=40)
        state = self.control.status()
        self.assertEqual(state["phase"], "idle")
        self.assertEqual(state["pending"], 0)
        self.assertEqual(self.control.original_file.read_bytes(), first_bytes)
        self.assertIn(state["last_result"]["state"], ("accepted", "rejected"))
        if state["last_result"]["state"] == "accepted":
            self.assertTrue(self.control.adapted_file.exists())
            fresh = SRTChatEngine(self.root)
            self.assertTrue(fresh._load_math())
            self.assertEqual(fresh.math_checkpoint_path, self.control.adapted_file)
            self.control.rollback()
            self.assertEqual(fresh.root / 'checkpoints/srt_m15_toy.pt', self.engine.math_checkpoint_path)
            self.assertFalse(self.control.adapted_file.exists())
        else:
            self.assertFalse(self.control.adapted_file.exists())


if __name__ == "__main__":
    unittest.main()
