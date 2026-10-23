"""Tests for staged training and exact checkpoint continuation."""
import argparse
import hashlib
from pathlib import Path
import tempfile
import unittest

import torch

from train_bilingual_stages import can_promote, run


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples" / "bilingual_toy_sft.jsonl"


def args_for(output, *, stages=1, resume=""):
    return argparse.Namespace(
        data=str(DATA), allow_demo_data=True, output=str(output),
        resume=str(resume), stages=stages, steps_per_stage=2,
        batch_size=2, context=16, dim=16, heads=4, experts=2,
        reasoning_steps=1, lr=0.001, dev_fraction=0.2,
        seed=42, device="cpu",
    )


class StagedTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_language_aware_promotion(self):
        old = {"zh": {"token_loss": 3.0}, "en": {"token_loss": 3.0}}
        good = {"zh": {"token_loss": 2.8}, "en": {"token_loss": 3.01}}
        bad = {"zh": {"token_loss": 2.2}, "en": {"token_loss": 3.6}}
        self.assertTrue(can_promote(good, old))
        self.assertFalse(can_promote(bad, old))

    def test_staged_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as home:
            root = Path(home)
            direct = root / "direct"
            staged = root / "staged"
            full = run(args_for(direct, stages=2))
            first = run(args_for(staged, stages=1))
            self.assertEqual(first["total_steps"], 2)
            self.assertTrue((staged / "latest.pt").is_file())
            resumed = run(args_for(staged, stages=1, resume=staged / "latest.pt"))
            self.assertEqual(full["total_steps"], resumed["total_steps"])
            self.assertEqual(resumed["total_steps"], 4)
            self.assertEqual(resumed["stage"], 2)
            self.assertEqual(len(resumed["history"]), 2)
            a = torch.load(direct / "latest.pt", weights_only=True)
            b = torch.load(staged / "latest.pt", weights_only=True)
            self.assertEqual(a["optimizer"]["param_groups"], b["optimizer"]["param_groups"])
            for name in a["model"]:
                self.assertTrue(torch.equal(a["model"][name], b["model"][name]),
                                f"Mismatch in {name}")
            self.assertEqual(full["history"], resumed["history"])

    def test_cycle_sampler_exact_resume_and_coverage(self):
        with tempfile.TemporaryDirectory() as home:
            root = Path(home)
            continuous = args_for(root / "continuous", stages=3)
            continuous.sampling = "cycle"
            result1 = run(continuous)
            self.assertEqual(result1["sampling"], "cycle")
            self.assertGreater(result1["cumulative_unique_window_coverage"]["zh"], 0)
            first = args_for(root / "staged", stages=1)
            first.sampling = "cycle"
            run(first)
            resume = args_for(root / "staged", stages=2,
                              resume=root / "staged/latest.pt")
            resume.sampling = "cycle"
            result2 = run(resume)
            self.assertEqual(result2["total_steps"], 6)
            self.assertEqual(result1["history"], result2["history"])
            self.assertEqual(result1["cumulative_unique_window_coverage"],
                             result2["cumulative_unique_window_coverage"])
            for language in ("zh", "en"):
                self.assertLessEqual(result1["cumulative_unique_window_coverage"][language],
                                     result1["window_counts"][language])
            a = torch.load(root / "continuous/latest.pt", weights_only=True)
            b = torch.load(root / "staged/latest.pt", weights_only=True)
            for key in a["model"]:
                self.assertTrue(torch.equal(a["model"][key], b["model"][key]), key)

    def test_cross_run_artifact_carries_forward_best(self):
        with tempfile.TemporaryDirectory() as home:
            root = Path(home)
            previous = root / "previous"
            run(args_for(previous, stages=1))
            latest = torch.load(previous / "latest.pt", weights_only=True)
            # On fixed data and seed this batch should improve the dev score;
            # if it does not, there is legitimately no promoted best yet.
            next_output = root / "next_workflow_run"
            report = run(args_for(next_output, stages=1,
                                  resume=previous / "latest.pt"))
            self.assertEqual(report["stage"], 2)
            self.assertEqual(report["total_steps"], 4)
            self.assertTrue((next_output / "latest.pt").is_file())
            if latest["best_stage"] > 0:
                self.assertTrue((next_output / "best.pt").is_file())
                best = torch.load(next_output / "best.pt", weights_only=True)
                self.assertGreater(best["stage"], 0)

    def test_wrong_data_and_hyperparameters_cannot_resume(self):
        with tempfile.TemporaryDirectory() as home:
            path = Path(home)
            run(args_for(path / "one"))
            resume = path / "one" / "latest.pt"
            cfg = args_for(path / "two", resume=resume)
            cfg.lr = .005
            with self.assertRaisesRegex(ValueError, "mismatch"):
                run(cfg)
            cfg = args_for(path / "two", resume=resume)
            cfg.seed = 13
            with self.assertRaisesRegex(ValueError, "mismatch"):
                run(cfg)

    def test_training_limits(self):
        with tempfile.TemporaryDirectory() as home:
            cfg = args_for(home)
            cfg.stages = 0
            with self.assertRaises(ValueError):
                run(cfg)
            cfg.stages = 1
            cfg.batch_size = 1
            with self.assertRaises(ValueError):
                run(cfg)


if __name__ == "__main__":
    unittest.main()
