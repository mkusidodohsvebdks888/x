"""Guarded opt-in learning for the *toy arithmetic* branch of SRT-M15.

Only verified single-digit addition may enter the queue. The original
checkpoint's test split is never used for training, gating or evaluation;
a separate portion of its original training split is reserved as a guard.
A candidate model learns offline and is swapped in only after checks pass.
Unlabeled language turns never create gradients or pseudo-labels.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import threading
import time
import uuid

import torch

from .learning import OnlineLearner, accuracy
from .tasks import ArithmeticTokenizer


def _utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class AutoLearningController:
    """Opt-in verified learning with a bounded queue and versioned checkpoints.

    The chat server's model_lock protects active inference model swaps. This
    controller's state_lock protects its persistent queue and UI status.
    """

    def __init__(self, engine, model_lock, *, threshold=3, steps=18, lr=0.001,
                 guard_size=12, max_attempts=12, cooldown_seconds=10.0):
        if threshold < 1 or steps < 1 or guard_size < 1:
            raise ValueError("threshold, steps and guard_size must be positive")
        self.engine = engine
        self.model_lock = model_lock
        self.state_lock = threading.RLock()
        self.threshold = threshold
        self.steps = steps
        self.lr = lr
        self.max_attempts = max_attempts
        self.cooldown_seconds = cooldown_seconds
        self.model_root = Path(engine.root)
        self.state_file = self.model_root / "checkpoints" / "autolearn_state.json"
        self.adapted_file = self.model_root / "checkpoints" / "srt_m15_toy_adapted.pt"
        self.previous_file = self.model_root / "checkpoints" / "srt_m15_toy_previous.pt"
        self.original_file = self.model_root / "checkpoints" / "srt_m15_toy.pt"
        self._worker = None
        self._running = False
        self._last_started = 0.0
        self._practice_attempt_limit = None
        self._cancel_training = threading.Event()
        self._train_set = set()
        self._guard_set = set()
        self._test_set = set()
        self._load_original_splits(guard_size)
        self._state = {
            "enabled": False, "pending": [], "attempted": [], "attempts": 0,
            "accepted_updates": 0, "last_result": None, "updated_at": None,
        }
        if self.state_file.is_file():
            try:
                saved = json.loads(self.state_file.read_text(encoding="utf-8"))
                for key in ("enabled", "attempts", "accepted_updates", "last_result", "updated_at"):
                    if key in saved:
                        self._state[key] = saved[key]
                for key in ("pending", "attempted"):
                    entries = saved.get(key, [])
                    self._state[key] = [list(p) for p in entries
                                        if isinstance(p, list) and len(p) == 2 and tuple(p) in self._train_set]
            except (OSError, ValueError, TypeError):
                print("WARNING: unable to read previous learning queue; starting with an empty queue")
        self._state["enabled"] = bool(self._state["enabled"])
        self._state["attempts"] = max(0, int(self._state["attempts"]))
        self._state["accepted_updates"] = max(0, int(self._state["accepted_updates"]))

    def _load_original_splits(self, guard_size):
        if not self.original_file.is_file():
            return
        source = torch.load(self.original_file, map_location="cpu", weights_only=True)
        # Checkpoints produced by train.py record disjoint original train/test pairs.
        raw_train = [tuple(p) for p in source.get("train_pairs", [])]
        raw_test = [tuple(p) for p in source.get("test_pairs", [])]
        if not raw_train or not raw_test:
            return  # Fail closed: do not train if data provenance cannot be verified.
        if set(raw_train) & set(raw_test):
            raise ValueError("Original training and held-out examples overlap")
        # Fixed guard subset comes *only* from the original training split.
        if len(raw_train) <= guard_size + 2:
            return
        self._guard_set = set(raw_train[-guard_size:])
        self._train_set = set(raw_train[:-guard_size])
        self._test_set = set(raw_test)

    def _persist_locked(self):
        self._state["updated_at"] = _utc_now()
        _atomic_json(self.state_file, self._state)

    def status(self):
        with self.state_lock:
            # An eligible queue might have accumulated during the short
            # cooldown; trigger it on the next UI status poll.
            self._maybe_start_locked()
            return {
                "enabled": self._state["enabled"],
                "phase": "training" if self._running else "idle",
                "pending": len(self._state["pending"]),
                "threshold": self.threshold,
                "attempts": self._state["attempts"],
                "max_attempts": self.max_attempts,
                "remaining_attempts": max(0, self.max_attempts - self._state["attempts"]),
                "accepted_updates": self._state["accepted_updates"],
                "last_result": self._state["last_result"],
                "updated_at": self._state["updated_at"],
                "eligible_pairs": len(self._train_set),
                "guard_pairs": len(self._guard_set),
                "reserved_test_pairs": len(self._test_set),
                "adapted_checkpoint": self.adapted_file.is_file(),
                "can_rollback": self.adapted_file.is_file(),
                "practice_attempt_limit": self._practice_attempt_limit,
            }

    def set_enabled(self, enabled: bool):
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be a boolean")
        with self.state_lock:
            if enabled and not self._train_set:
                raise ValueError("Learning disabled: original verified train/test split is missing")
            self._state["enabled"] = enabled
            self._persist_locked()
            if enabled:
                self._maybe_start_locked()
        return self.status()

    def set_practice_attempt_limit(self, limit):
        """Optionally cap all newly scheduled training attempts for a practice run.

        Install the cap *before* enabling learning; remove it only when learning
        has been disabled, so a queued batch cannot bypass the limit.
        """
        with self.state_lock:
            if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int)
                                      or limit < self._state["attempts"]):
                raise ValueError("Invalid practice attempt cap")
            self._practice_attempt_limit = limit

    def cancel_running(self):
        """Request a cooperative stop at the next training step boundary."""
        self._cancel_training.set()

    @staticmethod
    def _check_candidate(baseline, candidate, *, guard_pairs, distributed):
        """Reject nonfinite weights and any changes outside the fast adapters."""
        fast_ids = {id(p) for p in candidate.fast_parameters()}
        base_parameters = dict(baseline.named_parameters())
        base_buffers = dict(baseline.named_buffers())
        for name, parameter in candidate.named_parameters():
            if parameter.is_floating_point() and not bool(torch.isfinite(parameter).all()):
                return False, "nonfinite_parameters"
            if id(parameter) not in fast_ids:
                before = base_parameters[name]
                if not torch.equal(before, parameter):
                    return False, "protected_parameter_changed"
        for name, buffer in candidate.named_buffers():
            if buffer.is_floating_point() and not bool(torch.isfinite(buffer).all()):
                return False, "nonfinite_buffers"
            if not torch.equal(base_buffers[name], buffer):
                return False, "protected_buffer_changed"
        # Also smoke-test the actual hard-routed path. Finite parameters alone
        # do not guarantee finite activations or a valid prediction shape.
        try:
            example_pairs = list(guard_pairs[: min(4, len(guard_pairs))])
            inputs, _ = ArithmeticTokenizer.batch(example_pairs, distributed=distributed)
            candidate.eval()
            with torch.inference_mode():
                out = candidate(inputs, hard_routes=True)
                logits = out["receiver_logits"] if distributed else out["logits"]
                if (tuple(logits.shape) != (len(example_pairs), ArithmeticTokenizer.NUM_CLASSES)
                        or not bool(torch.isfinite(logits).all())):
                    return False, "invalid_model_outputs"
        except (ValueError, RuntimeError, KeyError, TypeError):
            return False, "invalid_model_outputs"
        return True, "ok"

    def observe(self, reply):
        """Observe a chat result; only *wrong*, independently verified math enters queue."""
        with self.state_lock:
            if not self._state["enabled"]:
                return "disabled"
            if reply.mode != "math" or reply.observed_pair is None:
                return "not_trainable"
            pair = tuple(reply.observed_pair)
            if pair in self._test_set:
                return "reserved_test"
            if pair in self._guard_set:
                return "guard_only"
            if pair not in self._train_set:
                return "unknown_split"
            if reply.prediction == reply.expected:
                return "correct"
            if reply.expected != ArithmeticTokenizer.verify(*pair):
                return "unverified"
            if list(pair) in self._state["pending"]:
                return "already_queued"
            if list(pair) in self._state["attempted"]:
                return "already_attempted"
            self._state["pending"].append(list(pair))
            self._persist_locked()
            self._maybe_start_locked()
            return "queued"

    def _maybe_start_locked(self, force=False):
        if self._running or not self._state["enabled"] or not self._train_set:
            return False
        if not self._state["pending"]:
            return False
        if not force and len(self._state["pending"]) < self.threshold:
            return False
        if self._state["attempts"] >= self.max_attempts:
            return False
        if (self._practice_attempt_limit is not None and
                self._state["attempts"] >= self._practice_attempt_limit):
            return False
        if time.monotonic() - self._last_started < self.cooldown_seconds:
            return False
        batch = [tuple(pair) for pair in self._state["pending"]]
        self._running = True
        self._cancel_training.clear()
        self._last_started = time.monotonic()
        self._state["attempts"] += 1
        self._state["last_result"] = {"state": "training", "at": _utc_now()}
        self._persist_locked()
        self._worker = threading.Thread(target=self._train_worker, args=(batch,), name="srt-learning", daemon=True)
        self._worker.start()
        return True

    def start_now(self):
        with self.state_lock:
            if not self._state["enabled"]:
                raise ValueError("Enable autonomous learning first")
            if self._running:
                raise ValueError("A candidate is already training")
            if not self._state["pending"]:
                raise ValueError("No verified training examples are queued")
            if not self._maybe_start_locked(force=True):
                raise ValueError("Learning attempt limit or cooldown reached")
        return self.status()

    def _train_worker(self, batch):
        report = None
        accepted = False
        try:
            with self.model_lock:
                if not self.engine._load_math():
                    raise RuntimeError("Math checkpoint unavailable")
                baseline = copy.deepcopy(self.engine.math_model).cpu()
                candidate = copy.deepcopy(baseline)
                distributed = self.engine.math_distributed
                # Read metadata from the active checkpoint, not from user input.
                source = torch.load(self.engine.math_checkpoint_path, map_location="cpu", weights_only=True)

            guard_pairs = sorted(self._guard_set)
            # The live chat uses hard routing, so validate on the *same* route.
            # The internal learner's smooth-route guard is retained as an
            # additional check, not substituted for live-path measurements.
            baseline_train = accuracy(candidate, batch, distributed=distributed, hard_routes=True)
            baseline_guard = accuracy(candidate, guard_pairs, distributed=distributed, hard_routes=True)
            learner = OnlineLearner(candidate, device="cpu", distributed=distributed)
            result = learner.adapt(batch, guard_pairs, steps=self.steps, lr=self.lr,
                                   max_guard_drop=0.0, cancel_check=self._cancel_training.is_set)
            candidate_train = accuracy(candidate, batch, distributed=distributed, hard_routes=True)
            candidate_guard = accuracy(candidate, guard_pairs, distributed=distributed, hard_routes=True)
            basic_ok, basic_reason = self._check_candidate(
                baseline, candidate, guard_pairs=guard_pairs, distributed=distributed)
            cancelled = bool(result.get("cancelled") or self._cancel_training.is_set())
            accepted = bool(not cancelled and basic_ok and result["accepted"] and
                            candidate_train > baseline_train and
                            candidate_guard >= baseline_guard)
            report = {"state": "cancelled" if cancelled else "accepted" if accepted else "rejected",
                      "at": _utc_now(),
                      "train_before": round(baseline_train, 4),
                      "train_candidate": round(candidate_train, 4),
                      "guard_before": round(baseline_guard, 4),
                      "guard_candidate": round(candidate_guard, 4),
                      "basic_check": basic_reason,
                      "examples": len(batch), "updated_parameters": result["updated_parameters"]}
            if accepted and not self._cancel_training.is_set():
                source["model"] = {k: v.detach().cpu() for k, v in candidate.state_dict().items()}
                source["online_learning"] = {"updated_at": _utc_now(), "num_examples": len(batch),
                                             "fast_parameters_only": True}
                temp = self.adapted_file.with_suffix("." + uuid.uuid4().hex + ".tmp")
                try:
                    torch.save(source, temp)
                    # Inference sees old weights until all disk operations succeed.
                    with self.model_lock:
                        if self._cancel_training.is_set():
                            accepted = False
                            report["state"] = "cancelled"
                        else:
                            if self.adapted_file.is_file():
                                backup_tmp = self.previous_file.with_suffix("." + uuid.uuid4().hex + ".tmp")
                                try:
                                    shutil.copy2(self.adapted_file, backup_tmp)
                                    os.replace(backup_tmp, self.previous_file)
                                finally:
                                    backup_tmp.unlink(missing_ok=True)
                            os.replace(temp, self.adapted_file)
                            self.engine.math_model = candidate.eval()
                            self.engine.math_distributed = distributed
                            self.engine.math_checkpoint_path = self.adapted_file
                finally:
                    temp.unlink(missing_ok=True)
        except Exception as exc:
            accepted = False
            report = {"state": "error", "at": _utc_now(), "detail": type(exc).__name__,
                      "examples": len(batch)}
            print("Autonomous learning error:", repr(exc))
        finally:
            with self.state_lock:
                # Pause/cancellation must not consume verified errors; they
                # remain available when the owner explicitly starts again.
                if report and report["state"] == "cancelled":
                    self._state["last_result"] = report
                    self._running = False
                    self._persist_locked()
                    return
                for pair in batch:
                    if list(pair) in self._state["pending"]:
                        self._state["pending"].remove(list(pair))
                    if list(pair) not in self._state["attempted"]:
                        self._state["attempted"].append(list(pair))
                self._state["last_result"] = report
                if accepted:
                    self._state["accepted_updates"] += 1
                    self._state["attempted"] = []  # A newer model may benefit from new errors.
                self._running = False
                self._persist_locked()

    def rollback(self):
        """Revert the latest accepted update; baseline is never overwritten."""
        with self.state_lock:
            if self._running:
                raise ValueError("Wait until the current candidate finishes")
            if not self.adapted_file.is_file():
                raise ValueError("There is no adapted checkpoint to revert")
            with self.model_lock:
                # Move the last saved adapted checkpoint out of use, or restore baseline.
                self.adapted_file.unlink()
                if self.previous_file.is_file():
                    os.replace(self.previous_file, self.adapted_file)
                self.engine.math_model = None
                self.engine.math_checkpoint_path = None
                self.engine._load_math()
            self._state["last_result"] = {"state": "rolled_back", "at": _utc_now()}
            self._state["pending"] = []
            self._state["attempted"] = []
            self._persist_locked()
        return self.status()
