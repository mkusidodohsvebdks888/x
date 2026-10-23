"""Opt-in, resource-limited, offline single-digit addition self-practice.

The only practice candidates come from the training-eligible part of the
original math checkpoint. The independent guard and test splits are excluded
from question generation and from gradient updates. Only a deterministic
arithmetic verifier supplies labels. There are no network calls.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import math
import os
import random
import threading
import time
import uuid

from .tasks import ArithmeticTokenizer


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


@dataclass(frozen=True)
class PracticeConfig:
    """All limits apply to EACH click of 'Start practice'."""

    interval_seconds: float = 1.0
    questions_per_round: int = 4
    max_training_attempts: int = 2
    max_runtime_seconds: int = 180
    max_questions: int = 100

    @classmethod
    def from_request(cls, data):
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ValueError("Practice settings must be an object")
        allowed = set(cls.__dataclass_fields__)
        if set(data) - allowed:
            raise ValueError("Unknown practice setting")
        numeric_bounds = {
            "interval_seconds": (0.2, 60.0),
            "questions_per_round": (1, 12),
            "max_training_attempts": (1, 8),
            "max_runtime_seconds": (1, 3600),
            "max_questions": (1, 500),
        }
        values = {}
        for name, (lo, hi) in numeric_bounds.items():
            default = getattr(cls(), name)
            val = data.get(name, default)
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise ValueError(f"{name} must be numeric")
            if not math.isfinite(val) or not lo <= val <= hi:
                raise ValueError(f"{name} must be between {lo} and {hi}")
            if name != "interval_seconds" and (not isinstance(val, int) or isinstance(val, bool)):
                raise ValueError(f"{name} must be an integer")
            values[name] = val
        return cls(**values)


class AutonomousPracticeRunner:
    """Practice worker separated from the HTTP chat thread.

    Practice never touches the held-out or guard sets. It calls the existing
    verified-error learner and cannot update parameters directly.
    """

    def __init__(self, engine, model_lock, learning, *, seed=None):
        self.engine = engine
        self.model_lock = model_lock
        self.learning = learning
        self.state_lock = threading.RLock()
        self.file_path = Path(engine.root) / "checkpoints" / "practice_state.json"
        # Guard/test are excluded before ever being considered for sampling.
        self._pool = tuple(sorted(learning._train_set))
        if (set(self._pool) & learning._guard_set or
                set(self._pool) & learning._test_set or
                any(not (0 <= a <= 9 and 0 <= b <= 9) for a, b in self._pool)):
            raise ValueError("Overlapping or invalid practice dataset")
        self._random = random.Random(seed)
        self._deck = []
        self._stop = threading.Event()
        self._worker = None
        self._running = False
        self._started_at_monotonic = None
        self._attempt_baseline = 0
        self._session_questions = 0
        self._session_rounds = 0
        self._config = PracticeConfig()
        self._state = {
            "total_questions": 0,
            "total_errors": 0,
            "total_rounds": 0,
            "total_training_sessions": 0,
            "last_problem": None,
            "recent_problems": [],
            "last_stop_reason": "not_started",
            "last_started_at": None,
            "last_updated_at": None,
        }
        if self.file_path.is_file():
            try:
                data = json.loads(self.file_path.read_text(encoding="utf-8"))
                for name in ("total_questions", "total_errors", "total_rounds", "total_training_sessions"):
                    val = data.get(name)
                    if type(val) is int and 0 <= val <= 10000000:
                        self._state[name] = val
                for name in ("last_problem", "last_started_at", "last_updated_at"):
                    if name in data:
                        self._state[name] = data[name]
                history = data.get("recent_problems")
                if isinstance(history, list):
                    self._state["recent_problems"] = history[-8:]
                previous_reason = data.get("last_stop_reason")
                self._state["last_stop_reason"] = (previous_reason if previous_reason
                                                  else "interrupted_by_restart")
                # Restart never silently resumes autonomous computation.
                if self._state["last_started_at"]:
                    self.learning.set_enabled(False)
            except (ValueError, TypeError, OSError):
                self._state["last_stop_reason"] = "invalid_saved_state"

    def _persist_locked(self):
        self._state["last_updated_at"] = utc_now()
        atomic_write_json(self.file_path, self._state)

    def _next_pair(self):
        if not self._deck:
            self._deck = list(self._pool)
            self._random.shuffle(self._deck)
        return self._deck.pop()

    def _run_round(self, count):
        """Return number of actually completed questions; callable in tests."""
        done = 0
        for _ in range(count):
            if self._stop.is_set():
                break
            if (self._started_at_monotonic is not None and
                    time.monotonic() - self._started_at_monotonic >= self._config.max_runtime_seconds):
                break
            pair = self._next_pair()
            if pair not in self.learning._train_set:
                raise RuntimeError("Practice tried to select a reserved example")
            a, b = pair
            with self.model_lock:
                reply = self.engine.reply(f"{a}+{b}")
            if reply.mode != "math" or reply.observed_pair != pair:
                raise RuntimeError("Math model did not return a verified arithmetic result")
            expected = ArithmeticTokenizer.verify(a, b)
            if reply.expected != expected or reply.prediction is None:
                raise RuntimeError("Arithmetic verification mismatch")
            event = self.learning.observe(reply)
            if event not in ("correct", "queued", "already_queued", "already_attempted"):
                raise RuntimeError("Learning queue refused an eligible example: " + event)
            wrong = reply.prediction != expected
            row = {"operands": [a, b], "prediction": reply.prediction,
                   "verified_answer": expected, "wrong": wrong,
                   "learning_event": event, "at": utc_now()}
            with self.state_lock:
                self._state["total_questions"] += 1
                self._state["total_errors"] += int(wrong)
                self._session_questions += 1
                self._state["last_problem"] = row
                self._state["recent_problems"] = (self._state["recent_problems"] + [row])[-8:]
                self._persist_locked()
            done += 1
            if self.learning.status()["phase"] == "training":
                break  # Allow candidate training to run without practice CPU contention.
        if done:
            with self.state_lock:
                self._state["total_rounds"] += 1
                self._session_rounds += 1
                self._persist_locked()
        return done

    def start(self, settings=None):
        config = PracticeConfig.from_request(settings)
        with self.state_lock:
            if self._running:
                raise ValueError("Autonomous practice is already running")
            if not self._pool:
                raise ValueError("No verified training pool; the original checkpoint split is missing")
            learning_status = self.learning.status()
            if learning_status["phase"] == "training":
                raise ValueError("A model candidate is training; start practice after it finishes")
            if learning_status["remaining_attempts"] == 0:
                raise ValueError("Total learning attempt limit reached. No new practice run can train")
            with self.model_lock:
                if not self.engine._load_math():
                    raise ValueError("Original arithmetic checkpoint is missing")
            self._config = config
            self._attempt_baseline = learning_status["attempts"]
            self.learning.set_practice_attempt_limit(min(
                learning_status["max_attempts"],
                self._attempt_baseline + config.max_training_attempts))
            try:
                self.learning.set_enabled(True)
            except Exception:
                self.learning.set_practice_attempt_limit(None)
                raise
            self._running = True
            self._stop = threading.Event()
            self._started_at_monotonic = time.monotonic()
            self._session_questions = 0
            self._session_rounds = 0
            self._deck = []
            self._state["last_started_at"] = utc_now()
            self._state["last_stop_reason"] = None
            self._state["total_training_sessions"] += 1
            self._persist_locked()
            self._worker = threading.Thread(target=self._run, name="srt-autonomous-practice", daemon=True)
            self._worker.start()
        return self.status()

    def pause(self):
        with self.state_lock:
            if self._running:
                self._state["last_stop_reason"] = "manual_pause"
                self._stop.set()
                self._persist_locked()
        # Disabling before releasing the run budget prevents new candidate jobs.
        self.learning.set_enabled(False)
        self.learning.cancel_running()
        worker = self._worker
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=2)
        return self.status()

    def _run(self):
        reason = "stopped"
        cancel_candidate = False
        try:
            while not self._stop.is_set():
                elapsed = time.monotonic() - self._started_at_monotonic
                if elapsed >= self._config.max_runtime_seconds:
                    reason = "max_runtime"
                    cancel_candidate = True
                    break
                if self._session_questions >= self._config.max_questions:
                    reason = "max_questions"
                    break
                status = self.learning.status()
                if not status["enabled"]:
                    reason = "learning_disabled"
                    break
                if status["attempts"] - self._attempt_baseline >= self._config.max_training_attempts:
                    reason = "max_training_attempts"
                    break
                if status["remaining_attempts"] <= 0:
                    reason = "global_training_limit"
                    break
                if status["phase"] == "training":
                    self._stop.wait(min(self._config.interval_seconds, 0.25))
                    continue
                count = min(self._config.questions_per_round,
                            self._config.max_questions - self._session_questions)
                self._run_round(count)
                # Never sleep past the configured wall-clock budget.
                remaining = self._config.max_runtime_seconds - (time.monotonic() - self._started_at_monotonic)
                if remaining <= 0:
                    continue
                self._stop.wait(min(self._config.interval_seconds, remaining))
            if self._stop.is_set():
                reason = "manual_pause"
                cancel_candidate = True
        except Exception as exc:
            reason = "error_" + type(exc).__name__
            cancel_candidate = True
            print("Practice error:", repr(exc))
        finally:
            self.learning.set_enabled(False)
            if cancel_candidate:
                self.learning.cancel_running()
            # Do not drop the cap until learning is disabled (prevent runaway queue).
            self.learning.set_practice_attempt_limit(None)
            with self.state_lock:
                self._state["last_stop_reason"] = reason
                self._running = False
                self._persist_locked()

    def status(self):
        with self.state_lock:
            data = dict(self._state)
            running = self._running
            elapsed = (time.monotonic() - self._started_at_monotonic
                       if running and self._started_at_monotonic is not None else None)
            current = asdict(self._config)
            questions_session = self._session_questions
            rounds_session = self._session_rounds
            baseline_attempts = self._attempt_baseline
        learning = self.learning.status()
        return {
            "running": running,
            "phase": ("waiting_training" if learning["phase"] == "training" else "practicing") if running else "stopped",
            "last_stop_reason": data["last_stop_reason"],
            "configuration": current,
            "total_questions": data["total_questions"],
            "total_errors": data["total_errors"],
            "total_rounds": data["total_rounds"],
            "total_training_sessions": data["total_training_sessions"],
            "session_questions": questions_session,
            "session_rounds": rounds_session,
            "session_training_attempts": max(0, learning["attempts"] - baseline_attempts) if running else 0,
            "elapsed_seconds": round(elapsed, 1) if elapsed is not None else None,
            "last_problem": data["last_problem"],
            "recent_problems": data["recent_problems"],
            "training_attempts": learning["attempts"],
            "accepted_updates": learning["accepted_updates"],
            "pending": learning["pending"],
            "threshold": learning["threshold"],
            "last_training_result": learning["last_result"],
            "eligible_pairs": len(self._pool),
        }
