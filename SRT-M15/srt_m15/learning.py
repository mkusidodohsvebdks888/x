"""Verified, guarded online adaptation: only the three fast adapters are updated."""
import copy
import torch
import torch.nn.functional as F
from .tasks import ArithmeticTokenizer


@torch.no_grad()
def accuracy(model, pairs, device="cpu", batch_size=32, distributed=False, disable_communication=False, hard_routes=False):
    if not pairs:
        return 0.0
    model.eval()
    num = 0
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start:start + batch_size]
        tokens, target = ArithmeticTokenizer.batch(batch, device, distributed=distributed)
        out = model(tokens, disable_communication=disable_communication, hard_routes=hard_routes)
        scores = out["receiver_logits"] if distributed else out["logits"]
        num += int((scores.argmax(-1) == target).sum())
    return num / len(pairs)


class OnlineLearner:
    """Apply verified examples and roll back when guard-set performance deteriorates."""
    def __init__(self, model, device="cpu", distributed=False):
        self.model = model
        self.device = device
        self.distributed = distributed

    def adapt(self, verified_pairs, guard_pairs, *, steps=12, lr=0.002,
              max_guard_drop=0.05, cancel_check=None):
        if not verified_pairs or not guard_pairs:
            raise ValueError("need labeled examples and independent guard examples")
        if set(verified_pairs) & set(guard_pairs):
            raise ValueError("guard data must not overlap training data")
        model = self.model
        fast = list(model.fast_parameters())
        old = [p.detach().clone() for p in fast]
        before_guard = accuracy(model, guard_pairs, self.device, distributed=self.distributed)
        before_train = accuracy(model, verified_pairs, self.device, distributed=self.distributed)
        opt = torch.optim.Adam(fast, lr=lr)
        x, y = ArithmeticTokenizer.batch(verified_pairs, self.device, distributed=self.distributed)
        model.train()
        cancelled = False
        for _ in range(steps):
            if cancel_check is not None and cancel_check():
                cancelled = True
                break
            results = model(x)
            logits = results["receiver_logits"] if self.distributed else results["logits"]
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fast, 1.0)
            opt.step()
        after_guard = accuracy(model, guard_pairs, self.device, distributed=self.distributed)
        after_train = accuracy(model, verified_pairs, self.device, distributed=self.distributed)
        # A prototype guardrail; independent holdout tests still needed.
        cancelled = cancelled or (cancel_check is not None and cancel_check())
        accepted = (not cancelled and after_guard >= before_guard - max_guard_drop
                    and after_train >= before_train)
        if not accepted:
            with torch.no_grad():
                for p, saved in zip(fast, old):
                    p.copy_(saved)
        model.eval()
        return {"accepted": accepted, "cancelled": cancelled,
                "guard_before": before_guard, "guard_candidate": after_guard,
                "train_before": before_train, "train_candidate": after_train,
                "updated_parameters": sum(p.numel() for p in fast)}
