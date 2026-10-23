"""Pool-based active acquisition: request verified answers for uncertain toy problems.

This is a deterministic toy oracle, NOT unsupervised discovery or a world model.
"""
import torch
import torch.nn.functional as F
from .tasks import ArithmeticTokenizer


@torch.no_grad()
def choose_experiments(model, candidates, count=4, device="cpu", distributed=False):
    if not candidates:
        return []
    model.eval()
    tokens, _ = ArithmeticTokenizer.batch(candidates, device, distributed=distributed)
    result = model(tokens)
    scores = result["receiver_logits"] if distributed else result["logits"]
    entropy = -(F.softmax(scores, -1) * F.log_softmax(scores, -1)).sum(-1)
    # Disagreement among expert predictions is only a heuristic, not an accuracy estimate.
    experts = result["expert_logits"].argmax(-1)
    disagreement = (experts != experts[:, :1]).float().mean(-1)
    score = entropy + 0.25 * disagreement
    indices = torch.topk(score, min(count, len(candidates))).indices.tolist()
    return [(candidates[i], float(score[i].item())) for i in indices]


def verified_experiments(model, pool, count=4, device="cpu", distributed=False):
    selections = choose_experiments(model, pool, count, device, distributed=distributed)
    # Ask external arithmetic oracle; never train using model guesses as labels.
    return [{"operands": (a, b), "verified_answer": ArithmeticTokenizer.verify(a, b),
             "acquisition_score": score} for (a, b), score in selections]
