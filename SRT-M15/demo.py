"""Demonstrate prediction, routing, external memory, exploration, guarded learning."""
import argparse
from pathlib import Path
import torch
from srt_m15 import ArithmeticTokenizer, ModelConfig, SRTM15
from srt_m15.storage import ExternalMemory
from srt_m15.exploration import verified_experiments
from srt_m15.learning import OnlineLearner, accuracy


def run(checkpoint="checkpoints/srt_m15_toy.pt"):
    torch.set_num_threads(min(4, torch.get_num_threads()))
    if not Path(checkpoint).exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint}. Run train.py first.")
    data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = SRTM15(ModelConfig(**data["config"]))
    model.load_state_dict(data["model"])
    model.eval()
    distributed = data.get("task") == "distributed"
    print("=== SRT-M15 inference (single-digit addition only) ===")
    for a, b in [(1, 2), (4, 5), (7, 8)]:
        inputs, _ = ArithmeticTokenizer.batch([(a, b)], distributed=distributed)
        with torch.no_grad():
            out = model(inputs, hard_routes=True)
        route = (out["edge_probs"][0, -1] > 0.5).int().tolist()
        scores = out["receiver_logits"] if distributed else out["logits"]
        print(f"{a}+{b}: model={scores.argmax(-1).item()}  verified={a+b}  route={route}")

    print("\n=== External memory version history ===")
    store = ExternalMemory("checkpoints/knowledge.json")
    store.write("A", 5, source="simulation", verified=True)
    store.write("A", 12, source="simulation", verified=True)
    print("latest A:", store.read("A")["value"], "history:", len(store.history("A")))

    print("\n=== Active experiment selection and safe online update ===")
    train_pairs, heldout = data["train_pairs"], data["test_pairs"]
    # We split held-out pairs into acquisition pool and disjoint guard set.
    pool = heldout[:10]
    guard = heldout[10:]
    chosen = verified_experiments(model, pool, count=4, distributed=distributed)
    training_examples = [tuple(entry["operands"]) for entry in chosen]
    print("selected:", chosen)
    if not guard:
        print("Need non-empty guard set; online stage skipped")
        return
    result = OnlineLearner(model, distributed=distributed).adapt(training_examples, guard, steps=20, lr=0.001,
                                        max_guard_drop=0.10)
    print("online update:", result)
    print("held-out communication ablation:",
          {"enabled": accuracy(model, guard, distributed=distributed),
           "disabled": accuracy(model, guard, distributed=distributed, disable_communication=True)})
    print("NOTE: validation on a small guard set is noisy; this is not a proof of generalization.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/srt_m15_toy.pt")
    args = parser.parse_args()
    run(args.checkpoint)
