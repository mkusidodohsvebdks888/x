import tempfile
import unittest
from pathlib import Path
import torch
import torch.nn.functional as F
from srt_m15 import ModelConfig, SRTM15, ArithmeticTokenizer
from srt_m15.exploration import verified_experiments
from srt_m15.storage import ExternalMemory
from srt_m15.learning import OnlineLearner


class SRTM15Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(5)
        torch.set_num_threads(2)
        cls.cfg = ModelConfig(vocab_size=14, num_classes=19, dim=32,
                              reasoning_steps=2, adapter_dim=8)

    def test_distributed_fragments_force_communication_path(self):
        model = SRTM15(self.cfg)
        fragments, target = ArithmeticTokenizer.batch([(2, 3), (4, 5)], distributed=True)
        self.assertEqual(tuple(fragments.shape), (2, 3, 5))
        self.assertEqual(fragments[0, 2].tolist(), fragments[1, 2].tolist())
        on = model(fragments)
        off = model(fragments, disable_communication=True)
        self.assertEqual(tuple(on["receiver_logits"].shape), (2, 19))
        self.assertTrue(torch.allclose(off["receiver_logits"][0], off["receiver_logits"][1], atol=1e-5))

    def test_shapes_backward_router_gradients(self):
        model = SRTM15(self.cfg)
        x, y = ArithmeticTokenizer.batch([(2, 3), (4, 1)])
        out = model(x)
        self.assertEqual(tuple(out["logits"].shape), (2, 19))
        self.assertEqual(tuple(out["edge_probs"].shape), (2, 2, 3, 3))
        self.assertEqual(tuple(out["learning_relation_scores"].shape), (2, 3, 3))
        self.assertEqual(tuple(out["memory"].slots.shape), (2, 3, 3, 32))
        from srt_m15.model import learning_relation_loss
        (F.cross_entropy(out["logits"], y) +
         learning_relation_loss(out["learning_relation_logits"], out["expert_logits"], y)).backward()
        self.assertIsNotNone(model.backbone.embed.weight.grad)
        self.assertIsNotNone(model.communication.to_query.weight.grad)
        self.assertIsNotNone(model.learning_relation.student.weight.grad)
        self.assertTrue(torch.isfinite(model.communication.to_query.weight.grad).all())

    def test_hard_route_can_refuse_communication(self):
        model = SRTM15(self.cfg)
        with torch.no_grad():
            model.communication.edge_bias.fill_(-1000)
        x, _ = ArithmeticTokenizer.batch([(3, 6)])
        out = model(x, hard_routes=True)
        self.assertEqual(int((out["edge_probs"] > 0.5).sum().item()), 0)
        self.assertTrue(torch.isfinite(out["logits"]).all())

    def test_memory_updates_and_can_persist(self):
        model = SRTM15(self.cfg)
        x, _ = ArithmeticTokenizer.batch([(1, 9)])
        before = model.init_memory(1)
        after = model(x, memory=before)["memory"]
        self.assertGreater(float(after.slots.abs().sum().item()), 0.0)
        x2, _ = ArithmeticTokenizer.batch([(9, 1)])
        again = model(x2, memory=after.detach())["memory"]
        self.assertFalse(torch.allclose(after.slots, again.slots))

    def test_external_memory_versions(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "memory.json"
            storage = ExternalMemory(path)
            storage.write("foo", "old", source="user", verified=True)
            storage.write("foo", "rumor", source="unknown", verified=False)
            storage.write("foo", "new", source="user", verified=True)
            reopened = ExternalMemory(path)
            self.assertEqual(reopened.read("foo")["value"], "new")
            self.assertEqual(len(reopened.history("foo")), 3)

    def test_language_model_causality_and_generation(self):
        from srt_m15.language import SRTM15LanguageModel, CharVocabulary
        cfg = ModelConfig(vocab_size=10, num_classes=10, dim=32,
                          reasoning_steps=1, max_length=8)
        lm = SRTM15LanguageModel(cfg)
        a = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        b = torch.tensor([[1, 2, 3, 9]], dtype=torch.long)
        lm.eval()
        with torch.no_grad():
            logits_a = lm(a)["logits"]
            logits_b = lm(b)["logits"]
            output = lm.generate(a[:, :2], max_new_tokens=3)
        self.assertEqual(tuple(logits_a.shape), (1, 4, 10))
        self.assertTrue(torch.allclose(logits_a[:, :3], logits_b[:, :3], atol=1e-5))
        self.assertEqual(output.shape[1], 5)
        vocab = CharVocabulary.from_text("你好世界")
        self.assertEqual(vocab.decode(vocab.encode("你好")), "你好")

    def test_active_queries_and_online_guard(self):
        model = SRTM15(self.cfg)
        pool = [(1, 2), (3, 4), (4, 5), (7, 9)]
        selected = verified_experiments(model, pool, count=2)
        self.assertEqual(len(selected), 2)
        self.assertTrue(all(v["verified_answer"] == sum(v["operands"]) for v in selected))
        learn = OnlineLearner(model)
        before_backbone = model.backbone.embed.weight.detach().clone()
        result = learn.adapt([tuple(v["operands"]) for v in selected], [(8, 8), (0, 0)],
                             steps=3, lr=0.001, max_guard_drop=1.0)
        self.assertIn("accepted", result)
        self.assertTrue(torch.equal(before_backbone, model.backbone.embed.weight))


if __name__ == "__main__":
    unittest.main()
