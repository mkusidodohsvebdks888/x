"""SRT-M15 foundational architecture. No pretrained weights or Transformer library.

Components: hand-written causal attention, three independent recurrent experts,
receiver/sender conditioned sparse messaging, three-timescale differentiable memory,
and a task prediction head. This is a toy research model, not a chat model.
"""
from dataclasses import dataclass
from typing import Optional
import math

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 16
    max_length: int = 8
    num_classes: int = 19
    dim: int = 48
    heads: int = 4
    layers: int = 1
    num_experts: int = 3
    reasoning_steps: int = 3
    message_groups: int = 4
    adapter_dim: int = 12

    def __post_init__(self):
        if self.dim % self.heads or self.dim % self.message_groups:
            raise ValueError("dim must be divisible by heads and message_groups")
        if self.num_experts < 2 or self.reasoning_steps < 1:
            raise ValueError("at least two experts and one reasoning step required")


@dataclass
class MemoryState:
    # [batch, experts, timescales=3, dim]
    slots: torch.Tensor

    def detach(self):
        return MemoryState(self.slots.detach())


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        batch, seq, dim = x.shape
        head_dim = dim // self.heads
        q, k, v = self.qkv(x).reshape(batch, seq, 3, self.heads, head_dim).permute(2, 0, 3, 1, 4)
        score = (q @ k.transpose(-1, -2)) / math.sqrt(head_dim)
        causal = torch.ones(seq, seq, device=x.device, dtype=torch.bool).tril()
        p = score.masked_fill(~causal, torch.finfo(score.dtype).min).softmax(dim=-1)
        value = (p @ v).transpose(1, 2).contiguous().reshape(batch, seq, dim)
        return self.out(value)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, heads)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 3), nn.GELU(), nn.Linear(dim * 3, dim))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ff(self.ln2(x))


class SharedBackbone(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.pos = nn.Embedding(cfg.max_length, cfg.dim)
        self.blocks = nn.ModuleList([TransformerBlock(cfg.dim, cfg.heads) for _ in range(cfg.layers)])
        self.norm = nn.LayerNorm(cfg.dim)

    def forward_hidden(self, ids):
        b, t = ids.shape
        if t > self.cfg.max_length:
            raise ValueError(f"sequence length {t} exceeds {self.cfg.max_length}")
        x = self.embed(ids) + self.pos(torch.arange(t, device=ids.device))[None, :, :]
        for block in self.blocks:
            x = block(x)
        # The final token's causal receptive field includes all preceding tokens.
        return self.norm(x)

    def forward(self, ids):
        return self.forward_hidden(ids)[:, -1]


class ReasoningExpert(nn.Module):
    """Independent GRU recurrent reasoning state + small trainable fast adapter."""
    def __init__(self, dim: int, adapter_dim: int):
        super().__init__()
        self.recurrence = nn.GRUCell(dim, dim)
        self.layer_norm = nn.LayerNorm(dim)
        self.fast_down = nn.Linear(dim, adapter_dim, bias=False)
        self.fast_up = nn.Linear(adapter_dim, dim, bias=False)
        nn.init.normal_(self.fast_down.weight, std=0.02)
        nn.init.normal_(self.fast_up.weight, std=0.02)

    def forward(self, base, previous, memory):
        updated = self.recurrence(base + memory, previous)
        adjustment = self.fast_up(F.gelu(self.fast_down(updated)))
        return self.layer_norm(updated + adjustment)


class DynamicCommunication(nn.Module):
    """Receiver- and sender-aware gates; each directed edge can be disabled.

    Training uses continuous gates for gradients; `hard=True` uses actual binary
    link and message-group selection. A zero row means no incoming communication.
    """
    def __init__(self, dim: int, experts: int, groups: int):
        super().__init__()
        self.experts = experts
        self.groups = groups
        self.to_query = nn.Linear(dim, dim)
        self.to_key = nn.Linear(dim, dim)
        self.sender_groups = nn.Linear(dim, groups)
        self.receiver_groups = nn.Linear(dim, groups)
        self.to_message = nn.Linear(dim, dim)
        self.merger = nn.Linear(dim, dim)
        self.recv_gate = nn.Linear(2 * dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.edge_bias = nn.Parameter(torch.tensor(1.4))
        self.group_bias = nn.Parameter(torch.tensor(0.5))

    def forward(self, states, hard: bool = False, disabled: bool = False):
        b, n, dim = states.shape
        q = self.to_query(states)
        k = self.to_key(states)
        # receiver i queries sender j
        edge_logits = (q @ k.transpose(-1, -2)) / math.sqrt(dim) + self.edge_bias
        eye = torch.eye(n, device=states.device, dtype=torch.bool)[None]
        edge_probability = torch.sigmoid(edge_logits).masked_fill(eye, 0.0)
        send_logits = self.sender_groups(states)  # B, sender, groups
        recv_logits = self.receiver_groups(states)  # B, receiver, groups
        group_probability = torch.sigmoid(
            send_logits[:, None, :, :] + recv_logits[:, :, None, :] + self.group_bias
        )
        if hard:
            edge_mask = (edge_probability > 0.5).to(states.dtype)
            group_mask = (group_probability > 0.5).to(states.dtype)
        else:
            edge_mask = edge_probability
            group_mask = group_probability
        if disabled:
            edge_mask = torch.zeros_like(edge_mask)
        message = self.to_message(states).reshape(b, n, self.groups, dim // self.groups)
        weighted = edge_mask[..., None, None] * group_mask[..., None] * message[:, None]
        incoming = weighted.sum(dim=2).reshape(b, n, dim)
        # Do not normalize by incoming gates: fewer messages should mean less influence.
        change = self.merger(incoming)
        receive_fraction = edge_mask.sum(-1, keepdim=True).clamp(max=1.0)
        gate = torch.sigmoid(self.recv_gate(torch.cat([states, incoming], dim=-1)))
        # With all edges shut, merged bias must not modify the state.
        out = self.norm(states + 0.9 * receive_fraction * gate * change)
        # True cost excludes diagonal and counts selected groups on active edges.
        expected_cost = ((edge_probability[..., None] * group_probability).sum((1, 2, 3))
                         / (n * (n - 1) * self.groups)).mean()
        return out, edge_probability, group_probability, expected_cost


class LearningRelationController(nn.Module):
    """Learn which peer may provide useful *learning*, separate from message routing.

    Pairwise scores can independently be zero. Actual usefulness must be taught
    by validation labels/feedback; raw scores are not correctness probabilities.
    """
    def __init__(self, dim):
        super().__init__()
        self.student = nn.Linear(dim, dim)
        self.teacher = nn.Linear(dim, dim)
        self.bias = nn.Parameter(torch.tensor(-0.7))

    def forward(self, states):
        b, n, d = states.shape
        scores = (self.student(states) @ self.teacher(states).transpose(-1, -2)) / math.sqrt(d)
        diagonal = torch.eye(n, dtype=torch.bool, device=states.device)[None]
        return (scores + self.bias).masked_fill(diagonal, -15.0)


def learning_relation_loss(relation_logits, expert_logits, targets):
    """A toy supervised proxy: prefer correct peers if the student is wrong.

    It is NOT a meta-learned estimate of counterfactual teaching benefit.
    """
    b, n, _ = expert_logits.shape
    correct = expert_logits.detach().argmax(-1) == targets[:, None]
    usefulness = (~correct[:, :, None] & correct[:, None, :]).float()
    diagonal = ~torch.eye(n, dtype=torch.bool, device=expert_logits.device)[None]
    return F.binary_cross_entropy_with_logits(relation_logits[diagonal.expand(b, -1, -1)],
                                               usefulness[diagonal.expand(b, -1, -1)])


class TimescaleMemory(nn.Module):
    """Three differentiable timescales. Explicit state, not permanent learned facts."""
    def __init__(self, dim: int):
        super().__init__()
        self.read_logits = nn.Linear(dim, 3)
        self.write_logits = nn.Linear(dim, 3)
        self.candidate = nn.Linear(dim, dim)
        self.register_buffer("max_rates", torch.tensor([0.85, 0.20, 0.035]))

    def forward(self, h, memory):
        # h [B,N,D]; memory [B,N,3,D]
        read = torch.softmax(self.read_logits(h), dim=-1)
        retrieved = (memory * read[..., None]).sum(dim=-2)
        alpha = torch.sigmoid(self.write_logits(h)) * self.max_rates
        write = self.candidate(h).tanh()[..., None, :]
        next_memory = memory * (1 - alpha[..., None]) + write * alpha[..., None]
        return retrieved, next_memory


class SRTM15(nn.Module):
    """From-scratch hybrid multi-path neural prototype, task output = sum class."""
    def __init__(self, cfg: Optional[ModelConfig] = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        c = self.cfg
        self.backbone = SharedBackbone(c)
        self.experts = nn.ModuleList([ReasoningExpert(c.dim, c.adapter_dim) for _ in range(c.num_experts)])
        self.communication = DynamicCommunication(c.dim, c.num_experts, c.message_groups)
        self.learning_relation = LearningRelationController(c.dim)
        self.memory = TimescaleMemory(c.dim)
        self.expert_head = nn.Linear(c.dim, c.num_classes)
        self.combine = nn.Sequential(nn.Linear(c.dim * c.num_experts, c.dim), nn.GELU(), nn.Linear(c.dim, c.num_classes))

    def init_memory(self, batch: int, device=None):
        device = device or next(self.parameters()).device
        return MemoryState(torch.zeros(batch, self.cfg.num_experts, 3, self.cfg.dim, device=device))

    def fast_parameters(self):
        for expert in self.experts:
            yield from expert.fast_down.parameters()
            yield from expert.fast_up.parameters()

    def reason_from_features(self, base, memory: Optional[MemoryState] = None, *,
                             hard_routes=False, disable_communication=False):
        if base.ndim == 2:
            b, dim = base.shape
            expert_base = base[:, None, :].expand(-1, self.cfg.num_experts, -1)
        elif base.ndim == 3:
            b, n, dim = base.shape
            if n != self.cfg.num_experts:
                raise ValueError("distributed inputs must have one sequence per expert")
            expert_base = base
        else:
            raise ValueError("features need shape [B,D] or [B,N,D]")
        if memory is None:
            memory = self.init_memory(b, base.device)
        if memory.slots.shape != (b, self.cfg.num_experts, 3, dim):
            raise ValueError("memory state shape mismatch")
        h = torch.zeros(b, self.cfg.num_experts, dim, device=base.device, dtype=base.dtype)
        slots = memory.slots
        costs = []
        edge_history = []
        for _ in range(self.cfg.reasoning_steps):
            read, slots = self.memory(h, slots)
            h = torch.stack([expert(expert_base[:, i], h[:, i], read[:, i])
                             for i, expert in enumerate(self.experts)], dim=1)
            h, edges, groups, cost = self.communication(
                h, hard=hard_routes, disabled=disable_communication)
            costs.append(cost)
            edge_history.append(edges)
        expert_logits = self.expert_head(h)
        relationship = self.learning_relation(h)
        logits = self.combine(h.reshape(b, -1)) + expert_logits.mean(dim=1)
        return {
            "logits": logits,
            "receiver_logits": expert_logits[:, -1],
            "expert_logits": expert_logits,
            "learning_relation_logits": relationship,
            "learning_relation_scores": torch.sigmoid(relationship),
            "states": h,
            "memory": MemoryState(slots),
            "edge_probs": torch.stack(edge_history, dim=1),
            "group_probs": groups,
            "communication_cost": torch.stack(costs).mean(),
        }

    def forward(self, tokens, memory: Optional[MemoryState] = None, *, hard_routes=False,
                disable_communication=False):
        if tokens.ndim == 2:
            base = self.backbone(tokens)
        elif tokens.ndim == 3:
            b, n, seq = tokens.shape
            if n != self.cfg.num_experts:
                raise ValueError("tokens need one fragment per expert")
            base = self.backbone(tokens.reshape(b * n, seq)).reshape(b, n, -1)
        else:
            raise ValueError("tokens should be [B,T] or [B,N,T]")
        return self.reason_from_features(base, memory, hard_routes=hard_routes,
                                         disable_communication=disable_communication)
