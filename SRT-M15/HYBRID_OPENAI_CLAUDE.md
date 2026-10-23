# SRT-M15: OpenAI × Anthropic-inspired hybrid agent (experimental)

This module borrows **public engineering patterns**, not proprietary model weights, training data or hidden reasoning procedures.

## Architecture

```text
User request
    │
    ▼
Task planner (optional callback; strict JSON; limited steps)
    │
    ▼
Explicit tool registry ── read-only by default; writes need user approval
    │
    ▼
SRT-M15/local-or-external writer ── tools are untrusted evidence, not instructions
    │
    ▼
Evaluator (optional model callback; strict JSON)
    │
    ├── accept (model-reviewed, NOT fact-proven)
    └── bounded correction → re-review → accept / needs_review
    │
    ▼
Answer + provider + bounded event trace + assessment state
```

**OpenAI-inspired:** typed tools, execution guardrails, trace events, model/provider disclosure and explicit orchestration.  
**Anthropic-inspired:** simple composable workflows, planner/executor separation, evaluator–optimizer feedback, and the principle that assistant behavior should remain correctable and honest. This prototype is **not** an implementation of Anthropic's Constitutional AI training or RLAIF algorithm.

## What is working

- `srt_m15/hybrid_agent.py`: `HybridAgent`, `HybridConfig`, `Tool`, `HybridResult`, `from_chat_engine`.
- Explicit, separately injectable `writer`, `planner` and `reviewer` callbacks. They may use the same model or different models. No provider network calls are hidden in this module.
- Strict JSON planning with safe single-step fallback; unrecognized tools cannot execute.
- Configurable limits: planning steps, revisions, question/context/tool-output lengths.
- Registered read-only tool calls and per-run approvals for tools marked `read_only=False`.
- Model-review feedback loop and explicit result states: `not_reviewed`, `review_unavailable`, `model_review_passed`, `needs_review`, `needs_approval`, `model_error`.
- Minimal phase/status trace; tool exceptions do not leak raw exception messages.
- Existing chat engine behavior and external-backend opt-in remain unchanged.

**Not implemented:** true multi-agent delegation, long-term self-training, stored episodic memory, automatic exploration loop, integrated GUI/API routes, independent truth verification, provider SDK integrations, learned Constitutional AI, or actual GPT/Claude model weights.

## Running (existing local trained weights required)

Install the existing requirements and train/load checkpoints following `README.md`. From inside `SRT-M15/`:

```python
from srt_m15.chat import SRTChatEngine
from srt_m15.hybrid_agent import from_chat_engine

engine = SRTChatEngine(root=".", backend="local")
agent = from_chat_engine(engine)
result = agent.run("1+2")
print(result.answer)
print(result.provider, result.status)
print(result.trace)
```

This version routes a **single offline writer step**, preserving the current SRT-M15 prototype's numeric/text limitations. Calling it a hybrid agent does not make the underlying toy model Claude- or GPT-class.

## Add a real planner/reviewer (explicit optional integration)

Both callbacks must return strict JSON (without Markdown fences):

```python
import json
from srt_m15.hybrid_agent import HybridAgent, Tool, HybridConfig

planner = lambda prompt: json.dumps({
    "steps": [{"goal": "Look up a reference", "tool": "lookup", "input": "topic"}]
})
reviewer = lambda prompt: json.dumps({"ok": True, "feedback": "Model-only review"})
tools = {
    "lookup": Tool("lookup", "Look up a local reference", lambda key: "reference text")
}
agent = HybridAgent(
    writer=lambda prompt: "Draft based on the available evidence",
    planner=planner,
    reviewer=reviewer,
    tools=tools,
    config=HybridConfig(max_steps=3, max_revisions=1),
    provider="injected test model",
)
result = agent.run("What does the reference say?")
print(result.answer, result.status, result.trace)
```

To use actual model reasoning for planning and reviewing, supply callable clients for those roles. For example, the existing opt-in `DeepSeekAPIClient().ask(prompt)` can be a callback after you configure `DEEPSEEK_API_KEY`, but prompts sent to that callback leave your device. The assistant will not acquire those models' weights or train itself merely by calling an API.

For side-effectful tools, the caller must register a `read_only=False` Tool **and** explicitly pass its name in `approved_actions` to an individual `run(...)` invocation. In production, add stronger per-argument/resource scoping and confirmation; a simple tool-name approval is only a minimal research guardrail.

## Testing and research limitations

```bash
python -m unittest discover -s tests -p 'test_hybrid_agent.py' -v
```

The tests use deterministic fake callbacks so they do not require API keys or trained checkpoints, though the package itself depends on existing PyTorch imports. The local and remote model quality has **not** been benchmarked. AI model reviews are opinions, not independent objective verification. Never feed unverified answers directly into training labels.

## Next development milestones

1. Add a safe evaluation dataset and deterministic task-specific validators.
2. Integrate persistent task state and a bounded resumable queue.
3. Introduce task-specific read-only tools (local docs / retrieval / calculator), then secured execution connectors.
4. Separate reference-grounded critiques from stylistic critiques and measure both.
5. Only after evaluation: human-reviewed data curation and versioned training/rollback.

## Public research background

- OpenAI: New tools for building agents (Responses API, Agents SDK, guardrails, tracing).
- OpenAI: Instruction hierarchy research and prompt-injection robustness.
- Anthropic: Building effective agents (workflow patterns, evaluator–optimizer).
- Anthropic: Constitutional AI / AI feedback (research distinction between self-critique during training and at inference).

This code is an **original educational implementation** of publicly discussed system design patterns.