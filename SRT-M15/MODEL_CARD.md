# SRT-M15 toy model: Model Card

**Model type:** Research prototype of a scratch-initialized, task-specific multi-path network.

**Tasks:** (1) Distributed single-digit addition: expert A sees a, B sees b, C sees neither, only C is supervised. (2) Autoregressive char-level text prediction on small example Chinese corpus.

**Not:** a pretrained LLM, a conversational agent, general autonomous scientist, a proven novel architecture, or a reliable arithmetic engine.

**Architecture:** hand-written causal multi-head attention; Transformer backbone; 3 independent recurrent GRU branches; bidirectional sender/receiver-conditioned message routing with per-group gates; dynamic 3-rate differentiable memory; dense classifier; small online fast adapters.

**Training:** random init, synthetic 80/20 train-test split of 100 addition pairs; separate tiny autoregressive LM training path. Distributed training uses only expert C output to prevent other experts from bypassing communications. Toy accuracy may be low and unstable, depending on hyperparameters. Never use test examples to choose parameters without a fresh holdout test.

**Inferences:** route selection and memory states are diagnostics. No claim of actual subjective thinking or demonstrated emergent reasoning. External JSON memory is a separate deterministic module, not differentiably integrated in the model.

**Limitations:** all experts see same input; differentiation of expert roles is not guaranteed; hard route threshold at inference differs from smooth training; current entropy-based experiment selection is a heuristic; online update guard set is tiny; model architecture incurs significant cost relative to the tiny task; no world model, no semantic retrieval, relation scorer has only a supervised proxy in non-distributed mode, not actual meta-learned teacher value; memory states are not automatically persisted to disk.

### Opt-in Internet knowledge retrieval

The local UI can query the official Chinese or English Wikipedia public read-only MediaWiki API with explicit user permission and strict request/time/storage limits. It stores attributed excerpts and manual review metadata in SQLite, enabling retrieval of citations within the local chat UI. **It does not train or change neural-network parameters from retrieved text.** Source materials are unverified by default; this is retrieval-based memory, not demonstrated open-domain understanding. The offline verified single-digit arithmetic adaptation path is unchanged.

### Bilingual extension (zh/en)
The supervised UTF-8 byte question-answer training can now use labelled Chinese
and English samples, matched translation split groups, language-balanced updates,
and separate held-out validation for each language. Sample pairs require manual
approval. This enables bilingual toy training experiments, **not** reliable
multilingual conversations or generalized language understanding.
