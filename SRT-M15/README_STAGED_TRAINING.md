# SRT-M15 多批次训练与断点续训

这一功能已直接接入原生 SRT-M15 模型（**随机初始化的 PyTorch 实验模型**），而不只是将多轮 API 调用误称为训练。每批（stage）执行真实梯度更新，单独验证中文与英文，保存可以完整恢复的最新权重、优化器、随机采样状态及指标历史。

目前已提供 [10,000 条中英双语合成示例生成器](DATASET_CARD_BILINGUAL_V3_10000.md)（5,000 对平行话题），并保留旧版 1,000 / 260 / 40 条样本。**10,000 条合成模板数据仍不足以训练出普遍流畅的中文和英文**；数据未逐条人工审核，不能把测试 Token 准确率当成真实聊天能力。

## 本地运行

**首次运行前，请先生成 10,000 条数据：**

```bash
python generate_bilingual_10k.py
```

从 `SRT-M15/` 目录运行，安装 CPU/GPU 版 PyTorch：

```bash
python train_bilingual_stages.py \
  --data examples/bilingual_synthetic_v3_10000.jsonl \
  --allow-demo-data --context 256 --sampling cycle --batch-size 16 \
  --stages 10 --steps-per-stage 160 \
  --output checkpoints/bilingual_stages
```

继续另外 **2 批训练**：

```bash
python train_bilingual_stages.py \
  --data examples/bilingual_synthetic_v3_10000.jsonl \
  --allow-demo-data --context 256 --sampling cycle --batch-size 16 \
  --stages 10 --steps-per-stage 160 \
  --resume checkpoints/bilingual_stages/latest.pt \
  --output checkpoints/bilingual_stages
```

**两次执行的训练状态会连续：** 第二次读取第一轮的模型参数、AdamW 优化器状态、批次计数及 Python 随机采样器状态，而非随机重新初始化。第一批与之后的批次必须使用相同数据文件与相关超参数（尤其是每批步数、种子、学习率等），否则明确拒绝恢复。

## 训练文件

- `latest.pt`：最新完成的批次状态。即使本批表现变差，也保留这份续训基线。
- `best.pt`：仅当**中英文分别没有明显退化、宏平均留出集损失下降**时更新。若尚无批次达标，则可能不存在。
- `history.json`：每批训练损失、中文/英文验证 Token 指标、是否晋升最佳模型、已执行的批数与总步数。
- 保留原始训练数据的 SHA-256 指纹、词表、模型结构、随机采样状态。**该脚本不允许改变词表后直接恢复旧优化器状态**；扩展语料需另行设计受检验的迁移方案。

本地输出属于运行时权重，默认不提交到公开 Git 仓库。请勿将秘钥、用户私人对话、未授权素材加入训练。

## GitHub Actions：跨运行续训

工作流：`../../.github/workflows/srt-m15-staged-train.yml`。

在 GitHub 仓库的 **Actions → SRT-M15 staged bilingual training → Run workflow** 中，选 `main` 分支，填入（**新数据集需从新检查点开始，不能续接旧 40 条数据的 checkpoint**）：

1. `training_dataset`：选择 `synthetic_10000`（默认），或旧的 `synthetic_1000` / `synthetic_260` / `toy_40`。
2. `stages`：本次训练的批数（1、2、3、5、8 或 10；默认 10）。
3. `steps_per_stage`：每批更新步数（20、40、80、160；默认 160）。
4. `resume_run_id`：首次留空；续训填写**同一数据集与超参数**的上一次成功运行编号。

首次运行不填编号时，从零初始化训练。v3 数据会由工作流生成，并使用 `context=256`、`batch_size=16`、不放回循环采样（`cycle`）；v2 使用 `context=208`，旧版 `synthetic_260`/`toy_40` 仍为 `context=32`。此前以 `context=32` 训练 v2 所生成的权重不能直接与新配置续训。完成后 GitHub 会上传 `srt-m15-staged-<run_id>` artifact。第二次填写该 run id，工作流会读取前一次 artifact 的 `latest.pt` 并在此基础上继续训练，然后上传新的检查点。**继续续训要改填最新一次成功运行的编号**，并沿用相同的 `steps_per_stage` 和数据。

每次工作流设置执行时长限制，artifact 保留 14 天。超过保留期不能自动下载；应及时下载保存到自己的持久存储。GitHub Actions 受分钟数、CPU 能力和额度约束，不是长期 GPU 模型训练平台。

## 用真实中文/英文数据

普通训练必须提供显式审核的 JSONL，每行例如：

```json
{"language":"zh","prompt":"早上好","response":"早上好！今天有什么计划？","verified":true,"source":"human-review"}
{"language":"en","prompt":"Good morning","response":"Good morning! What's your plan today?","verified":true,"source":"human-review"}
```

每种语言至少四条且至少有一条留出验证；实际广泛中英文对话能力需要大得多、覆盖常见领域并独立验证的数据。不要使用 `--allow-demo-data` 来处理未审查的实际来源数据。这个脚本不会自动将不可靠的网页答案标记为已验证。

## 建议的研发路线

对于原生从零训练：先确保训练/恢复/评测真实可运行，再扩大模型与语料，采用更合适的子词或字节编码，并针对中文、英文及跨语言指令建立独立盲测；只有实测通过才声明“普遍正常交流”。

对于快速获得可用的中英对话：运行已加入的 `bilingual_runtime.py`（开放权重 Qwen 底座），之后在持久 GPU 上使用 `finetune_bilingual.py` 训练自己的 LoRA 适配器。**它是基于现有开放权重训练，不是从零训练 SRT-M15。**

## 状态解释

- **GitHub workflow success**：训练任务和文件保存过程成功。
- **loss 下降**：小范围预测指标改善。
- **真实通用双语能力**：必须在足够多未见场景下评估，不能由上述两个指标推出。
