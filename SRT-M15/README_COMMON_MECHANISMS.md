# SRT-M15：主流自回归语言模型的共同机制（可运行实验版）

本模块将多家大语言模型共享的**公开、可复现基础机制**整理为可执行的 PyTorch 训练链路。它**不是**复现 OpenAI/DeepSeek/Claude/Gemini 的非公开权重或完整训练工艺，也不声称达到它们的能力。

## 1. 哪些是共同的，哪些不是？

| 机制 | SRT-M15 对应文件 | 状态 |
| --- | --- | --- |
| Unicode 文本读取和训练语料准备 | `train_common.py` | ✅ |
| 将字符映射到离散 Token ID | `srt_m15/language.py` | ✅ 字符级实验；不是 BPE |
| Token Embedding、位置嵌入 | `srt_m15/model.py` | ✅ |
| 多头因果注意力（禁止看未来） | `srt_m15/model.py` | ✅ |
| Transformer 前馈层、激活函数 | `srt_m15/model.py` | ✅ |
| 残差连接、LayerNorm | `srt_m15/model.py` | ✅ |
| 多路递归推理、跨分支通信与内部状态 | `srt_m15/model.py` | ✅ SRT-M15 专属实验扩展；不是所有大模型共同采用 |
| 词表上的自回归预测 | `srt_m15/language.py` | ✅ |
| 预训练：下一 Token 预测 | `srt_m15/common_mechanisms.py`、`train_common.py` | ✅ |
| 指令 SFT：只对助手回答计算监督损失 | 同上 | ✅ |
| 交叉熵、自动求导、反向传播 | 同上 | ✅ |
| AdamW 参数优化、梯度裁剪 | 同上 | ✅ |
| 训练/验证按独立样本切分，窗口不跨样本 | 同上 | ✅ 基础防泄漏 |
| Token 损失、Token 准确率、困惑度评估 | 同上 | ✅ |
| 温度 / Top-k / Top-p 自回归采样 | 同上 | ✅ |
| 包含模型权重、配置、字符词表、评估指标的检查点 | `train_common.py` | ✅ |
| 经过验证的加法题在线适配与回滚 | `srt_m15/learning.py` 等 | ✅ 原有独立实验 |
| 版本化外部记忆与网络来源检索 | `srt_m15/storage.py` 等 | ✅ 原有功能；不是所有基础 LLM 的必需机制 |
| RLHF / PPO / DPO / GRPO | 本次不纳入统一训练器 | ⏳ 非所有模型共有；需要另行设计偏好/奖励数据与可靠验证 |
| 可扩展多 GPU 分布式训练、海量语料治理 | 无 | ❌ |
| 图像、音频与视频输入 | 无 | ❌ |
| 生产级安全对齐、基准测试与可审计评估 | 无 | ❌ |

`✅` 意味着**存在可运行的教育/研究实现**，不意味着已完成成熟大模型能力。

## 2. 模型的共同训练数据流

```text
文档 / 指令 JSONL
       ↓
先按独立样本划分训练集 / 验证集
       ↓
字符词表 → Token ID → 固定上下文窗口
       ↓
Embedding + 位置表示
       ↓
因果 Transformer + SRT-M15 多分支实验结构
       ↓
每个位置预测下一个 Token 的 logits
       ↓
仅对监督的 token 计算交叉熵
       ↓
反向传播 → 梯度裁剪 → AdamW 更新权重
       ↓
独立验证集评估 → 保存检查点
       ↓
采样生成（temperature、top_k、top_p）
```

**预训练与指令微调的关键差异**：预训练监督文本后续字符；指令微调将用户提示作为条件，仅对助手回答（包括结束换行）的字符计算损失。预测仍采用因果语言模型。

## 3. 运行环境

在 `SRT-M15/` 目录下执行：

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

### A. 从随机权重进行下一字符预训练

```bash
python train_common.py --mode pretrain --data examples/common_text_corpus.txt \
  --steps 30 --context 24 --dim 32 --batch-size 4 \
  --output checkpoints/common_pretrain.pt
```

### B. 从随机权重进行监督指令微调（演示）

```bash
python train_common.py --mode sft --data examples/common_instruction_examples.jsonl \
  --steps 30 --context 24 --dim 32 --batch-size 4 \
  --output checkpoints/common_sft.pt
```

### C. 接着训练先前兼容的检查点

```bash
python train_common.py --mode sft --data examples/common_instruction_examples.jsonl \
  --init-checkpoint checkpoints/common_pretrain.pt \
  --context 24 --steps 30 --output checkpoints/common_sft.pt
```

**注意：** 只有字符词表与上下文长度完全一致时才允许加载先前权重；两个示例数据源的字符集合往往**不同**，因此 C 只是使用方法的示意。实际继续训练，应为两个阶段准备同一套预先固定的词表（当前 CLI 不会自动扩展词表）。如不一致，程序会明确报错，绝不会偷偷重置权重。

### D. 用保存的权重生成文字

```bash
python generate.py --checkpoint checkpoints/common_pretrain.pt \
  --prompt "学习" --new-tokens 20 --temperature 0.9
```

上述提示词必须只包含训练字符词表中已有的字符，否则会报错；这正是小型字符模型与成熟分词器的区别。

## 4. 训练数据格式

**预训练文本**：纯 UTF-8，使用一个空白行分隔独立文档，至少两个非重复文档。

**指令数据**：UTF-8 JSONL，每行一个样本：

```json
{"prompt":"请解释神经网络。","response":"神经网络使用可训练参数处理输入信息。"}
```

务必使用你拥有训练使用权的数据，移除隐私信息，避免收集未获授权的个人内容。为了避免高相关内容污染验证集，真实训练应按来源、文档、时间或主题进行更严格的分组划分；内置实验仅进行按样本划分与去重。

## 5. 结果应该怎样解释

- `train_batch_loss`：当前训练小批次的交叉熵，不是全训练集指标。
- `validation.token_loss`：未参加梯度更新的样本平均下一字符损失。
- `validation.perplexity`：`exp(token_loss)`；仅在相同 Tokenizer / 词表设置下才适合横向比较。
- `validation.token_accuracy`：未参加训练样本的受监督字符预测准确率。
- 如果验证集损失变高，即使训练损失下降，也可能发生过拟合。
- 几十条演示数据不能让模型获得稳定的中文问答能力，也无法证明模型自主思考。

## 6. 测试内容

`tests/test_common_mechanisms.py` 包含指令提示掩码、数据集隔离、窗口与填充、交叉熵梯度、真实网络训练参数更新、验证指标、温度参数校验、Top-k/Top-p 采样、重复文档拒绝等实验。

## 7. 后续研究方向

基础共同机制完整串通后，再单独研究偏好对齐、规则奖励、推理验证器、基准任务、规模化 Tokenizer、可恢复优化器状态、分布式训练和多模态输入。特别注意：**外部记忆/自主提问不会自动更新模型参数**；参数学习必须实际执行优化步骤，并经过独立验证。

## 8. 模块边界

本次新增代码与原项目并行，不覆盖已有数学权重和中文/双语学习权重；也不改变已有网页后端的对话身份和 API 路由。模型仍是**从零训练的小规模实验系统**。
