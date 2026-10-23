# SRT-M15 | 1,000,000 条中英文对话数据：本地流式构建

**状态**：已实现百万目标数据的本地流式收集/清洗/去重/分片代码，**并未宣称已实际获取、验证或训练完 1,000,000 条**。数据来源必须自行核验许可、授权和质量。GitHub 只保存程序；完整数据默认保留在本地 `local_data/`（已加入 .gitignore）。

## 计划

| 语言 | 目标有效样本数 |
|---|---:|
| 中文（zh） | 500,000 |
| 英文（en） | 500,000 |
| **总计** | **1,000,000** |

此处“样本”指一对提示与助手回复（多轮对话中的每个助手回合可以成为一个样本），**不是** 100 万个独立人工对话。来源中的模型生成回答不能当作经过人工审核的正确答案。

## 先进行本地验证

```bash
cd SRT-M15
python -m unittest discover -s tests -p "test_build_conversation_million.py" -v
python -m pip install datasets
```

## 方案 A：使用自己拥有合法使用权的本地数据

单个文件为每行 JSON 对象的 UTF-8 JSONL（也可以 `.jsonl.gz`），例如：

```json
{"prompt":"你好","response":"你好！有什么可以帮你？"}
```

```bash
python build_conversation_million.py \
  --input path/to/chinese_dialogues.jsonl::zh \
  --input path/to/english_dialogues.jsonl::en \
  --target 1000000 --zh-target 500000 \
  --shard-size 10000
```

**注意**：每个输入参数只能传一种语言；若混合语言，请事先拆分文件并审核标签。

## 方案 B：显式选择 Hugging Face 公开数据集（联网，本机下载/流式读取）

下面只给出**可选择的候选来源**，不是版权/隐私保证。先在各自的数据集卡中核对最新授权、第三方模型输出使用条件，尤其在公开发布或商用时。脚本本身不会偷偷下载，必须手动写明数据集并使用 `--accept-source-terms` 表示已核查。示例：

```bash
python -m pip install datasets
python build_conversation_million.py \
  --hf-source guoxz/GX-Chinese-Instruct-1M-2508::default::train::zh \
  --hf-source HuggingFaceH4/ultrachat_200k::default::train_sft::en \
  --hf-source HuggingFaceH4/ultrachat_200k::default::train_gen::en \
  --accept-source-terms \
  --target 1000000 --zh-target 500000 \
  --shard-size 10000 \
  --output local_data/conversation_million
```

- `guoxz/GX-Chinese-Instruct-1M-2508`：仓库标注 Apache-2.0，共约 100 万条中文指令，数据卡称由 DeepSeek 生成，需另行审视第三方输出条款和质量。
- `HuggingFaceH4/ultrachat_200k`：仓库标注 MIT，原始数据为模型生成的多轮英文聊天，`train_sft` 和 `train_gen` 来自不同划分，可能出现相似问题。程序进行提示文本去重；不同对话内容仍可能存在语义重叠。
- 无法保证以上两源必然通过过滤凑齐 50 万中英各一半；**是否达标以 `manifest.json` 为准**。不足 100 万返回非零退出码且明确报告缺口，禁止把下载量当作实际有效样本。
- 读入单行示例只支持 `instruction/input/output`、`input/target`、`inputs/targets`、`prompt/response` 或 `messages/conversations` 标准字段；不支持的格式需要先转换。

## 输出目录

```text
local_data/conversation_million/
  manifest.json
  dedup.sqlite3
  shards/
    train/zh/part-00000.jsonl.gz ...
    train/en/part-00000.jsonl.gz ...
    dev/zh/...       dev/en/...
    test/zh/...      test/en/...
```

- 约 90% train、5% dev、5% test，使用会话首轮问题的稳定 hash 做分组；多轮子样本尽量不跨集合。
- 每个 JSONL 行标记 `verified=false` 和 `review_status=unreviewed`；并保存 `source`。
- 用 SQLite 在本地做提示去重；过滤简单的邮箱、中国大陆手机号。**这不是完整的敏感信息清除**，仍需单独抽样检查。
- 使用 gzip 分片，避免单个 1M 行文本文件过大。实际磁盘占用取决于文本平均长度，几百 MB 到数 GB 均可能，建议保留额外缓存空间。
- 生成器当前采用单次全量运行；中断后使用新空目录重启，不应手动拼接未验证的残片。
- 文件写到本地，不会自动上传 GitHub。

## 与 SRT-M15 训练的关系

现有 `train_bilingual_stages.py` 会一次性读取完整 JSONL、创建字级词表和内存窗口；并要求中英文按 topic 成对。此处真实语料通常不成对、规模也很大，**不要直接把百万数据拼起来喂给旧训练脚本**。

后续需要升级：
1. 训练器按分片流式读取，无须把所有窗口加载到内存；
2. 固定并版本化 tokenizer（推荐子词或字节，而不是按新数据每次重建字符词表）；
3. 支持跨批次保存优化器、训练进度、数据位置和随机状态；
4. 独立验证中英文未见对话、长上下文、身份一致性和错误修正；
5. 扩大模型容量或适配合适的开放权重底座。**一百万条数据本身不能保证通用聊天能力**。

## 公开资料示例及许可核查入口

- https://huggingface.co/datasets/guoxz/GX-Chinese-Instruct-1M-2508
- https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k

