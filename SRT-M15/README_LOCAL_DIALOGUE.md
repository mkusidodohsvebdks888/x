# SRT-M15 本地双语对话实验（新增数据批次）

本批次专门补足原先 110,000 条合成数学/结构化任务缺乏对话上下文的问题。

- 数据生成器：`generate_local_dialogues.py`。从确定性场景模板产生中英文成对问答，包含自我介绍、多轮记忆、纠错、信息缺失时澄清、简短总结、身份回答、礼貌表达等。
- 对话测试器：`chat_staged.py`。读取**真实训练出的** staged checkpoint，直接运行本地神经网络，不调用 OpenAI、DeepSeek 等 API。
- 数据明确标为 `synthetic=true`、`verified=false`、`review_status=unreviewed`。新数据仍需要人工审查；不得宣称达到可靠通用聊天能力。

## 运行（在 `SRT-M15/` 目录）

```bash
python -m pip install -r requirements.txt
python generate_local_dialogues.py
python train_bilingual_stages.py --data examples/local_dialogue_synthetic_v1.jsonl --allow-demo-data --context 256 --sampling cycle --batch-size 4 --max-eval-pairs 32 --stages 1 --steps-per-stage 20 --output checkpoints/local_dialogues
python chat_staged.py --checkpoint checkpoints/local_dialogues/latest.pt
```

以上先进行 **20 步小规模训练管线测试**，**不是**充分训练。请在训练完成后检查 `checkpoints/local_dialogues/history.json` 的中英文分语言留出集 token loss、token accuracy，再逐步增加步数并验证是否有改善；CPU 训练可能很慢。每批会存储 `latest.pt` 和可能出现的 `best.pt`。

若电脑资源允许，可以从零开启较长训练（不同超参数的作业要使用**新目录**）：

```bash
python train_bilingual_stages.py --data examples/local_dialogue_synthetic_v1.jsonl --allow-demo-data --context 256 --sampling cycle --batch-size 8 --max-eval-pairs 64 --stages 5 --steps-per-stage 80 --output checkpoints/local_dialogues_long
python chat_staged.py --checkpoint checkpoints/local_dialogues_long/latest.pt --prompt "你是谁？"
```

如果继续同一个较长训练作业，保持相同数据文件、模型、batch size、context、学习率、步数等配置，使用 `--resume checkpoints/local_dialogues_long/latest.pt` 和相同 `--output` 追加批次。例如第二次运行相同命令并添加 `--resume` 参数。旧的 110k checkpoint 与本数据集词表不一致，**不能直接使用 --resume 续训**。

## 实际交互能力验收（独立人工测试）

必须用与模板不同的语句测试：基本寒暄、身份一致性、开放问题、纠错、跨轮记忆、中文/英文语法、无法回答时的诚实表达。检查至少各 50 个未见问题，并记录符合要求比例；只看训练 loss 下降不足以证明模型会对话。

**当前限制**：脚本为字符级词表，未知字会直接拒绝；上下文仅 256 字符；多轮聊天不保证稳定；生成可能为空或重复。改善这些限制需要更高质量自然对话数据、更合理分词、模型容量及评测，而非无限重复模板数据。训练与推理均为本地运行，不依赖云 GPU。
