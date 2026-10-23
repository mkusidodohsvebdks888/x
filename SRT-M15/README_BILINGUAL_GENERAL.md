# SRT-M15 中英双语能力路线：当前可运行版本

目标：日常中文、日常英文都可以交流和完成简单任务。本仓库**尚未证明** SRT-M15 自研小模型已经达到这一水平。项目提供两条严格区分的路线。

## A. 原生 SRT-M15 从零训练：可验证教学实验

- 原模型：`srt_m15/model.py`，独立开发的轻量神经网络，从随机权重初始化。
- 数据：`examples/bilingual_toy_sft.jsonl`，40 条**演示**数据（20 中文 + 20 英文），不是大规模真实对话语料。
- GitHub Actions：`.github/workflows/srt-m15-bilingual-train.yml`，推送相关文件后运行一次有上限的 CPU 训练，保存最多 14 天的 artifact（checkpoint + 评测 JSON）。
- 测试命令（需 CPU PyTorch）：

```bash
cd SRT-M15
python train_common.py --mode sft --data examples/bilingual_toy_sft.jsonl \
  --steps 30 --batch-size 2 --context 32 --dim 16 --heads 4 \
  --experts 2 --reasoning-steps 1 --output checkpoints/bilingual_toy_sft.pt
python evaluate_bilingual_toy.py --checkpoint checkpoints/bilingual_toy_sft.pt \
  --data examples/bilingual_toy_sft.jsonl --output checkpoints/bilingual_toy_eval.json
```

- 这是**真实训练**，但模型尺寸和数据不足以支持流畅通用对话；它的验证指标只有留出集字符预测损失和准确率。
- 训练权重属于 GitHub Actions artifact，不会自动推送到 Git 仓库，不会自动进入现有 GUI/API 的默认本地 `SRT-M15` 路由。若需要部署该检查点，必须单独接入兼容推理路径，不能将它误当成成熟 Chat 模型。

## B. 实用双语对话：公开预训练模型作为底座

`bilingual_runtime.py` 用预训练的 **Qwen/Qwen2.5-0.5B-Instruct** 提供可运行的中英文本对话（首轮需从模型仓库下载权重）。这套实际语言能力**来自 Qwen 预训练模型**，不是本项目自主从零训练得到的，也不是 GPT/Claude 的模型权重。小模型在复杂问答、推理及长篇写作方面仍有明显限制。

基础模型许可：Qwen2.5-0.5B-Instruct 的 Hugging Face model card 标注 Apache-2.0。使用前核实模型仓库的完整 LICENSE，并保留许可说明。

### 本地聊天（可先运行）

安装 Python 3.10+ 和 CPU/GPU 版 PyTorch，并执行：

```bash
cd SRT-M15
python -m pip install -r requirements.txt -r requirements_bilingual.txt
python bilingual_runtime.py --prompt "你好，请用中文介绍你自己"
python bilingual_runtime.py --prompt "Hello! Please introduce yourself in English."
python bilingual_runtime.py
```

有 CUDA 显卡可选 `--device cuda`。首次启动下载外部权重，需要网络和足够的内存；后续已缓存时可以离线运行。若希望提升基础能力，可显式选择更大的兼容开源指令模型，例如：

```bash
python bilingual_runtime.py --model Qwen/Qwen2.5-1.5B-Instruct
```

更大模型通常更占内存、推理更慢，并不保证每个问题正确。模型标识与训练来源需保持透明。

## C. GPU LoRA 训练（可选，默认不启动）

训练以**预训练 Qwen 模型**为基础，更新 LoRA 适配器，并不是从零训练 SRT-M15 网络。这是实际可用于更高水平中英文交流的训练工程路线。

准备你已审核的数据 `learning_data/bilingual_verified.jsonl`（请勿提交隐私和密钥）。格式一行一个 JSON：

```json
{"language":"zh","prompt":"请介绍一下监督学习","response":"监督学习通过带有目标标签的数据训练模型。","verified":true,"source":"human-review"}
{"language":"en","prompt":"Explain supervised learning","response":"Supervised learning trains models using labeled examples.","verified":true,"source":"human-review"}
```

每种语言至少 4 条记录；真正有效的训练应该用上千到数万以上经过授权和核查的高质量、多主题、多轮样本，并预留足够的独立测试数据。此处的 4 条只是脚本的最低合法输入，不是达到流利水平的条件。

```bash
python finetune_bilingual.py --data learning_data/bilingual_verified.jsonl \
  --model Qwen/Qwen2.5-0.5B-Instruct --max-steps 80
```

- **必须有 CUDA GPU**，没有 GPU 时会明确报错并停止，不会伪装训练成功。
- `bilingual_data.py` 验证语种、样本来源、标注、重复提问；中英文分别做训练/验证划分。
- 仅回答部分计算损失，并对中英文留出集分别评估 loss/perplexity。
- 候选 LoRA 仅在验证损失整体改进且两种语言均没有超出 3% 的相对退化时保存。这只是微型指标，不代表达到「普遍中英文都能用」。
- 权重放在本地 `checkpoints/bilingual_lora/`，结果放在 `checkpoints/bilingual_lora_report.json`。二者默认**不上传 GitHub**，也不会自行持续训练。

用已经通过门槛的 LoRA 适配器聊天：

```bash
python bilingual_runtime.py --adapter checkpoints/bilingual_lora
```

只验证流程的演示数据也能训练，但必须显式传入 `--allow-demo-data`；此模式不应被宣传为可靠的能力训练：

```bash
python finetune_bilingual.py --data examples/bilingual_toy_sft.jsonl \
  --allow-demo-data --max-steps 20
```

## D. 从「能输出」到「普遍中文/英文流畅」的验收

分别评测独立的**中文**和**英文**真实场景：日常对话、多轮理解、问答准确性、指令遵循、常用写作、改写、翻译、错误纠正、对模糊输入的回应等。额外检查代码混入、幻觉、拒答是否合适及跨语言切换。不能仅靠训练 loss 或少量问答精确匹配宣布通过。

逐级建议：

1. **流程已通**：GPU/CPU 实际训练、每语言留出集报告、检查点可以恢复。
2. **可用基线**：预训练模型正常处理常见中文、英文问题；做人工对话检查。
3. **微调有收益**：验证指标和未见测试共同提升；不得以验证集反复调参并声称它仍是独立测试。
4. **准生产**：扩大合法数据覆盖，盲测数百条以上独立中英对话，评估多轮与稳健性；符合使用场景再上线。

## E. 自动化与限制

GitHub Actions 的 CPU Toy 训练属于有上限的单次作业，**不会无限循环**。若需多轮可靠训练，必须有独立、持久的 GPU 计算资源与受控的版本化权重存储。GitHub 是**代码/工作流与测试**平台，不等于无上限训练 GPU；请勿把测试用 CPU 训练重复运行当成持续模型能力提升。

每次改动会触发原有的项目单元测试；新 CPU 实验工作流也会针对相关文件自动执行。训练或工具执行过程的权限必须保持明确，不能让外部网页或模型输出直接变成可信训练标签。

本项目绝不声称「当前模型已经可以理解普遍中文和英文」。只有对话样本实测、独立评测与训练来源证明才能支持那样的结论。
