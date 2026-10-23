# SRT-M15 双语训练实际结果（2026-10-23）

**说明：实际跑过训练，但尚未达到通用中文/英文聊天能力。** 下方数据来自 GitHub Actions 日志和独立留出的示例问题，不是推测。

## 从零初始化的 SRT-M15 小模型：真实 CPU 训练

- 工作流：`.github/workflows/srt-m15-bilingual-train.yml`
- 执行记录：`https://github.com/mkusidodohsvebdks888/x/actions/runs/39919202773`
- 结果：**success**；训练模型 25,525 参数、30 steps、32 条训练样本、8 条留出样本。
- 训练文件：`examples/bilingual_toy_sft.jsonl`（40 条演示数据）。
- 模型检查点和评测报告保存在该 Actions run 的 artifact `srt-m15-bilingual-toy-39919202773`（14 天保留）。

| 指标 | 起始 | 30 步后 |
|---|---:|---:|
| 综合留出集 Token loss | 5.7795 | 5.5498 |
| 综合留出集 Token 准确率 | 0.588% | 1.471% |
| 中文留出集 Token 准确率 | — | 3.614% |
| 英文留出集 Token 准确率 | — | 0.778% |
| 中文留出集数量 | — | 4 |
| 英文留出集数量 | — | 4 |

这些指标说明**确实进行了梯度训练，并降低了少量数据上的损失**，但 Token 准确率远不足以形成正常对话，且留出集仅 8 条。严禁将此标作“中英文训练完成”。

## 实用双语运行底座：预训练权重

- 工作流：`https://github.com/mkusidodohsvebdks888/x/actions/runs/39919464625`。
- 结果：**success**。
- 使用模型：`Qwen/Qwen2.5-0.5B-Instruct`，公开预训练模型（Open-weight base），**非 SRT-M15 原生训练参数**。
- 测试了中文、英文两条独立输入，两个方向均产生了对应语言的回答；生成长度只设置 32 tokens，因此回答有截断。这只证明运行通路正常，不意味着达到语言能力验收目标。
- 使用入口：`python bilingual_runtime.py`。

## 下一步真正需要的条件

通用中文/英文能力的实际路径是：使用已具有跨语言能力的基础模型 → 构建有版权/使用许可、标注可靠的多主题双语数据集 → 在持久 CUDA GPU 上进行 LoRA → 分语言回归测试、人工盲测和可控迭代。当前仓库已新增 `finetune_bilingual.py` 和数据质量门槛，但**尚未实际完成 GPU LoRA 微调**（此会话没有可用的持久训练 GPU）。

## 持续训练约束

GitHub Actions 的单次 CPU 实验不等于无限制、持续的通用大模型训练。不要将自动编排或模型自评误认为它能自主获得强语言能力。需要真实 GPU 环境、可持续保存的模型权重和人工核对数据。
