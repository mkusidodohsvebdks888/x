# SRT-M15 双语训练数据包 v1（助手原创合成示例）

**数据位置：** `examples/bilingual_synthetic_v1.jsonl`  
**生成日期：** 2026-10-23  
**来源与权利说明：** 由助手为此项目编写的短句对话示例，没有从 OpenAI、Claude、DeepSeek 等模型私有数据或训练权重中提取内容，也没有转载第三方文章。数据属于 **AI 生成的合成数据（synthetic）**，并非由母语者逐条审定、也不是可靠的事实参考库。使用前仍需人工检查事实、表达习惯以及适用许可要求。

## 一、规模与覆盖

- **130 组话题，每组中文、英文各 1 条，总计 260 条 JSONL 训练候选样本。**
- 13 个日常/学习主题，各 10 组话题：问候礼仪、日常生活、学习方法、中英互译、文字写作、基础数学、计算机技术、旅行、购物、工作协作、计划管理、逻辑推理、情绪与生活支持。
- 每条记录包含 `language`、`prompt`、`response`、`topic_id`、`category`、`split`、`verified`、`source`、`review_status`、`synthetic`。
- 目前只有**单轮**问答；不包含语音数据、复杂多轮对话、广泛专业领域或足以训练通用语言模型的大规模语料。
- 不是「130 万句」或者「普遍中文和英文全覆盖」，请勿扩大宣传。

## 二、来源和质量

示例标记为：

```json
{"language":"zh","prompt":"什么是函数？","response":"函数把一段可重复使用的操作组织起来，并可以接收输入、返回结果。","topic_id":"technology-03","category":"technology","split":"auto","verified":false,"source":"assistant-authored-synthetic-2026-10-23","review_status":"unreviewed","synthetic":true}
```

这里明确使用 `verified=false`，是因为**合成文本并不自动等于人类核对的事实或语言参考答案**。现有数据过滤器默认拒绝未经审查的实际训练样本。为了实验训练，必须显式指定 `--allow-demo-data`，使误用风险更明显。

如果要将数据投入正式模型质量提升，应由人类审核并为合格记录填写真实的审核来源；只有审核后的记录才可以移除演示开关。**不能将所有 `verified` 一键批量改为 true 来绕过验证。**

## 三、严格的中英平行数据拆分

`bilingual_data.py` 会在检测到 `topic_id` 字段时，按话题 **group split**，确保中文和英文的同一条话题绝不会被分到训练集与验证集两侧，避免平行翻译造成数据泄漏。

默认 `dev_fraction=0.2`，期望得到：

| 类型 | 中文 | 英文 | 话题组 |
|---|---:|---:|---:|
| 训练集 | 104 | 104 | 104 |
| 验证集 | 26 | 26 | 26 |
| 合计 | 130 | 130 | 130 |

训练、验证的词预测得分仍只能用于早期实验，**不能单独用来判定对话流利度、事实正确性或真实泛化能力**。原生 SRT-M15 现有很小的参数量、较短上下文和字符词表，尤其不适合据此宣称覆盖普遍的中英文。

## 四、在 GitHub 分批训练

进入仓库的 Actions 页面，找到 `SRT-M15 staged bilingual training` 并执行 `Run workflow`：

- `training_dataset` 选择 `synthetic_260`（默认）。
- `stages` 选择 `2`。
- `steps_per_stage` 选择 `20`。
- 首次 `resume_run_id` 留空；跨运行续训必须使用**相同数据文件与超参数**，填写新数据集第一轮成功运行的编号。原 `toy_40` 训练检查点由于数据 SHA、词表和模型设置不同，**不能直接在新数据集上接着训练**。

本地 CPU 训练：

```bash
cd SRT-M15
python train_bilingual_stages.py \
  --data examples/bilingual_synthetic_v1.jsonl \
  --allow-demo-data --stages 2 --steps-per-stage 20
```

工作流上传 `latest.pt`、`best.pt`（若晋升通过）和 `history.json`，在 Actions 的 Artifacts 下载。

## 五、扩展到广泛中英文能力的后续要求

1. 将单轮对话扩展为多轮交互，增加语序、语法、推理和未见问题的覆盖。
2. 获取明确授权的公开数据（含许可证/来源信息），而非无来源抓取。
3. 做真实语料清洗、去重、跨语言近似重复检测、训练/验证/测试隔离。
4. 原生模型需更大的合适架构与训练计算预算；实用路线可以在已合法取得的开放权重基础上使用 LoRA（要明确披露底座）。
5. 用未见中文及英文真实场景、人工盲评和统一指标进行独立验收。

**结论：这是一套明确可审计、可直接用于 SRT-M15 数据管道调试的第一批原创中英双语样本，不是通用模型训练所需的完整数据集。**
