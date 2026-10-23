# SRT-M15：可验证的增量学习（实验版）

这一版解决的是 **已审核知识进入训练、词表增量扩展、旧知识回放、留出集检测、防遗忘检查、权重回滚**。它不能把 10 万参数量级的字符模型直接变成通用大模型，也不能证明模型具有自主意识或可靠的开放领域推理能力。

## 1. 基础监督训练

手机用户建议直接打开 **`SRT_M15_Continual_Learning_Colab.ipynb`**（独立学习 Notebook），第一步从仓库 `main` 下载最新项目，不需要手动复制旧内嵌源码。旧版 `SRT_M15_Phone_Colab.ipynb` 仍可以用来进行加法和初始语言实验。

在项目根目录运行：

```bash
python train_qa.py --steps 1600 --eval-every 200 --device auto
```

训练后产生 `checkpoints/srt_m15_qa.pt`。这个训练器只对回答部分计算损失，留出训练中未出现的问句测试。**不要使用旧的 `train_lm.py` 检查点冒充监督问答权重。**

## 2. 准备已经核对过的学习数据

复制 `learning_data/verified_qa.example.jsonl` 为 `learning_data/verified_qa.jsonl`，一行一个 JSON，例如：

```json
{"question":"什么是过拟合","answer":"过拟合指在训练数据上表现较好但在新数据上表现较差。","verified":true,"source":"human-reviewed","split":"train"}
{"question":"请解释过拟合","answer":"过拟合指在训练数据上表现较好但在新数据上表现较差。","verified":true,"source":"human-reviewed","split":"dev"}
```

`verified=true` 必须表示数据已经由用户或可信来源审核，而不是模型自行判定；`source` 记录出处。`verified=false`、缺失该字段的候选答案**不会进入训练**。`split` 可以是 `train`、`dev` 或 `auto`：训练集与验证集的规范化问题不得重复。自动分割会尽量把同一答案的不同提问形式分为训练和验证；只有一个问题的知识点无法构成可信的泛化测试，将阻止本次自动训练。

建议每个新知识点提供至少 **2 种不同问法**（更多更好），显式固定留出集，后续不要偷偷把留出样本拿来训练。不要把密码、个人隐私、未授权文档提交到公开仓库。

## 3. 增量学习 + 防遗忘

```bash
python continual_qa.py \
  --data learning_data/verified_qa.jsonl \
  --steps 300 --batch-size 8 --replay-ratio 0.6 \
  --eval-every 50 --device auto
```

训练逻辑：

1. 加载已有监督 QA 检查点；允许新数据增加未见汉字，并只扩展必要的输入与输出词表参数，保留原权重。
2. 每一批把旧训练样本（默认约 60%）和已审核的新训练样本混合。
3. 用**未参与本轮训练**的新提问验证泛化；用旧知识留出集和以往通过验证的留出集评估遗忘。
4. 只有新增留出集精确匹配数量**实际提高**，同时旧正确数量不下降且旧损失未过多增加，才保存候选权重。否则只输出报告，继续使用原权重。
5. 通过后生成 `checkpoints/srt_m15_qa_continual.pt`。默认聊天会优先加载已经存在的 UTF-8 字节 QA 检查点，然后才尝试增量字符 QA；设置环境变量 `SRT_PREFER_CONTINUAL_QA=1` 则优先加载通过门槛的增量字符 QA。没有这些权重时回退至基础字符 QA 和旧语言模型。

这里的“防遗忘”只是对明确评测集的门槛检查，不是数学保证。答案精确匹配过于严格，但适合当前早期、小规模、固定答案的实验。门槛不通过并不意味着模型完全没有学习；意味着没有足够证据允许替换正式权重。

查看 `checkpoints/learning_report.json`：其中保存训练/验证数量、新增字符、旧任务与新任务指标、是否通过以及各题实际预测。已有检查点不包含训练数据本身时，无法全面验证它曾学过的所有知识。

## 4. 测试和问题排查

```bash
python -m unittest discover -s tests -p test_learning_pipeline.py -v
python train_qa.py --eval-only --output checkpoints/srt_m15_qa.pt
```

`continual_qa.py` 需要在 SRT-M15 根目录运行；独立增量学习 Notebook 第 1 步会从 GitHub 下载源码；缺少 QA 检查点时先训练。新问题答案过长超过检查点的 `max_length` 时应缩短数据或重新训练具有更长上下文的模型，不能直接改变旧模型的最大长度而假装兼容。

训练会消耗 CPU/GPU；手机适合用 Colab 控制训练而不是在手机本机运行。模型训练权重与 GitHub 代码是不同的：更新代码并不会自动训练或上传新权重。**本版本不自动把用户的训练数据或权重推送到 GitHub。**

## 5. 后续研究方向

基础语义编码和更大语料的预训练仍是主要瓶颈；扩展到子词 Tokenizer、更严格的跨主题测试、检索式外部知识、可审计的自问自答任务队列，属于后续阶段。不应把外部 API 生成的答案直接当作可靠训练标签。