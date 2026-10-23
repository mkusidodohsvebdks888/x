# SRT-M15 下一阶段：发现知识缺口 → 自主提问 → 人工审核 → 本地训练

已新增两个**离线**工具：

1. `diagnose_local_gaps.py`：用独立人工准备的 JSONL 测试原生 `SRT-M15` 已训练的 staged 权重，把可能回答欠佳的问题写入候选清单。它使用简单词汇重合度做**问题定位**，不判断答案真实性。
2. `review_deepseek_learning.py`：读取 `autonomous_deepseek_web.py` 的本地草稿，标记格式/敏感信息风险，要求人工审核依据、来源使用许可确认后才能导出已审核的数据；审核记录追加写入，不覆盖原草稿。

## 第 1 步：可选本地评估，发现问题

准备 `local_data/manual_eval.jsonl`，每行包含 `language`、`prompt` 和由你核实的 `reference`：

```json
{"language":"zh","prompt":"你是谁？","reference":"我是 SRT-M15，一个实验性人工智能模型。"}
{"language":"en","prompt":"Who are you?","reference":"I am SRT-M15, an experimental AI model."}
```

运行：

```bash
python diagnose_local_gaps.py --checkpoint checkpoints/local_dialogues/latest.pt --cases local_data/manual_eval.jsonl
```

输出 `local_data/deepseek_web_questions/local_gaps.jsonl` 仅包含问题，不包含参考答案。模型、字表及 checkpoint 必须已在你本机训练完成。本脚本不访问 DeepSeek、不修改权重。

**诊断的局限**：同义改写可能使字符重合度很低；某些错误答案也可能分数较高。该工具只能筛出值得人工查看的候选，不能当作真实正确率报告。

## 第 2 步：由 SRT-M15 向 DeepSeek 网页自主提问

登录必须由你自己在可见浏览器完成：

```bash
python deepseek_web_cli.py --login
python autonomous_deepseek_web.py --dry-run --topics "机器学习,英语" --max-questions 5
python autonomous_deepseek_web.py --start --topics "机器学习,英语" --max-questions 5
```

来源为 `DeepSeek Web`，默认输出草稿至：

```text
local_data/deepseek_web_questions/drafts.jsonl
```

当前提问模块只支持主题计划或可选原生小模型提问，**不会自动把 `local_gaps.jsonl` 作为网页发送队列**。可先根据诊断结果人工选择安全的学习主题。严禁把密码、私人对话等敏感信息通过主题发送给网站。

## 第 3 步：逐条检查并明确批准

先扫描（不会修改任何数据）：

```bash
python review_deepseek_learning.py scan --limit 20
```

命令会输出记录 ID、问题、回答预览和格式风险。你需要另外核查事实正确性、数据使用权、个人信息和是否含错误指导，不能只根据自动扫描决定批准。

确认某条记录后：

```bash
python review_deepseek_learning.py approve "<草稿ID>" --rights-confirmed --evidence "已独立核对原始资料来源、答案正确性与数据使用权"
```

如果答案需要人工重新整理，可以先保存 UTF-8 文件，再加 `--answer-file path/to/edited_answer.txt`。若网页原回答被截断，该条禁止直接导出；必须重新获取完整材料后形成新的可靠审核记录。

如回答不可靠：

```bash
python review_deepseek_learning.py reject "<草稿ID>" --reason "无法证实结论"
```

审核记录保存在 `local_data/deepseek_web_questions/reviews.jsonl`，重新审核同一记录会以最后一条有效决定为准。编辑或替换原草稿后，旧批准会因指纹不匹配而失效。

## 第 4 步：导出训练数据

在中文、英文分别至少各有若干条经人工审批的数据后：

```bash
python review_deepseek_learning.py export --output local_data/deepseek_web_questions/reviewed_sft.jsonl
```

只有明确人工批准、检查来源使用权并记录审核依据的记录才会以 `verified:true` 输出。导出会调用原有 `bilingual_data.py` 的正式校验和 train/dev 拆分，任何语言数量不足或记录不合法都会失败；不会伪造通过。

**注意**：`verified:true` 表示人工审核流程完成，而不意味着所有答案都可用作普遍事实证明；尤其不能仅因 DeepSeek 回答流畅就批准。

## 第 5 步：小批量本地训练与人工验收

```bash
python train_bilingual_stages.py --data local_data/deepseek_web_questions/reviewed_sft.jsonl --context 128 --batch-size 2 --stages 1 --steps-per-stage 20 --output checkpoints/reviewed_deepseek_demo
python chat_staged.py --checkpoint checkpoints/reviewed_deepseek_demo/latest.pt
```

这是在**已审核小数据集上从零训练原生 SRT-M15 实验网络**，不是让现有权重无条件融合 DeepSeek 能力，也不是百万数据完整训练。原先来自不同数据/词表的 staged checkpoint 无法直接 `--resume` 到这个新数据集。

比较训练前后的中英文独立问题测试成绩，检查错误、过拟合、多轮记忆，再决定继续迭代。**仅训练 Loss 下降不能证明能正常对话。**

## 风险边界与测试

- 永远不自动批准模型回答，不在无人监管下把 DeepSeek 输出变成训练标签。
- 网页失败即停止，避免重复发送；自主网页请求数量和时长限制保持不变。
- 不将本地账户、浏览器 Cookie、密码、私人数据、训练草稿和审核日志提交 GitHub。
- 当前测试文件：`tests/test_review_deepseek_learning.py`。建议本地运行：

```bash
python -m unittest discover -s tests -p "test_review_deepseek_learning.py" -v
```

上述测试只覆盖离线数据审核与导出逻辑，不能替代网页联通测试、许可审查或模型实际训练验证。
