# SRT-M15：将本地 Qwen 的回答整理为训练数据

这套功能采用 **Teacher → Student（教师模型数据蒸馏）**：让已经下载在本机的 Qwen 回答一组中英文问题，保存为 SRT-M15 可读取的指令微调（SFT）草稿。

**明确区分：** Qwen 生成的文字 ≠ Qwen 训练语料 ≠ Qwen 权重。你获取的是 Qwen 针对具体问题生成的示例答案，**不会复制其模型参数**，更不能保证从零训练的 SRT-M15 因而达到 Qwen 的语言能力。

## 版权、来源和真实性

- 当前所用的 `qwen3:4b-instruct` 对应模型在官方 Ollama 模型库注明 Apache-2.0，Qwen 开源仓库同样描述 Qwen3 权重采用 Apache-2.0。仍应保留 Qwen/阿里云的作者归属、原模型版本信息并查看具体许可证、模型卡和适用要求。
- 本功能只调用**本机** Ollama 的可见本地接口 `http://127.0.0.1:11434/api/chat`；不会爬取 Qwen 网页、搜集其私人训练数据、伪造源数据许可。
- 教师回答有可能含有错误、偏见或重复内容。全部输出 `verified:false`、`review_status:unreviewed`，不会自动进入经过验证的正式训练集。
- `local_data/` 已被 `.gitignore` 忽略，因此生成的数据默认不会提交 GitHub。

## 1. 在 Windows + RTX 3080 本机准备

安装并启动 Ollama；首次下载模型：

```powershell
ollama pull qwen3:4b-instruct
```

在 GitHub 项目 `SRT-M15` 目录中运行：

```powershell
git pull origin main
python -m pip install -r requirements.txt
```

## 2. 生成第一批双语教师回答

先**离线预览**要问 Qwen 的题目：

```powershell
python qwen_teacher_data.py --dry-run --pairs 20
```

真正调用本地 Qwen：

```powershell
python qwen_teacher_data.py --pairs 20
```

这表示 **20 组成对的中英文任务，共最多 40 条问答**，并不是 20 条总记录。默认输出：

```text
local_data/qwen_teacher/qwen_sft_unreviewed.jsonl
local_data/qwen_teacher/qwen_sft_unreviewed.manifest.json
```

再次生成前，请指定不同的输出文件，以免覆盖：

```powershell
python qwen_teacher_data.py --pairs 100 --output local_data/qwen_teacher/batch_002.jsonl
```

单次批次限制在 5～200 个双语任务（最多 400 条记录），可多批次运行。当前内置的有限主题和模板会产生近似重复；它们**不能代替大规模高质量自然语言语料**。如果 Qwen 不可用或数据格式不符合检查，脚本会停止或跳过，不会假装已生成成功。

## 3. 人工检查 Qwen 数据并挑选合格训练样本

现有审核程序 `review_deepseek_learning.py` 已扩展为**兼容本地 Ollama 数据**，且会保留原始来源，避免把 Qwen 数据错标成 DeepSeek。

```powershell
python review_deepseek_learning.py --drafts local_data/qwen_teacher/qwen_sft_unreviewed.jsonl --reviews local_data/qwen_teacher/reviews.jsonl scan --limit 20
```

逐条检查事实、对话自然程度、信息完整性、是否适合用于训练、来源授权，并审核通过某条记录：

```powershell
python review_deepseek_learning.py --drafts local_data/qwen_teacher/qwen_sft_unreviewed.jsonl --reviews local_data/qwen_teacher/reviews.jsonl approve "<样本ID>" --rights-confirmed --evidence "人工核对答案和来源归属，并确认可用于本地训练"
```

该命令需要把占位符 `<样本ID>` 换为扫描时输出的真实 ID。对于不适用的问题，可以使用 `reject "<样本ID>" --reason "原因"`。未审核记录绝不会自动变成验证训练样本。

当中英文样本**都达到原训练器的最低要求**后：

```powershell
python review_deepseek_learning.py --drafts local_data/qwen_teacher/qwen_sft_unreviewed.jsonl --reviews local_data/qwen_teacher/reviews.jsonl export --output local_data/qwen_teacher/reviewed_sft.jsonl
```

输出中的来源标记将指向 `human-reviewed-local-ollama:qwen3:4b-instruct`，用于追溯训练样本。

## 4. 用审核通过的 Qwen 教师数据训练原生 SRT-M15

如果你已经有足够的已审核中英文示例：

```powershell
python train_bilingual_stages.py --data local_data/qwen_teacher/reviewed_sft.jsonl --device cuda --batch-size 4 --context 128 --stages 1 --steps-per-stage 20 --max-eval-pairs 16 --output checkpoints/qwen_student_test
```

训练完可以测试：

```powershell
python chat_staged.py --checkpoint checkpoints/qwen_student_test/latest.pt
```

这里 **CUDA 实际更新的是 SRT-M15 自研 PyTorch 模型权重**；Qwen 仅负责提前生成文字答案。由于原生模型极小且随机初始化，这个测试只有 20 步，用于验证管线，**不意味着 SRT-M15 从此会流畅聊天**。

为了尽快获得真正可用的中文英文聊天，仍可同时运行 `python start_local.py --chat-backend ollama`，其回答来自独立本机 Qwen 模型，不是自研权重。

## 5. 真实进度

已编写并提交到 GitHub 的内容：

- `qwen_teacher_data.py`：使用本地 Qwen 生成双语 SFT 草稿，支持拒绝基本低质回答、检测重复、模型来源标识、固定 train/dev 分区和 SHA-256 清单。
- `review_deepseek_learning.py`：现在保留 Qwen 和 DeepSeek 的不同来源。
- `tests/test_qwen_teacher_data.py`：使用模拟本机 Qwen 进行离线单元测试，覆盖生成、审核路径及数据限制。

**尚未**在你的 RTX 3080 电脑上实际调用 Ollama、生成真实 Qwen 答案、审核数据或训练权重。执行完上述命令并检查 manifest 后，才能统计实际有效样本数。