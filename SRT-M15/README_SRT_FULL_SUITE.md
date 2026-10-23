# SRT-M15 | Qwen 教师数据 → 本地 GPU 训练 → 中文英文对话：完整 Windows 套件

这个套件面向 **Windows + NVIDIA RTX 3080**，将此前散落的 Qwen/Ollama 接入、双语数据生成、人工审核、训练、检查点、原生聊天和效果检查整合为一组**可执行的本地命令及一个 Windows 菜单**。不使用远程 DeepSeek API，也不需要将训练数据上传到 GitHub。

## 关键事实

- **马上能聊天**：使用本机安装的 **Qwen3 4B Instruct** 作为对话引擎，运行 `python srt_full_pipeline.py chat`；它不是 SRT-M15 原生参数。
- **真正更新 SRT-M15 参数**：先把本地 Qwen 的回答整理成训练示例，经人工审核，调用 `python srt_full_pipeline.py train ...` 在 RTX 3080 上训练原生 SRT-M15。原生字符级模型目前非常小，**不会因为此流程就变得和 Qwen 一样流畅**。
- **数据并未预生成**：仓库保存源代码、测试和验收案例；你的电脑需安装并运行 Ollama，才会实际产生 Qwen 的教学数据。所有生成数据默认留在本地 `local_data/`。
- **不自动认证回答**：Qwen 的输出不是事实真值，审核和核查数据使用权是显式步骤；不允许后台自动从未审核草稿训练“正式权重”。

## 目录与组成

```text
SRT-M15/
  SRT_M15_FULL_SUITE_WINDOWS.bat    # 双击打开主菜单
  srt_full_pipeline.py             # 一套 CLI 管理器
  qwen_teacher_data.py             # 从本地 Ollama 收集双语问答
  review_deepseek_learning.py      # 人工审核门禁（现兼容本地 Qwen）
  merge_reviewed_sft.py            # 合并多个已审数据批次
  train_bilingual_stages.py        # 真正更新原生 SRT-M15 PyTorch 参数
  chat_staged.py                   # 原生 SRT-M15 聊天实验
  evaluate_student_chat.py         # 独立双语聊天人工验收日志
  start_local.py                   # 图形化本地 Ollama/Qwen 聊天界面
  srt_m15/ollama_local.py          # 本机 127.0.0.1:11434 Ollama 适配
  examples/conversation_acceptance_v1.jsonl # 不参与训练的对话验收题
  local_data/                     # 本地生成，git 忽略
  checkpoints/                    # 本地权重；不能当作自动学习证明
```

## 0. Windows 首次安装

在电脑安装 Git、Python 3.11、Ollama，并从仓库更新：

```powershell
git clone https://github.com/mkusidodohsvebdks888/x.git
cd x\SRT-M15
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
ollama pull qwen3:4b-instruct
```

如果项目此前已克隆，运行 `git pull origin main`，不要重复克隆。上述 PyTorch 安装是否支持 CUDA 取决于安装源及 NVIDIA 驱动，先运行诊断命令确认。如果 `torch.cuda.is_available()` 是 `False`，需要根据本机驱动安装官方支持的 CUDA 构建，再进行 GPU 训练；**聊天模式使用 Ollama 的 GPU，不依赖 Python PyTorch 是否识别 CUDA**。

双击 `SRT_M15_FULL_SUITE_WINDOWS.bat` 即可看到聊天、生成、人工审核、GPU 训练和验收的菜单。所有选项都按需同步运行，按 Ctrl+C 可中断当前命令，不会自动创建后台任务。

## 1. 环境自检

```powershell
.\.venv\Scripts\python.exe srt_full_pipeline.py doctor
```

应查看 Python、PyTorch、CUDA 和本机 Ollama 模型名单。**这一步只检查，完全不训练，也不上传数据。**

## 2. 立即使用中文/英文聊天（无需训练）

```powershell
.\.venv\Scripts\python.exe srt_full_pipeline.py chat
```

浏览器打开 `http://127.0.0.1:8765/`，可连续对话并使用“新对话”清空历史。回复显示来源 `Ollama 本地 / qwen3:4b-instruct`，不能错当成 SRT-M15 自研模型的训练成果。

## 3. 本地 Qwen 生成第一批真实教学草稿

```powershell
.\.venv\Scripts\python.exe srt_full_pipeline.py generate --pairs 40 --offset 0 --preview
.\.venv\Scripts\python.exe srt_full_pipeline.py generate --pairs 40 --offset 0
```

第一条仅预览、不会向 Ollama 发出请求；第二条会尝试生成 **40 组中英文示例，上限 80 条记录**，输出：

```text
local_data/qwen_teacher/batch_000000_000040.jsonl
local_data/qwen_teacher/batch_000000_000040.manifest.json
```

数据格式为 JSONL，每条包含 `language`、`prompt`、`response`、`source`、`split` 等，并显式标记 `verified:false`、`review_status:unreviewed`。实际成功数请看 `manifest.json`，不允许虚报 80 条。

第二批示例（使用不同索引避免直接重复）：

```powershell
.\.venv\Scripts\python.exe srt_full_pipeline.py generate --pairs 40 --offset 40
```

生成器当前使用有限主题及模板，批次超过一定数量后会出现大量语义重复，因此**不要靠无限变更索引凑 100 万条**，应逐步引入经过授权的自然多轮对话语料和高质量人工数据。

## 4. 人工审核、剔除不可靠回答

先扫描：

```powershell
.\.venv\Scripts\python.exe srt_full_pipeline.py review --drafts local_data/qwen_teacher/batch_000000_000040.jsonl --reviews local_data/qwen_teacher/reviews.jsonl
```

实际批准某条样本时，用扫描输出的 ID 代替下面示例：

```powershell
.\.venv\Scripts\python.exe review_deepseek_learning.py --drafts local_data/qwen_teacher/batch_000000_000040.jsonl --reviews local_data/qwen_teacher/reviews.jsonl approve "qwen-pair-000000-zh" --rights-confirmed --evidence "我已人工审阅该答案、独立核对内容，并检查数据的使用许可"
```

如果答案有问题，应该先改写审核，或使用 `reject` 标记拒绝。程序只能自动检查格式、部分隐私和回声重复；真实性、自然程度及授权须由人检查。**不允许为节约步骤而对所有 Qwen 输出批量打上 verified=true**。

导出审核通过的数据：

```powershell
.\.venv\Scripts\python.exe srt_full_pipeline.py export --drafts local_data/qwen_teacher/batch_000000_000040.jsonl --reviews local_data/qwen_teacher/reviews.jsonl --output local_data/qwen_teacher/reviewed_sft.jsonl
```

如果中文或英文的有效样本不足、train/dev 划分失效，程序会拒绝导出。要把不同批次合成一个已审数据集，请分别审核/导出后运行：

```powershell
.\.venv\Scripts\python.exe srt_full_pipeline.py merge --input local_data/qwen_teacher/reviewed_batch_001.jsonl local_data/qwen_teacher/reviewed_batch_002.jsonl --output local_data/qwen_teacher/reviewed_merged.jsonl
```

注意：审核多个批次时，请为每个批次指定**不同的 reviews.jsonl**。因为对原始草稿的核验与其 ID/指纹绑定，单独导出后再合并最清晰。以上批次文件名仅为示例，必须以实际产出的文件为准。

## 5. RTX 3080 GPU 真正训练自研 SRT-M15

当 `reviewed_sft.jsonl` 已实际生成并通过严格校验后：

```powershell
.\.venv\Scripts\python.exe srt_full_pipeline.py train --data local_data/qwen_teacher/reviewed_sft.jsonl --device cuda --steps 60 --stages 2
```

训练器进行反向传播并真实更新自研模型权重。每阶段记录中英文独立 dev 结果，按验证损失选择是否提升最佳版本，输出：

```text
checkpoints/qwen_student/latest.pt
checkpoints/qwen_student/best.pt (仅有更优候选时出现)
checkpoints/qwen_student/history.json
```

如果仍处于演示开发阶段，需要检验“Qwen 生成 → 训练”代码管线，但尚未完成人工审核，可显式使用 `demo-train`（**不应视为可信模型训练成果**）：

```powershell
.\.venv\Scripts\python.exe srt_full_pipeline.py demo-train --data local_data/qwen_teacher/batch_000000_000040.jsonl --device cuda --steps 20
```

如要继续相同训练作业、保持**同一数据文件内容及所有超参数**：

```powershell
.\.venv\Scripts\python.exe srt_full_pipeline.py train --data local_data/qwen_teacher/reviewed_sft.jsonl --device cuda --steps 60 --stages 2 --resume
```

**严禁把不同数据集的 checkpoint 直接 resume**，因为字符词表和数据哈希不匹配。每次迭代数据不同，采用新作业目录，或以后升级为固定 tokenizer 的增量训练器。

## 6. 使用原生 SRT-M15 参数聊天并人工验收

```powershell
.\.venv\Scripts\python.exe srt_full_pipeline.py student
.\.venv\Scripts\python.exe srt_full_pipeline.py evaluate
```

原生 `student` 模式加载 SRT-M15 自身训练后的 `best.pt` 或回退 `latest.pt`，不会调用 Qwen；`evaluate` 输出到 `local_data/evaluation/student_acceptance.json`，其中包含 12 项中英文独立对话场景及**待人工评估**回复。缺失权重或不能生成文本必须明确报错，不能宣称验收通过。

## 7. 运行离线自动测试

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_qwen_teacher_data.py" -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_srt_full_pipeline.py" -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_review_deepseek_learning.py" -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_ollama_local.py" -v
```

测试通过也只能证明**代码与模拟输入的兼容性**，不能替代在 RTX 3080 上的 Ollama 真人交互、训练性能、独立准确率和显存检测。

## 完整流程概要

```text
                     Windows + RTX 3080
                            │
            ┌───────────────┴──────────────┐
            │                              │
       立即可用聊天                  训练原生模型
            │                              │
     Ollama 本地 Qwen            Qwen 生成双语教学草稿
            │                              │
    SRT-M15 网页对话                  人工审核和导出
                                           │
                                     train_bilingual_stages
                                           │
                                      保存 latest/best
                                           │
                               原生 SRT-M15 chat_staged
                                           │
                                 独立验收和继续迭代
```

**目标边界**：你提出的“能够正常对话”在短期内使用本地已训练的 Qwen 可以实现；对于“原生 SRT-M15 从零训练后独立流畅对话”，这套脚本是研究和训练基础设施，并不承诺成熟模型的语言能力。
