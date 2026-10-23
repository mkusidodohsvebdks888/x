# SRT-M15 | 自主架构底层模型（可运行实验版）

> **110,000 条双语合成实验数据 + 多批次训练：** 在旧 10k 数据上[新增 100k](DATASET_CARD_BILINGUAL_V4_110000.md)，详见[分批训练说明](README_STAGED_TRAINING.md)。支持可复现生成、十批真实梯度更新、无放回循环、中文/英文固定留出样本评估、断点续训与 GitHub Actions 模型产物。**样本未经逐条人工审核，模板数据不等于真实自然语言能力；本项目原生实验模型还不具备普遍流畅双语对话能力。**


这是一个**从随机权重开始训练**的 PyTorch 神经网络原型，不使用任何外部预训练模型或模型 API。我们自行实现了因果注意力、Transformer Block、多路径递归推理、选择性双向消息、三时间尺度的内部记忆。另附外部版本化 JSON 记忆、受验证保护的在线适配与主动挑选训练样本的示例。

**重要：有两条可运行训练路径。** 一条研究 0~9 整数加法的多分支通信，另一条研究字符级语言生成；两者都不是成熟的聊天大模型。 有些机制已可运行，但自主世界模型、开放领域推理、自动生成可靠监督、元学习和自主发现全新算法都**没有实现**。请勿把它视为已完成或验证的新型通用 AI。

## 1. 文件结构

```text
SRT_M15_Starter/
  srt_m15/model.py       Transformer + 三个循环推理分支 + 动态双向通信 + 三尺度内部记忆
  srt_m15/tasks.py       数字编码、分布式算术任务与验证器
  srt_m15/language.py    字符级自回归语言模型
  srt_m15/storage.py     外部版本化存储（JSON，非语义检索）
  srt_m15/learning.py    在线更新 fast adapter、验证集检查和回滚
  srt_m15/exploration.py 根据熵与多分支分歧选择新实验，由确定性验证器给答案
  train.py               从零训练分布式算术模型
  train_lm.py            从零训练字符语言模型
  generate.py            生成示例文字
  demo.py                推理 + 通信 + 外部记忆 + 主动学习演示
  tests/test_smoke.py    基础单元测试
  phone_colab.ipynb      手机 Colab 自包含源码 Notebook
```

## 2. 本地电脑启动

需要 Python 3.10+，推荐 PyTorch 2.1+。

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
python train.py --steps 180 --task distributed --device auto
python demo.py
python train_lm.py --steps 65 --device auto
python generate.py --prompt "你好" --new-tokens 25
```

快速验证：`python train.py --steps 20 --device cpu`。普通非分布式任务可用 `--task addition --output checkpoints/addition.pt` 训练。

输出位于 `checkpoints/srt_m15_toy.pt`、`checkpoints/metrics.json` 和 `checkpoints/knowledge.json`。训练日志包含训练集与未见过的留出集准确率。留出集准确率可能较低，不能保证泛化。

## 3. 手机运行（Google Colab）

1. 手机上打开 https://colab.research.google.com/，上传 `SRT_M15_Phone_Colab.ipynb`。
2. Notebook **已经嵌入源码**，不需要另外上传 ZIP。
3. 依次运行测试、训练、推理和模型文件下载代码单元。Colab 可能中断。
4. 小模型使用 CPU 即可，也支持可用的云端 GPU。

## 4. 核心架构（实际已写代码的部分）

```text
加法输入（可拆成 A/B/C 三份） -> Token Embedding + Position Embedding
         -> 手写因果自注意力的 Transformer 主干
         -> 3 个参数独立的 GRU 推理分支（循环 3 次）
             |      |       |
             +-- 动态接收/发送相关性打分 --+
             +-- 各特征组选择性消息传递 --+
             +-- 3 个时间尺度的可写入状态 --+
         -> 普通任务：分支合并 -> 19 类答案预测
         -> 分布式任务：仅网络 C 预测（C 仅通过通信获得两边操作数）
         -> 字符语言模型：逐位置预测下一个字符

另有：学习关系评分网络（代理目标，不等于真正的教师收益元学习）
      可验证的算术答案 -> 在线更新各分支小适配器 -> 守护集检查 -> 保留/回滚
      推理不确定性与分支分歧 -> 选择新问题 -> 由确定性算术验证器获取标签
      外部事实 -> JSON 版本历史存储/读取 (不自动接入主干)
```

**动态交流**训练时用连续概率门控，推理时可用阈值启用/关闭边；它不保证自动缩短 GPU 执行时间。**三层记忆**只在显式传入 `MemoryState` 时跨调用延续，和外部 JSON 存储不是同一个东西。**在线学习**只更新每个分支的 `fast_down` / `fast_up`，默认不修改主干参数。真正的教师收益元学习、跨网络梯度学习、自主实验世界模型尚未实现，预留后续研究阶段。

## 5. 关于“真正创造新架构”

代码是**自研实现 + 已知研究思想的实验组合**，不是已经得到科研验证的原创算法。下一步实验应依次测试：

- 单分支 / 三分支、固定消息 / 自由路由、无记忆 / 三尺度记忆、禁用在线更新 / 在线更新。
- 控制训练数据、参数量和实际计算量（包含多分支循环与通信开销），比较留出集准确率、损失、吞吐、推理成本以及遗忘。
- 在更严格任务（多位加法、逻辑任务、不同数据分布、跨任务记忆）上测试，避免 100 道小题导致的结论偏差。

## 6. 安全与隐私

在线学习仅在传入**外部验证过**的正确标签时更新参数，不允许从模型自己的猜测直接写入训练标签。JSON 外部记忆是本地文件，未实现加密、权限、审计、个人信息处理，不要直接用来保存秘密。

许可证：示例代码仅作学习与研究用途。

## 7. 本地实测结果（演示实验，不能外推）

分布式加法模型在 80 道训练题、20 道留出题、180 步训练的测试中：训练集准确率约 **91.25%**，留出集约 **60%**。在演示程序抽取的 10 道守护集样本中，通信开启约 **80%**，关闭通信约 **10%**。注意：这项任务人为分配了信息，目的是验证确实发生了跨网络通信，并不是优于已有 Transformer 的证据。

字符级语言模型在极小、重复的中文实验语料中，65 步训练损失从约 **4.42** 降到约 **0.26**。生成的文字只是统计模式复现，不能证明语言理解或通用推理。

## Local chat with guarded autonomous learning

See [README_LOCAL.md](README_LOCAL.md). The local browser interface provides an **opt-in** queue of externally verified single-digit addition errors; candidate fast-adapter updates are evaluated separately, persisted, and reversible. Unverified natural-language chat is never used as training supervision.

## Local offline autonomous arithmetic practice

The local browser interface can now practice independently on single-digit addition after the user clicks **Start practice**. It generates questions exclusively from the train-eligible split, checks the actual neural network output against a deterministic addition oracle, queues verified errors, performs bounded candidate fast-adapter training, validates the guard split, accepts only improved candidates, persists approved checkpoints, and supports pause and rollback. No external model API or Colab runtime is used. See [README_SELF_PRACTICE.md](README_SELF_PRACTICE.md) for exact controls, limits, and scientific caveats.

## 新增：受控联网自主学习（本地知识记忆，不自动修改权重）

在浏览器页面「联网自主学习」区域，明确授权后可从中/英文维基百科的公开只读 API 获取带来源摘要，自动保存到本地 SQLite，供聊天引用、人工审核与导出。自带间隔、请求数、运行时长和笔记数上限，重启后不会自动联网。

**这不会直接训练中文神经网络参数**；已有的确定性验证加法自主练习仍可独立运行。完整说明：[README_INTERNET_LEARNING.md](README_INTERNET_LEARNING.md)。


## 新增：中文问答与受控参数学习（本地）

见 [`README_LANGUAGE_LEARNING.md`](README_LANGUAGE_LEARNING.md)。此功能在原有聊天输入框增加 UTF-8 字节词表试验、人工校对问答、来源分组的独立验证和语言检查点回滚。原始中文及数学权重不会被覆盖。20条演示问答需要人工逐条批准后才可训练。

## 中英双语受控学习扩展

新版提供中文/English 问答草稿、24组中英配对演示、翻译分组防泄漏、按语言均衡取样、两种语言分别检验及聊天语种选择。完整使用方法见 [`README_BILINGUAL_LEARNING.md`](README_BILINGUAL_LEARNING.md)。已有中文训练数据和原始数学/字符权重保留不变；这仍然是小型学习实验而非可靠的中英对话模型。


## 新增：主流自回归语言模型共同机制训练链路

参见 [README_COMMON_MECHANISMS.md](README_COMMON_MECHANISMS.md)。
在现有手写 Transformer、多分支网络上加入可独立运行的预训练、仅监督助手回答的指令 SFT、独立验证、交叉熵、AdamW、梯度裁剪、采样及可复用检查点：

```bash
python train_common.py --mode pretrain --data examples/common_text_corpus.txt --steps 30
python train_common.py --mode sft --data examples/common_instruction_examples.jsonl --steps 30
python -m unittest discover -s tests -v
```

这是字符级、小数据、从零训练的学习实验，不是 ChatGPT、DeepSeek 等商用大模型的复刻；RLHF、DPO、GRPO、多 GPU 训练和多模态仍不在本次模块范围内。
