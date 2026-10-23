# SRT-M15 无服务器 GPU 分批训练（手机也能启动）

已为原生 SRT-M15 加入 [Colab GPU Notebook](SRT_M15_110k_GPU_Colab.ipynb)。不需要单独购买服务器，GPU 由 Colab 运行时临时提供（**免费 GPU 是否可用、额度、会话时长不保证**）。模型的持久训练检查点储存在**用户授权挂载的 Google Drive**，不是 GitHub 历史或公开网页。

## 手机操作：Colab 训练原生 SRT-M15

1. 用浏览器打开 [Colab](https://colab.research.google.com/)，选择 GitHub → 仓库 `mkusidodohsvebdks888/x` → 文件 `SRT-M15/SRT_M15_110k_GPU_Colab.ipynb`；如果找不到，可先查看 GitHub 中的 notebook 并从 Colab 的 GitHub 入口打开。Colab 需要用户自己登录/授权。
2. 在 Colab 中设置「运行时 → 更改运行时类型 → GPU」。如果没有免费 GPU，可以过段时间再尝试，或者使用有 GPU 的其他环境。
3. 从上到下运行单元格：挂载 Drive → 下载最新 GitHub `main` → 确认 GPU → 本地生成 110k 中英双语合成语料 → 校验训练设置 → 分批训练 → 查看中文、英文分别评估结果。
4. 每一批完成后写入 `MyDrive/SRT-M15/original_110k_gpu/` 的 `latest.pt`、`best.pt`（若有）、`history.json`。下一次 Colab 重新打开、重新挂载**同一个** Drive 目录即可恢复。
5. 默认每次请求新增 10 批，每批 320 步，每步 32 个中/英文平衡的训练窗口；修改 `BATCHES_THIS_SESSION` 可以缩短一次 Colab 会话，**不更改固定的 `STEPS_PER_BATCH`、`BATCH_SIZE` 等续训参数**。

> Notebook 从 GitHub 克隆当时最新版，版本更新可能改变生成器或训练设置；代码内用 **数据 SHA256 + 关键训练超参数** 检查是否允许接续。如果版本改变导致不兼容，请使用新的 Drive 子目录；不要删除或覆盖旧权重。
>
> **注意：** 训练成本主要受上下文长度、批量大小和 GPU 条件影响，而不是只看记录条数。大规模合成数据由模板与可校验数学题构成，尚未经过逐条人工核查。验证 loss 改进≠掌握普遍自然语言。该工作流只训练原生随机初始化 SRT-M15 小网络。

## 我没有 GPU，怎么办？

- 继续用 [GitHub Actions CPU 工作流](../.github/workflows/srt-m15-staged-train.yml) 运行小型验证，它已支持保存检查点。
- 使用 Colab 免费 GPU（实际能否分配不保证）；也可将相同训练脚本放到允许 GPU 的 Kaggle Notebook 中，**但 Kaggle 工作目录不会自动持久保留**，必须手动导出 checkpoint 或保存为受权限控制的数据集并在下次恢复。不要误认为 Kaggle 断线后有自动恢复保障。
- 如果只想日常流畅中英文对话，而不是训练从零模型，可使用 `bilingual_runtime.py` 的开放预训练底座；这与原生 SRT-M15 权重独立。

## Kaggle 手动示例（可选）

在 Kaggle Notebook 启用 GPU，把项目仓库和先前已保存的训练检查点放入输入目录后，先运行：

```bash
git clone --depth 1 https://github.com/mkusidodohsvebdks888/x.git
cd x/SRT-M15
python generate_bilingual_110k.py
python train_bilingual_stages.py \
  --data examples/bilingual_synthetic_v4_110000.jsonl \
  --allow-demo-data --device cuda --sampling cycle \
  --context 256 --batch-size 32 --max-eval-pairs 512 \
  --stages 1 --steps-per-stage 320 \
  --output checkpoints/kaggle_110k
```

继续训练时在相同命令中加上 `--resume <受信任的latest.pt文件路径>`，并确保 **生成器版本、数据哈希及所有固定超参数一致**。每轮的 `latest.pt`、`best.pt` 和 `history.json` 应及时备份到有权限控制的持久存储。

## 安全与成本说明

Colab 与 Kaggle 的免费配额、网络和可用机型随平台变化；本项目不执行无上限 GPU 任务，也不自动购买算力。Drive 授权仅在 Colab 本人账户中进行；项目代码不会读取您的 Gmail 或其他 Drive 文件。当前 Notebook 只写入指定的 `MyDrive/SRT-M15/original_110k_gpu` 训练目录。

**下一阶段**：采集有明确使用许可且更自然、多样的中英语料，建立不共享生成模板的独立问答评估集，再考虑增大模型容量或进行预训练底座的 LoRA 微调。
