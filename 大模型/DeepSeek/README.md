# DeepSeek 模型（独立目录）

- 所属分类：`大模型/DeepSeek/`
- 与 SRT-M15：**无关联；不共享代码、训练数据或权重**
- 当前 GitHub 内的内容：模型来源、清单与自包含下载脚本（**没有完整权重**）
- 许可：具体以各模型官方 LICENSE 为准（蒸馏模型还应核对底座许可）

## 已登记的官方模型源

| 名称 | 原发布方模型仓库 |
| --- | --- |
| R1-Distill-1.5B | [deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) |
| R1-Distill-7B | [deepseek-ai/DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) |
| R1-Distill-Llama-8B | [deepseek-ai/DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B) |
| R1-671B | [deepseek-ai/DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) |
| V3-671B | [deepseek-ai/DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3) |

## 独立打包与下载（仅备选工具，不会自动执行）

`models.json` 保留每个版本的官方来源；`download.py` 可在**任何有合适存储容量的环境**按需下载，与本仓库任何其他项目都没有依赖关系。

```bash
pip install huggingface_hub
python download.py --list
python download.py --model R1-Distill-1.5B --dry-run
python download.py --model R1-Distill-1.5B --destination /your/storage/DeepSeek/R1-Distill-1.5B
```

如只需要模型配置、不下载权重，可加 `--metadata-only`。为确保版本可复现，可加 `--revision <完整 commit SHA>`。受限制模型需自行取得访问资格及 `HF_TOKEN`，不要上传任何令牌。

**本目录不是发布方权重镜像，不会自动部署、下载或占用本机硬盘。** 如需再次发布权重，必须另行解决存储限额、许可证和来源完整性校验。
