# Mistral AI 模型（独立目录）

- 所属分类：`大模型/Mistral/`
- 与 SRT-M15：**无关联；不共享代码、训练数据或权重**
- 当前 GitHub 内的内容：模型来源、清单与自包含下载脚本（**没有完整权重**）
- 许可：本目录所列模型页面声明 Apache-2.0，具体以原始许可证为准

## 已登记的官方模型源

| 名称 | 原发布方模型仓库 |
| --- | --- |
| Mistral-7B-Instruct-v0.3 | [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) |
| Mistral-Small-3.1-24B | [mistralai/Mistral-Small-3.1-24B-Instruct-2503](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503) |

## 独立打包与下载（仅备选工具，不会自动执行）

`models.json` 保留每个版本的官方来源；`download.py` 可在**任何有合适存储容量的环境**按需下载，与本仓库任何其他项目都没有依赖关系。

```bash
pip install huggingface_hub
python download.py --list
python download.py --model Mistral-7B-Instruct-v0.3 --dry-run
python download.py --model Mistral-7B-Instruct-v0.3 --destination /your/storage/Mistral/Mistral-7B-Instruct-v0.3
```

如只需要模型配置、不下载权重，可加 `--metadata-only`。为确保版本可复现，可加 `--revision <完整 commit SHA>`。受限制模型需自行取得访问资格及 `HF_TOKEN`，不要上传任何令牌。

**本目录不是发布方权重镜像，不会自动部署、下载或占用本机硬盘。** 如需再次发布权重，必须另行解决存储限额、许可证和来源完整性校验。
