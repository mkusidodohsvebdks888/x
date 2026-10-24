# Microsoft Phi 模型（独立目录）

- 所属分类：`大模型/Microsoft-Phi/`
- 与 SRT-M15：**无关联；不共享代码、训练数据或权重**
- 当前 GitHub 内的内容：模型来源、清单与自包含下载脚本（**没有完整权重**）
- 许可：本目录所列模型页面声明 MIT，具体以原始许可证为准

## 已登记的官方模型源

| 名称 | 原发布方模型仓库 |
| --- | --- |
| Phi-4-mini-instruct | [microsoft/Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct) |
| Phi-4 | [microsoft/phi-4](https://huggingface.co/microsoft/phi-4) |

## 独立打包与下载（仅备选工具，不会自动执行）

`models.json` 保留每个版本的官方来源；`download.py` 可在**任何有合适存储容量的环境**按需下载，与本仓库任何其他项目都没有依赖关系。

```bash
pip install huggingface_hub
python download.py --list
python download.py --model Phi-4-mini-instruct --dry-run
python download.py --model Phi-4-mini-instruct --destination /your/storage/Microsoft-Phi/Phi-4-mini-instruct
```

如只需要模型配置、不下载权重，可加 `--metadata-only`。为确保版本可复现，可加 `--revision <完整 commit SHA>`。受限制模型需自行取得访问资格及 `HF_TOKEN`，不要上传任何令牌。

**本目录不是发布方权重镜像，不会自动部署、下载或占用本机硬盘。** 如需再次发布权重，必须另行解决存储限额、许可证和来源完整性校验。
