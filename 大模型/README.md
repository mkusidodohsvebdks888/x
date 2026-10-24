# 大模型：各厂商独立模型归档目录

本目录与 `SRT-M15/` **无联系、无依赖，也不集成任何模型**。每家模型按独立文件夹分类；这里只提交**来源清单、官方链接、独立下载脚本**，目前**没有把任何大模型权重上传到 GitHub**。

> GitHub 常规 Git 文件上限为 100 MiB，而实际权重多为 GB/TB 级；不能把一个下载脚本或模型链接当成已经上传的模型本体。模型权重建议保留在原发布方的 Hugging Face/ModelScope，或另行使用有配额的模型/对象存储。

## 分类

| 独立分类 | 厂商 | 已登记模型数 | 状态 |
| --- | --- | ---: | --- |
| [DeepSeek](DeepSeek/) | DeepSeek | 5 | 独立官方源目录 |
| [Qwen](Qwen/) | 阿里 Qwen | 4 | 独立官方源目录 |
| [OpenAI-gpt-oss](OpenAI-gpt-oss/) | OpenAI gpt-oss | 2 | 独立官方源目录 |
| [Meta-Llama](Meta-Llama/) | Meta Llama | 2 | 独立官方源目录，部分需授权 |
| [Google-Gemma](Google-Gemma/) | Google Gemma | 2 | 独立官方源目录，部分需授权 |
| [Mistral](Mistral/) | Mistral AI | 2 | 独立官方源目录 |
| [Microsoft-Phi](Microsoft-Phi/) | Microsoft Phi | 2 | 独立官方源目录 |
| [ZAI-GLM](ZAI-GLM/) | 智谱 / Z.ai GLM | 2 | 独立官方源目录 |
| [Kimi](Kimi/) | Moonshot AI Kimi | 1 | 独立官方源目录 |
| [MiniMax](MiniMax/) | MiniMax AI | 1 | 独立官方源目录 |
| [Yi](Yi/) | 零一万物 Yi | 2 | 独立官方源目录 |
| [GPT-4o-API-only](GPT-4o-API-only/) | OpenAI GPT-4o | 0 | **非开放权重，仅说明，不可下载** |

## 使用说明

1. 进入某个厂商子目录查看 `README.md` 和 `models.json`，其中的链接直接指向原发布方。
2. 每个目录自带独立 `download.py`，可自行指定**外部服务器、网盘挂载或其他存储路径**，不会自动运行。
3. `catalog.json` 是全目录机器可读索引。文件为官方入口清单，不代表本 GitHub 仓库托管了任何模型权重。
4. **GPT-4o 不提供可公开下载的权重**；OpenAI 可下载的是 [gpt-oss](OpenAI-gpt-oss/) 系列，两者不同。

## 为什么不直接上传模型权重？

GitHub 普通 Git 拒绝单文件超过 100 MiB 的提交；Git LFS 也受单文件大小、存储/流量配额等限制，且当前连接器不支持将多 GB Hugging Face 权重传入 LFS。本次仅上传可维护的独立模型**分类包**。如将来需要存储权重，宜使用 Hugging Face 模型仓库或专门的对象存储，并根据各模型许可决定能否再分发。

## 重要约定

- 各模型的架构、参数、训练数据、许可证和版本均相互独立，**不会被融合为 SRT-M15 的模型能力**。
- 不把 API Key、HF_TOKEN、用户凭据提交到 GitHub。
- 每个模型的许可/门禁状态以模型原始发布页面为准；“开放权重”不必然等于“OSI 开源许可证”。
- 下载源来自已列明的官方发布账号；本次未验证每个模型快照的哈希，也未实际下载权重。
