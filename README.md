# SRT-M15 实验项目

本仓库包含 **SRT-M15**：从随机权重开始训练的小型 Transformer 研究原型，含多分支协作、简单算术任务、字符级文本生成、问答训练、可选的 DeepSeek 网页/API 桥接，以及新增的**本地 HTTP API 服务**。

> SRT-M15 并不是预训练通用大模型，也不能因接入 DeepSeek 就把 DeepSeek 的回答视为自己的模型能力。

## 入口

- [手机 Google Colab Notebook](SRT-M15/SRT_M15_Phone_Colab.ipynb)：独立包含底层 Python 源代码，按步骤测试与训练。
- [SRT-M15 本地 API 部署指南](SRT-M15/API_README.md)：本地启动、Bearer Key、`/v1/models`、`/v1/chat/completions`。
- [DeepSeek 网页/API 接入指南](SRT-M15/WEB_DEEPSEEK.md)：可选外部模型，不会自动更新本地权重。
- [本地模型 API 源代码](SRT-M15/api_server.py)：FastAPI 非流式、单轮的基础兼容接口。
- [问答监督训练](SRT-M15/train_qa.py)：将问答训练和推理分开。

## 电脑本地快速开始

```bash
cd SRT-M15
python bootstrap.py
python -m pip install -r requirements_api.txt
```

准备 `checkpoints/srt_m15_qa.pt` 等训练权重后，设置 `SRT_API_KEY` 环境变量，再运行：

```bash
python -m uvicorn api_server:app --host 127.0.0.1 --port 8000
```

请参阅 API 部署指南获取 Windows、Linux/macOS 的密钥配置方法和接口调用示例。

**安全提示：** 默认只监听本机，不要在公网直接运行开发服务，也不要把 API Key、第三方凭证或网页登录配置目录上传至仓库。
