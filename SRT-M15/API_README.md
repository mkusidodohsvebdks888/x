# SRT-M15 本地 API 服务

本项目在**已训练好的 SRT-M15 模型**外增加 FastAPI HTTP 服务，提供 OpenAI Chat Completions 风格的**基础兼容接口**，不改动 Transformer 网络结构或模型权重。API 服务本身不会让模型变得更聪明。

## 1. 在电脑部署（直接克隆 GitHub 项目）

项目原始底层源码打包在 \`SRT_M15_Phone_Colab.ipynb\` 里。现在提供 \`bootstrap.py\`，无需 Colab 或下载额外资源，就能把底层 Python 模块提取到克隆后的项目目录：

```bash
git clone https://github.com/mkusidodohsvebdks888/x.git
cd x/SRT-M15
python bootstrap.py
python -m pip install -r requirements_api.txt
```

`bootstrap.py` 只提取预设的底层源文件，**不会覆盖**本仓库更新过的 `chat.py`、`deepseek_api.py` 等模块。

注意：训练权重 **不在 GitHub**，请先用本项目 Notebook 或本地 `train_qa.py` 训练，或把之前保存的权重文件放进 `checkpoints/`（常见名称：`srt_m15_qa.pt`、`srt_m15_toy.pt`）。没有语言模型权重时，文本问答 API 会返回 503，并非返回一个假装成功的答案。

## 2. 设置 API Key 并启动

请自行设置**至少 16 字符**的私密密钥，不要写进代码或 GitHub。示例（请替换为随机长密钥）：

Linux/macOS:

```bash
export SRT_API_KEY="your-own-very-long-random-secret"
python -m uvicorn api_server:app --host 127.0.0.1 --port 8000
```

Windows PowerShell:

```powershell
$env:SRT_API_KEY = "your-own-very-long-random-secret"
python -m uvicorn api_server:app --host 127.0.0.1 --port 8000
```

默认只监听**当前电脑**。不要把开发服务直接暴露公网；跨设备访问应增加 HTTPS 反向代理、限速、访问控制和日志审计。

## 3. OpenAI 风格接口

- `GET /health`：查看服务是否启动以及**权重是否已加载在磁盘上**。
- `GET /v1/models`：列出可调用的模型 ID。
- `POST /v1/chat/completions`：提交单条用户消息并获得模型回答。

所有接口均要求 `Authorization: Bearer <SRT_API_KEY>`。

基本请求（请在另一终端执行）：

```bash
curl http://127.0.0.1:8000/v1/chat/completions   -H "Authorization: Bearer your-own-very-long-random-secret"   -H "Content-Type: application/json"   -d '{"model":"srt-m15","messages":[{"role":"user","content":"你是谁"}],"stream":false}'
```

请求成功时返回：

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 0,
  "model": "srt-m15",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "此处为训练后的模型实际生成文本"},
    "finish_reason": "stop"
  }],
  "srt_provider": "SRT-M15",
  "srt_mode": "text"
}
```

**重要限制**：当前只是**非流式、单轮、文本**版本；`stream:true`、多轮历史、system message、工具调用均未实现，不会偷偷忽略历史。对于未支持的参数或模式会给出明确的 4xx 错误，`max_tokens` 最大 80，`temperature` 只能省略或指定 0。某些 OpenAI SDK 可能自动添加未支持参数，接入时需关闭对应功能。

很多支持自定义 OpenAI 接口的软件可以尝试配置：

```text
Base URL: http://127.0.0.1:8000/v1
API Key: <your SRT_API_KEY>
Model ID: srt-m15
Stream: false
```

具体能否使用取决于客户端是否允许关闭流式输出、system 消息和多轮功能。

## 4. 可选：让 API 转发给 DeepSeek

默认 **srt-m15 永远代表 SRT-M15 本地权重**。想启用外部大模型，必须**显式**开启相应开关，并使用单独的模型 ID：

- `SRT_API_ENABLE_DEEPSEEK_API=1`：额外启用 `srt-m15-deepseek-api`，需要 `DEEPSEEK_API_KEY`。
- `SRT_API_ENABLE_DEEPSEEK_WEB=1`：额外启用 `srt-m15-deepseek-web`，需要本机桌面浏览器、Playwright、手动登录，以及可用的 DeepSeek 网页结构。运行前安装 `requirements_web.txt` 并执行 `python -m playwright install chromium`。

网页版不保证稳定，不适用于纯手机 Colab。**不要在共享公网服务器上开启持有个人网页登录会话的浏览器桥接**。外部模型返回会通过 `srt_provider` 清楚表明来源，也不会自动写入本地模型权重。

## 5. 测试与手机 Colab

在本机执行离线测试（不实际访问 DeepSeek）：

```bash
python -m unittest discover -s tests -p 'test_api_server.py' -v
```

Notebook 第 6 步会同步写出最新的 `api_server.py` 和测试文件。Colab 可以在第 12 步用 TestClient 本地测试 API，但 Colab 内 `127.0.0.1` 指向**Colab 云服务器**，并非你的手机。若要让其他设备调用，必须在持续运行的电脑或服务器上部署并建立安全网络入口。

## 6. 后续扩展方向

流式输出 SSE、多轮上下文、模型预热、请求队列、速率限制、请求指标、在线版本发布与测试、生产日志审计。这些尚未实现，请不要将当前基础接口描述成完整 OpenAI API 服务。
