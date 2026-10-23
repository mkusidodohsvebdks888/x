# SRT-M15 在 RTX 3080 上实现可用的本地中英文对话

**本阶段唯一目标：先能正常对话。** 无需再生成百万训练数据，也无需调用 DeepSeek 网页版、租 GPU 或购买 API Key。

> 重要区分：此模式使用本地独立的开源 **Qwen3 4B Instruct 权重**，由 Ollama 在本机加载；SRT-M15 仍是你的项目名称、界面和实验平台，**不是说原生随机初始化的 SRT-M15 网络已经学会了流畅中文英文。** 原有自研模型、参数、学习和训练代码全部保留。页面顶部和每条回复均明确显示底层推理来源。

## Windows + NVIDIA RTX 3080：首次准备

1. 安装官方 [Ollama for Windows](https://ollama.com/download/windows)。确认 `ollama --version` 可运行、后台本机服务就绪。
2. 下载一次本地模型：

   ```powershell
   ollama pull qwen3:4b-instruct
   ```

   该 Q4 模型在 Ollama 模型库标示约 **2.5 GB 下载量**，RTX 3080 的常见 10GB/12GB 显存通常足够运行这一小模型，但实际 GPU 显存占用还与上下文及其他进程有关。通过 `ollama ps` 检查是否有 GPU 加速。可以选择 `qwen2.5:3b` 做轻量备用，但切换时也要下载相应模型。
3. 安装 Python 3.10+（推荐 3.11）并在 SRT-M15 目录运行：

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

   原有 SRT-M15 部分模块用到 PyTorch，因此仍需要本地 Python 依赖。聊天实际推理由 Ollama 服务执行，**不依赖本 Python PyTorch 的 CUDA 安装**。
4. 启动网页聊天（以下两种选一种）：

   ```powershell
   .\.venv\Scripts\python.exe start_local.py --chat-backend ollama
   ```

   或双击 `start_chat_windows.bat`。

5. 浏览器打开 `http://127.0.0.1:8765/`，直接输入中文/英文；支持保存本页内的短期对话上下文（浏览器标签页内，最多保留最近 8 轮历史）。点击“新对话”可以清空历史。网页后端仅监听 `127.0.0.1`，不会让局域网或公网直接访问。

## 模型与数据的区别

- `--chat-backend ollama`：立即使用本机预训练的 Qwen 模型聊天，回复标示“**Ollama 本地 / qwen3:4b-instruct**”。这不是训练自研神经网络。
- 不加 `--chat-backend`：保留 SRT-M15 原生科研版聊天/个位数加法；目前不具备稳定通用会话能力。
- 现有百万数据收集、DeepSeek 网页问答、训练审核、神经网络参数文件**都不会因为你启动 Ollama 就被修改**。
- 不会上传对话到远程模型 API；内容经 HTTP 回环地址发送给本机 Ollama 服务。请不要自行把服务端口暴露到公网。

## 最短验证

开启网页对话后依次问：

1. `你好，你能介绍一下自己吗？`
2. `我叫小明，请记住我的名字。`
3. `我刚才说我叫什么？`
4. `Please explain the difference between Python and Java in two sentences.`

第 3 个问题应正确利用本标签页之前的对话内容。点击“新对话”后再问自己的名字，模型不应该仍假装确定知道。

## 常见问题

**找不到 Ollama**：先在 PowerShell 运行 `ollama --version`。模型必须用 `ollama pull qwen3:4b-instruct` 下载到本机。

**提示无法连接 127.0.0.1:11434**：确认 Ollama 已启动并且没有占用端口。可以运行 `ollama serve`（如果后台服务已运行，不必重复启动）。

**模型未找到**：运行 `ollama list` 确认标签。若切换模型，启动参数 `--ollama-model` 必须与已下载的标签一致。

**运行慢/显存不足**：关闭其他占显存软件；尝试 `ollama pull qwen2.5:3b` 后运行 `python start_local.py --chat-backend ollama --ollama-model qwen2.5:3b`。

**我坚持必须是 SRT-M15 自己训练出来的权重才能聊天**：那是独立的长期研究任务，需要更高质量语言数据、适当的模型大小、分词、训练预算和人工对话验收。不能把 Qwen 代理的输出称为 SRT-M15 训练成果。

## 离线开发测试

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_ollama_local.py" -v
```

该测试用 mock 模拟本机 Ollama，校验本地地址、请求形状、多轮上下文、错误和页面来源提示。**目前仅代码已提交 GitHub，尚未在你的 Windows / RTX 3080 上进行真实启动和效果验收**。
