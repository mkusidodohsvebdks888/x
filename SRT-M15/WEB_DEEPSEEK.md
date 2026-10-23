# SRT-M15 接入 DeepSeek 网页版（无需 API Key）

SRT-M15 本地神经网络与 DeepSeek 是不同的模型。远程回答必须清楚标记来源，**远程回答不会自动更新 SRT-M15 的权重**。

## 方式 A：本地电脑直接操作 DeepSeek 网页（无需 API Key）

网页版入口：https://chat.deepseek.com/

需要 Windows/macOS/Linux 图形桌面、Python 和 Chromium 浏览器。首次安装：

```bash
pip install playwright
python -m playwright install chromium
```

运行以下两条命令（在已提取完整 SRT-M15 项目的文件夹）：

```bash
python deepseek_web_cli.py --login
python deepseek_web_cli.py --ask "请介绍神经网络"
```

首次登录会打开可见浏览器，你亲自登录 DeepSeek，登录信息只保存在本机独立浏览器配置目录（默认 `~/.srt_m15/deepseek_web`）。桥接仅操作网页可见输入框并读取可见回答，**不通过逆向工程请求网页私有 API、不导出 cookie、不处理验证码**。网页版页面改版、登录限制、网络中断、回复超时都可能导致调用失败。

普通聊天也可以在本地 Python 使用：

```python
from srt_m15.chat import SRTChatEngine

engine = SRTChatEngine(backend="web")
reply = engine.reply("什么是 Transformer")
print(reply.provider, reply.text)
```

`backend="local"` 只使用 SRT-M15；`backend="web"` 使用网页版；`backend="api"` 使用官方 API；`backend="auto"` 优先在简单计算和身份问题使用本地网络，其余使用 `SRT_EXTERNAL_BACKEND`（默认为 `api`）。`/raw 你是谁` 始终测试本地神经网络原始生成。

**重要：网页桥接不适用于手机浏览器+Google Colab。** Colab 代码运行在远程服务器，不能自动控制你手机上登录的网页。手机 Colab 需要稳定自动调用时，建议官方 DeepSeek API，或者通过你自己电脑上运行的授权后端转发（必须另建服务，不能将本机浏览器调试端口暴露到公网）。

## 方式 B：官方 API（可用于 Colab）

如已有 DeepSeek API Key，可使用环境变量 `DEEPSEEK_API_KEY`，无需把密钥写入 GitHub：

```python
import os
from getpass import getpass
os.environ["DEEPSEEK_API_KEY"] = getpass("DeepSeek API Key: ")
from srt_m15.chat import SRTChatEngine
print(SRTChatEngine(backend="api").reply("你好").text)
```

API 基础地址为 `https://api.deepseek.com`，本项目使用 `deepseek-flash` 默认模型（可用 `DEEPSEEK_MODEL` 环境变量修改）。API 与网页版额度分开。

## 训练自主回答（无需 DeepSeek）

在 Colab 执行第 9 步的 `train_qa.py`，让 SRT-M15 通过答案监督学习。训练完成后重新打开第 8 步聊天框。问答准确度由实际评测决定，小实验模型不等于通用大模型。
