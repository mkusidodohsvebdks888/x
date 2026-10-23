# SRT-M15：流式 API + 多轮消息 + 2GB Linux 服务器 HTTPS

**当前交付：** GitHub `main/SRT-M15` 已增加严格的 `/v1/chat/completions` **SSE 真正增量流式输出**、无服务端持久状态的**多轮消息输入**、独立的分批训练模型 `srt-m15-staged` 加载路径，并保留既有单轮 API。本文给出可用于 **2GB RAM** 的保守部署示例，但**尚未替用户登录服务器、配置域名或启动公网服务**。

## 1. 支持范围

| Model ID | 来源 | `stream:true` | `messages` 多轮 | 说明 |
| --- | --- | --- | --- | --- |
| `srt-m15` | 旧版 SRT 原生已安装语言权重 | ✅ 真正逐 token 解码 | ✅ 文本拼接，受词表与上下文限制 | 单轮非流式保留原本数学/文本分流 |
| `srt-m15-staged` | 原生多批训练 `latest.pt` / `best.pt` | ✅ 真正逐字符解码 | ✅ 实验性文本历史 | **仅在服务器配置 `SRT_STAGED_CHECKPOINT` 后出现** |
| `srt-m15-deepseek-api` | 显式配置的外部 DeepSeek API | ✅ 转发真实上游 SSE delta | ✅ 角色消息原样发给上游 | 外部模型能力**不是** SRT 自研权重 |
| `srt-m15-deepseek-web` | 显式配置的桌面网页桥接 | ❌ | ❌ | 不在公网服务器推荐启用 |

注意：本地小模型没有 GPT/Claude 那样经过多轮聊天训练的语义能力。这里的「多轮」只是把你发送的历史格式化为模型的文本输入，**不会自动替用户跨请求保存会话**。客户端每轮必须提交完整的最近历史。若字符词表缺失、历史超过小模型上下文（例如分批模型 `context=256`）或超过 API 字符预算，则返回 `422` 或 `400`，**不会偷偷删掉历史或假装理解系统指令**。

- 单条消息限 400 字符；最多 32 条消息，所有消息正文合计限 600 字符。
- 消息必须以用户结尾，并按「可选的最前面 1 条 system → user → assistant → user …」交替排列。
- `max_tokens` 限 1–80。当前仅允许省略 `temperature` 或显式设置为 `0`；本地增量推理使用确定性最大概率 token。
- SSE 帧遵循 `data: {"object":"chat.completion.chunk",...}`；流完成发 `data: [DONE]`。若流启动后推理失败，则发送 `event: error`，不能把前面的 HTTP 200 当作回答成功。
- 本地对话同时仅执行一个推理（使用锁），避免 2GB 内存下多个请求叠加。请求高峰可能排队，生产环境应另设访问限速。

## 2. 本地启动（先验证、再配置公网）

```bash
git clone https://github.com/mkusidodohsvebdks888/x.git
cd x/SRT-M15
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt -r requirements_api.txt
export SRT_API_KEY="请替换成由密码管理器生成的至少32位随机字符串"
# 服务器上已放入 GitHub Actions 训练的 latest.pt 后再启用：
export SRT_STAGED_CHECKPOINT="/path/to/trusted/latest.pt"
python -m uvicorn api_server:app --host 127.0.0.1 --port 8000 --workers 1
```

不设置 `SRT_STAGED_CHECKPOINT` 则只提供旧版本地模型 `srt-m15`；如果旧版语言权重也没安装，聊天请求会报 `503`。**GitHub Actions artifact 不会被 API 自动拉取**，需要自己下载 `latest.pt` 或 `best.pt` 并安全上传服务器。只加载自己训练生成、来源可信的权重文件。

检查模型列表：

```bash
curl -s http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer $SRT_API_KEY"
```

### 非流式多轮请求

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $SRT_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"srt-m15-staged","stream":false,"max_tokens":48,"messages":[
    {"role":"user","content":"你好"},
    {"role":"assistant","content":"你好！"},
    {"role":"user","content":"刚才的问候是什么？"}
  ]}'
```

注意：上述多轮输入可能超过原生小模型的词表/可用上下文，导致 `422`；即使能生成字符也未必回答正确。可先从简单单轮开始测试。

### 真正的流式 SSE 请求

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $SRT_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"srt-m15-staged","stream":true,"max_tokens":24,
       "messages":[{"role":"user","content":"你好"}]}'
```

`curl -N` 关闭客户端缓冲，便于观察逐 token/字输出。客户端需要忽略首帧 `role`、拼接后续 `delta.content`，在 `[DONE]` 后结束；若收到 `event: error` 应标记请求失败。

## 3. 你的 2GB 内存服务器：推荐拓扑

```text
浏览器 / 第三方应用
         │
         ▼
   https://api.example.com
         │
         ▼
   Caddy (80/443, HTTPS 自动证书)
         │
         ▼
   Uvicorn 127.0.0.1:8000（仅本机）
         │
         ▼
   FastAPI + SRT-M15 小模型（1 个 worker, 串行推理）
```

**可行，但有前提：**
- **原生 SRT-M15 小模型**的 CPU 推理 + FastAPI + Caddy，一般适合先在 2GB RAM 上小并发测试。
- 最好关闭不必要的常驻程序，保留系统内存，使用 `--workers 1`，必要时配置 1–2GB swap（swap 不是 RAM，速度明显慢）。
- 仓库自带的 `Qwen2.5-0.5B-Instruct` 默认 CPU **float32** 加载，光权重就接近 2GB，连同 PyTorch、词表、操作系统和反向代理通常**放不进 2GB RAM**。不要在 2GB 服务器直接启动该默认精度版本；若确实要使用，应采用显式量化模型 + 内存实测，或者外部推理 API。
- **2GB RAM 不适合训练大型语言模型**；训练继续交给 GitHub/Colab，服务器仅承载小模型推理或中转外部 API。
- 2GB 是内存（RAM）假设。若你说的是 2GB **硬盘空间**，则需要额外评估依赖安装、缓存及日志空间。

## 4. Ubuntu/Debian 部署示例（需要你在服务器执行）

提前将 DNS `api.example.com` 的 A/AAAA 记录解析到你的服务器，并在安全组与防火墙放通 **80、443**，确保没有其他服务占用这些端口。Caddy 会在具备域名和网络可达性的情况下自动申请、续期 HTTPS 证书。

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git caddy
sudo useradd --system --create-home --home-dir /opt/srt-m15 --shell /usr/sbin/nologin srtm15
sudo -u srtm15 git clone --depth 1 https://github.com/mkusidodohsvebdks888/x.git /opt/srt-m15/repo
sudo -u srtm15 python3 -m venv /opt/srt-m15/.venv
sudo -u srtm15 /opt/srt-m15/.venv/bin/pip install -r /opt/srt-m15/repo/SRT-M15/requirements.txt -r /opt/srt-m15/repo/SRT-M15/requirements_api.txt
```

将可信的 `latest.pt` 放在 `/opt/srt-m15/repo/SRT-M15/checkpoints/staged_latest.pt`，由 `srtm15` 用户可读。

新建 `/etc/srt-m15.env`（**权限必须受保护**，不要提交到 GitHub）：

```ini
SRT_API_KEY=请替换成至少32位随机字符串
SRT_MODEL_ROOT=/opt/srt-m15/repo/SRT-M15
SRT_STAGED_CHECKPOINT=/opt/srt-m15/repo/SRT-M15/checkpoints/staged_latest.pt
```

```bash
sudo chown root:root /etc/srt-m15.env
sudo chmod 600 /etc/srt-m15.env
sudo cp /opt/srt-m15/repo/SRT-M15/deploy/srt-m15-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now srt-m15-api
sudo systemctl status srt-m15-api
```

将 [`deploy/Caddyfile.example`](deploy/Caddyfile.example) 复制到 `/etc/caddy/Caddyfile`，并把域名占位符 `api.example.com` 替换为真实域名：

```bash
sudo caddy validate --config /etc/caddy/Caddyfile
sudo systemctl enable --now caddy
sudo systemctl reload caddy
```

然后从外部客户端测试 `https://你的域名/v1/models`。**只有确认 HTTPS 访问成功后**，才把接口给第三方软件使用。不要把 `127.0.0.1:8000` 或带密钥的 HTTP API 暴露到公网。

### 2GB RAM 的额外保护

提供的 `deploy/srt-m15-api.service` 将服务内存限制为 `1500M`，只启动一个 Uvicorn worker，并保留少量并发余量。可查看资源：

```bash
free -h
sudo systemctl status srt-m15-api
sudo journalctl -u srt-m15-api -n 50 --no-pager
```

如果因资源不足被杀掉，不应反复自动扩容内存；应先测量消耗，降低并发，必要时迁移推理后端。日志和 API 响应不应回传敏感的密钥或完整异常堆栈。

## 5. 与 OpenAI 兼容客户端的真实区别

目前只实现 **Chat Completions API 的一部分**，并非完整 OpenAI API 服务；不提供 `/v1/responses`、Tools/function calls、图片语音、服务端消息历史数据库、完整 Tokens usage、并行推理，客户端发送额外字段也会报参数错误。部分 SDK 默认发送 system / developer、tools 或 `temperature>0`，须相应调整。

**模型权重与 API 来源必须分开：** `srt-m15-staged` 加载你自行训练产生的原生 SRT 权重；`srt-m15-deepseek-api` 则是对 DeepSeek 的代理调用，不能宣传为原生 SRT 的新能力。

最新程序：[api_server.py](api_server.py)、[srt_m15/api_streaming.py](srt_m15/api_streaming.py)、[srt_m15/deepseek_api.py](srt_m15/deepseek_api.py)。
