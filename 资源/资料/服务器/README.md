# 🖥️ 免费服务器资源

> 更新核查：2026-10-27｜内容：官方入口、免费条件、部署用途和主要限制。  
> **特别说明**：免费服务器 ≠ 免费 GPU；“永久免费额度”也不等于保证永不回收、稳定 24 小时可用。所有名额、价格、可用地区和资格以官方实时页面为准。

[← 返回资料](../) · [← 返回资源](../../)

## 🟢 1. 长期免费云主机（有虚拟机 / 可运行自定义后端）

| 平台 | 免费资源 | 适用场景 | 注意事项 | 官方信息 |
| --- | --- | --- | --- | --- |
| **Oracle Cloud Always Free** | Ampere A1 ARM：每月合计 **1,500 OCPU 小时 + 9,000 GB 内存小时**，相当于约 **2 OCPU / 12GB RAM**；另有 Always Free 存储额度 | Linux、Docker、FastAPI、小型量化 CPU 模型 | 无免费 GPU；受区域库存、账户验证与闲置实例回收政策影响；架构为 ARM | [官网](https://www.oracle.com/cloud/free/) · [官方说明](https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm) |
| **Google Cloud Compute Engine Free Tier** | 指定美国地区每月累计约 **1 台 e2-micro** 的使用时长、**30GB 标准持久磁盘**和有限出站流量 | 轻量 API、反向代理、定时任务 | 只在指定地区和规格内免费；无 GPU；超出网络、存储或其他额度可能计费 | [官网](https://cloud.google.com/free) · [官方免费条款](https://docs.cloud.google.com/free/docs/free-cloud-features) |

## 🟡 2. 免费 Web / API 托管（不等于独立 VPS）

| 平台 | 免费资源 | 适用场景 | 主要限制 | 官方信息 |
| --- | --- | --- | --- | --- |
| **Render** | Free Web Service；每个 workspace 每月合计 **750 免费实例小时** | Python/Node.js 后端、演示 API | 约 15 分钟无请求会休眠；重启后临时文件不持久；免费数据库另有限制 | [官网](https://render.com/) · [免费计划说明](https://render.com/docs/free) |
| **Koyeb** | 每个组织 **1 个免费 Web 实例**，约 **0.1 vCPU / 512MB RAM / 2GB SSD** | 小型 Flask/FastAPI、Webhook | 资源小；不是 GPU；区域与资格有限制 | [官网](https://www.koyeb.com/) · [规格说明](https://www.koyeb.com/docs/reference/instances) |
| **Cloudflare Workers** | 免费套餐目前约 **100,000 次请求/天** | API 网关、Token 鉴权、代理转发、轻量逻辑 | 是 Serverless，不是 Linux 虚拟机；CPU 时间/内存受限，不能运行完整大模型 | [官网](https://workers.cloudflare.com/) · [限制](https://developers.cloudflare.com/workers/platform/limits/) |
| **Hugging Face Spaces** | Static Spaces 对所有用户免费；CPU Basic 硬件标价免费，但创建运行 Gradio/Docker 的计算型 Space 受账户套餐限制 | 模型展示、网页演示 | 免费账号不能默认获得无限制的计算型托管服务；需看 ZeroGPU 资格 | [官网](https://huggingface.co/spaces) · [Spaces 文档](https://huggingface.co/docs/hub/spaces-overview) |

## 🟠 3. 免费 GPU 算力（主要用于实验，不是 24 小时独占 GPU 服务器）

| 平台 | 免费或试用额度 | 适合 | 主要限制 | 官方信息 |
| --- | --- | --- | --- | --- |
| **Kaggle Notebooks** | 免费 P100 等 GPU；官方文档举例为 **每周约 30 小时**，可随资源调整 | 训练、微调、Notebook 实验 | 会话有时限、需排队；不适合长期公网 API | [Notebook](https://www.kaggle.com/code) · [GPU 使用说明](https://www.kaggle.com/docs/efficient-gpu-usage) |
| **Google Colab** | 可免费申请 GPU/TPU 运行时，但**没有保证的固定时长和 GPU 型号** | 模型实验、训练、量化 | 免费计算受动态配额、运行时间与反滥用规则限制 | [Colab](https://colab.research.google.com/) · [官方 FAQ](https://research.google.com/colaboratory/faq.html) |
| **Lightning AI** | 免费 CPU Studio；新用户最多 **30 Lightning credits**（通常注册先得 5，添加付款卡可能再得 25）；官网称最多约 **80 GPU 小时**，依 GPU 价格而变 | 小模型微调、训练、实验 | 试用积分非永久免费；免费 Studio 需定期重启；使用付费资源前核查计费 | [官网](https://lightning.ai/) · [计费说明](https://lightning.ai/pricing) |
| **Hugging Face ZeroGPU** | 符合条件的免费个人账号最多可托管 **2 个 ZeroGPU Spaces**；共享 48GB/96GB GPU；免费用户的使用配额约 **5 分钟/天** | Gradio AI 演示、短时间推理 | 共享排队、单次调用有约束；不可当作持续独占 GPU | [官方文档](https://huggingface.co/docs/hub/spaces-zerogpu) |

## 🎓 4. 学生云资源 / 申请资助

| 渠道 | 用途 | 说明 |
| --- | --- | --- |
| **GitHub Student Developer Pack** | 学生可申请云资源、开发工具和合作伙伴优惠 | 需通过学生身份审核；包含的优惠会更新，并不保证赠送大型 GPU。[查看官方优惠](https://education.github.com/pack/) |
| **Hugging Face Community GPU Grant** | 为创新性公开 Space 申请额外算力 | 可从 Space 的 Settings 中查找 GPU 资助申请，审核通过与资源规格均不保证。[官方介绍](https://huggingface.co/docs/hub/spaces-overview) |

## 🤖 5. 如果是为了部署 DeepSeek 模型

| 目标 | 优先考虑 | 预期 |
| --- | --- | --- |
| 免费搭一个自己的 API 地址 | Render / Koyeb / Cloudflare Workers | 运行后端或 API 网关，不代表自带免费 DeepSeek 推理算力 |
| 长期运行较小的量化模型 | Oracle Always Free（CPU）/ 自己的电脑 | 性能依具体模型、CPU 指令集和内存而变；大模型不适用 |
| 训练 / 微调较小的 DeepSeek 蒸馏模型 | Kaggle / Colab / Lightning AI | 按免费额度分批实验，保存权重 |
| 给别人演示 GPU 模型 | Hugging Face ZeroGPU | 有共享 GPU 与配额限制 |
| 独立部署**完整 DeepSeek-V4.1-Flash** | 大型多 GPU 服务器、科研 GPU 资源或算力赞助 | **上述常规免费套餐不足以保证完整模型长期运行** |

## ✅ 使用前检查

- 确认“永久免费额度”“首次注册送积分”“限时试用”属于哪一种。
- 核对付款卡要求；为云账单设置预算、限额和告警。
- 关注 CPU 架构、内存、GPU 显存、磁盘、流量与对外开放端口。
- 免费平台通常可能休眠、回收、重启或限流；不要依赖它存放唯一一份模型权重。
- 永远不要把 API Key、登录 Cookie、服务器密码写入公开 GitHub 仓库。
- **本目录仅收集官方资源链接与使用限制，不代表已为该 GitHub 账号注册或领取这些服务。**

---

**维护约定**：新增免费服务器时，请写清官方链接、免费资源、时效、配置、是否支持 GPU、适合什么用途和重要限制。建议每次实际使用前复核官方说明。
