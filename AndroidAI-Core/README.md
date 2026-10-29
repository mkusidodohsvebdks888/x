# AndroidAI-Core — 安卓 AI 双执行器 Redmi 13C 外卖辅助版

独立 Android 工程。仅包含 **AI 对话、执行记录、设置** 三个界面。参考 [AndroidAssistant (Linodx)](https://github.com/xnylh502/AndroidAssistant) 的 Agent/Termux RUN_COMMAND 架构，不包含其 APK 逆向、Root、Frida、文件工作台、编辑器等扩展功能。

## 当前能力

- DeepSeek 官方 OpenAI 兼容 `chat/completions` API，结构化工具调用，最多 24 轮，默认 `deepseek-chat`（可编辑）。
- `cli_connection_test`：验证 Termux 桥接与真实命令回调。
- `cli_device_info`：使用 Termux 查询手机型号、Android 版本。
- `cli_environment`：查询 Android、Termux 环境、CPU 架构、MIUI/HyperOS 属性。
- `cli_tools`：只读检查 Git、Python、Node、Curl 和 jq 是否可用。
- `cli_disk_space`：查询 Termux 数据目录磁盘空间。
- `cli_list_files`：只允许查看 Termux `home`、`tmp` 或已通过 `termux-setup-storage` 授权的下载目录，拒绝任意路径或任意 Shell 字符串。
- `android_get_status`：Android 系统版本、Termux 安装及权限状态。
- `android_open_app`：通过公开 Android Intent 启动可见 APP；系统设置可通过 `com.android.settings` 打开。
- 简洁本地 UI：对话、日志、设置。日志最多保留最近 100 条，不保留 API 密钥或用户完整提示词。
- DeepSeek API Key 用 Android Keystore AES/GCM 加密保存在设备；工程和 APK 不内置 API Key。
- 无 Root、无通知监听、无自动支付或自动发货、无任意命令执行；无障碍服务仅用于用户主动开启的外卖辅助会话。

## 手机安装

1. 下载 GitHub Actions 构建产出的 APK 或从 [GitHub Releases](https://github.com/mkusidodohsvebdks888/x/releases) 下载 `AndroidAI-Core-v0.5.apk`。**只有构建任务成功发布后才有下载文件**。
2. Android 8.0+ 手机上允许该来源安装 APK，点击安装。
3. 打开 APP → 设置 → 输入自己的 DeepSeek API Key → 保存。
4. 如需 CLI，安装 GitHub/F-Droid 版本 Termux（建议用相同来源安装 Termux 相关插件）。在 Termux 中执行：

   ```sh
   mkdir -p ~/.termux
   echo 'allow-external-apps=true' >> ~/.termux/termux.properties
   termux-reload-settings
   ```

   再回到 APP 设置页授予 `RUN_COMMAND` 权限。
5. 设置页面可直接复制 Termux 授权配置脚本，并在 Termux 粘贴执行；也可以点击“测试 Termux 连接”和“检查 Termux 环境”。失败会显示提示，成功或失败都记录在“执行记录”。
6. 如果要让 Termux 访问下载目录，请在 Termux 自行运行 `termux-setup-storage`，按系统提示授权。APP 不会主动索取所有文件权限。
7. 红米 13C 的 MIUI/HyperOS 系统可能限制后台服务。遇到连接超时，先打开 Termux，再到系统应用设置按需检查 Termux 和 Android AI Core 的后台省电策略。**不保证锁屏后台执行，也不要求 Root**。
8. 在对话输入「查看 Android 状态」「用 Termux 查询设备型号」或「打开系统设置」。

注意：普通 APK 无法获得 `adb shell` 全部权限。Termux 工具执行能力受 Android 后台服务、文件沙盒和授权限制。若未安装/授权 Termux，Android 原生工具仍可用。

## 开发构建

在包含 Android SDK Platform 35、JDK 17、Gradle 8.7 的机器执行：

```sh
gradle :app:assembleDebug
```

安装包输出：`app/build/outputs/apk/debug/app-debug.apk`。

仓库内提供 GitHub Actions 自动构建流程，构建成功后上传 APK artifact 并发布**公开调试版** Release。该调试版使用 CI 的 debug key 签名，仅适合测试；不同 CI 构建之间签名可能变化，升级可能需先卸载旧调试版（请注意备份个人设置）。

## 模块分离

```text
app/src/main/java/dev/androidai/core/
  agent/AgentClient.kt              DeepSeek Tool Calls
  agent/AgentRuntime.kt             多轮 Agent Runtime / 手动恢复
  tools/ToolRegistry.kt             工具清单、权限边界、分发
  termux/TermuxBridge.kt             Termux RUN_COMMAND
  termux/TermuxResultReceiver.kt    异步结果
  storage/SecureSettings.kt         Android Keystore
  storage/ExecutionLogStore.kt      最小执行记录
  ui/MainActivity.kt                三页面 UI
  workflow/PlaybookManager.kt       PokeClaw 原生流程模板管理组件（Apache-2.0）
  workflow/WorkflowVerifier.kt      独立操作核验层
  assets/playbooks/*.md             内置工作流模板
```

这里的 MVP **暂未迁移为“Termux 内启动 HTTP API Server”**；我们保留原项目更简单可靠的 RUN_COMMAND 作为基础闭环，后续仅在确认需要时才讨论 API 桥接。不要误认为已实现后台无人值守服务。

## 开源说明

参考项目 [AndroidAssistant (Linodx)](https://github.com/xnylh502/AndroidAssistant)，原仓库使用 MIT License，版权声明 `Copyright (c) 2026 linbo`。本项目保留许可证及来源说明；与原作者项目独立，不会随上游删除或变更自动改变。


## Redmi 13C / Termux 诊断功能

本版本增加了连接测试按钮、Termux 版本检测、MIUI/HyperOS 系统属性诊断、已安装 CLI 工具检测、Termux 空间查询、下载目录只读浏览及可复制的配置命令。

Termux 必须从官方 GitHub 或 F-Droid 来源安装。不要混用不同签名的 Termux 与其扩展包。
设置页面的“复制基本工具安装命令”只复制 `pkg update && pkg install -y git python curl`，**不会自动执行**。

升级测试 APK 时注意：GitHub Actions 每次构建的调试签名可能不同。若系统提示签名不一致，先备份密钥，再卸载旧版安装新版。

## v0.3.0 外卖 APP 操作（前台单次辅助）

- 设置页新增外卖 APP 选择入口（美团 / 美团外卖 / 饿了么 / 淘宝闪购 / 京东外卖，实际安装包名可能因版本而异）。
- 点击外卖 APP 名称启动**临时授权会话**，20 分钟后自动失效；重启 APP 也会取消。可随时在设置页手动结束。
- 必须由用户在系统设置里手动授权 `Android AI Core` 无障碍服务；授权后该服务在用户发起工具请求之前不会进行任何点击操作。
- DeepSeek 仅在会话有效时获得 `android_ui_scan`、`android_ui_click_text`、`android_ui_search_text`、`android_ui_scroll` 工具。界面读取限用户选定的外卖 APP。
- 每次点击前检查当前前台包名与敏感确认按钮。**订单提交、确认支付、付款、资金操作不自动执行**，遇到相关界面时终止会话，交给用户本人。
- 不支持无障碍树无法访问的特殊绘制页面，无法保证任何版本外卖 APP 都能完成选餐。仅供用户主动操作时辅助，不保证锁屏后台运行。
- 红米 13C 侧载 APK 的无障碍服务可能被 Android 系统标注为“受限制的设置”，请自行确认来源是否可信、按系统授权提示操作，不使用绕过方式。

### 手机端操作示例

1. 在“设置”中配置 DeepSeek API Key。
2. 安装好外卖 APP，在 Android 系统无障碍设置中允许 Android AI Core。
3. 回到 Android AI Core 设置页，选择已安装的外卖 APP，开启本次辅助授权。
4. 回到 Android AI Core AI 对话页输入：“在美团帮我找一份鸡肉饭，选择合适的餐品并准备到确认订单前。”
5. DeepSeek 会尝试打开被选定的外卖 APP 并读取可访问控件。若 APP 只支持图片按钮或需要手动验证，可能需要你自行完成。
6. 到订单提交／支付页面由你本人检查真实商家、商品、配送地址和合计费用，再手动提交与付款。

**API Key 和配送地址都不应该写入 GitHub。不要把短信验证码、支付密码交给 AI。**


## v0.4.0 — Codex 式多轮执行器（Agent Runtime）

本版将原有简单的最多 16 轮循环替换为独立的 `AgentRuntime`。仍使用你配置的 DeepSeek API Key、同一套 Termux 和 Android 工具以及仅有的三个页面。

- 用户输入一次任务后，AI 自动循环“判断 → 调用工具 → 读取结果 → 继续判断”，工具结果通过 `tool_call_id` 返回模型。
- **单次任务上限**：最多 24 轮 DeepSeek 请求、64 次工具调用和 10 分钟；实际任务完成即停止，**并非无限调用 API**。
- **可见进度**：AI 对话页显示当前轮数、工具计数与最近 7 个步骤；执行记录按工具记录成功或失败。
- **失败控制**：网络错误与 HTTP 429/5xx 有最多 2 次有限重试；同一工具、相同参数连续 3 次获得相同结果时停止以防重复操作；不会把失败伪装为完成。
- **停止按钮**：取消协程，防止启动后续 API 和工具步骤。Termux 外部已开始运行的命令未必能立即中止。
- **隐私**：不会持久化完整聊天记录或无障碍页面扫描内容，不记录 API 密钥或搜索词。
- **安全**：完全复用现有工具权限，仍不能自动付款、提交订单、执行任意 Shell、Root 等。外卖辅助依然需要用户临时授权。

**尚未实现**：逐 token 流式生成、任务队列、后台持久任务、崩溃后自动恢复、插件动态安装。这是 Codex *风格的多轮执行*，不是完整的 Codex 客户端。若手机锁屏/切后台，MIUI/HyperOS 可能中止进程。

### 验证连续执行

1. 手机安装 v0.4 测试 APK（与旧版的 CI 调试签名可能不同）。
2. 设置 DeepSeek API Key；需要 Termux 命令时先在系统中授权 Termux。
3. 在 AI 对话输入：“先查询 Android 状态，再检查 Termux 连接和环境，最后汇总结果。”
4. 观察 “第 N 轮 · 已执行 M 个工具” 的进度、执行记录中的工具状态；任务完成后不应继续调用 API。
5. 在执行途中点击“停止”测试取消；触发网络异常时应只有限重试。

## v0.5.0 — 独立完整流程模块（workflow）

**目标**：让用户发出一次请求后，AI 能按流程操作、观察、验证和继续，而不是把某次点击当作整项任务完成。无需增加第四个主页面。

- 在 GitHub 项目中独立创建 `app/src/main/java/dev/androidai/core/workflow/`。
- **完整移植 PokeClaw 的原生 PlaybookManager 组件**，改名空间并去除其 XLog 依赖，加载只读内置 Markdown 流程；依法保留 Apache-2.0 许可证、原始版权与改动说明，详见 `third_party/pokeclaw-playbook/`。
- 新增 `WorkflowVerifier`：每次外卖界面点击、搜索输入或滚动后，使用只读页面快照验证页面是否发生变化，不盲目重试改变状态的操作。
- 新增“蜜雪冰城／美团外卖选餐”和“应用内搜索”流程模板，由 AgentRuntime 自动匹配。流程严格要求先确认饮品与规格、查询真实界面、核验购物车，到提交订单或付款时交还给用户。
- **人工接管／恢复**：页面短暂加载自动限时重读；验证码、登录、无障碍断开或无法验证加购结果时暂停。返回应用点击“返回美团并继续（先检查页面）”后重新扫描，再由模型决定下一步。
- 每轮最多执行一次修改界面的工具操作；模型要求的多次点击会分解到后续轮次，避免基于旧页面连续盲点。
- 单任务仍限制 24 轮 API、64 个工具调用和 10 分钟，不开放任意 Shell、Root、验证码绕过、自动提交订单或付款。

**完整项目选择说明**：PokeClaw 是完整的独立 Android 应用，不能在不重构其应用 ID、模型/权限运行时与服务的情况下直接嵌入另一 APK。本次采用其可独立移植的完整工作流组件，而不是把整个 PokeClaw APK 当作插件嵌入。Mobilerun/Open-AutoGLM 通常需要外部主机或 ADB 运行时，也没有直接嵌入当前手机 APK。

**真实能力边界**：构建成功仅证明工程可编译，不代表已经在红米 13C 的实际美团页面跑完任务。自绘页面、页面文字变化不明显、系统后台策略仍可能中断；未确认的点击会要求人工核对。模块在 APP 进程重启后不能恢复内存检查点。详见 `docs/WORKFLOW_MODULE.md`。
