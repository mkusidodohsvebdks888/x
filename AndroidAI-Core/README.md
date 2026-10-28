# AndroidAI-Core — 安卓 AI 双执行器最小版

独立 Android 工程。仅包含 **AI 对话、执行记录、设置** 三个界面。参考 [AndroidAssistant (Linodx)](https://github.com/xnylh502/AndroidAssistant) 的 Agent/Termux RUN_COMMAND 架构，不包含其 APK 逆向、Root、Frida、文件工作台、编辑器等扩展功能。

## 当前能力

- DeepSeek 官方 OpenAI 兼容 `chat/completions` API，结构化工具调用，最多 6 轮，默认 `deepseek-chat`（可编辑）。
- `cli_device_info`：使用 Termux 的真实 Shell 查询手机型号、Android 版本。
- `cli_list_files`：只允许查看 Termux `home` 或 `tmp`，拒绝任意路径或任意 Shell 字符串。
- `android_get_status`：Android 系统版本、Termux 安装及权限状态。
- `android_open_app`：通过公开 Android Intent 启动可见 APP；系统设置可通过 `com.android.settings` 打开。
- 简洁本地 UI：对话、日志、设置。日志最多保留最近 100 条，不保留 API 密钥或用户完整提示词。
- DeepSeek API Key 用 Android Keystore AES/GCM 加密保存在设备；工程和 APK 不内置 API Key。
- 无 Root、无无障碍服务、无通知监听、无自动支付或自动发货、无任意命令执行。

## 手机安装

1. 下载 GitHub Actions 构建产出的 APK 或从 [GitHub Releases](https://github.com/mkusidodohsvebdks888/x/releases) 下载 `AndroidAI-Core-v0.1.apk`。**只有构建任务成功发布后才有下载文件**。
2. Android 8.0+ 手机上允许该来源安装 APK，点击安装。
3. 打开 APP → 设置 → 输入自己的 DeepSeek API Key → 保存。
4. 如需 CLI，安装 GitHub/F-Droid 版本 Termux（建议用相同来源安装 Termux 相关插件）。在 Termux 中执行：

   ```sh
   mkdir -p ~/.termux
   echo 'allow-external-apps=true' >> ~/.termux/termux.properties
   termux-reload-settings
   ```

   再回到 APP 设置页授予 `RUN_COMMAND` 权限。
5. 在对话输入「查看 Android 状态」「用 Termux 查询设备型号」或「打开系统设置」。

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
  tools/ToolRegistry.kt             工具清单、权限边界、分发
  termux/TermuxBridge.kt             Termux RUN_COMMAND
  termux/TermuxResultReceiver.kt    异步结果
  storage/SecureSettings.kt         Android Keystore
  storage/ExecutionLogStore.kt      最小执行记录
  ui/MainActivity.kt                三页面 UI
```

这里的 MVP **暂未迁移为“Termux 内启动 HTTP API Server”**；我们保留原项目更简单可靠的 RUN_COMMAND 作为基础闭环，后续仅在确认需要时才讨论 API 桥接。不要误认为已实现后台无人值守服务。

## 开源说明

参考项目 [AndroidAssistant (Linodx)](https://github.com/xnylh502/AndroidAssistant)，原仓库使用 MIT License，版权声明 `Copyright (c) 2026 linbo`。本项目保留许可证及来源说明；与原作者项目独立，不会随上游删除或变更自动改变。
