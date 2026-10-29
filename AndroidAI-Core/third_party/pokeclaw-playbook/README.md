# PokeClaw PlaybookManager 移植声明

来源：PokeClaw（agents-io/PokeClaw）：https://github.com/agents-io/PokeClaw
原文件：app/src/main/java/io/agents/pokeclaw/agent/PlaybookManager.kt
原版权：Copyright 2026 PokeClaw (agents.io). All rights reserved.
许可证：Apache License Version 2.0，见本目录 LICENSE。

移植修改：修改 Kotlin package，XLog 替换为 Android SDK Log，移除包含完整用户问题的日志，增加 playbook 文件名/ID/长度检查；仅加载 APK 内置 assets/playbooks。参照其 open-and-search 模板的工作方式，将模板改写为 AndroidAI-Core 的现有工具命名；具体文本并非原文件逐字复制。

只移植完整可重用的 PlaybookManager 组件，不包含 PokeClaw 的整款 APK/本地大模型/其他依赖。主项目仍为独立 AndroidAI-Core。本项目不代表 PokeClaw 官方认可。
