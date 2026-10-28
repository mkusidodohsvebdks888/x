# 第一版验收

- [ ] GitHub Actions `assembleDebug` 成功，生成真实 APK
- [ ] Android 8.0+ 真机能安装并显示 3 个页面
- [ ] 未设置 API Key 时提示进入设置
- [ ] API Key 设置并保存后能调用 DeepSeek
- [ ] `android_get_status` 返回设备状态
- [ ] `android_open_app` 打开系统设置
- [ ] 安装 Termux、配置 allow-external-apps 和 RUN_COMMAND 后 CLI 查询有结果
- [ ] 未安装 Termux、无授权、超时等情况有明确错误
- [ ] 执行日志可查看、清空，且不记录 API Key
- [ ] 模型无法通过工具执行任意 Shell 命令

安全边界：不主动请求 Root、无障碍、所有文件访问、读取通知、悬浮窗、安装 APK 等权限。

- [ ] Redmi 13C 真机测试 Termux 授权脚本可复制并执行
- [ ] “测试 Termux 连接”返回 TERMUX_OK，成功日志可查看
- [ ] “检查 Termux 环境”显示 Android、架构与系统属性
- [ ] CLI 工具检查、磁盘空间查询均不修改系统文件
- [ ] 下载目录未授权时明确提示 termux-setup-storage
- [ ] Termux 命令非零退出码能标记为失败
- [ ] MIUI/HyperOS 后台限制只提示检查，不强制修改省电策略
