# Xianyu-Automation — Codex / 开发代理约束

## 项目目标

这是 Android 原生、纯本地、只读的闲鱼页面识别项目。V1.0 只读取用户当前主动打开的闲鱼页面，提取可见买家、会话、消息和商品卡片线索，并保存到本地数据库。

## 不可突破的只读边界

- 只监听 com.taobao.idlefish 的 Accessibility 事件。
- 只允许读取 AccessibilityEvent、rootInActiveWindow 和节点只读属性。
- 禁止调用 AccessibilityNodeInfo.performAction()。
- 禁止调用 AccessibilityService.performGlobalAction()。
- 禁止调用 dispatchGesture()、takeScreenshot()、MediaProjection 或 OCR。
- 禁止执行点击、长按、输入、粘贴、复制、滚动、返回、打开页面等 UI 操作。
- 禁止使用 RemoteInput、ADB input、UIAutomator 点击或其他自动化工具写入闲鱼页面。
- 禁止实现发消息、付款、下单、确认发货、退款、评价、商品编辑等操作。
- 禁止调用闲鱼私有 HTTP / WebSocket 接口、逆向协议或读取 Cookie / Token。
- V1.0 不添加 INTERNET 权限，不添加远程 API、云同步或模型调用。
- 不为补齐缺失数据而暗中加入截图、OCR、网络请求或后台页面导航。

## 允许的读取

- 读取节点的 text、contentDescription、viewIdResourceName、className、boundsInScreen、isVisibleToUser 等属性。
- 遍历节点的子节点；处理空节点、不可见节点和异常。
- 用户手动打开会话或手动滚动后，读取当前窗口新出现的可访问内容。
- 将必要的结构化识别结果写入本地 Room / SQLite。
- 在本地界面展示识别结果、证据和待确认状态。

## 身份与匹配规则

- 昵称不得作为买家唯一键。
- 本地 UUID 不得伪装成平台买家 ID、会话 ID 或商品 ID。
- 只有页面明确暴露时才记录平台 ID。
- 标题和价格匹配仅用于生成候选；存在歧义时必须进入人工确认。
- 不确定的消息方向、商品关联或身份统一标记为 unknown / pending_confirmation。
- 不得编造置信度、平台字段或测试结果。

## 隐私与日志

- 日志不写入聊天正文、买家昵称、商品完整内容、平台 ID、Cookie 或 Token。
- 不保存完整 Accessibility 节点树；只保存完成识别所需的最小字段。
- 不保存截图或录屏。
- 测试数据使用本人控制的测试账号与会话；提交前脱敏。
- 数据库只存本机；提供清除本地数据的方式。不要把运行期数据库、日志、导出数据提交到 Git。

## 开发与测试流程

1. 修改前检查本目录 README 与 docs。
2. 新增任何 Accessibility 能力前，先更新 READ_ONLY_POLICY.md 并检查是否仍符合只读边界。
3. 为解析器使用脱敏的本地 fixture / mock 测试，不要求真实账号。
4. 真机测试必须由用户手动打开闲鱼、手动切换会话；程序只读取页面。
5. 每项测试记录通过 / 失败 / 未测和证据。没有设备时标记未测。
6. 每次交付说明新增原因、项目帮助、推进内容、验证结果；区分已写代码、已编译、已安装、已真机验证。

## 禁止误报

- 不得声称通知/无障碍事件已收到完整聊天记录。
- 不得把 rootInActiveWindow 可读取等同于所有闲鱼页面都可读取。
- 不得把昵称、控件 ID、节点路径或本地 UUID 当成平台稳定身份。
- 不得把商品标题相似当作商品 ID 已确认。
- 不得把本地解析测试说成闲鱼真实页面验收。
