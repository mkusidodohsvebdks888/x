# AndroidAI-Core 工作流模块（v0.5.0）

## 模块目标

用户一次描述目标后，系统自动在“观察 → 规划 → 执行 → 再观察 → 核验 → 继续或暂停”之间循环。
不仅询问模型是否完成，还根据实际工具结果决定是否继续。执行失败或无法核验时优先暂停，不重复有副作用的操作。

## 复用的开源组件

PokeClaw / PlaybookManager（Apache-2.0），源文件路径：
https://github.com/agents-io/PokeClaw/blob/main/app/src/main/java/io/agents/pokeclaw/agent/PlaybookManager.kt

它负责从 APK 本地 assets/playbooks 解析带 YAML frontmatter 的 .md 流程，匹配用户任务。
移植组件为完整的 playbook 管理器代码，经过包名和日志适配；它不是完整 PokeClaw APP。
原版权与许可证在 third_party/pokeclaw-playbook 下。

## 工作流目录

- workflow/PlaybookManager.kt：加载、校验、匹配本地流程模板。
- workflow/WorkflowVerifier.kt：一次 UI 动作前后读取页面，有限等待、状态反馈。
- assets/playbooks/takeout-meituan.md：美团／蜜雪冰城选餐流程。
- assets/playbooks/search-in-app.md：应用内搜索流程。

AgentRuntime 在任务开始时加载最匹配流程的说明，和安全系统提示一起发送给 DeepSeek。
ToolRegistry 在 Android 执行器上增加操作后核验，但不扩展权限。

## 外卖场景示例

“帮我在美团选择蜜雪冰城一杯饮品，放购物车，到提交订单前停止。”

1. 如果没给出饮品种类或规格，DeepSeek 应询问用户。
2. 核查外卖授权和无障碍，打开用户选择的外卖 APP。
3. 读取真实页面，找到搜索入口。每次 UI 动作后必须重新读取屏幕。
4. 精确选店、选餐和规格，观察购物车变化。
5. 点击已执行而页面无可验证变化时停止自动操作，用户检查后才能继续。
6. 到登录验证码、地址、订单提交或付款时交由用户处理，不能静默越过。

## 恢复和取消

页面加载阶段仅尝试重读，不重复点击。页面不匹配、无障碍断开、无法确认是否加购等产生可恢复中断：
按 APP 的“返回美团并继续（先检查页面）”，先重新观察并继续由模型规划；不会重放上次操作。
权限失效或敏感付款页面不可自动恢复。
用户点击停止则取消后续工具循环。

恢复点为当前 APP 进程内的内存对象，不包含跨进程持久恢复，也不支持锁屏后台无人值守。

## 风险与边界

对第三方外卖 APP 的适配没有任何“保证成功”的承诺。若控件不提供无障碍文本，识别会失败；
目前不支持视觉截图、OCR、坐标盲点、验证码绕过、自动提交订单、代付。
真实商品价格和是否加入购物车只能从可访问 UI 获取并验证，不可编造。
执行日志不保存页面原文、支付信息和 API 密钥；建议不要在用户任务中输入地址和密码。
