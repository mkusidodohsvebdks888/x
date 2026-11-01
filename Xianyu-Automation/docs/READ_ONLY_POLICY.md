# V1.0 只读安全策略

## 1. 核心原则

程序只读取用户当前主动打开的闲鱼页面，并把识别结果保存到本地。程序不得改变闲鱼页面状态，不得代表用户与买家互动，不得执行交易。

## 2. 允许的行为

| 类别 | 允许内容 |
|---|---|
| 页面事件 | 监听指定包名的 Accessibility 事件 |
| 节点读取 | 读取当前窗口可访问节点的文本和属性 |
| 页面解析 | 识别列表、聊天详情和商品卡片 |
| 数据处理 | 本地解析、匹配、去重、分类 |
| 本地存储 | Room / SQLite 写入识别结果 |
| 用户界面 | 展示、筛选、人工确认或删除本地识别记录 |

## 3. 明确禁止的行为

| 类别 | 禁止内容 |
|---|---|
| UI 写入 | 点击、长按、输入、粘贴、复制、拖动、滚动、手势 |
| 导航 | 自动返回、打开聊天、打开商品、切换页面 |
| 消息 | 自动发送、撤回或回复 |
| 交易 | 下单、付款、确认收货、发货、退款、评价 |
| 平台数据访问 | 私有 HTTP / WebSocket、逆向协议、隐藏接口、Cookie / Token |
| 图像采集 | 截屏、录屏、MediaProjection、OCR |
| 远程通信 | 云端 API、AI API、远程日志、云同步 |
| 设备特权 | Root、ADB input、隐藏系统接口 |

## 4. 代码审查禁用项

项目源码中不得出现用于闲鱼操作的调用：

- AccessibilityNodeInfo.performAction(...)
- AccessibilityService.performGlobalAction(...)
- dispatchGesture(...)
- RemoteInput.addResultsToIntent(...)
- MediaProjection / takeScreenshot(...)
- Runtime.exec("input ...") 或等价 shell 输入
- UIAutomator / Espresso 中用于点击、输入或滚动闲鱼的测试代码

普通 Android 本地数据库写入不属于闲鱼写操作，允许用于保存识别结果。

## 5. 权限最小化

V1.0 只申请实现本地 APP 和读取闲鱼可访问节点所需的权限。不得声明 INTERNET、录音、相机、通讯录、短信、位置或存储全盘访问权限，除非未来另有明确需求并重新审查范围。

Accessibility 服务仅过滤 com.taobao.idlefish。不读取其他 APP 的页面。

## 6. 不确定性处理

- 不可见字段：unknown。
- 平台 ID 未暴露：保持 null。
- 商品匹配冲突：pending_confirmation。
- 页面类型无法确定：UNKNOWN。
- 消息方向不确定：UNKNOWN。
- 不允许通过网络、截图或 OCR 绕过上述限制。

## 7. 变更流程

任何拟新增的权限、网络访问、截图能力、自动化动作或平台接口访问，都必须先：
1. 停止相关实现。
2. 更新需求与 READ_ONLY_POLICY。
3. 明确新增能力、必要性和影响。
4. 获得用户明确确认。
5. 独立设计测试与回滚方式。

未经确认，不得把未来版本能力提前加入 V1.0。
