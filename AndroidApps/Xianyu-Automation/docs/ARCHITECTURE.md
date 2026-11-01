# 系统架构

## 1. 目标

V1.0 在 Android 手机上运行。用户主动打开闲鱼并操作页面；应用通过只读 AccessibilityService 观察当前窗口，解析可见的聊天和商品卡片，保存本地识别结果。

不建设服务器，不连接闲鱼私有接口，不接入 AI，不执行页面操作。

## 2. 分层结构

    Xianyu Android App
      |
      +-- AccessibilityService
      |     +-- package filter: com.taobao.idlefish
      |     +-- event observer
      |     +-- current-window reader
      |
      +-- Page Snapshot
      |     +-- visible node text
      |     +-- content description
      |     +-- view ID / class / bounds
      |
      +-- Screen Classifier
      |     +-- CHAT_LIST
      |     +-- CHAT_DETAIL
      |     +-- PRODUCT_CARD_VISIBLE
      |     +-- OTHER / UNKNOWN
      |
      +-- Parsers
      |     +-- ChatListParser
      |     +-- ChatDetailParser
      |     +-- ProductCardParser
      |     +-- MessageParser
      |
      +-- Recognition
      |     +-- BuyerRecognizer
      |     +-- ConversationRecognizer
      |     +-- ProductMatcher
      |     +-- IntentClassifier
      |
      +-- Repository
      |     +-- evidence validation
      |     +-- deduplication
      |     +-- pending confirmation
      |
      +-- Room / SQLite
      |
      +-- Local UI

## 3. 数据流

1. 用户手动打开闲鱼或手动切换页面。
2. AccessibilityService 收到闲鱼包名的窗口事件。
3. 服务读取当前可访问窗口的节点树，不执行任何节点动作。
4. Snapshot 层提取必要的可见字段；不保存完整节点树。
5. ScreenClassifier 判断当前页面类型。
6. 对应 Parser 提取买家、消息、商品卡片线索。
7. Recognition 层按证据等级匹配身份和商品。
8. 证据不足的数据标记为 unknown / pending_confirmation。
9. Repository 对结果去重并写入 Room。
10. 本地 UI 展示记录与需要人工确认的项目。

## 4. 页面与事件处理

- 只处理 event.packageName == com.taobao.idlefish 的事件。
- 对连续的 TYPE_WINDOW_CONTENT_CHANGED 事件做短时间合并，避免同一页面重复解析。
- 事件合并仅用于减少重复读取，不通过自动滚动或页面导航获取更多数据。
- 用户手动滚动后，程序可读取当前窗口新出现的可访问节点。
- 页面切换、节点失效、空窗口和权限撤销都必须安全失败。
- 页面未提供完整历史消息时，只记录当前实际可见内容。

## 5. 识别策略

### 买家

优先读取页面明确暴露的稳定平台身份字段；若不可见，只保存本地 UUID 与显示昵称，身份状态为 unverified。

### 会话

平台会话 ID 仅在页面明确提供时记录。否则建立本地会话记录，并保留来源页面和时间线索，不承诺跨页面/跨设备稳定。

### 商品

商品 ID 仅在商品卡片或页面可见属性明确暴露时记录。否则以标题、价格等字段生成候选；候选冲突时必须人工确认。

### 消息意图

V1.0 使用本地关键词 / 正则规则。低确定性归为 UNKNOWN，不调用远程模型。

## 6. 运行与隐私

- 纯本地运行。
- 不声明 INTERNET 权限。
- 不保存截图、录屏、Cookie 或 Token。
- 日志仅记录事件类型、解析结果数量、错误类别和耗时，不记录消息正文与身份数据。
- Room 数据库为运行期数据，不提交到 Git。
- 数据保留和清除方式在 UI 中提供。

## 7. 技术限制

AccessibilityService 只能读取当前系统向服务暴露的节点。闲鱼可能使用自绘控件、WebView、自定义渲染或动态内容，导致节点缺失、字段不稳定或内容不完整。

因此：
- 不保证读取所有聊天消息。
- 不保证获得稳定买家 ID / 会话 ID / 商品 ID。
- 不通过隐藏网络接口补齐数据。
- 无法读取的字段明确标记 unknown。
