# AccessibilityService 设计

## 1. 服务定位

AccessibilityService 是 V1.0 的主要页面读取入口。它只观察用户主动打开的闲鱼页面，不替用户操作闲鱼。

目标包名：com.taobao.idlefish。

## 2. 允许监听的事件

- TYPE_WINDOW_STATE_CHANGED
- TYPE_WINDOW_CONTENT_CHANGED
- TYPE_VIEW_SCROLLED
- TYPE_VIEW_TEXT_CHANGED

事件只作为“页面可能变化”的信号。收到事件后，确认包名并读取当前窗口；不得根据事件自动执行导航或滚动。

## 3. 只读节点属性

允许读取：

- text
- contentDescription
- viewIdResourceName
- className
- boundsInScreen
- isVisibleToUser
- isClickable（仅读取属性，不执行点击）
- childCount / getChild(index)

读取节点后及时释放或避免长期持有节点引用；内部 Snapshot 仅保存必要字段。

## 4. 服务配置参考

以下是后续 Android 工程的配置参考，不代表服务已实现：

    <accessibility-service
        android:accessibilityEventTypes="typeWindowStateChanged|typeWindowContentChanged|typeViewScrolled|typeViewTextChanged"
        android:packageNames="com.taobao.idlefish"
        android:accessibilityFeedbackType="feedbackGeneric"
        android:notificationTimeout="300"
        android:canRetrieveWindowContent="true"
        android:accessibilityFlags="flagReportViewIds" />

不设置 canPerformGestures。不配置截图能力。Manifest 不添加 INTERNET 权限。

## 5. 事件处理伪代码

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {
        if (event?.packageName?.toString() != TARGET_PACKAGE) return

        val root = rootInActiveWindow ?: return
        val snapshot = readOnlySnapshot(root)
        parserQueue.submit(snapshot)
    }

readOnlySnapshot 只能读取节点字段和遍历子节点。不得在此函数或其下游调用任何 UI 操作 API。

## 6. 重要限制

- rootInActiveWindow 可能为空。
- 页面可能只暴露部分节点。
- 消息可能因虚拟列表而只保留当前可见部分。
- 用户手动滚动后才可能出现更早消息；服务不能自动滚动。
- 页面更新事件可能非常频繁，需合并解析任务。
- 控件 ID 和节点层级可能随闲鱼版本变化。
- 读取到昵称不等于读取到买家唯一 ID。
- 读取到商品标题不等于读取到商品 ID。

## 7. 验证方式

使用本人控制的测试账号和测试会话：
1. 用户手动打开闲鱼聊天列表。
2. 验证服务仅在闲鱼包名事件触发。
3. 用户手动打开测试聊天。
4. 验证节点 Snapshot 与屏幕实际可见内容一致。
5. 用户手动滚动后再次验证新出现的可见内容。
6. 检查服务没有点击、输入、滚动、发送或交易动作。
