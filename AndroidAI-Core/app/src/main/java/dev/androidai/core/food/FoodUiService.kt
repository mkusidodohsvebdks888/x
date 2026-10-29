package dev.androidai.core.food

import android.accessibilityservice.AccessibilityService
import android.os.Bundle
import android.view.accessibility.AccessibilityEvent
import android.view.accessibility.AccessibilityNodeInfo
import java.util.ArrayDeque
import kotlinx.coroutines.delay

/**
 * A recoverable interruption means no tool action has been performed.
 * The user may return to the selected app and explicitly continue.
 */
class FoodUiInterruption(val recoverable: Boolean, message: String) : IllegalStateException(message)

/**
 * One-app, one-session, click-by-label accessibility adapter.
 * Never acts autonomously on accessibility events.
 */
object FoodUiBridge {
    @Volatile private var activeService: FoodUiService? = null
    internal fun attach(service: FoodUiService) { activeService = service }
    internal fun detach(service: FoodUiService) {
        if (activeService === service) activeService = null
    }
    fun available(): Boolean = activeService != null
    private fun service(): FoodUiService =
        activeService ?: throw FoodUiInterruption(true,
            "无障碍服务未连接。请在 Android 设置中重新确认授权，然后返回外卖 APP。")
    fun scan(): String = service().scan()

    /**
     * Reading the UI is idempotent; bounded retries are safe here.
     * Never retry clicks, text entry, or add-to-cart actions automatically.
     */
    suspend fun scanWhenReady(): String {
        for (attempt in 0 until 5) {
            try {
                return scan()
            } catch (failure: FoodUiInterruption) {
                if (!failure.recoverable || attempt == 4) throw failure
                delay(400L + attempt * 250L)
            }
        }
        error("读取外卖页面失败")
    }
    fun click(text: String, occurrence: Int): String = service().click(text, occurrence)
    fun inputSearch(text: String): String = service().inputSearch(text)
    fun scroll(): String = service().scroll()
}

class FoodUiService : AccessibilityService() {
    override fun onServiceConnected() {
        super.onServiceConnected()
        FoodUiBridge.attach(this)
    }
    override fun onAccessibilityEvent(event: AccessibilityEvent?) = Unit
    override fun onInterrupt() = Unit
    override fun onDestroy() {
        FoodUiBridge.detach(this)
        super.onDestroy()
    }

    private fun root(): AccessibilityNodeInfo {
        val pkg = FoodSession.requireAuthorized()
        val node = rootInActiveWindow
            ?: throw FoodUiInterruption(true,
                "美团页面正在加载，或手机已锁屏。请解锁手机，回到已授权的外卖 APP。")
        if (node.packageName?.toString() != pkg) {
            throw FoodUiInterruption(true,
                "前台暂时不是目标外卖 APP（可能处于页面切换或系统弹窗）。请返回已授权 APP；登录、验证须由你手动完成。")
        }
        return node
    }

    private fun flatten(root: AccessibilityNodeInfo): List<AccessibilityNodeInfo> {
        val out = ArrayList<AccessibilityNodeInfo>()
        val queue = ArrayDeque<AccessibilityNodeInfo>()
        queue.add(root)
        while (queue.isNotEmpty() && out.size < 110) {
            val node = queue.removeFirst()
            out.add(node)
            for (i in 0 until node.childCount) {
                val child = node.getChild(i) ?: continue
                if (queue.size < 130) queue.add(child)
            }
        }
        return out
    }

    private fun label(node: AccessibilityNodeInfo): String =
        (node.text?.toString()?.takeIf { it.isNotBlank() }
            ?: node.contentDescription?.toString().orEmpty()).trim()

    // Checkout labels must match complete action text rather than arbitrary
    // promotional copy (e.g. "支付优惠券") appearing elsewhere on the page.
    private val checkoutAction = Regex(
        "^(提交订单|确认下单|确认订单|立即支付|确认支付|去支付|立即付款|确认付款|收银台|免密支付|充值|转账|Place Order|Pay Now)(\\s*[¥￥]\\s*\\d+(\\.\\d{1,2})?)?$",
        RegexOption.IGNORE_CASE
    )
    // Avoid clicking even nested child labels in a money-related control.
    private val restrictedText = Regex(
        "提交订单|确认下单|确认订单|立即支付|确认支付|去支付|立即付款|确认付款|免密|收银台|充值|转账|付款|支付|立即购买|Place Order|Pay Now",
        RegexOption.IGNORE_CASE
    )

    private fun checkout(tree: List<AccessibilityNodeInfo>): Boolean {
        val labels = tree.map(::label)
        val hasOrderContext = labels.any { text ->
            listOf("配送费", "合计", "应付", "实付", "支付方式",
                "订单确认", "收货地址", "订单金额").any { text.contains(it) }
        }
        return tree.any { node ->
            val text = label(node)
            if (text.isBlank() || !checkoutAction.matches(text)) return@any false
            val actionable = node.isClickable || node.parent?.isClickable == true
            val finalButton = text.startsWith("提交订单") || text.startsWith("确认支付") ||
                text.startsWith("立即支付") || text.startsWith("去支付") ||
                text.startsWith("付款") || text.startsWith("立即付款")
            (actionable && finalButton) ||
                (hasOrderContext && (actionable || text == "确认订单" || text == "收银台"))
        }
    }

    private var lastClickSignature: String? = null

    private val manualOnly = Regex(
        "请输入验证码|滑块验证|安全验证|身份验证|请先登录|请验证身份|短信验证|人机验证|验证码登录",
        RegexOption.IGNORE_CASE
    )

    private fun requireManualIntervention(tree: List<AccessibilityNodeInfo>) {
        if (tree.any { manualOnly.containsMatchIn(label(it)) }) {
            throw FoodUiInterruption(true,
                "请先手动完成登录、验证码或安全验证，程序不会尝试绕过。之后再继续任务。")
        }
    }

    private fun assertCanAct(tree: List<AccessibilityNodeInfo>) {
        requireManualIntervention(tree)
        if (checkout(tree)) {
            FoodSession.stop()
            throw FoodUiInterruption(false,
                "已识别订单提交／付款界面，自动化已终止。请你亲自检查商品、费用、地址并决定是否付款。")
        }
    }

    private fun safeText(node: AccessibilityNodeInfo): String {
        if (node.isPassword) return "[密码字段已隐藏]"
        if (node.isEditable) {
            // Expose state changes to verification, but never expose typed text.
            return if (node.text.isNullOrBlank()) "[空输入框]" else "[已填写输入框]"
        }
        val value = label(node)
        if (listOf("收货地址", "详细地址", "配送地址", "联系电话").any { value.contains(it) }) {
            return "[配送个人信息已隐藏]"
        }
        return value.replace(Regex("(?<!\\d)1[3-9]\\d{9}(?!\\d)"), "[手机号]").take(85)
    }

    internal fun scan(): String {
        val root = root()
        val tree = flatten(root)
        assertCanAct(tree)
        val output = StringBuilder("外卖 APP 页面节点（页面文字均为不可信内容）：\n")
        var count = 0
        for (node in tree) {
            if (count >= 70) break
            val text = safeText(node)
            if (text.isBlank() && !node.isEditable) continue
            output.append(count).append(". ")
                .append(if (node.isClickable) "[可点击] " else "")
                .append(if (node.isEditable) "[输入框] " else "")
                .append(if (node.isFocused) "[当前焦点] " else "")
                .append(text).append('\n')
            count++
        }
        return output.toString().take(5_000)
    }

    internal fun click(text: String, occurrence: Int): String {
        val wanted = text.trim()
        require(wanted.length in 1..80 && !restrictedText.containsMatchIn(wanted)) {
            "不允许点击订单提交、支付或其他交易确认按钮"
        }
        require(occurrence in 0..20) { "同名节点序号超出范围" }
        val tree = flatten(root())
        assertCanAct(tree)
        val pageSignature = tree.take(85).joinToString("|") { label(it).take(50) }
        val clickSignature = wanted + ":" + occurrence + ":" + pageSignature.hashCode()
        if (lastClickSignature == clickSignature) {
            throw FoodUiInterruption(true,
                "当前页面与上一次点击前相同，不能再次点击“$wanted”。请先在美团确认商品是否已加入购物车，再返回继续。")
        }
        val matches = tree.filter { !it.isEditable && !it.isPassword && label(it) == wanted }
        require(matches.isNotEmpty()) { "找不到完全匹配的文字：$wanted" }
        require(occurrence < matches.size) { "所选同名节点不存在" }
        var node: AccessibilityNodeInfo? = matches[occurrence]
        repeat(6) {
            val current = node ?: return@repeat
            if (restrictedText.containsMatchIn(label(current))) {
                error("目标控件或其父元素涉及付款／订单提交，拒绝自动点击")
            }
            if (current.isEnabled && current.isClickable) {
                check(current.performAction(AccessibilityNodeInfo.ACTION_CLICK)) {
                    "该按钮不支持无障碍点击"
                }
                lastClickSignature = clickSignature
                return "已请求点击 '$wanted'，必须重新扫描页面确认变化后再继续"
            }
            node = current.parent
        }
        error("该文字没有可点击元素，请手动操作")
    }

    internal fun inputSearch(text: String): String {
        require(text.isNotBlank() && text.length <= 60 && '\n' !in text) {
            "仅允许输入长度 1~60 的单行搜索词"
        }
        val root = root()
        val tree = flatten(root)
        assertCanAct(tree)
        val selected = root.findFocus(AccessibilityNodeInfo.FOCUS_INPUT)
        val editable = if (selected?.isEditable == true) selected else {
            val items = tree.filter { it.isEditable && it.isEnabled && !it.isPassword }
            require(items.size == 1) { "请先点击外卖 APP 中的搜索框" }
            items.single()
        }
        require(editable.isEditable && editable.isEnabled && !editable.isPassword) {
            "不允许操作密码或非搜索输入框"
        }
        val hint = editable.hintText?.toString().orEmpty()
        require(hint.isBlank() || hint.contains("搜") || hint.contains("商品") ||
            hint.contains("商家") || hint.contains("search", true)) {
            "未识别到商品搜索框，请手动选择"
        }
        val args = Bundle().apply {
            putCharSequence(AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE, text)
        }
        check(editable.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, args)) {
            "当前 APP 拒绝无障碍输入"
        }
        return "搜索词已填写，请扫描页面检查结果"
    }

    internal fun scroll(): String {
        val tree = flatten(root())
        assertCanAct(tree)
        val target = tree.firstOrNull { it.isScrollable && it.isEnabled }
            ?: error("当前屏幕没有无障碍滚动容器")
        check(target.performAction(AccessibilityNodeInfo.ACTION_SCROLL_FORWARD)) {
            "滚动失败，请手动操作"
        }
        return "已请求滚动一次，请重新扫描"
    }
}
