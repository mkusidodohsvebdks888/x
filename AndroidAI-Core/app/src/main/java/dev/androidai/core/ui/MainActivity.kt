package dev.androidai.core.ui

import android.app.Activity
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.provider.Settings
import android.content.pm.PackageManager
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.os.Bundle
import android.text.InputType
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import dev.androidai.core.agent.AgentClient
import dev.androidai.core.agent.ChatTurn
import dev.androidai.core.storage.ExecutionLogStore
import dev.androidai.core.storage.SecureSettings
import dev.androidai.core.termux.TermuxBridge
import dev.androidai.core.termux.TermuxCommand
import dev.androidai.core.termux.TermuxSetup
import dev.androidai.core.tools.ToolRegistry
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/** Three pages only: Chat, Tool Logs, Settings. No editor or file manager UI. */
class MainActivity : Activity() {
    private lateinit var settings: SecureSettings
    private lateinit var logs: ExecutionLogStore
    private val uiScope = CoroutineScope(SupervisorJob() + Dispatchers.Main)
    private val messages = mutableListOf<ChatTurn>()
    private var page = 0
    private var inputDraft = ""
    private var running = false
    private var activeJob: Job? = null

    private val dark = Color.rgb(24, 33, 49)
    private val muted = Color.rgb(108, 120, 138)
    private val blue = Color.rgb(40, 103, 226)
    private val background = Color.rgb(246, 248, 252)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        settings = SecureSettings(this)
        logs = ExecutionLogStore(this)
        window.statusBarColor = Color.WHITE
        window.navigationBarColor = Color.WHITE
        @Suppress("DEPRECATION")
        window.decorView.systemUiVisibility = View.SYSTEM_UI_FLAG_LIGHT_STATUS_BAR or View.SYSTEM_UI_FLAG_LIGHT_NAVIGATION_BAR
        render()
    }

    override fun onDestroy() {
        activeJob?.cancel()
        uiScope.cancel()
        super.onDestroy()
    }

    private fun dp(px: Int): Int = (px * resources.displayMetrics.density).toInt()

    private fun shape(color: Int, radius: Int = 14): GradientDrawable =
        GradientDrawable().apply { setColor(color); cornerRadius = dp(radius).toFloat() }

    private fun label(value: String, size: Float = 15f, bold: Boolean = false, color: Int = dark): TextView =
        TextView(this).apply {
            text = value
            textSize = size
            setTextColor(color)
            if (bold) typeface = Typeface.DEFAULT_BOLD
        }

    private fun button(text: String, onClick: () -> Unit, primary: Boolean = true): Button =
        Button(this).apply {
            this.text = text
            textSize = 13f
            isAllCaps = false
            setTextColor(if (primary) Color.WHITE else dark)
            background = shape(if (primary) blue else Color.rgb(231, 235, 243), 12)
            setOnClickListener { onClick() }
        }

    private fun render() {
        val shell = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(this@MainActivity.background)
        }
        val heading = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(dp(18), dp(16), dp(18), dp(12))
            setBackgroundColor(Color.WHITE)
        }
        heading.addView(label("Android AI Core", 20f, true), LinearLayout.LayoutParams(0, -2, 1f))
        heading.addView(label(if (TermuxBridge.installed(this)) "● Termux 已安装" else "○ Termux 未安装", 11f,
            color = if (TermuxBridge.installed(this)) Color.rgb(19, 143, 83) else muted))
        shell.addView(heading)

        val tabs = LinearLayout(this).apply { setPadding(dp(10), dp(8), dp(10), dp(10)); setBackgroundColor(Color.WHITE) }
        listOf("AI 对话", "执行记录", "设置").forEachIndexed { index, title ->
            val tab = TextView(this).apply {
                text = title
                textSize = 14f
                gravity = Gravity.CENTER
                setPadding(dp(4), dp(11), dp(4), dp(11))
                setTextColor(if (page == index) blue else muted)
                if (page == index) typeface = Typeface.DEFAULT_BOLD
                background = shape(if (page == index) Color.rgb(234, 242, 255) else Color.WHITE, 9)
                setOnClickListener { page = index; render() }
            }
            tabs.addView(tab, LinearLayout.LayoutParams(0, -2, 1f))
        }
        shell.addView(tabs)

        val content = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(dp(14), dp(14), dp(14), dp(14)) }
        shell.addView(content, LinearLayout.LayoutParams(-1, 0, 1f))
        when (page) {
            0 -> drawChat(content)
            1 -> drawLogs(content)
            else -> drawSettings(content)
        }
        setContentView(shell)
    }

    private fun drawChat(content: LinearLayout) {
        val scroll = ScrollView(this)
        val feed = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        if (messages.isEmpty()) {
            feed.addView(label("你可以让 DeepSeek 查询设备信息、调用 Termux CLI，或打开手机 APP。", 16f, color = muted).apply {
                setPadding(dp(12), dp(42), dp(12), dp(18))
            })
        }
        messages.forEach { message ->
            val bubble = label(message.text, 15f, color = if (message.role == "user") Color.WHITE else dark).apply {
                background = shape(if (message.role == "user") blue else Color.WHITE)
                setPadding(dp(14), dp(12), dp(14), dp(12))
            }
            val params = LinearLayout.LayoutParams(-2, -2).apply {
                gravity = if (message.role == "user") Gravity.END else Gravity.START
                setMargins(0, dp(5), 0, dp(9))
            }
            feed.addView(bubble, params)
        }
        if (running) feed.addView(label("正在调用 DeepSeek 或执行工具…", 13f, color = muted))
        scroll.addView(feed)
        content.addView(scroll, LinearLayout.LayoutParams(-1, 0, 1f))
        scroll.post { scroll.fullScroll(View.FOCUS_DOWN) }

        val composer = LinearLayout(this).apply { gravity = Gravity.CENTER_VERTICAL; orientation = LinearLayout.HORIZONTAL }
        val input = EditText(this).apply {
            setText(inputDraft)
            hint = "输入要执行的任务…"
            textSize = 14f
            minLines = 1
            maxLines = 3
            background = shape(Color.WHITE)
            setPadding(dp(12), dp(10), dp(12), dp(10))
        }
        composer.addView(input, LinearLayout.LayoutParams(0, -2, 1f))
        val action = button(if (running) "停止" else "发送", {
            if (running) {
                activeJob?.cancel()
                logs.add("agent", "cancelled", "已停止等待后续步骤；Termux 中已启动的命令可能继续执行")
                running = false
                render()
            } else {
                inputDraft = input.text.toString()
                send()
            }
        })
        composer.addView(action, LinearLayout.LayoutParams(dp(84), dp(52)).apply { leftMargin = dp(8) })
        content.addView(composer)
    }

    private fun send() {
        val input = inputDraft.trim()
        if (input.isBlank()) return
        if (settings.readApiKey().isBlank()) {
            page = 2
            Toast.makeText(this, "请先设置 DeepSeek API Key", Toast.LENGTH_LONG).show()
            render()
            return
        }
        messages.add(ChatTurn("user", input))
        inputDraft = ""
        running = true
        render()
        activeJob = uiScope.launch {
            try {
                val key = settings.readApiKey()
                val answer = AgentClient(ToolRegistry(this@MainActivity)).ask(key, settings.model, messages.toList()) { tool, success, output ->
                    withContext(Dispatchers.Main) {
                        logs.add(tool, if (success) "success" else "error", output)
                        if (page == 1) render()
                    }
                }
                messages.add(ChatTurn("assistant", answer))
            } catch (_: CancellationException) {
                messages.add(ChatTurn("assistant", "已暂停后续操作。正在运行的外部命令不一定已终止。"))
            } catch (error: Exception) {
                messages.add(ChatTurn("assistant", "执行失败：${error.message ?: "未知错误"}"))
            } finally {
                running = false
                activeJob = null
                render()
            }
        }
    }

    private fun drawLogs(content: LinearLayout) {
        val header = LinearLayout(this).apply { gravity = Gravity.CENTER_VERTICAL }
        header.addView(label("工具执行记录", 18f, true), LinearLayout.LayoutParams(0, -2, 1f))
        header.addView(button("清空", { logs.clear(); render() }, primary = false))
        content.addView(header)
        val scroll = ScrollView(this)
        val rows = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        val entries = logs.read()
        if (entries.isEmpty()) rows.addView(label("还没有工具调用记录。", 14f, color = muted).apply {
            setPadding(0, dp(24), 0, 0)
        })
        entries.forEach { entry ->
            val group = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                background = shape(Color.WHITE)
                setPadding(dp(14), dp(12), dp(14), dp(12))
            }
            group.addView(label("${entry.tool}   ·   ${entry.status}", 14f, true,
                if (entry.status == "success") Color.rgb(23, 135, 74) else dark))
            group.addView(label(entry.displayTime(), 12f, color = muted))
            group.addView(label(entry.detail, 12f, color = muted).apply { setPadding(0, dp(6), 0, 0) })
            rows.addView(group, LinearLayout.LayoutParams(-1, -2).apply { bottomMargin = dp(10) })
        }
        scroll.addView(rows)
        content.addView(scroll, LinearLayout.LayoutParams(-1, 0, 1f))
    }

    private fun drawSettings(content: LinearLayout) {
        val scroll = ScrollView(this)
        val form = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        form.addView(label("DeepSeek API", 18f, true))
        form.addView(label("接口：https://api.deepseek.com/chat/completions", 12f, color = muted).apply {
            setPadding(0, dp(5), 0, dp(12))
        })
        form.addView(label("API Key", 13f, true))
        val apiField = EditText(this).apply {
            setText(settings.readApiKey())
            hint = "sk-..."
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_PASSWORD
            setSingleLine(true)
        }
        form.addView(apiField, LinearLayout.LayoutParams(-1, dp(54)))
        form.addView(label("模型名称", 13f, true).apply { setPadding(0, dp(12), 0, 0) })
        val modelField = EditText(this).apply { setText(settings.model); setSingleLine(true) }
        form.addView(modelField, LinearLayout.LayoutParams(-1, dp(54)))
        form.addView(button("保存 API 设置", {
            runCatching {
                settings.saveApiKey(apiField.text.toString())
                settings.model = modelField.text.toString()
            }.onSuccess {
                Toast.makeText(this, "已保存（API Key 使用 Android Keystore 加密）", Toast.LENGTH_LONG).show()
            }.onFailure {
                Toast.makeText(this, "保存失败：${it.message}", Toast.LENGTH_LONG).show()
            }
        }), LinearLayout.LayoutParams(-1, dp(52)).apply { topMargin = dp(12) })

        form.addView(label("Termux 执行器", 18f, true).apply { setPadding(0, dp(28), 0, dp(8)) })
        form.addView(label("安装：${if (TermuxBridge.installed(this)) "已安装" else "未安装"}", 14f))
        form.addView(label("RUN_COMMAND 权限：${if (TermuxBridge.permissionGranted(this)) "已授权" else "未授权"}", 14f))
        form.addView(label("Termux 版本：${TermuxBridge.versionName(this) ?: "尚未安装"}", 13f, color = muted))
        form.addView(label("Redmi 13C 适配：普通权限运行，不需要 Root。手机可能运行 MIUI 或 HyperOS，请按实际系统设置检查后台限制。", 12f,
            color = muted).apply { setPadding(0, dp(10), 0, dp(8)) })
        form.addView(label("请先在 Termux 手动设置 allow-external-apps=true，随后重新加载配置，再授予 RUN_COMMAND。", 12f,
            color = muted).apply { setPadding(0, dp(8), 0, dp(8)) })
        form.addView(button("复制 Termux 授权配置命令", {
            copyToClipboard("Termux 配置", TermuxSetup.enableExternalApps)
        }, primary = false), LinearLayout.LayoutParams(-1, dp(50)).apply { bottomMargin = dp(8) })
        form.addView(button("复制基本工具安装命令（可选）", {
            copyToClipboard("Termux 基本工具", TermuxSetup.installBasicTools)
        }, primary = false), LinearLayout.LayoutParams(-1, dp(50)).apply { bottomMargin = dp(8) })
        form.addView(button("请求 Termux 执行权限", {
            if (!TermuxBridge.installed(this)) Toast.makeText(this, "请先安装 Termux", Toast.LENGTH_LONG).show()
            else requestPermissions(arrayOf("com.termux.permission.RUN_COMMAND"), 701)
        }, primary = false), LinearLayout.LayoutParams(-1, dp(52)))
        form.addView(button("测试 Termux 连接", {
            runTermuxCheck()
        }), LinearLayout.LayoutParams(-1, dp(52)).apply { topMargin = dp(10) })
        form.addView(button("检查 Termux 环境", {
            runTermuxCheck(TermuxCommand.ENVIRONMENT)
        }, primary = false), LinearLayout.LayoutParams(-1, dp(52)).apply { topMargin = dp(8) })
        form.addView(button("打开本应用系统设置", {
            startActivity(Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS,
                Uri.fromParts("package", packageName, null)))
        }, primary = false), LinearLayout.LayoutParams(-1, dp(52)).apply { topMargin = dp(8) })
        form.addView(label("如遇到锁屏或切换应用后执行失败，请在小米手机的应用信息中按需检查 Termux 与 Android AI Core 的省电策略、后台运行和权限。系统可能主动限制服务，无法保证锁屏执行。", 12f,
            color = muted).apply { setPadding(0, dp(10), 0, dp(8)) })
        form.addView(label("默认仅支持预先登记的只读 CLI 命令与 APP 启动。没有 Root、任意 Shell 或自动支付功能。", 12f,
            color = muted).apply { setPadding(0, dp(18), 0, dp(8)) })
        scroll.addView(form)
        content.addView(scroll)
    }

    private fun copyToClipboard(title: String, value: String) {
        val manager = getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        manager.setPrimaryClip(ClipData.newPlainText(title, value))
        Toast.makeText(this, "已复制，请在 Termux 中粘贴并执行", Toast.LENGTH_LONG).show()
    }

    private fun runTermuxCheck(command: TermuxCommand = TermuxCommand.PING) {
        uiScope.launch {
            try {
                val output = withContext(Dispatchers.IO) {
                    TermuxBridge.run(this@MainActivity, command)
                }
                logs.add("termux." + command.name.lowercase(), "success", output)
                Toast.makeText(this@MainActivity, "连接成功，已写入执行记录", Toast.LENGTH_LONG).show()
                page = 1
                render()
            } catch (e: Exception) {
                val info = e.message ?: "未知错误"
                logs.add("termux." + command.name.lowercase(), "error", info)
                Toast.makeText(this@MainActivity, info.take(160), Toast.LENGTH_LONG).show()
            }
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == 701) {
            Toast.makeText(this,
                if (grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED) "Termux 权限已授予" else "Termux 权限未授予",
                Toast.LENGTH_SHORT).show()
            render()
        }
    }
}
