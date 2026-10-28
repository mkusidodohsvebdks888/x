package dev.androidai.core.agent

import dev.androidai.core.tools.ToolRegistry
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL
import kotlin.coroutines.coroutineContext

data class ChatTurn(val role: String, val text: String)

/** Minimal OpenAI-compatible, multi-round DeepSeek tool calling agent. */
class AgentClient(private val registry: ToolRegistry) {
    suspend fun ask(apiKey: String, model: String, history: List<ChatTurn>, report: suspend (String, Boolean, String) -> Unit): String {
        require(apiKey.isNotBlank()) { "请先在设置中填写 DeepSeek API Key" }
        val messages = JSONArray().put(JSONObject().put("role", "system").put("content",
            "你是 Android 手机本地自动化助手。仅可调用已注册工具。可以检测Termux连接、读取设备信息、Termux环境和工具状态、磁盘空间、列出授权目录或按用户请求打开APP。" +
                "绝不编造执行结果，不运行任意 Shell 命令，不进行安装、删除、付款或发送消息。" +
                "操作失败时说明原因。请用简体中文回答。"))
        history.takeLast(12).forEach { turn ->
            messages.put(JSONObject().put("role", turn.role).put("content", turn.text))
        }
        repeat(6) {
            coroutineContext.ensureActive()
            val result = request(apiKey, model, messages, registry.definitions())
            val choice = result.getJSONArray("choices").getJSONObject(0)
            val message = choice.getJSONObject("message")
            val calls = message.optJSONArray("tool_calls")
            if (calls == null || calls.length() == 0) return message.optString("content").ifBlank { "已完成。" }
            messages.put(message)
            for (i in 0 until calls.length()) {
                coroutineContext.ensureActive()
                val call = calls.getJSONObject(i)
                val function = call.getJSONObject("function")
                val toolName = function.getString("name")
                val args = runCatching { JSONObject(function.optString("arguments", "{}")) }.getOrElse { JSONObject() }
                val value = runCatching { registry.execute(toolName, args) }
                val ok = value.isSuccess
                val output = value.getOrElse { "错误：${it.message}" }.take(5_000)
                report(toolName, ok, output)
                messages.put(JSONObject().put("role", "tool")
                    .put("tool_call_id", call.getString("id"))
                    .put("content", output))
            }
        }
        return "已达到 6 轮工具调用上限，已停止执行。请缩小任务范围。"
    }

    private suspend fun request(apiKey: String, model: String, messages: JSONArray, tools: JSONArray): JSONObject =
        withContext(Dispatchers.IO) {
            val url = URL("https://api.deepseek.com/chat/completions")
            val connection = url.openConnection() as HttpURLConnection
            try {
                connection.requestMethod = "POST"
                connection.connectTimeout = 20_000
                connection.readTimeout = 65_000
                connection.doOutput = true
                connection.setRequestProperty("Content-Type", "application/json; charset=utf-8")
                connection.setRequestProperty("Authorization", "Bearer $apiKey")
                val payload = JSONObject().put("model", model).put("messages", messages)
                    .put("tools", tools).put("tool_choice", "auto")
                connection.outputStream.use { it.write(payload.toString().toByteArray(Charsets.UTF_8)) }
                val status = connection.responseCode
                val body = (if (status in 200..299) connection.inputStream else connection.errorStream)
                    ?.bufferedReader()?.use { it.readText() } ?: ""
                if (status !in 200..299) error("DeepSeek HTTP $status: ${body.take(320)}")
                JSONObject(body)
            } finally {
                connection.disconnect()
            }
        }
}
