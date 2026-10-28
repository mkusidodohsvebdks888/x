package dev.androidai.core.agent

import dev.androidai.core.tools.ToolRegistry
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL

data class ChatTurn(val role: String, val text: String)

/**
 * DeepSeek transport and a fresh Codex-style runtime per user task.
 * Network requests only happen when the runtime needs another decision.
 */
class AgentClient(private val registry: ToolRegistry) {
    suspend fun ask(
        apiKey: String,
        model: String,
        history: List<ChatTurn>,
        report: suspend (AgentEvent) -> Unit
    ): String {
        require(apiKey.isNotBlank()) { "请先在设置中填写 DeepSeek API Key" }
        return AgentRuntime(
            registry = registry,
            request = { messages, tools -> request(apiKey, model, messages, tools) },
            event = report
        ).run(history)
    }

    private suspend fun request(
        apiKey: String,
        model: String,
        messages: JSONArray,
        tools: JSONArray
    ): JSONObject = withContext(Dispatchers.IO) {
        val connection = URL("https://api.deepseek.com/chat/completions")
            .openConnection() as HttpURLConnection
        try {
            connection.requestMethod = "POST"
            connection.connectTimeout = 20_000
            connection.readTimeout = 65_000
            connection.doOutput = true
            connection.setRequestProperty("Content-Type", "application/json; charset=utf-8")
            connection.setRequestProperty("Authorization", "Bearer " + apiKey)
            val payload = JSONObject()
                .put("model", model)
                .put("messages", messages)
                .put("tools", tools)
                .put("tool_choice", "auto")
            connection.outputStream.use { stream ->
                stream.write(payload.toString().toByteArray(Charsets.UTF_8))
            }
            val status = connection.responseCode
            val body = (if (status in 200..299) connection.inputStream else connection.errorStream)
                ?.bufferedReader()?.use { it.readText() } ?: ""
            if (status !in 200..299) throw AgentHttpException(status, body)
            JSONObject(body)
        } finally {
            connection.disconnect()
        }
    }
}
