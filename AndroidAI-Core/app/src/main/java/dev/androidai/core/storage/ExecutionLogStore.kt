package dev.androidai.core.storage

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

data class ToolLog(val tool: String, val status: String, val detail: String, val timestamp: Long) {
    fun displayTime(): String = SimpleDateFormat("MM-dd HH:mm:ss", Locale.getDefault()).format(Date(timestamp))
}

/** Stores only short tool results, not chat prompts or credentials. */
class ExecutionLogStore(context: Context) {
    private val prefs = context.getSharedPreferences("execution_logs", Context.MODE_PRIVATE)

    fun read(): List<ToolLog> = runCatching {
        val array = JSONArray(prefs.getString("entries", "[]"))
        (0 until array.length()).map { index ->
            val row = array.getJSONObject(index)
            ToolLog(row.optString("tool"), row.optString("status"), row.optString("detail"), row.optLong("timestamp"))
        }.reversed()
    }.getOrElse { emptyList() }

    fun add(tool: String, status: String, detail: String) {
        val array = JSONArray(prefs.getString("entries", "[]"))
        array.put(JSONObject().put("tool", tool.take(80)).put("status", status.take(30))
            .put("detail", detail.take(400)).put("timestamp", System.currentTimeMillis()))
        val trim = JSONArray()
        for (i in (array.length() - 100).coerceAtLeast(0) until array.length()) trim.put(array.get(i))
        prefs.edit().putString("entries", trim.toString()).apply()
    }

    fun clear() { prefs.edit().remove("entries").apply() }
}
