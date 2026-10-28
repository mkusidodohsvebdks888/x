package dev.androidai.core.tools

import android.app.Activity
import android.content.Intent
import android.os.Build
import android.provider.Settings
import dev.androidai.core.termux.TermuxBridge
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject

/** Fixed, inspectable tool registry. The model cannot supply arbitrary shell commands. */
class ToolRegistry(private val activity: Activity) {
    fun definitions(): JSONArray = JSONArray().apply {
        put(def("cli_device_info", "Read basic Android details using a fixed Termux shell command"))
        put(def("cli_list_files", "List either Termux home or temporary directory", JSONObject()
            .put("type", "object")
            .put("properties", JSONObject().put("directory", JSONObject().put("type", "string")
                .put("enum", JSONArray(listOf("home", "tmp")))))
            .put("required", JSONArray(listOf("directory")))
            .put("additionalProperties", false)))
        put(def("android_get_status", "Check current Android version and Termux availability"))
        put(def("android_open_app", "Open one installed Android app by its package name", JSONObject()
            .put("type", "object")
            .put("properties", JSONObject().put("package_name", JSONObject().put("type", "string")
                .put("description", "Android package id e.g. com.android.settings")))
            .put("required", JSONArray(listOf("package_name")))
            .put("additionalProperties", false)))
    }

    private fun def(name: String, description: String, parameters: JSONObject =
        JSONObject().put("type", "object").put("properties", JSONObject()).put("additionalProperties", false)
    ): JSONObject = JSONObject().put("type", "function").put("function", JSONObject()
        .put("name", name).put("description", description).put("parameters", parameters))

    suspend fun execute(name: String, args: JSONObject): String = when (name) {
        "cli_device_info" -> TermuxBridge.run(activity, "getprop ro.product.model; getprop ro.build.version.release")
        "cli_list_files" -> {
            val command = when (args.optString("directory")) {
                "home" -> "ls -la \"\$HOME\""
                "tmp" -> "ls -la \"\$PREFIX/tmp\""
                else -> error("directory 仅允许 home 或 tmp")
            }
            TermuxBridge.run(activity, command)
        }
        "android_get_status" -> JSONObject()
            .put("android_sdk", Build.VERSION.SDK_INT)
            .put("device_model", Build.MODEL)
            .put("termux_installed", TermuxBridge.installed(activity))
            .put("termux_permission_granted", TermuxBridge.permissionGranted(activity)).toString()
        "android_open_app" -> withContext(Dispatchers.Main) {
            val packageName = args.optString("package_name")
            require(packageName.matches(Regex("[A-Za-z0-9_]+(\\.[A-Za-z0-9_]+)+"))) {
                "需要合法的 Android package_name"
            }
            val intent = if (packageName == "com.android.settings") {
                Intent(Settings.ACTION_SETTINGS)
            } else {
                activity.packageManager.getLaunchIntentForPackage(packageName)
                    ?: error("找不到可启动的 APP：$packageName")
            }
            activity.startActivity(intent)
            "已请求打开 $packageName（需在手机界面确认结果）"
        }
        else -> error("未注册的工具：$name")
    }
}
