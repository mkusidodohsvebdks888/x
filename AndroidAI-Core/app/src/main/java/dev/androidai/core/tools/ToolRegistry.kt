package dev.androidai.core.tools

import android.app.Activity
import android.content.Intent
import android.os.Build
import android.provider.Settings
import dev.androidai.core.termux.TermuxBridge
import dev.androidai.core.termux.TermuxCommand
import dev.androidai.core.food.FoodSession
import dev.androidai.core.food.FoodUiBridge
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject

/** Fixed tool registry. Never pass user/model supplied strings into the shell. */
class ToolRegistry(private val activity: Activity) {
    fun definitions(): JSONArray = JSONArray().apply {
        put(def("cli_connection_test", "Test whether Termux RUN_COMMAND works; read-only"))
        put(def("cli_device_info", "Read Android manufacturer, model and system version using Termux"))
        put(def("cli_environment", "Diagnose Termux and Android runtime; useful for Redmi 13C"))
        put(def("cli_tools", "Check whether git, python, node, curl and jq are installed; do not install"))
        put(def("cli_disk_space", "Read filesystem free space from Termux without changing any files"))
        put(def("cli_list_files", "List limited filenames in one approved Termux folder", JSONObject()
            .put("type", "object")
            .put("properties", JSONObject().put("directory", JSONObject().put("type", "string")
                .put("enum", JSONArray(listOf("home", "tmp", "downloads")))))
            .put("required", JSONArray(listOf("directory")))
            .put("additionalProperties", false)))
        put(def("android_get_status", "Check Android SDK version and whether Termux is installed and authorized"))
        put(def("android_food_status", "Read current opt-in takeout assistant status and accessibility availability"))
        if (FoodSession.current() != null) {
            put(def("android_ui_scan", "Read the accessible screen of the authorized takeout app. Stop before order submission."))
            put(def("android_ui_click_text", "Click a visible item by its EXACT accessible label. Never submit an order or pay.", JSONObject()
                .put("type", "object")
                .put("properties", JSONObject()
                    .put("text", JSONObject().put("type", "string"))
                    .put("occurrence", JSONObject().put("type", "integer").put("minimum", 0).put("maximum", 20)))
                .put("required", JSONArray(listOf("text")))
                .put("additionalProperties", false)))
            put(def("android_ui_search_text", "Fill the focused takeout-app search box (not passwords, addresses or payment forms)", JSONObject()
                .put("type", "object")
                .put("properties", JSONObject().put("text", JSONObject().put("type", "string")))
                .put("required", JSONArray(listOf("text")))
                .put("additionalProperties", false)))
            put(def("android_ui_scroll", "Scroll forward on the authorized takeout app screen"))
        }
        put(def("android_open_app", "Open an installed Android app by package name", JSONObject()
            .put("type", "object")
            .put("properties", JSONObject().put("package_name", JSONObject().put("type", "string")
                .put("description", "Android package id, e.g. com.android.settings")))
            .put("required", JSONArray(listOf("package_name")))
            .put("additionalProperties", false)))
    }

    private fun def(
        name: String,
        description: String,
        parameters: JSONObject = JSONObject().put("type", "object")
            .put("properties", JSONObject()).put("additionalProperties", false)
    ): JSONObject = JSONObject().put("type", "function").put(
        "function", JSONObject().put("name", name)
            .put("description", description).put("parameters", parameters)
    )

    suspend fun execute(name: String, args: JSONObject): String = when (name) {
        "cli_connection_test" -> TermuxBridge.run(activity, TermuxCommand.PING)
        "cli_device_info" -> TermuxBridge.run(activity, TermuxCommand.DEVICE_INFO)
        "cli_environment" -> TermuxBridge.run(activity, TermuxCommand.ENVIRONMENT)
        "cli_tools" -> TermuxBridge.run(activity, TermuxCommand.TOOLS)
        "cli_disk_space" -> TermuxBridge.run(activity, TermuxCommand.DISK_SPACE)
        "cli_list_files" -> {
            val command = when (args.optString("directory")) {
                "home" -> TermuxCommand.LIST_HOME
                "tmp" -> TermuxCommand.LIST_TMP
                "downloads" -> TermuxCommand.LIST_DOWNLOADS
                else -> error("directory 仅允许 home、tmp 或 downloads")
            }
            TermuxBridge.run(activity, command)
        }
        "android_food_status" -> JSONObject()
            .put("food_app", FoodSession.current() ?: "inactive")
            .put("accessibility_enabled", FoodUiBridge.available())
            .put("payment_automation", false).toString()
        "android_ui_scan" -> withContext(Dispatchers.Main) { FoodUiBridge.scan() }
        "android_ui_click_text" -> withContext(Dispatchers.Main) {
            val result = FoodUiBridge.click(args.optString("text"), args.optInt("occurrence", 0))
            kotlinx.coroutines.delay(650)
            result
        }
        "android_ui_search_text" -> withContext(Dispatchers.Main) {
            val result = FoodUiBridge.inputSearch(args.optString("text"))
            kotlinx.coroutines.delay(500)
            result
        }
        "android_ui_scroll" -> withContext(Dispatchers.Main) {
            val result = FoodUiBridge.scroll()
            kotlinx.coroutines.delay(400)
            result
        }
        "android_get_status" -> JSONObject()
            .put("android_sdk", Build.VERSION.SDK_INT)
            .put("android_release", Build.VERSION.RELEASE)
            .put("manufacturer", Build.MANUFACTURER)
            .put("device_model", Build.MODEL)
            .put("termux_installed", TermuxBridge.installed(activity))
            .put("termux_version", TermuxBridge.versionName(activity) ?: "not installed")
            .put("termux_permission_granted", TermuxBridge.permissionGranted(activity))
            .toString()
        "android_open_app" -> withContext(Dispatchers.Main) {
            val packageName = args.optString("package_name")
            val selected = FoodSession.current()
            if (selected != null) require(packageName == selected) {
                "外卖辅助会话仅允许打开已选 APP：" + selected
            }
            require(packageName.matches(Regex("[A-Za-z0-9_]+(\\.[A-Za-z0-9_]+)+"))) {
                "需要合法的 Android package_name"
            }
            val intent = if (packageName == "com.android.settings") {
                Intent(Settings.ACTION_SETTINGS)
            } else {
                activity.packageManager.getLaunchIntentForPackage(packageName)
                    ?: error("找不到可启动的 APP：" + packageName)
            }
            activity.startActivity(intent)
            kotlinx.coroutines.delay(900)
            "已请求打开 " + packageName + "（请在手机界面检查操作结果）"
        }
        else -> error("未注册的工具：" + name)
    }
}
