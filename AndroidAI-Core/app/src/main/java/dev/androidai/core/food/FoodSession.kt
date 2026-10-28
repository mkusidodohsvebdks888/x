package dev.androidai.core.food

import android.content.Context
import android.os.SystemClock

data class FoodApp(val title: String, val packageName: String)

/** Temporary, explicit user authorization for ONE takeout app. */
object FoodSession {
    val supported = listOf(
        FoodApp("美团", "com.sankuai.meituan"),
        FoodApp("美团外卖", "com.sankuai.meituan.takeoutnew"),
        FoodApp("饿了么", "me.ele"),
        FoodApp("淘宝闪购", "com.taobao.taobao"),
        FoodApp("京东外卖", "com.jingdong.app.mall")
    )

    @Volatile private var allowedPackage: String? = null
    @Volatile private var expiresAt: Long = 0L

    fun current(): String? {
        val pkg = allowedPackage ?: return null
        if (SystemClock.elapsedRealtime() >= expiresAt) {
            stop()
            return null
        }
        return pkg
    }

    fun start(context: Context, pkg: String) {
        require(supported.any { it.packageName == pkg }) { "不支持此 APP" }
        require(context.packageManager.getLaunchIntentForPackage(pkg) != null) {
            "无法启动这个外卖 APP，请先安装并打开一次"
        }
        allowedPackage = pkg
        expiresAt = SystemClock.elapsedRealtime() + 20L * 60_000L
    }

    fun requireAuthorized(): String =
        current() ?: error("请在设置页开启本次外卖辅助（20分钟后自动失效）")

    fun stop() {
        allowedPackage = null
        expiresAt = 0L
    }
}
