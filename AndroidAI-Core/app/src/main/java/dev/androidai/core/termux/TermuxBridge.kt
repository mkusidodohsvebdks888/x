package dev.androidai.core.termux

import android.Manifest
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.withTimeout
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicInteger

/** Thin bridge adapted from the RUN_COMMAND design in AndroidAssistant (Linodx). */
object TermuxBridge {
    private const val RUN_PERMISSION = "com.termux.permission.RUN_COMMAND"
    private const val PACKAGE = "com.termux"
    private val nextId = AtomicInteger(4100)
    private val pending = ConcurrentHashMap<Int, CompletableDeferred<String>>()

    fun installed(context: Context): Boolean = runCatching {
        @Suppress("DEPRECATION")
        context.packageManager.getPackageInfo(PACKAGE, 0)
        true
    }.getOrDefault(false)

    fun permissionGranted(context: Context): Boolean =
        context.checkSelfPermission(RUN_PERMISSION) == PackageManager.PERMISSION_GRANTED

    suspend fun run(context: Context, command: String): String {
        if (!installed(context)) error("未安装 Termux，请安装 F-Droid 或 GitHub 发布版")
        if (!permissionGranted(context)) error("未授权 Termux RUN_COMMAND，请在设置页授权")
        val id = nextId.incrementAndGet()
        val response = CompletableDeferred<String>()
        pending[id] = response
        val flags = PendingIntent.FLAG_ONE_SHOT or
            (if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) PendingIntent.FLAG_MUTABLE else 0)
        val callback = PendingIntent.getBroadcast(context, id,
            Intent(context, TermuxResultReceiver::class.java).putExtra("request_id", id), flags)
        val runIntent = Intent("com.termux.RUN_COMMAND")
            .setClassName(PACKAGE, "com.termux.app.RunCommandService")
            .putExtra("com.termux.RUN_COMMAND_PATH", "/data/data/com.termux/files/usr/bin/bash")
            .putExtra("com.termux.RUN_COMMAND_ARGUMENTS", arrayOf("-lc", command))
            .putExtra("com.termux.RUN_COMMAND_WORKDIR", "/data/data/com.termux/files/home")
            .putExtra("com.termux.RUN_COMMAND_BACKGROUND", true)
            .putExtra("com.termux.RUN_COMMAND_PENDING_INTENT", callback)
        try {
            context.startService(runIntent)
            return withTimeout(30_000L) { response.await() }
        } catch (_: TimeoutCancellationException) {
            error("Termux 命令超时；命令可能仍在运行，请手动检查")
        } catch (e: Exception) {
            error("Termux 执行失败：${e.message}. 请检查 allow-external-apps=true 和 RUN_COMMAND 授权")
        } finally {
            pending.remove(id)
            callback.cancel()
        }
    }

    internal fun completed(id: Int, bundle: Bundle?) {
        val b = bundle ?: run {
            pending.remove(id)?.complete("无执行结果；请检查 Termux 设置")
            return
        }
        val stdout = b.getString("stdout").orEmpty()
        val stderr = b.getString("stderr").orEmpty()
        val exitCode = b.getInt("exitCode", -999)
        val error = b.getString("errmsg").orEmpty()
        val output = "exitCode=$exitCode\n$stdout\n$stderr\n$error".take(5_000)
        pending.remove(id)?.complete(output)
    }
}
