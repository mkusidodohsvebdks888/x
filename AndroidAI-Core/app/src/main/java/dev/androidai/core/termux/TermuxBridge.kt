package dev.androidai.core.termux

import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.withTimeout
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicInteger

/**
 * Strict Termux RUN_COMMAND bridge. AI can select only predefined TermuxCommand
 * values and cannot supply shell strings. No root, remote server, or polling.
 */
object TermuxBridge {
    private const val RUN_PERMISSION = "com.termux.permission.RUN_COMMAND"
    private const val PACKAGE = "com.termux"
    private const val HOME = "/data/data/com.termux/files/home"
    private const val BASH = "/data/data/com.termux/files/usr/bin/bash"
    private const val ERROR_OK = -1
    private val nextId = AtomicInteger(5100)
    private val pending = ConcurrentHashMap<Int, CompletableDeferred<TermuxResult>>()

    private data class TermuxResult(
        val stdout: String,
        val stderr: String,
        val exitCode: Int?,
        val errorCode: Int,
        val errorMessage: String,
    )

    fun installed(context: Context): Boolean = runCatching {
        @Suppress("DEPRECATION")
        context.packageManager.getPackageInfo(PACKAGE, 0)
        true
    }.getOrDefault(false)

    fun versionName(context: Context): String? = runCatching {
        @Suppress("DEPRECATION")
        context.packageManager.getPackageInfo(PACKAGE, 0).versionName
    }.getOrNull()

    fun permissionGranted(context: Context): Boolean =
        context.checkSelfPermission(RUN_PERMISSION) == PackageManager.PERMISSION_GRANTED

    suspend fun run(
        context: Context,
        command: TermuxCommand,
        timeoutMillis: Long = 45_000L,
    ): String {
        if (!installed(context)) {
            error("未安装 Termux。请先从 F-Droid 或 Termux 官方 GitHub 安装并打开一次。")
        }
        if (!permissionGranted(context)) {
            error("缺少 RUN_COMMAND 权限。请在本应用的设置页授权，或到应用信息的其他权限中手动允许。")
        }

        val id = nextId.incrementAndGet()
        val reply = CompletableDeferred<TermuxResult>()
        pending[id] = reply
        val flags = PendingIntent.FLAG_ONE_SHOT or
            (if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) PendingIntent.FLAG_MUTABLE else 0)
        val callback = PendingIntent.getBroadcast(
            context, id,
            Intent(context, TermuxResultReceiver::class.java).putExtra("request_id", id),
            flags,
        )

        val intent = Intent("com.termux.RUN_COMMAND")
            .setClassName(PACKAGE, "com.termux.app.RunCommandService")
            .putExtra("com.termux.RUN_COMMAND_PATH", BASH)
            .putExtra("com.termux.RUN_COMMAND_ARGUMENTS", arrayOf("-lc", command.shell))
            .putExtra("com.termux.RUN_COMMAND_WORKDIR", HOME)
            .putExtra("com.termux.RUN_COMMAND_BACKGROUND", true)
            .putExtra("com.termux.RUN_COMMAND_COMMAND_LABEL", "AndroidAI-Core " + command.name)
            .putExtra("com.termux.RUN_COMMAND_PENDING_INTENT", callback)

        try {
            if (context.startService(intent) == null) {
                error("系统没有启动 Termux 服务。请先手动打开 Termux 后重试。")
            }
            val result = withTimeout(timeoutMillis.coerceIn(5_000L, 90_000L)) { reply.await() }
            if (result.errorCode != ERROR_OK || result.exitCode != 0) {
                val detail = (result.errorMessage + "\n" + result.stderr + "\n" + result.stdout)
                    .trim().take(800)
                error(
                    "Termux 执行未成功 (exit=" + (result.exitCode?.toString() ?: "missing") +
                        ", err=" + result.errorCode + "): " + detail
                )
            }
            val output = buildString {
                append(result.stdout.trim())
                if (result.stderr.isNotBlank()) {
                    if (isNotBlank()) append("\n")
                    append(result.stderr.trim())
                }
            }
            return output.take(5_000).ifBlank { "命令执行成功（无输出）" }
        } catch (_: TimeoutCancellationException) {
            error("Termux 回调超时。命令可能仍在运行；请在 Termux 中检查并关闭省电限制后重试。")
        } catch (cancel: CancellationException) {
            throw cancel
        } catch (failure: Exception) {
            val message = failure.message.orEmpty()
            if (message.startsWith("Termux ") || message.startsWith("系统没有")) throw failure
            error(
                "Termux 启动失败：" + message.take(500) +
                    "；请检查 allow-external-apps=true、RUN_COMMAND 权限，并先打开 Termux。"
            )
        } finally {
            pending.remove(id)
            callback.cancel()
        }
    }

    internal fun completed(id: Int, bundle: Bundle?) {
        val target = pending.remove(id) ?: return
        if (bundle == null) {
            target.complete(TermuxResult("", "", null, 1, "回调没有结果数据"))
            return
        }
        target.complete(
            TermuxResult(
                stdout = bundle.getString("stdout").orEmpty(),
                stderr = bundle.getString("stderr").orEmpty(),
                exitCode = if (bundle.containsKey("exitCode")) bundle.getInt("exitCode") else null,
                errorCode = bundle.getInt("err", ERROR_OK),
                errorMessage = bundle.getString("errmsg").orEmpty(),
            )
        )
    }
}
