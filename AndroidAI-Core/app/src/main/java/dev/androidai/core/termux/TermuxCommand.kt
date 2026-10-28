package dev.androidai.core.termux

/**
 * The complete read-only command whitelist. Tool arguments never become shell text.
 * On Redmi 13C, let Termux run with normal app privileges; root is not needed.
 */
enum class TermuxCommand(val shell: String) {
    PING("printf 'TERMUX_OK\n'; printf 'Model: '; getprop ro.product.model; printf 'Android: '; getprop ro.build.version.release"),
    DEVICE_INFO(
        "printf 'Manufacturer: '; getprop ro.product.manufacturer; " +
        "printf 'Model: '; getprop ro.product.model; " +
        "printf 'Android: '; getprop ro.build.version.release; " +
        "printf 'SDK: '; getprop ro.build.version.sdk; " +
        "printf 'Architecture: '; uname -m"
    ),
    ENVIRONMENT(
        "printf 'Termux command bridge: READY\n'; " +
        "printf 'Android: '; getprop ro.build.version.release; " +
        "printf 'Manufacturer: '; getprop ro.product.manufacturer; " +
        "printf 'Model: '; getprop ro.product.model; " +
        "printf 'CPU architecture: '; uname -m; " +
        "printf 'Shell: '; command -v bash; " +
        "printf 'MIUI: '; getprop ro.miui.ui.version.name; " +
        "printf 'HyperOS: '; getprop ro.mi.os.version.name"
    ),
    TOOLS(
        "for tool in git python node curl jq; do " +
        "if command -v \"\$tool\" >/dev/null 2>&1; then " +
        "printf '%s: installed\n' \"\$tool\"; " +
        "else printf '%s: not installed\n' \"\$tool\"; fi; done"
    ),
    DISK_SPACE("df -h /data/data/com.termux/files/home"),
    LIST_HOME("ls -A /data/data/com.termux/files/home | head -n 40"),
    LIST_TMP("ls -A /data/data/com.termux/files/usr/tmp | head -n 40"),
    LIST_DOWNLOADS(
        "if test -d /data/data/com.termux/files/home/storage/downloads; then " +
        "ls -A /data/data/com.termux/files/home/storage/downloads | head -n 40; " +
        "else printf '下载目录尚未授权。在 Termux 手动执行 termux-setup-storage\n'; exit 3; fi"
    ),
}

/** Commands pasted by the user in Termux, never executed automatically by the app. */
object TermuxSetup {
    val enableExternalApps = """
        mkdir -p ~/.termux
        touch ~/.termux/termux.properties
        if grep -q '^allow-external-apps=' ~/.termux/termux.properties; then
          sed -i 's/^allow-external-apps=.*/allow-external-apps=true/' ~/.termux/termux.properties
        else
          printf 'allow-external-apps=true\n' >> ~/.termux/termux.properties
        fi
        termux-reload-settings
    """.trimIndent()

    const val installBasicTools = "pkg update && pkg install -y git python curl"
}
