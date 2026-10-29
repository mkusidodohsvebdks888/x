package dev.androidai.core.workflow

import dev.androidai.core.food.FoodUiBridge
import dev.androidai.core.food.FoodUiInterruption
import kotlinx.coroutines.delay

/**
 * Observe -> Act -> Observe verification for user-authorized foreground UI.
 *
 * Calls to scanWhenReady() are read-only. Clicking, typing and scrolling are
 * NEVER repeated by this class, because a second attempt could duplicate cart
 * items, select wrong variants or trigger a purchase unintentionally.
 */
object WorkflowVerifier {
    suspend fun actAndObserve(
        toolName: String,
        action: () -> String
    ): String {
        val before = FoodUiBridge.scanWhenReady()
        val actionResult = action()

        // Wait for the app to settle using read-only observations.
        var after = before
        for (attempt in 0 until 4) {
            delay(450L + attempt * 150L)
            after = FoodUiBridge.scanWhenReady()
            if (after != before) break
        }

        if (after == before) {
            // A click could have been accepted even when UI text didn't change.
            // Don't claim success or repeat the action. Ask for re-observation.
            throw FoodUiInterruption(true,
                "动作 " + toolName + " 已请求执行，但页面快照没有可验证的变化。" +
                    "请亲自确认商品和购物车状态，避免重复加购。")
        }

        return buildString {
            append(actionResult)
            append("\n工具：").append(toolName)
            append("\n核验状态：已读取到操作后变化的页面。")
            append("\n操作后页面摘要：\n")
            append(after.take(1_800))
            append("\n提示：核验到页面变化不等于已购买或已成功加购；请核对购物车内容。")
        }
    }

}
