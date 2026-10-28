package dev.androidai.core.termux

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent

class TermuxResultReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        @Suppress("DEPRECATION")
        val bundle = intent.getBundleExtra("result")
        TermuxBridge.completed(intent.getIntExtra("request_id", -1), bundle)
    }
}
