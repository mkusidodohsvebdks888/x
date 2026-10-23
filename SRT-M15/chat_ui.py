"""SRT-M15 notebook chat UI: local, DeepSeek Web, or official API.

Web mode requires a visible browser on a local desktop, not remote Colab.
External answers are prominently labeled; they do not train the local model.
"""
from html import escape
from IPython.display import display
import ipywidgets as widgets
from srt_m15.chat import SRTChatEngine

try:
    from google.colab import output
    output.enable_custom_widget_manager()
except ImportError:
    pass

engine = SRTChatEngine()
messages = []

history = widgets.HTML(
    value="<div style='padding:12px'>试试：<b>你是谁</b>、<b>1+9=多少</b>。"
          "本地答案来自实验模型；DeepSeek 答案将注明外部来源。</div>",
    layout=widgets.Layout(width="100%")
)
textbox = widgets.Textarea(
    value="", placeholder="输入问题，例如：你是谁",
    description="提问：", layout=widgets.Layout(width="100%", height="100px")
)
backend_select = widgets.Dropdown(
    options=[
        ("SRT-M15 本地神经网络", "local"),
        ("DeepSeek 网页（仅本机电脑）", "web"),
        ("DeepSeek API（可在 Colab 使用）", "api"),
        ("自动路由（本地/外部）", "auto"),
    ],
    value="local", description="回答方式：",
    layout=widgets.Layout(width="100%")
)
send = widgets.Button(description="发送问题", button_style="primary", icon="paper-plane")
clear = widgets.Button(description="清空对话", icon="trash")
status = widgets.HTML(
    value="<small>默认只使用 SRT-M15。切换到外部模式才会将问题发送给 DeepSeek。"
          "网页版需要本地电脑手动登录，不支持从 Colab 接管手机浏览器。</small>"
)


def render_history():
    bubbles = []
    for who, msg in messages[-30:]:
        is_user = who == "你"
        bg = "#e5f2ff" if is_user else "#f0f3f5"
        safe = escape(msg).replace("\n", "<br>")
        bubbles.append(
            f'<div style="margin:8px 4px;padding:12px;background:{bg};'
            f'border-radius:12px"><b>{escape(who)}</b><br>{safe}</div>'
        )
    history.value = (
        '<div style="max-height:400px;overflow-y:auto">'
        + "".join(bubbles) + "</div>"
    )


def on_send(_):
    question = textbox.value.strip()
    if not question:
        status.value = "<small>请先输入问题。</small>"
        return
    textbox.value = ""
    send.disabled = True
    try:
        result = engine.reply(question, backend=backend_select.value)
        reply = result.text
        label = result.provider + ("（错误）" if result.mode == "error" else "")
        status.value = (
            f"<small>本次回答来源：<b>{escape(result.provider)}</b>。"
            "外部回答不是 SRT-M15 独立生成，也不会自动训练它。</small>"
        )
    except Exception as exc:
        reply = f"模型加载或推理错误：{type(exc).__name__}: {exc}"
        label = "系统提示"
    finally:
        send.disabled = False
    messages.extend([("你", question), (label, reply)])
    render_history()


def on_clear(_):
    messages.clear()
    render_history()


send.on_click(on_send)
clear.on_click(on_clear)
display(widgets.VBox([
    widgets.HTML("<h3>SRT-M15 多来源问答实验</h3>"),
    history, backend_select, textbox, widgets.HBox([send, clear]), status
], layout=widgets.Layout(width="100%")))
