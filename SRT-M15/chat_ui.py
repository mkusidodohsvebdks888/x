"""Run with: %run chat_ui.py  (Colab notebook with source extracted).

This is a research UI, using ONLY two locally trained SRT-M15 toy networks.
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
history = widgets.HTML(value="<div style='padding:12px'>示例：<b>你是谁</b>、<b>1+9=多少</b>，或 <b>/raw 你是谁</b> 查看模型原始生成。</div>", layout=widgets.Layout(width="100%"))
textbox = widgets.Textarea(value="", placeholder="输入中文、数字或标点，例如：1+9=多少", description="提问：", layout=widgets.Layout(width="100%", height="96px"))
send = widgets.Button(description="发送问题", button_style="primary", icon="paper-plane")
clear = widgets.Button(description="清空对话", icon="trash")
status = widgets.HTML(value="<small>研究原型：身份说明由程序规则提供；输入 /raw 你是谁 可测试神经网络的原始字符生成。</small>")

def render_history():
    bubbles = []
    for who, msg in messages[-30:]:
        label = "你" if who == "user" else "SRT-M15"
        bg = "#e5f2ff" if who == "user" else "#f0f3f5"
        safe = escape(msg).replace("\n", "<br>")
        bubbles.append(f'<div style="margin:8px 4px;padding:12px;background:{bg};border-radius:12px"><b>{label}</b><br>{safe}</div>')
    history.value = '<div style="max-height:340px;overflow-y:auto">' + ''.join(bubbles) + '</div>'

def on_send(_):
    question = textbox.value.strip()
    if not question:
        status.value = "<small>请先输入问题。</small>"
        return
    textbox.value = ""
    send.disabled = True
    try:
        reply = engine.reply(question).text
    except Exception as exc:
        reply = f"模型加载或推理错误：{type(exc).__name__}: {exc}"
    finally:
        send.disabled = False
    messages.extend([("user", question), ("bot", reply)])
    render_history()

def on_clear(_):
    messages.clear()
    render_history()
send.on_click(on_send)
clear.on_click(on_clear)
display(widgets.VBox([widgets.HTML("<h3>SRT-M15 手机聊天实验</h3>"), history, textbox, widgets.HBox([send, clear]), status], layout=widgets.Layout(width="100%")))
