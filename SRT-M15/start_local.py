"""Start SRT-M15 on your computer with a direct browser input box.

No external model API. The UI is bound to localhost only. If run from the
public GitHub repo, the script extracts the bundled source
from SRT_M15_Phone_Colab.ipynb one time into ./runtime_model.
"""
import argparse
import base64
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import threading
import webbrowser
import zipfile

ROOT = Path(__file__).resolve().parent

HTML_PAGE = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SRT-M15 | Local Chat</title>
<style>
:root{color-scheme:light;font-family:Inter,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
*{box-sizing:border-box}body{margin:0;background:#f7f7f5;color:#242526}
.app{display:flex;flex-direction:column;min-height:100dvh;max-width:860px;margin:auto}
.header{display:flex;align-items:center;gap:13px;padding:24px 22px 14px;border-bottom:1px solid #e5e6e7}
.logo{display:grid;place-items:center;width:42px;height:42px;border-radius:14px;background:#242526;color:white;font-weight:800;font-size:17px}
h1{font-size:19px;letter-spacing:-.4px;margin:0} .small{font-size:12px;color:#687076;margin:5px 0 0}
.status{margin-left:auto;font-size:12px;color:#617062;padding:7px 10px;border-radius:18px;background:#e4ede5;white-space:nowrap}
.learnbar{margin:0 22px 0;padding:13px 15px;background:#fff;border:1px solid #e4e6e8;border-radius:0 0 14px 14px}
.learn-top{display:flex;align-items:center;gap:12px;justify-content:space-between;flex-wrap:wrap}
.learn-title{font-size:14px;font-weight:700}.learn-title small{font-size:11px;font-weight:400;color:#6b747c}
.learn-toggle{display:flex;align-items:center;gap:7px;font-size:12px;cursor:pointer}
.learn-toggle input{accent-color:#245e47;width:19px;height:19px}
.learn-info{font-size:12px;color:#69747a;line-height:1.7;margin-top:8px;white-space:pre-wrap}
.learn-actions{display:flex;gap:8px;flex-wrap:wrap;margin-top:9px}
.learn-actions button{font-size:12px;border-radius:9px;border:1px solid #d5dcda;padding:7px 10px;background:#f7f9f8;cursor:pointer}
.learn-actions button:disabled{opacity:.5;cursor:not-allowed}
.practicebar{margin:10px 22px 0;padding:15px;background:#fff;border:1px solid #dfe3e6;border-radius:14px;box-shadow:0 3px 12px #00000007}
.practice-title{display:flex;align-items:center;justify-content:space-between;gap:12px;font-size:15px;font-weight:700}
.practice-pill{font-weight:600;font-size:11px;color:#5e696d;background:#eef0f1;padding:6px 9px;border-radius:18px}
.practice-config{display:grid;grid-template-columns:repeat(5,minmax(90px,1fr));gap:9px;margin-top:12px}
.practice-config label{font-size:11px;color:#596168;line-height:1.35;display:flex;flex-direction:column;gap:5px}
.practice-config input{width:100%;padding:8px 9px;border:1px solid #d7dcdf;border-radius:9px;font-size:13px;outline-color:#445a50;background:#fafbfb;color:#242526}
.practice-actions{display:flex;flex-wrap:wrap;gap:9px;margin-top:11px}
.practice-actions button{font-size:13px;font-weight:700;padding:10px 13px;border-radius:10px;border:1px solid #cad1d4;background:#fff;color:#242526;cursor:pointer}
.practice-actions button:first-child{background:#263e35;color:white;border-color:#263e35}
.practice-actions button:disabled{opacity:.44;cursor:not-allowed}
.practice-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin-top:12px}
.practice-stat{background:#f5f7f6;border-radius:10px;padding:10px 12px}
.practice-value{font-weight:750;font-size:19px;font-variant-numeric:tabular-nums}
.practice-label{font-size:11px;color:#69737a;margin-top:2px}
.practice-log{font-size:12px;color:#586468;line-height:1.6;margin-top:10px;white-space:pre-wrap;overflow-wrap:anywhere}
@media(max-width:640px){.practice-config{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:520px){.practicebar{margin:10px 13px 0}.practice-grid{grid-template-columns:repeat(3,minmax(0,1fr))}.practice-value{font-size:16px}}


.webbar{margin:10px 22px 0;padding:15px;background:#fff;border:1px solid #dfe3e6;border-radius:14px;box-shadow:0 3px 12px #00000007}
.webtop{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:9px}
.webtop strong{font-size:15px}
.webdesc{font-size:12px;color:#66747a;line-height:1.6;margin:7px 0 0}
.webconfig{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:9px;margin:13px 0 0}
.webconfig label{display:flex;flex-direction:column;font-size:11px;gap:5px;color:#596168}
.webconfig input,.webconfig select,.webtopics,.websearch input{width:100%;min-width:0;padding:9px;border:1px solid #d7dcdf;border-radius:9px;font-size:13px;background:#fafbfb;color:#242526;font:inherit}
.webtopics{resize:vertical;margin-top:7px;min-height:55px}
.webactions{display:flex;gap:9px;flex-wrap:wrap;margin-top:11px}
.webactions button,.websearch button,.review-button{background:#fff;color:#253c34;border:1px solid #cad1d4;border-radius:9px;padding:9px 11px;font-size:12px;cursor:pointer;font-weight:650}
.webactions button:first-child,.websearch button{background:#263e35;color:white;border-color:#263e35}
.webactions button:disabled,.websearch button:disabled{opacity:.45;cursor:default}
.webstats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;margin-top:12px}
.webstat{background:#f5f7f6;border-radius:10px;padding:10px 12px}
.webstat b{display:block;font-size:18px}.webstat span{font-size:11px;color:#6a747a}
.websearch{margin-top:12px;display:flex;gap:7px;align-items:center;flex-wrap:wrap}
.websearch input{flex:1 1 200px}.websearch a{font-size:12px;color:#285744;text-decoration:underline}
.weblast{margin-top:10px;font-size:12px;line-height:1.6;color:#59666a;overflow-wrap:anywhere;white-space:pre-wrap}
.webnotes{margin-top:10px;display:grid;gap:8px}
.webnote{padding:11px;border:1px solid #e2e6e7;border-radius:10px;background:#fafbf9}
.webnote a{font-size:13px;color:#27644c;word-break:break-word}.webnote p{font-size:12px;line-height:1.6;color:#525d62;white-space:pre-wrap;overflow-wrap:anywhere;max-height:7.4em;overflow:hidden;margin:5px 0}
.webnote small{font-size:11px;color:#7a8588}.webnote button{margin:5px 6px 0 0;padding:5px 8px;font-size:11px}
@media(max-width:640px){.webconfig{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:520px){.webbar{margin:10px 13px 0}.webstats{grid-template-columns:repeat(2,minmax(0,1fr))}}

.qabar{margin:10px 22px 0;padding:15px;background:#fff;border:1px solid #dfe3e6;border-radius:14px;box-shadow:0 3px 12px #00000007}
.qabar h2{font-size:15px;margin:0 0 7px}.qabar .qahelp{font-size:12px;color:#5d6a6f;line-height:1.6;margin:5px 0 12px}
.qaform{display:grid;gap:9px}.qaform label{font-size:12px;display:flex;flex-direction:column;gap:5px;color:#52615f}
.qaform input,.qaform textarea,.qaform select,.qaconfig input,.qaconfig select{width:100%;border:1px solid #d7dcdf;background:#fafbfb;padding:9px 10px;border-radius:9px;font:inherit;font-size:13px;color:#222}
.qaform textarea{resize:vertical;min-height:65px}.qaactions{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin:9px 0}
.qaactions button,.qacard button{padding:8px 11px;border-radius:9px;border:1px solid #cad1d4;background:#f7f9f8;font-size:12px;cursor:pointer;color:#244b39}
.qaactions button:first-child{color:white;background:#263e35;border-color:#263e35}.qaactions button:disabled,.qacard button:disabled{opacity:.4}
.qastats{font-size:12px;color:#536266;line-height:1.7;white-space:pre-wrap;margin:8px 0}
.qacards{display:grid;gap:8px;max-height:360px;overflow:auto}.qacard{border:1px solid #e4e9e8;padding:9px;border-radius:9px;font-size:12px;white-space:pre-wrap;overflow-wrap:anywhere}
.qacard small{display:block;color:#68747a;margin-top:5px}.qaconfig{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;margin-top:8px}
.qaconfig label{font-size:11px;display:flex;flex-direction:column;gap:5px}
@media(max-width:520px){.qabar{margin:10px 13px 0}.qaconfig{grid-template-columns:repeat(2,minmax(0,1fr))}}

.chat{flex:1;padding:27px 22px;display:flex;flex-direction:column;gap:18px;min-height:320px;overflow-y:auto}
.message{max-width:87%;padding:13px 16px;border-radius:16px;white-space:pre-wrap;line-height:1.65;overflow-wrap:anywhere;font-size:14.5px}
.message.user{align-self:flex-end;background:#222427;color:#fff;border-bottom-right-radius:5px}
.message.bot{align-self:flex-start;background:white;border:1px solid #e4e6e8;border-bottom-left-radius:5px}
.examples{display:flex;flex-wrap:wrap;gap:8px;margin-top:8px}
.sample{border:1px solid #dde0e3;background:white;padding:10px 13px;border-radius:10px;cursor:pointer;color:#4b5054}
.composer{padding:16px 22px 24px;background:#f7f7f5;border-top:1px solid #e5e6e7}
.frame{border:1px solid #d6d8db;border-radius:16px;background:white;padding:13px 14px;box-shadow:0 5px 18px #00000006}
textarea{width:100%;resize:vertical;min-height:70px;max-height:220px;border:0;outline:0;background:transparent;font:inherit;color:#202427;line-height:1.5}
.actions{display:flex;justify-content:space-between;align-items:center;gap:10px}
.hint{font-size:11px;color:#868b90}button.send{color:white;background:#222427;border:none;border-radius:11px;padding:11px 18px;font-weight:600;cursor:pointer}
button.send:disabled{opacity:.5;cursor:wait}.error{color:#b0473e}.disclaimer{font-size:12px;text-align:center;color:#858a8f;margin:11px 0 0}
@media(max-width:520px){.header{padding:16px 13px}.chat{padding:19px 13px}.composer{padding:12px 13px 18px}.message{max-width:95%}.hint{font-size:10px}.learnbar{margin:0 13px}}
</style>
</head>
<body>
<div class="app">
 <header class="header"><div class="logo">SRT</div><div><h1>SRT-M15</h1><p class="small" id="sub"></p></div><span class="status" id="status">Local</span></header>
 <section class="learnbar" aria-label="Autonomous learning controls">
  <div class="learn-top"><span class="learn-title">自学习实验 <small>仅限通过验证的个位数加法</small></span>
   <label class="learn-toggle"><input type="checkbox" id="learning-toggle">启用自动学习</label></div>
  <div id="learning-info" class="learn-info" aria-live="polite">正在读取学习状态...</div>
  <div class="learn-actions"><button type="button" id="train-now">手动训练队列</button><button type="button" id="rollback">撤销上次更新</button></div>
 </section>
 <section class="practicebar" aria-label="Autonomous practice controls">
  <div class="practice-title"><span>自主练习 <span style="font-size:11px;color:#6b747c;font-weight:400">自动出题 · 验证 · 受控训练</span></span><span id="practice-state" class="practice-pill">未启动</span></div>
  <div class="practice-config">
   <label>练习间隔（秒）<input id="practice-interval" type="number" min="0.2" max="60" step="0.1" value="1"></label>
   <label>每轮题数<input id="practice-round" type="number" min="1" max="12" step="1" value="4"></label>
   <label>最多训练次数<input id="practice-attempt-limit" type="number" min="1" max="8" step="1" value="2"></label>
   <label>最长运行（秒）<input id="practice-duration" type="number" min="1" max="3600" step="1" value="180"></label>
   <label>最多练习题数<input id="practice-question-limit" type="number" min="1" max="500" step="1" value="100"></label>
  </div>
  <div class="practice-actions"><button type="button" id="practice-start">▶ 开始自主练习</button><button type="button" id="practice-pause" disabled>⏸ 暂停</button></div>
  <div class="practice-grid" aria-live="polite">
   <div class="practice-stat"><div class="practice-value" id="practice-questions">0</div><div class="practice-label">练习题数</div></div>
   <div class="practice-stat"><div class="practice-value" id="practice-errors">0</div><div class="practice-label">预测错误</div></div>
   <div class="practice-stat"><div class="practice-value" id="practice-pending">0</div><div class="practice-label">待学习样本</div></div>
   <div class="practice-stat"><div class="practice-value" id="practice-attempts">0</div><div class="practice-label">训练尝试次数</div></div>
   <div class="practice-stat"><div class="practice-value" id="practice-accepted">0</div><div class="practice-label">接受更新次数</div></div>
   <div class="practice-stat"><div class="practice-value" id="practice-rounds">0</div><div class="practice-label">练习轮次</div></div>
  </div>
  <div class="practice-log" id="practice-last">只有训练池的个位数加法参与练习；保留检验集与测试集不用于训练。</div>
 </section>
 <section class="webbar" aria-label="Bounded Internet research">
  <div class="webtop"><strong>联网自主学习 <span id="web-state" class="practice-pill">未启用</span></strong>
   <label class="learn-toggle"><input type="checkbox" id="web-enabled">明确允许联网</label></div>
  <p class="webdesc">仅通过 HTTPS 使用中/英文维基百科的公开只读接口；自动收集主题摘要、保存出处、可以人工审核。<b>不会自动把网页文本写入神经网络权重。</b>学习主题会发送给 Wikimedia，默认关闭。</p>
  <label style="display:block;font-size:12px;color:#596168;margin-top:10px">自主学习主题（英文逗号或中文逗号分隔，最多10个）</label>
  <textarea id="web-topics" class="webtopics" maxlength="500" rows="2">人工智能, 神经网络, 机器学习</textarea>
  <div class="webconfig">
   <label>百科语言<select id="web-language"><option value="zh">中文</option><option value="en">English</option></select></label>
   <label>最短检索间隔（秒）<input id="web-interval" type="number" min="8" max="120" step="1" value="12"></label>
   <label>最多联网请求<input id="web-max-requests" type="number" min="1" max="30" step="1" value="12"></label>
   <label>最长运行（秒）<input id="web-duration" type="number" min="10" max="3600" step="1" value="180"></label>
   <label>本次最多新资料<input id="web-max-records" type="number" min="1" max="60" step="1" value="24"></label>
  </div>
  <div class="webactions"><button id="web-start" type="button">🌐 开始联网学习</button><button id="web-pause" type="button" disabled>⏸ 暂停联网</button></div>
  <div class="webstats" aria-live="polite">
   <div class="webstat"><b id="web-requests">0</b><span>本次网络请求</span></div>
   <div class="webstat"><b id="web-saved">0</b><span>累计资料记录</span></div>
   <div class="webstat"><b id="web-reviewed">0</b><span>已人工审核</span></div>
   <div class="webstat"><b id="web-errors">0</b><span>联网失败次数</span></div>
  </div>
  <div class="websearch"><input id="web-query" maxlength="60" placeholder="输入检索主题，如：Transformer"><button id="web-search-btn" type="button">立即检索</button><a href="/api/web/export" download="srt_m15_reviewed_qa.jsonl">导出已审核资料</a></div>
  <div id="web-last" class="weblast">尚未启动。你也可以在聊天框输入“联网搜索：神经网络”。</div>
  <div id="web-notes" class="webnotes"></div>
 </section>


 <section class="qabar" aria-label="Human reviewed language training">
  <h2>中文 + English · 双语持续学习 <span id="qa-state" class="practice-pill">未训练</span></h2>
  <p class="qahelp">使用同一个 UTF-8 字节模型学习中文与英文；中英文配对示例保持相同训练/检验分区，并分别检查两种语言的验证损失。<b>网页摘要不是训练真值；所有问答必须逐条校对并批准。</b>本模型规模小，不能保证中英文流畅回答。</p>
  <div class="qaform">
   <input type="hidden" id="qa-edit-id">
   <label>已审核百科资料编号（可选）<input type="number" min="1" id="qa-note-id" placeholder="从上方已审核资料复制 ID；留空则表示手工编写"></label>
   <div class="qaactions"><button type="button" id="qa-suggest">从已审核资料生成草稿</button></div>
   <label>训练语种<select id="qa-language"><option value="zh">中文 zh</option><option value="en">English en</option></select></label>
   <label>对应翻译分组（可选，同一中英对填写同一个代号）<input id="qa-group" maxlength="64" placeholder="例如：greeting_01"></label>
   <label>问题（请校对）<input id="qa-question" maxlength="60" placeholder="例：你是谁？"></label>
   <label>答案（请先核实，建议简短）<textarea id="qa-answer" maxlength="95" rows="2" placeholder="例：我是一个从零训练的实验模型。"></textarea></label>
  </div>
  <div class="qaactions"><button id="qa-save" type="button">保存待审核问答</button><button id="qa-bilingual-examples" type="button">导入24组中英双语示例（48条）</button><button id="qa-examples" type="button">旧版20条中文示例</button><span class="qahelp">编码合计最多160字节；不允许超长静默截断。全部示例为待审核草稿。</span></div>
  <div class="qastats" id="qa-stats">正在读取已审核语料...</div>
  <div class="qacards" id="qa-recent"></div>
  <div class="qaconfig"><label>训练语言<select id="qa-mode"><option value="bilingual">中文 + English（均衡学习）</option><option value="zh">仅中文</option><option value="en">仅 English</option><option value="auto">旧版自动</option></select></label><label>训练步数<input id="qa-steps" type="number" min="1" max="100" value="60"></label><label>最长训练秒数<input id="qa-runtime" type="number" min="15" max="240" value="150"></label></div>
  <div class="qaactions"><button id="qa-train" type="button">🧠 训练候选语言模型</button><button id="qa-stop" type="button">暂停训练</button><button id="qa-rollback" type="button">回滚语言权重</button></div>
  <div class="qastats" id="qa-result">双语训练要求中文和英文各至少3条训练与2条独立检验样本，并通过双语检验。</div>
 </section>

 <main class="chat" id="chat"><div class="message bot" id="intro"></div><div class="examples" id="examples"><button class="sample" data-prompt="math">1+9=?</button><button class="sample" data-prompt="zh">你好</button><button class="sample" data-prompt="hello">Hello</button></div></main>
 <footer class="composer"><div class="frame"><label class="hint" for="chat-language">回复语言：</label><select id="chat-language" aria-label="Chat language" style="border:1px solid #d6d8db;border-radius:7px;padding:6px;background:#fff"><option value="auto">自动识别</option><option value="zh">中文</option><option value="en">English</option></select><textarea id="message" maxlength="400" aria-label="Your message"></textarea>
 <div class="actions"><span class="hint" id="hint"></span><button id="reset-chat" type="button" style="padding:8px;border:1px solid #d6d8db;border-radius:9px;background:#fff;cursor:pointer">新对话</button><button class="send" id="send" type="button"></button></div></div><p class="disclaimer" id="disclaimer"></p></footer>
</div>
<script>
'use strict';
const $=id=>document.getElementById(id);
$('sub').textContent='\u672c\u5730\u8fd0\u884c\u00b7\u81ea\u7814\u7f51\u7edc\u00b7\u4e0d\u8c03\u7528\u5916\u90e8\u5927\u6a21\u578b';
$('intro').textContent='\u4f60\u597d\uff01\u8fd9\u662f SRT-M15 \u5b9e\u9a8c\u6a21\u578b\u3002\n\n\u53ef\u6d4b\u8bd5\u4e24\u4e2a 0\uff5e9 \u6574\u6570\u76f8\u52a0\uff08\u5982 1+9=\u591a\u5c11\uff09\uff0c\u6216\u4f7f\u7528\u5c0f\u578b\u5b57\u7b26\u6a21\u578b\u5c1d\u8bd5\u6587\u672c\u7eed\u5199\u3002';
$('message').placeholder='\u76f4\u63a5\u8f93\u5165\u95ee\u9898\uff0c\u4f8b\u5982\uff1a1+9=\u591a\u5c11';
$('hint').textContent='Enter \u53d1\u9001  \u00b7  Shift+Enter \u6362\u884c';
$('send').textContent='\u53d1\u9001';
$('disclaimer').textContent='\u6ce8\u610f\uff1a\u8fd9\u662f\u7814\u7a76\u539f\u578b\uff0c\u5b57\u7b26\u7eed\u5199\u4e0d\u7b49\u4e8e\u901a\u7528\u95ee\u7b54\u80fd\u529b\u3002';
$('status').textContent='\u6b63\u5728\u68c0\u67e5\u6a21\u578b';
function bubble(role, value){const n=document.createElement('div');n.className='message '+role;n.textContent=value;$('chat').appendChild(n);n.scrollIntoView({behavior:'smooth',block:'end'});}
let working=false;
const chatHistory=[];
$('reset-chat').addEventListener('click',()=>{chatHistory.length=0;bubble('bot','已开始新对话，之前的上下文不会继续发送给模型。');});
async function submit(){const question=$('message').value.trim();if(!question||working)return;
 $('message').value='';bubble('user',question);working=true;$('send').disabled=true;
 try{const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:question,language:$('chat-language').value,history:chatHistory})});
 const data=await r.json();if(!r.ok)throw new Error(data.error||('HTTP '+r.status));
 const answer=data.text||'';
 bubble('bot',(data.provider?'【'+data.provider+'】\n':'')+answer);
 if(answer){chatHistory.push({role:'user',content:question},{role:'assistant',content:answer});if(chatHistory.length>16)chatHistory.splice(0,chatHistory.length-16);}
 }
 catch(e){bubble('bot','\u6267\u884c\u9519\u8bef\uff1a'+e.message);}
 finally{working=false;$('send').disabled=false;$('message').focus();refreshStatus();}}
$('send').addEventListener('click',submit);
$('message').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();submit();}});
for(const el of document.querySelectorAll('.sample')){el.addEventListener('click',()=>{
 $('message').value=el.dataset.prompt==='math'?'1+9=\u591a\u5c11':el.dataset.prompt==='hello'?'Hello':'\u4f60\u597d';
 $('chat-language').value=el.dataset.prompt==='hello'?'en':el.dataset.prompt==='zh'?'zh':'auto';$('message').focus();
});}
async function postLearning(path, body={}){
 const res=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
 const data=await res.json();if(!res.ok)throw new Error(data.error||('HTTP '+res.status));return data;
}

let qaSwitching=false;
function renderQA(q){
 if(!q)return;
 const running=!!q.running;
 $('qa-state').textContent=running?'训练候选中':(q.trained_checkpoint?'已训练·实验模型':'尚未有字节问答权重');
 const tc=q.train_by_language||{},vc=q.holdout_by_language||{},ac=q.approved_by_language||{};
 $('qa-stats').textContent=`已批准 ${q.approved||0}（中文 ${ac.zh||0} / EN ${ac.en||0}） · 待审核 ${q.draft||0}\n中文训练 ${tc.zh||0} / 检验 ${vc.zh||0}；English 训练 ${tc.en||0} / 检验 ${vc.en||0}\n训练尝试 ${q.attempts||0} · 已接受 ${q.accepted_updates||0} 次`;
 const mode=$('qa-mode').value;
 const hasEnough=mode==='bilingual'?((tc.zh||0)>=3&&(tc.en||0)>=3&&(vc.zh||0)>=2&&(vc.en||0)>=2):
  mode==='zh'?((tc.zh||0)>=4&&(vc.zh||0)>=2):
  mode==='en'?((tc.en||0)>=4&&(vc.en||0)>=2):((q.training_samples||0)>=4&&(q.validation_samples||0)>=2);
 $('qa-train').disabled=running||qaSwitching||!hasEnough;
 $('qa-stop').disabled=!running;
 $('qa-rollback').disabled=running||!q.can_rollback;
 const last=q.last_result||{};
 const stateText={not_started:'尚未训练',training:'候选正在训练',accepted:'已接受更新',rejected:'候选未通过检查',cancelled:'训练已暂停',timeout:'超过时长上限',error:'候选训练失败',rolled_back:'已恢复上一版'};
 const perf=last.per_language||{};
 const line=lang=>perf[lang]?`\n${lang.toUpperCase()}：训练 ${perf[lang].train_before} → ${perf[lang].train_after}；检验 ${perf[lang].validation_before} → ${perf[lang].validation_after}`:'';
 $('qa-result').textContent='最近结果：'+(stateText[last.state]||last.state||'尚无')+(last.reason?' · '+last.reason:'')+
 (last.train_before!==undefined?`\n训练损失 ${last.train_before} → ${last.train_after}；独立验证损失 ${last.validation_before} → ${last.validation_after}`:'')+
 line('zh')+line('en')+
 '\n保留原始数学和字符模型权重；本阶段仍不能可靠回答一般问题。';
 const cards=$('qa-recent');cards.replaceChildren();
 for(const item of (q.recent||[])){
  const div=document.createElement('div');div.className='qacard';
  const head=document.createElement('strong');head.textContent='问题：'+item.question;div.append(head);
  const ans=document.createElement('div');ans.textContent='答案：'+item.answer;div.append(ans);
  const meta=document.createElement('small');meta.textContent='编号 '+item.id+' · '+(item.language==='en'?'English':'中文')+(item.group_key?' · 译文组 '+item.group_key:'')+' · '+({draft:'待审核',approved:'已人工批准',rejected:'已拒绝'}[item.status]||item.status)+' · 来源：'+(item.source_title||'手工编写');div.append(meta);
  if(item.status==='draft'||item.status==='approved'){
   const edit=document.createElement('button');edit.textContent='修改问答';edit.type='button';
   edit.onclick=()=>{
    $('qa-edit-id').value=item.id;$('qa-note-id').value=item.source_note_id||'';
    $('qa-question').value=item.question;$('qa-answer').value=item.answer;
    $('qa-language').value=item.language||'zh';$('qa-group').value=item.group_key||'';
    $('qa-save').textContent='保存修改（重置审核）';$('qa-question').scrollIntoView({behavior:'smooth'});
   };div.append(edit);
   const choices=item.status==='draft'?[[true,'批准训练样本'],[false,'拒绝']]:[[false,'撤销批准']];
   for(const [approved,title] of choices){
    const button=document.createElement('button');button.textContent=title;button.type='button';
    button.onclick=()=>reviewQA(item.id,approved);div.append(button);
   }
  }
  cards.append(div);
 }
}
async function suggestQA(note_id){
 try{
  const info=await postLearning('/api/qa/suggest',{note_id});
  $('qa-note-id').value=info.note_id;$('qa-question').value=info.question;$('qa-answer').value=info.answer;
  $('qa-language').value=info.language||'zh';$('qa-group').value='';
  bubble('bot',info.notice);
 }catch(e){bubble('bot','生成资料草稿失败：'+e.message);}
}
$('qa-suggest').addEventListener('click',()=>suggestQA(Number($('qa-note-id').value)));
$('qa-mode').addEventListener('change',()=>refreshStatus());
$('qa-bilingual-examples').addEventListener('click',async()=>{
 if(!confirm('导入24组中英双语配对示例，共48条草稿。示例不会自动获批；需要你检查每一条。继续？'))return;
 try{const r=await postLearning('/api/qa/bilingual-examples');bubble('bot','已新增双语示例草稿：'+r.new_drafts+' 条，共 '+r.pairs+' 对；请逐条审核。');await refreshStatus();}
 catch(e){bubble('bot','导入双语示例失败：'+e.message);}
});
$('qa-examples').addEventListener('click',async()=>{
 if(!confirm('将添加20条由开发者编写的演示问答，全部仍为待审核状态。需要你逐条批准。'))return;
 try{const result=await postLearning('/api/qa/examples');bubble('bot','已新增演示草稿：'+result.new_drafts+' 条。请逐条审核，不能直接作为真实知识。');await refreshStatus();}
 catch(e){bubble('bot','导入演示草稿失败：'+e.message);}
});
$('qa-save').addEventListener('click',async()=>{
 try{
  const note=$('qa-note-id').value.trim(),edit=$('qa-edit-id').value.trim();
  const payload={question:$('qa-question').value,answer:$('qa-answer').value,language:$('qa-language').value,group_key:$('qa-group').value};
  if(edit){await postLearning('/api/qa/update',{id:Number(edit),...payload});}
  else {await postLearning('/api/qa/draft',{...payload,note_id:note?Number(note):null});}
  $('qa-edit-id').value='';$('qa-save').textContent='保存待审核问答';
  bubble('bot','问答草稿已保存。请逐条人工批准，批准前不会训练。');await refreshStatus();
 }catch(e){bubble('bot','保存问答失败：'+e.message);}
});
async function reviewQA(id,approved){
 try{await postLearning('/api/qa/review',{id,approved});await refreshStatus();}
 catch(e){bubble('bot','问答审核失败：'+e.message);}
}
$('qa-train').addEventListener('click',async()=>{
 if(!confirm('仅使用逐条人工批准的问答进行有上限的本地训练。候选可能因验证未通过而被拒绝。是否开始？'))return;
 qaSwitching=true;
 try{await postLearning('/api/qa/train',{steps:Number($('qa-steps').value),max_runtime_seconds:Number($('qa-runtime').value),language_mode:$('qa-mode').value});await refreshStatus();}
 catch(e){bubble('bot','语言训练启动失败：'+e.message);}
 finally{qaSwitching=false;await refreshStatus();}
});
$('qa-stop').addEventListener('click',async()=>{
 try{await postLearning('/api/qa/stop');await refreshStatus();}catch(e){bubble('bot','暂停失败：'+e.message);}
});
$('qa-rollback').addEventListener('click',async()=>{
 if(!confirm('回滚语言模型候选权重？数学与原始字符模型不受影响。'))return;
 try{await postLearning('/api/qa/rollback');bubble('bot','语言模型权重已回滚。');await refreshStatus();}catch(e){bubble('bot','回滚失败：'+e.message);}
});

let switching=false;
function configFromUI(){
 return {
  interval_seconds:Number($('practice-interval').value),
  questions_per_round:Number($('practice-round').value),
  max_training_attempts:Number($('practice-attempt-limit').value),
  max_runtime_seconds:Number($('practice-duration').value),
  max_questions:Number($('practice-question-limit').value)
 };
}
let practiceSwitching=false;
function renderPractice(p){
 const running=!!p.running;
 const reasonNames={not_started:'尚未开始',manual_pause:'已手动暂停',max_runtime:'达到时长上限',max_questions:'达到题数上限',max_training_attempts:'达到训练次数上限',global_training_limit:'达到累计训练上限',learning_disabled:'自动学习已关闭',interrupted_by_restart:'程序重启后保持暂停'};
 const phaseNames={practicing:'练习中',waiting_training:'正在训练候选模型',stopped:'已停止'};
 $('practice-state').textContent=phaseNames[p.phase]||'未启动';
 $('practice-start').disabled=running||practiceSwitching;
 $('practice-pause').disabled=!running||practiceSwitching;
 for(const n of ['practice-interval','practice-round','practice-attempt-limit','practice-duration','practice-question-limit'])$(n).disabled=running;
 $('practice-questions').textContent=p.total_questions||0;
 $('practice-errors').textContent=p.total_errors||0;
 $('practice-pending').textContent=p.pending||0;
 $('practice-attempts').textContent=p.training_attempts||0;
 $('practice-accepted').textContent=p.accepted_updates||0;
 $('practice-rounds').textContent=p.total_rounds||0;
 $('train-now').disabled=$('train-now').disabled||running;
 $('rollback').disabled=$('rollback').disabled||running;
 const last=p.last_problem;
 const train=p.last_training_result;
 const trainNames={training:'候选训练中',accepted:'已通过独立检验并保存',rejected:'候选未通过验证',cancelled:'训练已取消，样本保留',error:'训练出错'};
 const detail=last?('最近题目：'+last.operands[0]+' + '+last.operands[1]+' = '+last.verified_answer+' · 模型预测 '+last.prediction+' · '+(last.wrong?'错误':'正确')+' · '+(last.learning_event||'')):'尚无自主练习记录';
 const outcome=train?('最近候选：'+(trainNames[train.state]||train.state)):'尚未进行候选训练';
 const reason=p.last_stop_reason?('停止原因：'+(reasonNames[p.last_stop_reason]||p.last_stop_reason)):'本次运行：'+(p.session_questions||0)+' 题';
 $('practice-last').textContent=detail+'\n'+outcome+' · '+reason+'\n只采样训练池；检验集与测试集不出题、不参与梯度训练。';
}
$('practice-start').addEventListener('click',async()=>{
 if(practiceSwitching)return;practiceSwitching=true;
 try{await postLearning('/api/practice/start',configFromUI());await refreshStatus();}
 catch(e){bubble('bot','自主练习启动失败：'+e.message);}
 finally{practiceSwitching=false;await refreshStatus();}
});
$('practice-pause').addEventListener('click',async()=>{
 if(practiceSwitching)return;practiceSwitching=true;
 try{await postLearning('/api/practice/pause');await refreshStatus();}
 catch(e){bubble('bot','自主练习暂停失败：'+e.message);}
 finally{practiceSwitching=false;await refreshStatus();}
});
async function refreshStatus(){
 try{
 const r=await fetch('/api/status');if(!r.ok)throw new Error('HTTP '+r.status);
 const d=await r.json();
 $('status').textContent=d.chat_backend==='ollama'?'本地对话：'+d.ollama_model:(d.math_checkpoint&&d.language_checkpoint?'模型已就绪':'部分权重缺失');
 if(d.chat_backend==='ollama'){
  $('sub').textContent='SRT-M15 对话界面 · Ollama 本地 Qwen 推理（非 SRT-M15 原生权重）';
  $('intro').textContent='你好！当前启用独立的本地 Qwen 指令模型进行中英文多轮对话。SRT-M15 自研模型及训练面板仍保留为实验功能。';
  $('disclaimer').textContent='对话来源：本机 Ollama / '+d.ollama_model+'。此模式不会更新 SRT-M15 自研权重。';
  for(const panel of document.querySelectorAll('.learnbar,.practicebar,.webbar,.qabar'))panel.style.display='none';
 }
 const l=d.learning||{};
 if(!switching)$('learning-toggle').checked=!!l.enabled;
 renderPractice(d.practice||{});
 renderWeb(d.web||{});
 renderQA(d.qa||{});
 const states={training:'候选参数训练中',accepted:'已验证并保存新权重',rejected:'未通过检验，已丢弃候选权重',cancelled:'暂停后已取消候选训练',error:'训练失败，原权重保留',rolled_back:'已撤销上一次更新'};
 const latest=l.last_result?('  最近结果：'+(states[l.last_result.state]||l.last_result.state)):'';
 const trigger=l.enabled?'已开启':'已关闭';
 $('learning-info').textContent=trigger+' · 待学习样本：'+(l.pending||0)+' / '+(l.threshold||3)+' · 已接受更新：'+(l.accepted_updates||0)+' 次'+(l.phase==='training'?' · 训练进行中':'')+'\n仅使用验证器确认的错误加法；原测试集不参与学习。'+latest;
 $('train-now').disabled=!l.enabled||l.phase==='training'||!l.pending;
 $('rollback').disabled=l.phase==='training'||!l.can_rollback;
 }catch(e){$('status').textContent='状态检查失败';$('learning-info').textContent=e.message;}
}
$('learning-toggle').addEventListener('change',async e=>{
 switching=true;$('learning-toggle').disabled=true;
 try{await postLearning('/api/learning/toggle',{enabled:!!e.target.checked});}
 catch(error){bubble('bot','学习开关错误：'+error.message);e.target.checked=!e.target.checked;}
 finally{switching=false;$('learning-toggle').disabled=false;await refreshStatus();}
});
$('train-now').addEventListener('click',async()=>{
 try{await postLearning('/api/learning/run');await refreshStatus();}
 catch(e){bubble('bot','学习操作错误：'+e.message);}
});
$('rollback').addEventListener('click',async()=>{
 if(!confirm('确定撤销最近一次学习更新吗？原始模型权重始终保留。'))return;
 try{await postLearning('/api/learning/rollback');bubble('bot','已恢复上一版数学模型权重。');await refreshStatus();}
 catch(e){bubble('bot','回滚错误：'+e.message);}
});

let webSwitching=false, webSearchBusy=false, webConfigLoaded=false;
function webSettings(){return {
 topics:$('web-topics').value,language:$('web-language').value,
 interval_seconds:Number($('web-interval').value),max_requests:Number($('web-max-requests').value),
 max_runtime_seconds:Number($('web-duration').value),max_new_records:Number($('web-max-records').value)
};}
function renderWeb(w){
 const active=!!w.running, enabled=!!w.enabled;
 if(!webSwitching)$('web-enabled').checked=enabled;
 if(!webConfigLoaded && w.config){
  $('web-topics').value=w.config.topics||'';
  $('web-language').value=w.config.language||'zh';
  $('web-interval').value=w.config.interval_seconds||12;
  $('web-max-requests').value=w.config.max_requests||12;
  $('web-duration').value=w.config.max_runtime_seconds||180;
  $('web-max-records').value=w.config.max_new_records||24;
  webConfigLoaded=true;
 }
 $('web-state').textContent=active?'正在采集':(enabled?'已授权·暂停':'未授权');
 $('web-start').disabled=!enabled||active||webSwitching;
 $('web-pause').disabled=!active||webSwitching;
 $('web-search-btn').disabled=!enabled||webSearchBusy;
 for(const id of ['web-topics','web-language','web-interval','web-max-requests','web-duration','web-max-records'])$(id).disabled=active;
 $('web-requests').textContent=w.session_requests||0;
 $('web-saved').textContent=w.stored||0;
 $('web-reviewed').textContent=w.approved||0;
 $('web-errors').textContent=w.errors||0;
 const reasons={not_started:'未开始',manual_pause:'手动暂停',max_runtime:'达到运行时长上限',max_requests:'达到请求次数上限',max_new_records:'达到资料数量上限',topics_exhausted:'主题已探索完毕',too_many_errors:'网络连续失败，自动停止',rate_limited:'来源限流或不可用，自动停止',interrupted_by_restart:'重启后默认不自动联网'};
 $('web-last').textContent=(w.last_result||'尚无联网记录')+(w.last_topic?'\n最近学习主题：'+w.last_topic:'')+
 (w.last_error?'\n连接问题：'+w.last_error:'')+'\n'+(w.running?'联网学习中':(reasons[w.stop_reason]||'等待开始'))+
 ' · '+(w.pending_review||0)+'条待审核资料（不是已经验证的事实）';
 const container=$('web-notes'); container.replaceChildren();
 for(const note of (w.recent||[])){
  const card=document.createElement('article');card.className='webnote';
  const a=document.createElement('a'); a.textContent=note.title; a.target='_blank';a.rel='noopener noreferrer';
  try{const url=new URL(note.url);if(url.protocol==='https:'&&['zh.wikipedia.org','en.wikipedia.org'].includes(url.hostname))a.href=url.href;}catch(e){}
  card.append(a);
  const p=document.createElement('p');p.textContent=note.summary;card.append(p);
  const m=document.createElement('small'); m.textContent='来源：'+note.language+'维基百科 · '+note.fetched_at+' · '+({'pending':'待审核','approved':'人工已标记','rejected':'已拒绝'}[note.review]||note.review);card.append(m);
  if(note.review!=='rejected'){
   const yes=document.createElement('button');yes.className='review-button';yes.textContent=note.review==='approved'?'已标记审核':'标记已审核';
   yes.disabled=note.review==='approved';yes.onclick=()=>reviewWeb(note.id,true);card.append(yes);
   if(note.review==='approved'){
    const qa=document.createElement('button');qa.className='review-button';qa.textContent='生成问答草稿';qa.onclick=()=>suggestQA(note.id);card.append(qa);
   }
   const no=document.createElement('button');no.className='review-button';no.textContent='拒绝此资料';no.onclick=()=>reviewWeb(note.id,false);card.append(no);
  }
  container.append(card);
 }
}
async function reviewWeb(id, approved){
 try{await postLearning('/api/web/review',{id,approved});await refreshStatus();}
 catch(e){bubble('bot','资料审核操作失败：'+e.message);}
}
$('web-enabled').addEventListener('change',async(e)=>{
 if(webSwitching)return;webSwitching=true;$('web-enabled').disabled=true;
 try{await postLearning('/api/web/toggle',{enabled:!!e.target.checked});}
 catch(err){bubble('bot','联网开关错误：'+err.message);e.target.checked=!e.target.checked;}
 finally{webSwitching=false;$('web-enabled').disabled=false;await refreshStatus();}
});
$('web-start').addEventListener('click',async()=>{
 if(webSwitching)return;webSwitching=true;
 try{await postLearning('/api/web/start',webSettings());await refreshStatus();}
 catch(e){bubble('bot','联网学习启动失败：'+e.message);}
 finally{webSwitching=false;await refreshStatus();}
});
$('web-pause').addEventListener('click',async()=>{
 if(webSwitching)return;webSwitching=true;
 try{await postLearning('/api/web/pause');await refreshStatus();}
 catch(e){bubble('bot','联网暂停失败：'+e.message);}
 finally{webSwitching=false;await refreshStatus();}
});
$('web-search-btn').addEventListener('click',async()=>{
 if(webSearchBusy)return;
 webSearchBusy=true;$('web-search-btn').disabled=true;
 try{
  const query=$('web-query').value.trim();
  if(!query)throw new Error('请先输入检索主题');
  const data=await postLearning('/api/web/search',{query,language:$('web-language').value});
  let text='联网检索结果：'+query+'\n';
  for(const s of data.sources||[])text+='\n【'+s.title+'】\n'+s.summary.slice(0,430)+'\n来源：'+s.url+'\n';
  bubble('bot',data.found?text:'维基百科没有返回有效摘要，请更换关键词。');
 }catch(e){bubble('bot','联网检索失败：'+e.message);}
 finally{webSearchBusy=false;await refreshStatus();}
});

refreshStatus();setInterval(refreshStatus,3500);
</script>
</body></html>'''


def prepare_model_root():
    """Support both the full source ZIP and the GitHub notebook-only layout."""
    if all((ROOT / name).is_file() for name in ("srt_m15/model.py", "srt_m15/language.py", "srt_m15/tasks.py")):
        model_root = ROOT
    else:
        notebook = ROOT / "SRT_M15_Phone_Colab.ipynb"
        if not notebook.is_file():
            raise FileNotFoundError(
                "Missing model sources. Put SRT_M15_Phone_Colab.ipynb next to start_local.py, "
                "or use the complete SRT-M15 local ZIP."
            )
        model_root = ROOT / "runtime_model"
        if not (model_root / "srt_m15/model.py").is_file():
            notebook_data = json.loads(notebook.read_text(encoding="utf-8"))
            embedded = None
            for cell in notebook_data.get("cells", []):
                if cell.get("cell_type") == "code":
                    code = "".join(cell.get("source", []))
                    match = re.search(r'^SOURCE_ZIP_BASE64\s*=\s*"([A-Za-z0-9+/=]+)"', code, re.M)
                    if match:
                        embedded = match.group(1)
                        break
            if not embedded:
                raise RuntimeError("The notebook contains no SOURCE_ZIP_BASE64 source bundle")
            with zipfile.ZipFile(io.BytesIO(base64.b64decode(embedded))) as z:
                for item in z.infolist():
                    parts = PurePosixPath(item.filename).parts
                    if not parts or parts[0] != "SRT_M15_Starter" or ".." in parts:
                        raise ValueError("Unexpected archive entry")
                    relative = parts[1:]
                    if not relative or item.is_dir():
                        continue
                    target = model_root.joinpath(*relative)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(z.read(item))
            print("Extracted SRT-M15 source into", model_root)
        # Overlay integrations on the extracted notebook source. The notebook
        # may contain an older bundled copy of these small Python modules.
        for filename in ("chat.py", "learning.py", "autolearn.py", "practice.py", "web_learning.py"):
            source = ROOT / "srt_m15" / filename
            if not source.is_file():
                raise FileNotFoundError(f"Missing srt_m15/{filename} beside start_local.py")
            target = model_root / "srt_m15" / filename
            if not target.exists() or target.read_bytes() != source.read_bytes():
                shutil.copyfile(source, target)
    results = ROOT / "SRT_M15_Results.zip"
    if results.is_file():
        allowed = {"srt_m15_toy.pt", "srt_m15_lm.pt", "srt_m15_qa.pt"}
        with zipfile.ZipFile(results) as z:
            for name in z.namelist():
                rel = PurePosixPath(name)
                if len(rel.parts) == 2 and rel.parts[0] == "checkpoints" and rel.name in allowed:
                    folder = model_root / "checkpoints"
                    folder.mkdir(parents=True, exist_ok=True)
                    (folder / rel.name).write_bytes(z.read(name))
        print("Loaded previously saved model checkpoints from SRT_M15_Results.zip")
    if not (model_root / "checkpoints/srt_m15_toy.pt").is_file():
        print("NOTICE: math weights missing. Put SRT_M15_Results.zip beside start_local.py, or train locally.")
    if not (model_root / "checkpoints/srt_m15_lm.pt").is_file():
        print("NOTICE: language weights missing. Put SRT_M15_Results.zip beside start_local.py, or train locally.")
    return model_root


def run_server(model_root, port, open_browser, *, chat_backend="srt-m15",
               ollama_model="qwen3:4b-instruct"):
    sys.path.insert(0, str(model_root))
    from srt_m15.chat import SRTChatEngine
    from srt_m15.autolearn import AutoLearningController
    from srt_m15.practice import AutonomousPracticeRunner
    from srt_m15.web_learning import WebLearningController
    from srt_m15.supervised_language import ReviewedQAStore, LanguageLearningController
    engine = SRTChatEngine(model_root)
    if chat_backend not in ("srt-m15", "ollama"):
        raise ValueError("invalid chat backend")
    ollama = None
    if chat_backend == "ollama":
        from srt_m15.ollama_local import OllamaLocalClient
        ollama = OllamaLocalClient(model=ollama_model)
    guard = threading.RLock()
    learning = AutoLearningController(engine, guard)
    practice = AutonomousPracticeRunner(engine, guard, learning)
    web = WebLearningController(model_root)
    qa_store = ReviewedQAStore(model_root, web.store)
    qa = LanguageLearningController(model_root, qa_store, engine, guard)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, payload, mime="application/json; charset=utf-8"):
            data = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/":
                self._send(200, HTML_PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif self.path == "/api/web/export":
                content = web.store.export_approved_qa().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
                self.send_header("Content-Disposition", 'attachment; filename="srt_m15_reviewed_qa.jsonl"')
                self.send_header("Content-Length", str(len(content)))
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(content)
            elif self.path == "/api/status":
                self._send(200, {**engine.status(), "chat_backend": chat_backend,
                                 "ollama_model": (ollama_model if ollama is not None else None),
                                 "learning": learning.status(),
                                 "practice": practice.status(), "web": web.status(),
                                 "qa": qa.status()})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            allowed = {"/api/chat", "/api/learning/toggle", "/api/learning/run",
                       "/api/learning/rollback", "/api/practice/start", "/api/practice/pause",
                       "/api/web/toggle", "/api/web/start", "/api/web/pause",
                       "/api/web/search", "/api/web/review",
                       "/api/qa/suggest", "/api/qa/draft", "/api/qa/review", "/api/qa/update", "/api/qa/examples",
                       "/api/qa/bilingual-examples",
                       "/api/qa/train", "/api/qa/stop", "/api/qa/rollback"}
            if self.path not in allowed:
                return self._send(404, {"error": "not found"})
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                return self._send(415, {"error": "JSON required"})
            origin = self.headers.get("Origin")
            if origin and origin not in (f"http://127.0.0.1:{self.server.server_port}",
                                         f"http://localhost:{self.server.server_port}"):
                return self._send(403, {"error": "Invalid origin"})
            try:
                size = int(self.headers.get("Content-Length", ""))
                if not 0 < size <= (65536 if self.path == "/api/chat" else 8192):
                    raise ValueError("invalid payload size")
                body = json.loads(self.rfile.read(size).decode("utf-8"))
                if not isinstance(body, dict):
                    raise ValueError("JSON object required")
                if self.path == "/api/chat":
                    question = body["message"]
                    if not isinstance(question, str) or not 0 < len(question) <= 400:
                        raise ValueError("message must be 1..400 characters")
                    lang = body.get("language", "auto")
                    if lang not in ("auto", "zh", "en"):
                        raise ValueError("chat language must be auto, zh or en")
                    if ollama is not None:
                        history = body.get("history", [])
                        if not isinstance(history, list) or len(history) > 16 or len(history) % 2:
                            raise ValueError("history must be an even list of up to 16 messages")
                        for idx, item in enumerate(history):
                            required_role = "user" if idx % 2 == 0 else "assistant"
                            if (not isinstance(item, dict) or item.get("role") != required_role
                                    or not isinstance(item.get("content"), str)
                                    or not 1 <= len(item["content"].strip()) <= 4000):
                                raise ValueError("invalid conversation history")
                        language_hint = ("Please reply in English." if lang == "en" else
                                         "请用简体中文回答。" if lang == "zh" else
                                         "请使用与用户输入一致的语言回答，优先中文或英文。")
                        system_message = (
                            "You are a local instruction-tuned Qwen model accessed through "
                            "the SRT-M15 application interface. Do not claim your model weights "
                            "are SRT-M15's own trained weights. Be helpful and accurate. "
                            + language_hint
                        )
                        try:
                            with guard:
                                answer = ollama.chat(
                                    [{"role": "system", "content": system_message}]
                                    + history + [{"role": "user", "content": question}],
                                    max_tokens=256,
                                )
                        except Exception as exc:
                            from srt_m15.ollama_local import OllamaLocalError
                            if not isinstance(exc, OllamaLocalError):
                                raise
                            return self._send(503, {"error": str(exc)})
                        return self._send(200, {
                            "mode": "local_external_weights", "text": answer,
                            "provider": "Ollama 本地 / " + ollama_model,
                            "learning_event": "not_trainable"
                        })
                    trimmed = question.strip()
                    lookup = re.match(r"^(?:联网搜索|联网查询)\s*[:：]\s*(.+)$", trimmed, re.S)
                    if lookup:
                        results = web.search_once(lookup.group(1).strip())
                        cited = web.answer_from_memory(lookup.group(1).strip())
                        output = cited or "维基百科暂未找到符合要求的摘要，请尝试其他主题。"
                        return self._send(200, {"mode": "web_retrieval", "text": output,
                                                "web_search": results})
                    if web.enabled and not re.search(r"\d+\s*[+加]\s*\d+", trimmed):
                        memory_answer = web.answer_from_memory(trimmed)
                        if memory_answer is not None:
                            return self._send(200, {"mode": "web_memory", "text": memory_answer,
                                                    "learning_event": "not_trainable"})
                    with guard:
                        reply = engine.reply(question, language=lang)
                    queue_result = learning.observe(reply)
                    return self._send(200, {"mode": reply.mode, "text": reply.text,
                                            "learning_event": queue_result})
                if self.path == "/api/qa/examples":
                    return self._send(200, qa_store.add_demo_drafts())
                if self.path == "/api/qa/bilingual-examples":
                    return self._send(200, qa_store.add_bilingual_demo_drafts())
                if self.path == "/api/qa/suggest":
                    return self._send(200, qa_store.suggest_from_source(body["note_id"]))
                if self.path == "/api/qa/draft":
                    return self._send(200, qa_store.add_draft(body["question"], body["answer"], body.get("note_id"),
                                                               body.get("language", "zh"), body.get("group_key")))
                if self.path == "/api/qa/review":
                    return self._send(200, qa_store.review(body["id"], body["approved"]))
                if self.path == "/api/qa/update":
                    return self._send(200, qa_store.revise(body["id"], body["question"], body["answer"],
                                                              body.get("language"), body.get("group_key")))
                if self.path == "/api/qa/train":
                    if practice.status()["running"] or learning.status().get("phase") == "training":
                        raise ValueError("先暂停算术练习与算术训练，避免 CPU 竞争")
                    return self._send(200, qa.start(body))
                if self.path == "/api/qa/stop":
                    return self._send(200, qa.stop())
                if self.path == "/api/qa/rollback":
                    return self._send(200, qa.rollback())
                if self.path == "/api/web/toggle":
                    return self._send(200, web.set_enabled(body["enabled"]))
                if self.path == "/api/web/start":
                    return self._send(200, web.start(body))
                if self.path == "/api/web/pause":
                    return self._send(200, web.pause())
                if self.path == "/api/web/search":
                    return self._send(200, web.search_once(body["query"], body.get("language", "zh")))
                if self.path == "/api/web/review":
                    return self._send(200, web.review(body["id"], body["approved"]))
                if self.path == "/api/practice/start":
                    return self._send(200, practice.start(body))
                if self.path == "/api/practice/pause":
                    return self._send(200, practice.pause())
                if self.path == "/api/learning/toggle":
                    if body["enabled"] is False and practice.status()["running"]:
                        practice.pause()
                    return self._send(200, learning.set_enabled(body["enabled"]))
                if self.path == "/api/learning/run":
                    return self._send(200, learning.start_now())
                if practice.status()["running"]:
                    raise ValueError("Pause autonomous practice before rolling back weights")
                return self._send(200, learning.rollback())
            except (ValueError, TypeError, KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                return self._send(400, {"error": str(exc)[:220]})
            except Exception as exc:
                print("SRT-M15 server error:", repr(exc), file=sys.stderr)
                return self._send(500, {"error": "Model request failed; see terminal logs"})

        def log_message(self, fmt, *args):
            # Avoid persisting private chat text in log files.
            if self.path.startswith("/api/"):
                return
            super().log_message(fmt, *args)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    address = "http://127.0.0.1:%s/" % server.server_port
    print("SRT-M15 local interface:", address)
    if chat_backend == "ollama":
        print("Conversation backend: Ollama pretrained local model " + ollama_model)
        print("This does NOT train or change SRT-M15 native weights.")
    print("Press Ctrl+C in this terminal to stop it. The service is local only.")
    if open_browser:
        webbrowser.open(address)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        practice.pause()
        qa.shutdown()
        web.shutdown()
        server.server_close()


def main():
    p = argparse.ArgumentParser(description="SRT-M15 local learning and opt-in Wikipedia retrieval")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--chat-backend", choices=("srt-m15", "ollama"), default="srt-m15",
                   help="ollama enables locally downloaded pretrained conversation weights")
    p.add_argument("--ollama-model", default="qwen3:4b-instruct",
                   help="An installed local Ollama model; no model is downloaded implicitly")
    args = p.parse_args()
    try:
        model_root = prepare_model_root()
    except Exception as exc:
        print("Could not prepare model:", exc, file=sys.stderr)
        raise SystemExit(2)
    run_server(model_root, args.port, not args.no_browser,
               chat_backend=args.chat_backend, ollama_model=args.ollama_model)


if __name__ == "__main__":
    main()
