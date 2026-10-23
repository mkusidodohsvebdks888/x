# SRT-M15 | 本地自主提问 DeepSeek 网页版（可控实验）

SRT-M15 已经有一个可选的 [DeepSeek 可见浏览器桥接](WEB_DEEPSEEK.md)：`srt_m15/deepseek_web.py`。本功能在此基础上新增 `autonomous_deepseek_web.py`，使 SRT-M15 **在用户明确启动后，自行组织问题、逐条发给用户本人已登录的 DeepSeek 网页，并把回答记为待审核草稿**。

**这不是后台无限爬取，也不是直接训练 SRT-M15。** DeepSeek 的答复与本地模型参数是两回事。默认问题来自本地确定性学习计划，`--question-mode native` 允许已训练的原生小模型尝试自己提出问题，质量检查失败时退回固定计划。两种模式都不会自动把 DeepSeek 回答写进 SRT-M15 权重。

## 1. 环境要求

- Windows、macOS 或带图形桌面的 Linux；Python 3.10+；**本机电脑**，不是 iPhone、手机浏览器或 Colab 中的远程进程。
- 用户能正常访问 DeepSeek 网页，并自主在可见浏览器完成登录；不要向 SRT-M15 提供密码、验证码或浏览器 Cookie。
- 不绕过验证码、反机器人措施或访问网页版的未公开私有接口。网站界面改变可能导致运行失败。
- 使用前请自行确认网站允许的自动化访问方式；需要稳定、频繁的任务时，优先考虑官方 API。

```bash
cd SRT-M15
python -m pip install -r requirements.txt
python -m pip install playwright
python -m playwright install chromium
python deepseek_web_cli.py --login
```

上述登录步骤会打开一个独立 Chromium 浏览器。请在窗口中亲自登录，完成后按照终端提示回车。浏览器配置保存在本机 `~/.srt_m15/deepseek_web`，不上传 GitHub。

## 2. 先离线预览问题，不访问网页

```bash
python autonomous_deepseek_web.py --dry-run --topics "人工智能,英语,数学" --max-questions 5
```

## 3. 明确允许后启动自动提问

```bash
python autonomous_deepseek_web.py --start --topics "人工智能,英语" --language zh --max-questions 5 --interval 30 --max-minutes 15
```

可选：切换英文计划 `--language en`。**默认每轮至多 3 问；每次最多 20 问、最短间隔 20 秒、最多运行 60 分钟。** 这不是可靠的高频自动化，也不应尝试通过多开实例规避服务端限制。

如果已有实际训练产生的 `checkpoints/local_dialogues/latest.pt`，可尝试原生模型自己提问：

```bash
python autonomous_deepseek_web.py --start --question-mode native --checkpoint checkpoints/local_dialogues/latest.pt --topics "机器学习" --max-questions 3
```

原生模型太小或权重不兼容时，自动回退到本地规则提问，终端显示原始来源标签（`rule-based-planner` / `native-SRT-M15-unverified`）；不会把 DeepSeek 生成的问题伪装成 SRT-M15 原生推理。

**按 Ctrl+C 可停止本次运行。** 不会注册后台任务；重启后也不会自己恢复。网页请求遇到超时、验证码或读取失败会停止，避免在不知道网站是否收到了问题时无限重发。

## 4. 回答保存位置与使用限制

默认只在本机追加保存：

```text
SRT-M15/local_data/deepseek_web_questions/drafts.jsonl
```

每一行包含题目、网页回答、话题、语言、时间、独立会话编号及来源。所有记录明确标注：

```json
{"verified":false,"review_status":"unreviewed","training_approved":false}
```

它们**不会**自动进入神经网络训练，亦不会直接提交 GitHub。人工确认来源授权、隐私、回答是否正确且有用之后，才可以转换成经过审核的训练材料。尤其不要把 DeepSeek 对无法验证的问题的推测当作客观事实。

## 5. 本地离线测试

```bash
python -m unittest discover -s tests -p "test_autonomous_deepseek_web.py" -v
```

这些测试使用模拟网页客户端，覆盖话题边界、问题规划、任务上限、失败即停、草稿保存及人工审核默认关闭；**并不证明你的账号当前能成功登录网页版或其页面 DOM 选择器仍有效**。

## 6. 工作流程

```text
SRT-M15 本地计划问题（规则生成 / 可选原生权重）
       |
       v
人工点击启动 + 现有可见浏览器登录
       |
       v
DeepSeek 网页回答（严格标记外部来源）
       |
       v
本地保存待审核 JSONL 草稿
       |
       v
人工抽样核验 / 数据许可与隐私检查
       |
       v
另行构建训练集 -> 本地更新 SRT-M15 权重 -> 独立评估
```

**验收标准**：实际能够登录并问出第一题、读取并写入一条完整草稿、在触发限额或 Ctrl+C 后停止，且本地权重文件不改变。网页实际联通还需要你在自己的电脑上验证。
