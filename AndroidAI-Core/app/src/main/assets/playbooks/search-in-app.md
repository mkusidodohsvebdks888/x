---
id: search_in_app
name: 安卓 APP 搜索流程
triggers:
  - "在应用里搜索"
  - "帮我查找"
  - "搜索商品"
  - "search"
---

1. 先核对用户当前授权的 APP，用 android_open_app 启动目标。
2. 使用 android_ui_scan 检查前台页面与可访问控件。
3. 只点击真实可见的搜索入口，通过 android_ui_click_text 精确定位。
4. 通过 android_ui_search_text 写入用户提供的搜索词。
5. 每一步后必须核对工具返回的页面变化，不可凭空声称完成。
6. 账号登录、验证码、支付和无法识别的控件必须人工介入。
