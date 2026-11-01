# Xianyu-Automation

> Android 原生闲鱼只读识别项目。当前阶段：**V1.0 规划与文档初始化**；尚未包含可安装 APK、Gradle 工程或已验证的识别实现。

## 项目目标

用户主动打开闲鱼并浏览页面时，通过 Android AccessibilityService 读取当前可访问窗口中的可见内容，识别并整理买家、会话、聊天消息、商品卡片和消息意图。识别结果保存到手机本地数据库，供用户查看、核对和修正。

## V1.0 允许

- 只监听包名 com.taobao.idlefish 的无障碍事件。
- 读取当前窗口可访问节点及其文字、描述、控件 ID、类型和屏幕边界。
- 对当前页面做本地结构解析、买家/会话/商品关联和规则型意图分类。
- 将识别结果保存到本地 Room / SQLite。
- 用户手动打开闲鱼、切换会话和滚动页面；程序读取这些用户操作后显示的页面。

## V1.0 禁止

- 自动点击、输入、发送消息、复制/粘贴、滚动或手势操作。
- 自动打开聊天、商品详情或其他页面。
- 付款、下单、确认交易、发货、评价、退款等任何交易写操作。
- 调用闲鱼私有网络接口、逆向协议、读取 Cookie / Token 或后台抓取。
- 截屏、MediaProjection、OCR。
- 连接云端服务或 AI API；V1.0 不申请网络权限。
- 将买家昵称当作唯一身份，或在证据不足时自动绑定商品。

## 识别原则

1. 只读取当前窗口实际暴露的数据，不推断隐藏字段。
2. 平台买家 ID、会话 ID、商品 ID 只有在页面明确提供时才记录。
3. 昵称只是显示字段；本地 UUID 用于本地记录，不冒充平台 ID。
4. 商品绑定必须保存证据来源。标题/价格相似只产生候选，不作为无歧义的自动绑定依据。
5. 不确定的数据标记为 unknown 或 pending_confirmation。
6. 页面不可访问或字段缺失时，明确记录识别不完整，不用 OCR 或网络接口补齐。

## 目录规划

    Xianyu-Automation/
    ├── README.md
    ├── AGENTS.md
    ├── .gitignore
    └── docs/
        ├── ARCHITECTURE.md
        ├── ACCESSIBILITY.md
        ├── READ_ONLY_POLICY.md
        ├── DATABASE.md
        ├── RECOGNITION.md
        ├── TEST_PLAN.md
        └── ROADMAP.md

当前提交只初始化项目说明、开发约束和设计文档。后续再建立 Kotlin / Gradle 工程和可测试实现。

## 开发文档

- [系统架构](docs/ARCHITECTURE.md)
- [AccessibilityService 设计](docs/ACCESSIBILITY.md)
- [只读安全边界](docs/READ_ONLY_POLICY.md)
- [数据库设计](docs/DATABASE.md)
- [识别规则](docs/RECOGNITION.md)
- [测试计划](docs/TEST_PLAN.md)
- [开发路线图](docs/ROADMAP.md)

## 当前状态

| 项目 | 状态 |
|---|---|
| 运行方式与 V1.0 范围 | 已确定 |
| 只读权限设计 | 已规划，尚未实现 |
| 页面解析器 | 尚未开发 |
| Room 数据库 | 尚未开发 |
| Android 工程 / APK | 尚未创建 |
| 编译与真机验收 | 尚未执行 |

## 外部项目参考

以下项目只作为功能和架构研究资料，不在本次提交中复制其源码：

- [DoLovya/pyxianyu](https://github.com/DoLovya/pyxianyu)
- [zhinianboke/xianyu-auto-reply](https://github.com/zhinianboke/xianyu-auto-reply)
- [GuDong2003/xianyu-auto-reply-fix](https://github.com/GuDong2003/xianyu-auto-reply-fix)

本项目 V1.0 不集成上述项目的协议层能力，也不复用其账号登录、消息发送或自动发货代码。
