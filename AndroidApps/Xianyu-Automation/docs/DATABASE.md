# 本地数据库设计

## 1. 原则

V1.0 使用 Room / SQLite，所有数据保存在手机本地。数据库用于保存用户当前可见页面中读取到的识别结果，不连接云端。

不保存 Cookie、Token、密码、截图、录屏或完整 Accessibility 节点树。

## 2. 核心表

### buyers

| 字段 | 类型 | 说明 |
|---|---|---|
| local_id | TEXT PK | 本地 UUID |
| platform_buyer_id | TEXT NULL | 仅当页面明确暴露时填写 |
| display_name | TEXT NULL | 买家显示名称，不作为唯一键 |
| identity_status | TEXT | verified / unverified / unknown |
| first_seen_at | INTEGER | 首次观察时间 |
| last_seen_at | INTEGER | 最近观察时间 |

### conversations

| 字段 | 类型 | 说明 |
|---|---|---|
| local_id | TEXT PK | 本地 UUID |
| platform_chat_id | TEXT NULL | 页面明确暴露时填写 |
| buyer_local_id | TEXT FK NULL | 本地买家关联 |
| status | TEXT | observed / pending_confirmation / unknown |
| first_seen_at | INTEGER | 首次观察时间 |
| last_seen_at | INTEGER | 最近观察时间 |

### products

| 字段 | 类型 | 说明 |
|---|---|---|
| local_id | TEXT PK | 本地 UUID |
| platform_item_id | TEXT NULL | 页面明确暴露时填写 |
| title | TEXT NULL | 可见商品标题 |
| price_text | TEXT NULL | 原样保存可见价格文本 |
| image_description | TEXT NULL | 仅保存页面提供的描述文字 |
| observed_at | INTEGER | 观察时间 |

### messages

| 字段 | 类型 | 说明 |
|---|---|---|
| local_id | TEXT PK | 本地 UUID |
| conversation_local_id | TEXT FK | 所属本地会话 |
| source_message_id | TEXT NULL | 页面明确暴露时填写 |
| direction | TEXT | buyer / seller / unknown |
| content | TEXT | 可见消息正文 |
| message_type | TEXT | text / image / system / unknown |
| intent | TEXT | 规则分类结果 |
| observed_at | INTEGER | 页面观察时间 |
| dedupe_fingerprint | TEXT NULL | 本地去重指纹，不冒充平台 ID |

### conversation_product_bindings

| 字段 | 类型 | 说明 |
|---|---|---|
| local_id | TEXT PK | 本地 UUID |
| conversation_local_id | TEXT FK | 会话 |
| product_local_id | TEXT FK | 商品 |
| status | TEXT | confirmed / pending_confirmation / rejected |
| evidence_level | TEXT | explicit_id / visible_card / local_match / manual |
| created_at | INTEGER | 创建时间 |
| updated_at | INTEGER | 更新时间 |

### recognition_evidence

| 字段 | 类型 | 说明 |
|---|---|---|
| local_id | TEXT PK | 本地 UUID |
| entity_type | TEXT | buyer / conversation / product / message |
| entity_local_id | TEXT | 对应本地实体 |
| source | TEXT | accessibility_node / user_confirmation |
| field_name | TEXT | 被读取字段 |
| evidence_level | TEXT | explicit / inferred / manual |
| observed_at | INTEGER | 观察时间 |

## 3. 关联关系

    buyers 1 --- N conversations
    conversations 1 --- N messages
    conversations N --- N products (via conversation_product_bindings)
    entities 1 --- N recognition_evidence

## 4. 去重原则

- 平台 message ID 可见时优先使用。
- 平台 ID 不可见时，可用会话、本地方向、正文摘要和观察时间构造本地指纹。
- 本地指纹只用于降低重复记录，不保证跨设备或跨时间唯一。
- 同一条消息无法可靠判断是否重复时，保留来源和观察时间，不错误合并。

## 5. 隐私和数据生命周期

- 默认只在本机保存。
- 日志不包含消息正文、买家名称和平台 ID。
- 提供清除本地数据库和识别历史的入口。
- 后续若增加导出功能，导出前必须提供脱敏选项；V1.0 不做自动上传。
- Android 备份策略需在实际 Manifest 工程中明确配置，避免意外同步敏感聊天内容。
