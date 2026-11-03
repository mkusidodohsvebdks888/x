# SRT-M15 文档导航

> 本索引只为现有文件建立阅读路径，不修改模型、训练数据或检查点。SRT-M15 是自研实验型模型；外部 DeepSeek/Qwen 接口不等于本地模型已学会相关能力。

## 从这里开始

- [项目总览](README.md) / [仓库首页](../README.md)：研究定位、运行步骤和边界。
- [模型卡](MODEL_CARD.md)：模型能力与限制。
- [本地部署](README_LOCAL.md) / [本地对话](README_LOCAL_DIALOGUE.md)：电脑本地体验。
- [RTX 3080 聊天指南](README_CHAT_RTX3080.md)：显卡环境相关示例。
- [免费 GPU](README_FREE_GPU.md)：Colab / Kaggle 参考。
- [手机 Colab Notebook](SRT_M15_Phone_Colab.ipynb)：手机训练测试入口。

## API 与外部模型

- [API 启动说明](API_README.md)：本地 HTTP API。
- [流式 API 与 HTTPS 部署](API_STREAM_HTTPS_2GB.md)：SSE、多轮接口及低内存部署参考。
- [DeepSeek 网页/API 接入](WEB_DEEPSEEK.md)：外部模型连接。
- [自主提问 DeepSeek 网页实验](README_AUTONOMOUS_DEEPSEEK_WEB.md)：网页实验流程与人工审核边界。
- [Qwen 教师数据](README_QWEN_TEACHER_DATA.md)：教师输出与学生训练数据的区分。

## 训练与数据

- [分批训练与断点续训](README_STAGED_TRAINING.md)。
- [双语综合](README_BILINGUAL_GENERAL.md) / [双语学习](README_BILINGUAL_LEARNING.md) / [语言学习](README_LANGUAGE_LEARNING.md)。
- [百万对话实验](README_CONVERSATION_MILLION.md) / [11 万条合成数据卡](DATASET_CARD_BILINGUAL_V4_110000.md)。
- [训练状态快照](TRAINING_STATUS_20261023.md)：含日期的实验状态，不能代表当前运行结果。
- [110k GPU Colab](SRT_M15_110k_GPU_Colab.ipynb)：GPU 分批训练 Notebook。

## 研究与可扩展设计

- [已有模型共同机制](README_COMMON_MECHANISMS.md)。
- [完整实验模块说明](README_SRT_FULL_SUITE.md)。
- [自主练习](README_SELF_PRACTICE.md) / [审核再学习](README_REVIEW_AND_LEARN.md)。
- [互联网学习研究](README_INTERNET_LEARNING.md)。

## 维护约定

1. 新开发功能先维护项目总览和本索引，再补充专题文档。
2. 实验数据、训练检查点、LoRA 权重和密钥不要直接上传到公开仓库。
3. 区分**代码已实现**、**已本地验收**、**计划中**，避免将接口调用误写为模型已学会。
4. 每次训练单独记录数据版本、配置、检查点、评测指标与日期。
