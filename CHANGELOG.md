# Changelog

## 1.0.0

首个可用版本。

- 单文件代理，零第三方依赖，监听 OpenAI 兼容的 `/v1/chat/completions`
- SQLite 存向量与账本，按 `(project, ns)` 隔离
- 命中判定：实体一致 + 余弦阈值 + 词面门槛 + `max_tokens`
- 阈值自动校准：背景噪声下界、反馈偏置、抬升上限
- 反馈闭环：`/admin/feedback` 显式差评，命中后重问算隐式差评
- 流式响应缓存，命中时回放为 SSE
- 重复率与费用报告：文本 / Markdown / JSON，附 Web 看板
- 影子模式 `--dry-run`
- `selftest`（24 项）与 `demo`（生成演示账本）
