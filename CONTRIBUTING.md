# 参与开发

## 环境

Python 3.9 及以上。不需要安装任何第三方包；numpy 只是可选加速，装了会用，没装也能跑。
CI 会分别在 3.9 / 3.10 / 3.11 / 3.12 / 3.13，以及有 numpy / 没 numpy 的情况下跑一遍自检。

## 跑测试

```bash
python semcache.py selftest     # 起一个假上游 + 真代理，24 项检查，不需要网络和 key
python semcache.py demo --db /tmp/demo.db --calls 600
```

`selftest` 是唯一的测试入口，覆盖精确命中、归一化、近似改写、项目隔离、流式回放、
阈值校准、差评上调、报告算术，以及 `openai` embedder 通道。

## 代码结构

全部在 `semcache.py` 一个文件里，从上到下大致是：

1. 配置默认值、价格表、工具函数（归一化、实体抽取、token 估算）
2. 存储层 `Store`：`entries` / `events` / `calib` / `embed_cache` 四张表
3. embedder：`LocalEmbedder`（词面哈希）和 `OpenAIEmbedder`
4. 缓存引擎 `CacheEngine` 与阈值校准 `Calibrator`
5. HTTP 代理：`Upstream`、`Handler`、看板 HTML
6. 报告、CLI、自检、演示数据

## 改动原则

- **不引入第三方依赖**。numpy 只能作为可选加速，所有功能在没有 numpy 时也必须能跑。
- **保持单文件**。新增能力写在 `semcache.py` 里；如果确实需要拆分，先开 issue 讨论。
- **改了命中判定或阈值逻辑，必须补 `selftest` 用例**，并在 PR 里说明为什么。
  这两块直接影响"答错"风险，改动理由比代码本身重要。
- **报告格式变动**要同步更新 `README.md` 里的示例和 `docs/report-sample.*`。

## 提 PR 之前

```bash
python -m py_compile semcache.py
python semcache.py selftest
python semcache.py demo --db /tmp/demo.db --calls 600 --keep
```

三条都通过就可以提了。
