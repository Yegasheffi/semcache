"""用 OpenAI SDK 接 semcache 的最小示例。

先起代理：
    python semcache.py serve

再运行：
    pip install openai
    python examples/client.py
"""

from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8787/v1", api_key="sk-...")

QUESTION = "退款要多久到账？"

# 普通调用
for i in range(2):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": QUESTION}],
    )
    print(i, resp.choices[0].message.content[:40])

# 想看命中情况，用 with_raw_response 拿响应头
raw = client.chat.completions.with_raw_response.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": QUESTION}],
)
print("X-Semcache:", raw.headers.get("x-semcache"))
print("score:", raw.headers.get("x-semcache-score"))
print("tokens saved:", raw.headers.get("x-semcache-tokens-saved"))
