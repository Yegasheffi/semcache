#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
semcache.py — 单文件语义缓存代理（LLM Semantic Cache Proxy）

一句话：把重复的问题拦在本地，别让它再花钱去问模型。

特性
  1. 单文件、零第三方依赖（numpy 可选，有则更快）。`python3 semcache.py serve` 直接跑。
  2. SQLite 存向量（float32 BLOB），暴力余弦检索，万级条目毫秒级返回。
  3. 相似度阈值自动校准：用「同域随机样本的相似度分布」估计误命中风险，定下安全下界，
     再用线上命中/失败反馈做微调。不需要人工标注。
  4. 按项目隔离：不同 project 的向量物理分区，永不互相命中。
  5. 重复率报告：直接告诉你「这个月 41% 的调用是白花的钱，已经省下 ¥xxx」。
  6. 影子模式（--dry-run）：只统计不生效，先给客户看账再上线。

用法
  python3 semcache.py serve                      # 默认 http://127.0.0.1:8787
  python3 semcache.py serve --dry-run             # 影子模式，只记账不命中
  python3 semcache.py report --month 2025-10      # 重复率报告
  python3 semcache.py report --md                 # Markdown（贴给客户）
  python3 semcache.py calibrate --all             # 立刻重算所有阈值
  python3 semcache.py selftest                    # 自检
  python3 semcache.py demo --calls 4000           # 造假数据看报告效果

客户端接入：把 base_url 从 https://api.openai.com/v1 改成 http://127.0.0.1:8787/v1 即可。
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import logging
import math
import os
import random
import re
import shutil
import sqlite3
import ssl
import sys
import threading
import time
import unicodedata
import zlib
from array import array
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlsplit, parse_qs

try:  # 可选加速
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None

VERSION = "1.0.0"
LOG = logging.getLogger("semcache")

# --------------------------------------------------------------------------- #
# 默认配置
# --------------------------------------------------------------------------- #

DEFAULT_PRICES: Dict[str, Tuple[float, float]] = {
    # 模型前缀 -> (输入 $/1M tokens, 输出 $/1M tokens)
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "o4-mini": (1.10, 4.40),
    "o3-mini": (1.10, 4.40),
    "o3": (2.00, 8.00),
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-7-sonnet": (3.00, 15.00),
    "claude-sonnet-4": (3.00, 15.00),
    "claude-3-5-haiku": (0.80, 4.00),
    "claude-3-haiku": (0.25, 1.25),
    "claude-3-opus": (15.00, 75.00),
    "deepseek-chat": (0.27, 1.10),
    "deepseek-reasoner": (0.55, 2.19),
    "qwen-max": (1.60, 6.40),
    "qwen-plus": (0.40, 1.20),
    "qwen-turbo": (0.05, 0.20),
    "qwen-long": (0.07, 0.28),
    "glm-4-plus": (0.70, 0.70),
    "glm-4-air": (0.07, 0.07),
    "glm-4-flash": (0.00, 0.00),
    "moonshot-v1-8k": (1.68, 1.68),
    "moonshot-v1-32k": (3.36, 3.36),
    "doubao-pro-32k": (0.11, 0.28),
    "doubao-lite-32k": (0.04, 0.08),
    "ernie-4.0": (4.20, 12.60),
    "hunyuan-pro": (1.40, 1.40),
}

DEFAULTS: Dict[str, Any] = {
    "host": "127.0.0.1",
    "port": 8787,
    "upstream": "https://api.openai.com",
    "db": "~/.semcache/semcache.db",
    "embedder": "local",            # local | openai
    "embed_model": "text-embedding-3-small",
    "embed_base": "",               # 空 = 用 upstream
    "embed_key": "",                # 空 = 用客户端的 Authorization / upstream_key
    "local_dim": 2048,
    "threshold": 0.0,               # >0 表示人工指定，关闭自动校准
    "min_threshold": 0.0,           # 0 = 按 embedder 取默认下界
    "max_threshold": 0.995,
    "ttl": 30 * 86400,              # 缓存条目过期秒数，0 = 永不过期
    "max_entries": 200000,
    "max_temperature": 1.0,         # 高于此温度的请求不缓存
    "cache_stream": True,           # 流式响应也缓存，命中时回放成 SSE
    "max_embed_chars": 4000,
    "max_body": 32 * 1024 * 1024,
    "upstream_key": "",             # 代理自带 key（为空则透传客户端 key）
    "admin_token": "",              # 管理接口 / 仪表盘口令
    "usd_cny": 7.2,
    "mask_dynamic": True,           # 系统提示里的时间戳/UUID 掩码后再算命名空间
    "calib_every": 200,             # 每个命名空间每 N 次查询触发一次校准
    "calib_pairs": 4000,            # 校准采样对数
    "calib_z": 3.0,                 # 阈值 = mu + z*sigma
    "calib_margin": 0.02,           # 额外安全边际
    "max_floor_lift": 0.15,         # 阈值相对先验最多抬高多少
    "repeat_window": 240,           # 命中后 N 秒内同一问题再来 = 隐式差评
    "implicit_feedback": True,
    "lexical_gate": 0.6,            # 仅 local embedder：词面 Jaccard 门槛
    "lexical_cover": 0.8,           # 仅 local embedder：短句包含度门槛
    "request_timeout": 300,
    "project_keys": {},             # {"proj": "sk-前缀"}
    "prices": {},                   # 覆盖 DEFAULT_PRICES
    "dry_run": False,
    "log_level": "INFO",
}


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #

_CJK_RANGES = ((0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0x3040, 0x30FF))


def is_cjk(ch: str) -> bool:
    o = ord(ch)
    for lo, hi in _CJK_RANGES:
        if lo <= o <= hi:
            return True
    return False


def display_width(s: str) -> int:
    """中英混排的对齐宽度。"""
    w = 0
    for ch in s:
        if unicodedata.east_asian_width(ch) in ("W", "F"):
            w += 2
        elif unicodedata.combining(ch):
            w += 0
        else:
            w += 1
    return w


def pad(s: str, width: int, align: str = "left") -> str:
    s = str(s)
    gap = max(0, width - display_width(s))
    if align == "right":
        return " " * gap + s
    if align == "center":
        left = gap // 2
        return " " * left + s + " " * (gap - left)
    return s + " " * gap


def trunc(s: str, n: int) -> str:
    s = str(s).replace("\n", " ").replace("\r", " ")
    return s if len(s) <= n else s[: max(0, n - 1)] + "…"


def now() -> float:
    return time.time()


def day_of(ts: float) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(ts))


def month_of(ts: float) -> str:
    return time.strftime("%Y-%m", time.localtime(ts))


def display_path(path: str) -> str:
    """报告里尽量显示相对路径或 ~，别把本机绝对路径写进去。"""
    try:
        ap = os.path.abspath(os.path.expanduser(path))
        cwd = os.getcwd()
        if ap.startswith(cwd + os.sep):
            return os.path.relpath(ap, cwd)
        home = os.path.expanduser("~")
        if ap.startswith(home + os.sep):
            return "~" + ap[len(home):]
        return path
    except Exception:
        return path


def money(usd: float) -> str:
    if abs(usd) >= 1000:
        return "$%.0f" % usd
    if abs(usd) >= 1:
        return "$%.2f" % usd
    return "$%.3f" % usd


def cny(usd: float, rate: float) -> str:
    return money_cny(usd * rate)


def money_cny(v: float) -> str:
    if abs(v) >= 0.01:
        return "¥%.2f" % v
    return "¥%.4f" % v


def human_int(n: float) -> str:
    n = float(n)
    if n >= 1e9:
        return "%.2fB" % (n / 1e9)
    if n >= 1e6:
        return "%.2fM" % (n / 1e6)
    if n >= 1e4:
        return "%.1fk" % (n / 1e3)
    return "%d" % int(n)


def sha(text: str, n: int = 32) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


_WS_RE = re.compile(r"\s+")

_ENT_PATTERNS = [
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    re.compile(r"https?://[^\s，。；）)\]]+"),
    re.compile(r"\d+(?:[.,]\d+)*"),
    re.compile(r"\b[a-z0-9]*[a-z][a-z0-9_-]{11,}\b"),
]


def entities(raw: str) -> Tuple[str, ...]:
    """
    抽出「必须一字不差」的实体：数字/单号、邮箱、URL、长 ID。

    这是语义缓存最容易翻车的地方：'订单 12345 什么时候发货' 和
    '订单 67890 什么时候发货' 向量几乎一样，但答案完全不同。
    实体不一致就绝不命中。
    """
    if not raw:
        return ()
    text = unicodedata.normalize("NFKC", raw).lower()
    found = set()
    for pat in _ENT_PATTERNS:
        for m in pat.finditer(text):
            tok = m.group(0).strip(".,;:")
            if tok:
                found.add(tok)
    return tuple(sorted(found))


def normalize_text(s: str) -> str:
    """统一大小写/全半角，去掉标点与表情，只留词与空白。"""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s).lower()
    out = []
    for ch in s:
        cat = unicodedata.category(ch)
        if ch.isspace():
            out.append(" ")
        elif cat[0] in ("L", "N") or is_cjk(ch):
            out.append(ch)
        elif cat in ("Mn", "Mc"):
            out.append(ch)
        else:
            out.append(" ")
    s = _WS_RE.sub(" ", "".join(out)).strip()
    # 中文与数字/中文之间不留空格，让「订单 12345」和「订单12345」归一成同一个 key
    s = re.sub(r"(?<=[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff])\s+", "", s)
    s = re.sub(r"\s+(?=[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff])", "", s)
    return s


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_CJK_RUN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff]+")


def lex_tokens(norm: str) -> List[str]:
    """词面 token：拉丁词 + 中日韩二元组（中文没有空格的分词近似）。"""
    toks = _TOKEN_RE.findall(norm)
    for run in _CJK_RUN_RE.findall(norm):
        if len(run) == 1:
            toks.append(run)
        else:
            toks.extend(run[i : i + 2] for i in range(len(run) - 1))
    return toks


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    return inter / float(len(sa | sb))


def overlap(a: Iterable[str], b: Iterable[str]) -> float:
    """重叠系数：交集 / 较短集合。对「多一个字」的改写不敏感，对「换一个关键词」敏感。"""
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / float(min(len(sa), len(sb)))


def containment(query: Iterable[str], cand: Iterable[str]) -> float:
    """query 有多大比例出现在候选里（短问句被长上下文包住的情况）。"""
    sa, sb = set(query), set(cand)
    if not sa:
        return 0.0
    return len(sa & sb) / float(len(sa))


_MASK_PATTERNS = [
    re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2})?(\.\d+)?(Z|[+-]\d{2}:?\d{2})?"),
    re.compile(r"\d{4}[-/年]\d{1,2}[-/月]\d{1,2}日?"),
    re.compile(r"\b\d{2}:\d{2}(:\d{2})?\b"),
    re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"),
    re.compile(r"\b[0-9a-fA-F]{16,}\b"),
    re.compile(r"\b\d{10,13}\b"),  # epoch 秒/毫秒
]


def mask_dynamic(s: str) -> str:
    """把系统提示里的时间戳/UUID/长数字替换掉，避免每个请求都变成新命名空间。"""
    if not s:
        return ""
    for pat in _MASK_PATTERNS:
        s = pat.sub("<dyn>", s)
    return s


def est_tokens(text: str) -> int:
    """粗估 token：CJK 约 1 字 1 token，拉丁约 4 字符 1 token。"""
    if not text:
        return 0
    cjk = sum(1 for ch in text if is_cjk(ch))
    latin = len(text) - cjk
    return int(cjk + latin / 4.0) + 1


# --------------------------------------------------------------------------- #
# 价格
# --------------------------------------------------------------------------- #

class Prices:
    def __init__(self, overrides: Optional[Dict[str, Any]] = None):
        self.table: Dict[str, Tuple[float, float]] = dict(DEFAULT_PRICES)
        for k, v in (overrides or {}).items():
            try:
                if isinstance(v, (list, tuple)):
                    self.table[str(k).lower()] = (float(v[0]), float(v[1]))
                elif isinstance(v, dict):
                    self.table[str(k).lower()] = (float(v["in"]), float(v["out"]))
                elif isinstance(v, str) and "/" in v:
                    a, b = v.split("/", 1)
                    self.table[str(k).lower()] = (float(a), float(b))
            except Exception:
                LOG.warning("忽略无法解析的价格配置: %s=%r", k, v)
        # 长前缀优先
        self._keys = sorted(self.table.keys(), key=len, reverse=True)

    def rates(self, model: str) -> Tuple[float, float]:
        m = (model or "").lower()
        for k in self._keys:
            if m.startswith(k):
                return self.table[k]
        return (0.0, 0.0)

    def cost(self, model: str, prompt_tokens: float, completion_tokens: float) -> float:
        rin, rout = self.rates(model)
        return (prompt_tokens * rin + completion_tokens * rout) / 1_000_000.0


# --------------------------------------------------------------------------- #
# 存储层
# --------------------------------------------------------------------------- #

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS entries (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    project          TEXT NOT NULL,
    ns               TEXT NOT NULL,
    qhash            TEXT NOT NULL,
    prompt           TEXT NOT NULL,
    answer           BLOB NOT NULL,
    vec              BLOB NOT NULL,
    dim              INTEGER NOT NULL,
    model            TEXT,
    created_at       REAL NOT NULL,
    last_hit_at      REAL,
    hits             INTEGER NOT NULL DEFAULT 0,
    prompt_tokens    INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    finish_reason    TEXT,
    expires_at       REAL,
    ents             TEXT,
    meta             TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_entries_q ON entries(project, ns, qhash);
CREATE INDEX IF NOT EXISTS ix_entries_ns ON entries(project, ns);
CREATE INDEX IF NOT EXISTS ix_entries_lru ON entries(project, last_hit_at);

CREATE TABLE IF NOT EXISTS events (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    ts               REAL NOT NULL,
    day              TEXT NOT NULL,
    project          TEXT NOT NULL,
    ns               TEXT,
    model            TEXT,
    kind             TEXT NOT NULL,      -- hit | miss | shadow | bypass | error | feedback
    detail           TEXT,
    score            REAL,
    prompt_tokens    INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    cost_saved       REAL DEFAULT 0,
    cost_extra       REAL DEFAULT 0,
    latency_ms       REAL DEFAULT 0,
    prompt           TEXT,
    entry_id         INTEGER
);
CREATE INDEX IF NOT EXISTS ix_events_day ON events(day);
CREATE INDEX IF NOT EXISTS ix_events_proj_day ON events(project, day);

CREATE TABLE IF NOT EXISTS calib (
    project       TEXT NOT NULL,
    ns            TEXT NOT NULL,
    base          REAL NOT NULL,
    bias          REAL NOT NULL DEFAULT 0,
    eff           REAL NOT NULL,
    floor         REAL NOT NULL,
    n_null        INTEGER NOT NULL DEFAULT 0,
    mu            REAL NOT NULL DEFAULT 0,
    sigma         REAL NOT NULL DEFAULT 0,
    p99           REAL NOT NULL DEFAULT 0,
    lookups       INTEGER NOT NULL DEFAULT 0,
    hits          INTEGER NOT NULL DEFAULT 0,
    bad           INTEGER NOT NULL DEFAULT 0,
    status        TEXT NOT NULL DEFAULT 'bootstrap',
    updated_at    REAL NOT NULL,
    PRIMARY KEY (project, ns)
);

CREATE TABLE IF NOT EXISTS calib_log (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        REAL NOT NULL,
    project   TEXT NOT NULL,
    ns        TEXT NOT NULL,
    old_eff   REAL,
    new_eff   REAL,
    floor     REAL,
    mu        REAL,
    sigma     REAL,
    reason    TEXT
);

CREATE TABLE IF NOT EXISTS embed_cache (
    key       TEXT PRIMARY KEY,
    vec       BLOB NOT NULL,
    dim       INTEGER NOT NULL,
    created   REAL NOT NULL
);
"""


class Store:
    """SQLite 封装：线程本地连接 + 写锁。"""

    def __init__(self, path: str):
        self.path = os.path.abspath(os.path.expanduser(path))
        d = os.path.dirname(self.path)
        if d:
            os.makedirs(d, exist_ok=True)
        self._local = threading.local()
        self._lock = threading.RLock()
        with self._lock:
            c = self._conn()
            c.executescript(SCHEMA)
            c.commit()

    # -- 连接 -------------------------------------------------------------
    def _conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(self.path, timeout=30.0, check_same_thread=False)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA busy_timeout=15000")
            c.execute("PRAGMA synchronous=NORMAL")
            c.execute("PRAGMA foreign_keys=ON")
            self._local.conn = c
        return c

    def close(self) -> None:
        c = getattr(self._local, "conn", None)
        if c is not None:
            c.close()
        self._local.conn = None

    # -- 条目 -------------------------------------------------------------
    def get_entry(self, project: str, ns: str, qhash: str) -> Optional[sqlite3.Row]:
        cur = self._conn().execute(
            "SELECT * FROM entries WHERE project=? AND ns=? AND qhash=?", (project, ns, qhash)
        )
        return cur.fetchone()

    def get_entry_by_id(self, entry_id: int) -> Optional[sqlite3.Row]:
        return self._conn().execute("SELECT * FROM entries WHERE id=?", (entry_id,)).fetchone()

    def insert_entry(self, **kw: Any) -> int:
        cols = (
            "project ns qhash prompt answer vec dim model created_at last_hit_at "
            "prompt_tokens completion_tokens finish_reason expires_at ents meta"
        ).split()
        vals = [kw.get(c) for c in cols]
        with self._lock:
            c = self._conn()
            cur = c.execute(
                "INSERT OR REPLACE INTO entries (%s) VALUES (%s)"
                % (",".join(cols), ",".join("?" * len(cols))),
                vals,
            )
            c.commit()
            return int(cur.lastrowid or 0)

    def delete_entry(self, entry_id: int) -> None:
        with self._lock:
            c = self._conn()
            c.execute("DELETE FROM entries WHERE id=?", (entry_id,))
            c.commit()

    def touch_entry(self, entry_id: int, ts: float, hits: int) -> None:
        with self._lock:
            c = self._conn()
            c.execute(
                "UPDATE entries SET last_hit_at=?, hits=? WHERE id=?", (ts, hits, entry_id)
            )
            c.commit()

    def iter_ns(self, project: str, ns: str, limit: int = 50000) -> List[sqlite3.Row]:
        cur = self._conn().execute(
            "SELECT id,qhash,vec,dim,prompt,hits,prompt_tokens,completion_tokens,"
            "expires_at,ents FROM entries WHERE project=? AND ns=? ORDER BY id DESC LIMIT ?",
            (project, ns, limit),
        )
        return cur.fetchall()

    def count_entries(self, project: Optional[str] = None) -> int:
        if project:
            r = self._conn().execute(
                "SELECT COUNT(*) FROM entries WHERE project=?", (project,)
            ).fetchone()
        else:
            r = self._conn().execute("SELECT COUNT(*) FROM entries").fetchone()
        return int(r[0])

    def evict(self, project: str, max_entries: int, batch: int = 200) -> int:
        with self._lock:
            c = self._conn()
            n = c.execute("SELECT COUNT(*) FROM entries WHERE project=?", (project,)).fetchone()[0]
            if n <= max_entries:
                return 0
            kill = int(n - max_entries) + batch
            c.execute(
                "DELETE FROM entries WHERE id IN (SELECT id FROM entries WHERE project=? "
                "ORDER BY COALESCE(last_hit_at, created_at) ASC LIMIT ?)",
                (project, kill),
            )
            c.commit()
            return kill

    def purge_expired(self, ts: float) -> int:
        with self._lock:
            c = self._conn()
            cur = c.execute(
                "DELETE FROM entries WHERE expires_at IS NOT NULL AND expires_at < ?", (ts,)
            )
            c.commit()
            return int(cur.rowcount or 0)

    def all_namespaces(self) -> List[sqlite3.Row]:
        return self._conn().execute(
            "SELECT project, ns, COUNT(*) AS n FROM entries GROUP BY project, ns"
        ).fetchall()

    # -- 事件 -------------------------------------------------------------
    def add_event(self, **kw: Any) -> None:
        ts = kw.get("ts", now())
        cols = (
            "ts day project ns model kind detail score prompt_tokens completion_tokens "
            "cost_saved cost_extra latency_ms prompt entry_id"
        ).split()
        kw.setdefault("day", day_of(ts))
        kw.setdefault("project", "default")
        vals = [kw.get(c) for c in cols]
        with self._lock:
            c = self._conn()
            c.execute(
                "INSERT INTO events (%s) VALUES (%s)" % (",".join(cols), ",".join("?" * len(cols))),
                vals,
            )
            c.commit()

    # -- 校准 -------------------------------------------------------------
    def get_calib(self, project: str, ns: str) -> Optional[sqlite3.Row]:
        return self._conn().execute(
            "SELECT * FROM calib WHERE project=? AND ns=?", (project, ns)
        ).fetchone()

    def put_calib(self, project: str, ns: str, **kw: Any) -> None:
        with self._lock:
            c = self._conn()
            c.execute(
                "INSERT INTO calib (project,ns,base,bias,eff,floor,n_null,mu,sigma,p99,"
                "lookups,hits,bad,status,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(project,ns) DO UPDATE SET base=excluded.base, bias=excluded.bias, "
                "eff=excluded.eff, floor=excluded.floor, n_null=excluded.n_null, mu=excluded.mu, "
                "sigma=excluded.sigma, p99=excluded.p99, lookups=excluded.lookups, "
                "hits=excluded.hits, bad=excluded.bad, status=excluded.status, "
                "updated_at=excluded.updated_at",
                (
                    project,
                    ns,
                    float(kw.get("base", 0.9)),
                    float(kw.get("bias", 0.0)),
                    float(kw.get("eff", 0.9)),
                    float(kw.get("floor", 0.8)),
                    int(kw.get("n_null", 0)),
                    float(kw.get("mu", 0.0)),
                    float(kw.get("sigma", 0.0)),
                    float(kw.get("p99", 0.0)),
                    int(kw.get("lookups", 0)),
                    int(kw.get("hits", 0)),
                    int(kw.get("bad", 0)),
                    str(kw.get("status", "bootstrap")),
                    float(now()),
                ),
            )
            c.commit()

    def log_calib(self, project: str, ns: str, old_eff: float, new_eff: float,
                  floor: float, mu: float, sigma: float, reason: str) -> None:
        with self._lock:
            c = self._conn()
            c.execute(
                "INSERT INTO calib_log (ts,project,ns,old_eff,new_eff,floor,mu,sigma,reason) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (now(), project, ns, old_eff, new_eff, floor, mu, sigma, reason),
            )
            c.commit()

    # -- 嵌入缓存 ---------------------------------------------------------
    def embed_get(self, key: str) -> Optional[bytes]:
        r = self._conn().execute("SELECT vec FROM embed_cache WHERE key=?", (key,)).fetchone()
        return r[0] if r else None

    def embed_put(self, key: str, vec: bytes, dim: int) -> None:
        with self._lock:
            c = self._conn()
            c.execute(
                "INSERT OR REPLACE INTO embed_cache (key,vec,dim,created) VALUES (?,?,?,?)",
                (key, vec, dim, now()),
            )
            c.commit()

    # -- 统计 -------------------------------------------------------------
    def report_rows(self, month: Optional[str], project: Optional[str]) -> List[sqlite3.Row]:
        where = ["1=1"]
        args: List[Any] = []
        if month:
            where.append("substr(day,1,7)=?")
            args.append(month)
        if project:
            where.append("project=?")
            args.append(project)
        sql = (
            "SELECT project, "
            " SUM(CASE WHEN kind='hit' THEN 1 ELSE 0 END) AS hits, "
            " SUM(CASE WHEN kind='miss' THEN 1 ELSE 0 END) AS misses, "
            " SUM(CASE WHEN kind='shadow' THEN 1 ELSE 0 END) AS shadows, "
            " SUM(CASE WHEN kind='bypass' THEN 1 ELSE 0 END) AS bypasses, "
            " SUM(CASE WHEN kind='error' THEN 1 ELSE 0 END) AS errors, "
            " SUM(CASE WHEN kind='hit' THEN prompt_tokens+completion_tokens ELSE 0 END) AS saved_tokens, "
            " SUM(CASE WHEN kind='hit' THEN prompt_tokens ELSE 0 END) AS saved_in, "
            " SUM(CASE WHEN kind='hit' THEN completion_tokens ELSE 0 END) AS saved_out, "
            " SUM(cost_saved) AS saved_usd, SUM(cost_extra) AS extra_usd, "
            " SUM(CASE WHEN kind='shadow' THEN cost_saved ELSE 0 END) AS shadow_usd, "
            " SUM(CASE WHEN kind='miss' THEN latency_ms ELSE 0 END) AS miss_lat, "
            " SUM(CASE WHEN kind='hit' THEN latency_ms ELSE 0 END) AS hit_lat, "
            " MIN(day) AS first_day, MAX(day) AS last_day "
            "FROM events WHERE %s GROUP BY project ORDER BY saved_usd DESC" % " AND ".join(where)
        )
        return self._conn().execute(sql, args).fetchall()

    def hit_kinds(self, month: Optional[str], project: Optional[str]) -> List[sqlite3.Row]:
        where = ["kind='hit'"]
        args: List[Any] = []
        if month:
            where.append("substr(day,1,7)=?")
            args.append(month)
        if project:
            where.append("project=?")
            args.append(project)
        return self._conn().execute(
            "SELECT project, detail, COUNT(*) AS n FROM events WHERE %s "
            "GROUP BY project, detail" % " AND ".join(where), args
        ).fetchall()

    def daily_rows(self, month: Optional[str], project: Optional[str]) -> List[sqlite3.Row]:
        where = ["1=1"]
        args: List[Any] = []
        if month:
            where.append("substr(day,1,7)=?")
            args.append(month)
        if project:
            where.append("project=?")
            args.append(project)
        sql = (
            "SELECT day, SUM(CASE WHEN kind IN ('hit','miss') THEN 1 ELSE 0 END) AS calls, "
            " SUM(CASE WHEN kind='hit' THEN 1 ELSE 0 END) AS hits, SUM(cost_saved) AS usd "
            "FROM events WHERE %s GROUP BY day ORDER BY day" % " AND ".join(where)
        )
        return self._conn().execute(sql, args).fetchall()

    def top_prompts(self, month: Optional[str], project: Optional[str], limit: int = 10) -> List[sqlite3.Row]:
        where = ["kind='hit'"]
        args: List[Any] = []
        if month:
            where.append("substr(day,1,7)=?")
            args.append(month)
        if project:
            where.append("project=?")
            args.append(project)
        sql = (
            "SELECT prompt, model, COUNT(*) AS n, SUM(cost_saved) AS usd, "
            " MAX(score) AS best FROM events WHERE %s GROUP BY prompt ORDER BY n DESC LIMIT ?"
            % " AND ".join(where)
        )
        return self._conn().execute(sql, args + [limit]).fetchall()

    def miss_details(self, month: Optional[str], project: Optional[str]) -> List[sqlite3.Row]:
        where = ["kind='miss'"]
        args: List[Any] = []
        if month:
            where.append("substr(day,1,7)=?")
            args.append(month)
        if project:
            where.append("project=?")
            args.append(project)
        return self._conn().execute(
            "SELECT detail, COUNT(*) AS n, AVG(score) AS avg_score, MAX(score) AS max_score "
            "FROM events WHERE %s GROUP BY detail" % " AND ".join(where), args
        ).fetchall()

    def near_misses(self, month: Optional[str], project: Optional[str]) -> sqlite3.Row:
        where = ["kind='miss'", "detail='near'"]
        args: List[Any] = []
        if month:
            where.append("substr(day,1,7)=?")
            args.append(month)
        if project:
            where.append("project=?")
            args.append(project)
        sql = (
            "SELECT COUNT(*) AS n, AVG(score) AS avg_score, MAX(score) AS max_score, "
            " AVG(prompt_tokens) AS avg_in, AVG(completion_tokens) AS avg_out "
            "FROM events WHERE %s" % " AND ".join(where)
        )
        return self._conn().execute(sql, args).fetchone()

    def calib_all(self, project: Optional[str] = None) -> List[sqlite3.Row]:
        if project:
            return self._conn().execute(
                "SELECT * FROM calib WHERE project=? ORDER BY lookups DESC", (project,)
            ).fetchall()
        return self._conn().execute("SELECT * FROM calib ORDER BY lookups DESC").fetchall()

    def corpus_stats(self) -> Dict[str, Any]:
        c = self._conn()
        n = c.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
        ev = c.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        return {"entries": int(n), "events": int(ev)}

    def vacuum(self) -> None:
        with self._lock:
            c = self._conn()
            c.execute("VACUUM")
            c.commit()


# --------------------------------------------------------------------------- #
# 向量工具
# --------------------------------------------------------------------------- #

def vec_normalize(v: Sequence[float]) -> List[float]:
    s = 0.0
    for x in v:
        s += x * x
    s = math.sqrt(s)
    if s <= 1e-12:
        return list(v)
    return [x / s for x in v]


def vec_to_blob(v: Sequence[float]) -> bytes:
    return array("f", v).tobytes()


def blob_to_vec(b: bytes, dim: int) -> List[float]:
    a = array("f")
    a.frombytes(b)
    if len(a) != dim:  # 维度不一致（换过模型），返回空
        return []
    return a.tolist()


def cos_py(a: Sequence[float], b: Sequence[float]) -> float:
    s = 0.0
    for x, y in zip(a, b):
        s += x * y
    return s


def matmul(a: Any, b: Any) -> Any:
    """numpy 矩阵乘，屏蔽 Accelerate BLAS 误报的浮点异常标志。"""
    with _np.errstate(all="ignore"):
        return a @ b


# --------------------------------------------------------------------------- #
# Embedder
# --------------------------------------------------------------------------- #

class Embedder:
    """抽象：把文本变成 L2 归一化的向量。"""

    ident = "base"
    dim = 0

    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

    def cost_per_token(self) -> float:
        return 0.0


class LocalEmbedder(Embedder):
    """
    零依赖本地向量：词 + 中日韩字 n-gram 的带符号哈希投影。

    它不是神经网络，但对「近似重复」的问题（改写、语序、标点、多轮追加）
    命中率很高，且完全离线、零成本、可复现。真正跨语义的改写请用 openai embedder。
    """

    def __init__(self, dim: int = 2048):
        self.dim = int(dim)
        self.ident = "local:%d" % self.dim

    def _features(self, norm: str) -> Dict[str, float]:
        feats: Dict[str, float] = {}
        for t in _TOKEN_RE.findall(norm):
            feats["w:" + t] = feats.get("w:" + t, 0.0) + 1.0
        for run in _CJK_RUN_RE.findall(norm):
            if len(run) == 1:
                feats["c1:" + run] = feats.get("c1:" + run, 0.0) + 1.0
            else:
                for i in range(len(run) - 1):
                    g = "c2:" + run[i : i + 2]
                    feats[g] = feats.get(g, 0.0) + 1.0
                for i in range(len(run) - 2):
                    g = "c3:" + run[i : i + 3]
                    feats[g] = feats.get(g, 0.0) + 0.5
        # 拉丁词二元组，保留短语顺序信息
        words = _TOKEN_RE.findall(norm)
        for i in range(len(words) - 1):
            g = "b:" + words[i] + "_" + words[i + 1]
            feats[g] = feats.get(g, 0.0) + 0.6
        return feats

    def embed(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        dim = self.dim
        for text in texts:
            norm = normalize_text(text)
            vec = [0.0] * dim
            if norm:
                for f, c in self._features(norm).items():
                    h = hashlib.blake2b(f.encode("utf-8"), digest_size=8).digest()
                    idx = int.from_bytes(h[:4], "little") % dim
                    sign = 1.0 if (h[4] & 1) else -1.0
                    vec[idx] += sign * (1.0 + math.log(c)) if c > 1 else sign
            out.append(vec_normalize(vec))
        return out


class OpenAIEmbedder(Embedder):
    """OpenAI 兼容 /v1/embeddings（OpenAI、硅基流动、DashScope 兼容模式等都支持）。"""

    def __init__(self, base: str, key: str, model: str, dims: int = 0,
                 timeout: float = 30.0, batch: int = 64):
        self.base = base.rstrip("/")
        self.key = key
        self.model = model
        self.dims = int(dims or 0)
        self.timeout = timeout
        self.batch = batch
        self.ident = "openai:%s:%d" % (model, self.dims)
        self.dim = 1536

    def cost_per_token(self) -> float:
        m = self.model.lower()
        if "large" in m:
            return 0.13 / 1e6
        return 0.02 / 1e6

    def _one_batch(self, batch: List[str]) -> List[List[float]]:
        payload: Dict[str, Any] = {"model": self.model, "input": batch}
        if self.dims:
            payload["dimensions"] = self.dims
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.key,
            "Accept": "application/json",
        }
        path = "/embeddings" if self.base.endswith("/v1") else "/v1/embeddings"
        url = self.base + path
        status, data = http_request_json("POST", url, headers, body, self.timeout)
        if status != 200:
            raise RuntimeError("embedding API %s: %s" % (status, trunc(data.decode("utf-8", "replace"), 300)))
        obj = json.loads(data)
        items = sorted(obj.get("data", []), key=lambda d: d.get("index", 0))
        vecs = [list(map(float, it["embedding"])) for it in items]
        if vecs and len(vecs[0]) != self.dim:
            self.dim = len(vecs[0])
        return [vec_normalize(v) for v in vecs]

    def embed(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for i in range(0, len(texts), self.batch):
            out.extend(self._one_batch(texts[i : i + self.batch]))
        return out


def http_request_json(method: str, url: str, headers: Dict[str, str], body: Optional[bytes],
                      timeout: float) -> Tuple[int, bytes]:
    parts = urlsplit(url)
    path = parts.path or "/"
    if parts.query:
        path += "?" + parts.query
    if parts.scheme == "https":
        conn: http.client.HTTPConnection = http.client.HTTPSConnection(
            parts.hostname, parts.port, timeout=timeout, context=ssl.create_default_context()
        )
    else:
        conn = http.client.HTTPConnection(parts.hostname, parts.port, timeout=timeout)
    try:
        conn.request(method, path, body=body, headers=headers)
        r = conn.getresponse()
        return r.status, r.read()
    finally:
        conn.close()


class EmbeddingService:
    """embedder + 两级缓存（进程内 LRU + SQLite），避免重复算/重复调 API。"""

    def __init__(self, embedder: Embedder, store: Store, cache_size: int = 50000):
        self.embedder = embedder
        self.store = store
        self.cache_size = cache_size
        self._lru: Dict[str, List[float]] = {}
        self._order: List[str] = []
        self._lock = threading.RLock()
        self.embed_calls = 0
        self.embed_texts = 0
        self.embed_cost = 0.0

    def _key(self, text: str) -> str:
        return self.embedder.ident + "\x00" + sha(text, 40)

    def embed_one(self, text: str) -> Tuple[List[float], bool]:
        """返回 (向量, 是否命中本地缓存)。"""
        k = self._key(text)
        with self._lock:
            v = self._lru.get(k)
            if v is not None:
                return v, True
        blob = self.store.embed_get(k)
        if blob is not None:
            v = blob_to_vec(blob, self.embedder.dim)
            if v:
                self._remember(k, v)
                return v, True
        vecs = self.embedder.embed([text])
        v = vecs[0] if vecs else []
        self.embed_calls += 1
        self.embed_texts += 1
        self.embed_cost += self.embedder.cost_per_token() * est_tokens(text)
        if v:
            self.store.embed_put(k, vec_to_blob(v), len(v))
            self._remember(k, v)
        return v, False

    def _remember(self, k: str, v: List[float]) -> None:
        with self._lock:
            if k not in self._lru:
                self._order.append(k)
                if len(self._order) > self.cache_size:
                    old = self._order.pop(0)
                    self._lru.pop(old, None)
            self._lru[k] = v


# --------------------------------------------------------------------------- #
# 请求预处理（归一化 / 命名空间 / 可缓存性）
# --------------------------------------------------------------------------- #

@dataclass
class Prepared:
    ok: bool
    reason: str = ""
    project: str = "default"
    ns: str = ""
    text: str = ""
    full_text: str = ""
    qhash: str = ""
    model: str = ""
    stream: bool = False
    temperature: float = 1.0
    preview: str = ""
    n_messages: int = 0
    entities: Tuple[str, ...] = ()
    display: str = ""
    max_tokens: int = 0


@dataclass
class Answer:
    text: str
    finish_reason: str = "stop"
    has_tool_calls: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0


def _text_of_content(content: Any) -> Tuple[str, bool]:
    """返回 (纯文本, 是否含图片/音频等非文本片段)。"""
    if content is None:
        return "", False
    if isinstance(content, str):
        return content, False
    if isinstance(content, list):
        parts, media = [], False
        for p in content:
            if isinstance(p, str):
                parts.append(p)
                continue
            if not isinstance(p, dict):
                continue
            t = p.get("type")
            if t in ("text", "input_text"):
                parts.append(str(p.get("text", "")))
            elif t in ("image_url", "image", "input_audio", "audio", "file"):
                media = True
        return "\n".join(parts), media
    return str(content), False


def prepare(payload: Dict[str, Any], project: str, embedder_id: str,
            max_temperature: float = 1.0, mask_dynamic_on: bool = True,
            max_embed_chars: int = 4000) -> Prepared:
    """判定可缓存性并构造命名空间 / 查询文本。"""
    if not isinstance(payload, dict):
        return Prepared(ok=False, reason="bad-json")
    model = str(payload.get("model") or "")
    if not model:
        return Prepared(ok=False, reason="no-model")
    if payload.get("n", 1) not in (None, 1):
        return Prepared(ok=False, reason="n>1", model=model)
    if payload.get("logprobs"):
        return Prepared(ok=False, reason="logprobs", model=model)

    temp = payload.get("temperature")
    temp = 1.0 if temp is None else float(temp)
    if max_temperature is not None and temp > float(max_temperature):
        return Prepared(ok=False, reason="temperature=%.2f" % temp, model=model, temperature=temp)

    msgs = payload.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return Prepared(ok=False, reason="no-messages", model=model)

    sys_parts: List[str] = []
    convo_parts: List[str] = []
    clear_parts: List[str] = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "user")
        text, media = _text_of_content(m.get("content"))
        if media:
            return Prepared(ok=False, reason="multimodal", model=model, temperature=temp)
        if role in ("system", "developer"):
            sys_parts.append(text)
        else:
            convo_parts.append("%s: %s" % (role, text))
            clear_parts.append(text)
        # assistant 消息里带 tool_calls 也在 convo_parts 之外，不影响语义文本
    if not convo_parts:
        return Prepared(ok=False, reason="empty-prompt", model=model, temperature=temp)

    tools = payload.get("tools") or payload.get("functions")
    tools_txt = json.dumps(tools, sort_keys=True, ensure_ascii=False) if tools else ""
    rf = payload.get("response_format")
    rf_txt = json.dumps(rf, sort_keys=True, ensure_ascii=False) if rf else ""

    sys_txt = "\n".join(p for p in sys_parts if p)
    if mask_dynamic_on:
        sys_txt = mask_dynamic(sys_txt)
        tools_txt = mask_dynamic(tools_txt)

    raw_convo = "\n".join(convo_parts)
    raw_clear = "\n".join(clear_parts)
    full_text = normalize_text(raw_convo)
    display = normalize_text(raw_clear)
    ents = entities(raw_clear)
    if not full_text or not display:
        return Prepared(ok=False, reason="empty-prompt", model=model, temperature=temp)
    query_text = display[-max_embed_chars:] if max_embed_chars > 0 else display

    ns = sha(
        "\x1f".join([
            "v1", embedder_id, model.lower(),
            sha(sys_txt), sha(tools_txt), sha(rf_txt),
        ]),
        32,
    )
    qhash = sha(ns + "\x1f" + full_text, 40)
    return Prepared(
        ok=True,
        project=project,
        ns=ns,
        text=query_text,
        full_text=full_text,
        qhash=qhash,
        model=model,
        stream=bool(payload.get("stream")),
        temperature=temp,
        preview=trunc(display, 120),
        n_messages=len(msgs),
        entities=ents,
        display=display,
        max_tokens=int(payload.get("max_tokens") or payload.get("max_completion_tokens") or 0),
    )


def extract_answer(resp: Dict[str, Any]) -> Optional[Answer]:
    """从 OpenAI 风格的响应里取答案；不可缓存的情况返回 None。"""
    choices = resp.get("choices") or []
    if not choices:
        return None
    ch = choices[0] or {}
    finish = str(ch.get("finish_reason") or "")
    msg = ch.get("message") or ch.get("delta") or {}
    if ch.get("tool_calls") or msg.get("tool_calls") or finish in ("tool_calls", "function_call"):
        return None
    content = msg.get("content")
    if isinstance(content, list):
        content, _ = _text_of_content(content)
    if not isinstance(content, str) or not content.strip():
        return None
    usage = resp.get("usage") or {}
    return Answer(
        text=content,
        finish_reason=finish or "stop",
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
    )


def encode_answer(resp: Dict[str, Any]) -> bytes:
    return zlib.compress(json.dumps(resp, ensure_ascii=False, separators=(",", ":")).encode("utf-8"), 6)


def decode_answer(blob: bytes) -> Dict[str, Any]:
    return json.loads(zlib.decompress(blob).decode("utf-8"))


def assemble_sse(raw: bytes, model: str = "") -> Optional[Dict[str, Any]]:
    """把 SSE 流拼回一个完整的 chat.completion 响应（用于缓存流式结果）。"""
    text_parts: List[str] = []
    usage: Dict[str, Any] = {}
    finish = ""
    rid = ""
    created = 0
    mdl = model
    saw_tool = False
    for line in raw.split(b"\n"):
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data in (b"[DONE]", b""):
            continue
        try:
            obj = json.loads(data)
        except Exception:
            continue
        rid = obj.get("id") or rid
        created = obj.get("created") or created
        mdl = obj.get("model") or mdl
        if obj.get("usage"):
            usage = obj["usage"]
        for ch in obj.get("choices") or []:
            d = ch.get("delta") or {}
            if d.get("tool_calls") or ch.get("finish_reason") in ("tool_calls", "function_call"):
                saw_tool = True
            c = d.get("content")
            if isinstance(c, str) and c:
                text_parts.append(c)
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
    if saw_tool or not text_parts:
        return None
    if finish in ("", "length") and not text_parts:
        return None
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    if not completion_tokens:
        completion_tokens = est_tokens("".join(text_parts))
    return {
        "id": rid or "chatcmpl-cache",
        "object": "chat.completion",
        "created": created or int(now()),
        "model": mdl or model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "".join(text_parts)},
                "finish_reason": finish or "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def sse_replay(resp: Dict[str, Any], chunk_chars: int = 24) -> Iterable[bytes]:
    """把缓存的答案按 SSE 回放给客户端。"""
    rid = "chatcmpl-cache-" + sha(resp.get("id", "") + str(now()), 12)
    created = int(resp.get("created") or now())
    model = resp.get("model", "")
    ch = (resp.get("choices") or [{}])[0]
    text = (ch.get("message") or {}).get("content") or ""
    finish = ch.get("finish_reason") or "stop"

    def ev(obj: Dict[str, Any]) -> bytes:
        return b"data: " + json.dumps(obj, ensure_ascii=False).encode("utf-8") + b"\n\n"

    yield ev({
        "id": rid, "object": "chat.completion.chunk", "created": created, "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
    })
    for i in range(0, len(text), chunk_chars):
        yield ev({
            "id": rid, "object": "chat.completion.chunk", "created": created, "model": model,
            "choices": [{"index": 0, "delta": {"content": text[i : i + chunk_chars]}, "finish_reason": None}],
        })
    yield ev({
        "id": rid, "object": "chat.completion.chunk", "created": created, "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
    })
    usage = resp.get("usage")
    if usage:
        yield ev({
            "id": rid, "object": "chat.completion.chunk", "created": created, "model": model,
            "choices": [], "usage": usage,
        })
    yield b"data: [DONE]\n\n"


# --------------------------------------------------------------------------- #
# 阈值自动校准
# --------------------------------------------------------------------------- #

class Calibrator:
    """
    不需要人工标注的阈值校准，三条腿：

    1. 空模型下界（主）：取样同一命名空间里的向量两两余弦，估计「不相关问题」的
       相似度分布 (mu, sigma)。真正的语义重复必须明显高于这个背景噪声。
           floor = clip(mu + z*sigma, prior_floor, ceiling)
       其中 prior_floor / ceiling 是按 embedder 类型给的常识护栏。
    2. 反馈偏置（修正）：命中后用户重问 / 显式差评，说明这次相似度是误命中。
       bias 直接把阈值抬到「那次误命中的分数 + 余量」之上，随时间衰减。
    3. 缓慢下探（求省）：没有误命中信号时，base 每轮向 floor 靠一点，
       直到 floor + 一档余量，多了不给 —— 省钱但不越安全线。

    最终阈值 = clip(base + bias, 0, max_threshold)。
    """

    def __init__(self, store: Store, cfg: Dict[str, Any], embedder_id: str):
        self.store = store
        self.cfg = cfg
        self.embedder_id = embedder_id
        self._lock = threading.RLock()
        self._mem: Dict[Tuple[str, str], Dict[str, Any]] = {}

    # -- 先验 -------------------------------------------------------------
    def prior_floor(self) -> float:
        if self.cfg.get("min_threshold"):
            return float(self.cfg["min_threshold"])
        return 0.80 if self.embedder_id.startswith("openai") else 0.45

    def ceiling(self) -> float:
        return float(self.cfg.get("max_threshold", 0.995))

    def bootstrap_base(self) -> float:
        return 0.90 if self.embedder_id.startswith("openai") else 0.62

    def _defaults(self) -> Dict[str, Any]:
        b = float(self.cfg.get("threshold") or 0.0)
        if b > 0:
            return {"base": b, "bias": 0.0, "eff": b, "floor": min(b, self.prior_floor()),
                    "n_null": 0, "mu": 0.0, "sigma": 0.0, "p99": 0.0,
                    "lookups": 0, "hits": 0, "bad": 0, "status": "manual"}
        base = self.bootstrap_base()
        return {"base": base, "bias": 0.0, "eff": base, "floor": self.prior_floor(),
                "n_null": 0, "mu": 0.0, "sigma": 0.0, "p99": 0.0,
                "lookups": 0, "hits": 0, "bad": 0, "status": "bootstrap"}

    # -- 读写 -------------------------------------------------------------
    def get(self, project: str, ns: str) -> Dict[str, Any]:
        k = (project, ns)
        with self._lock:
            m = self._mem.get(k)
            if m is not None:
                return m
        row = self.store.get_calib(project, ns)
        if row is None:
            d = self._defaults()
        else:
            d = {kk: row[kk] for kk in row.keys() if kk not in ("project", "ns", "updated_at")}
            d["bias"] = float(d.get("bias") or 0.0)
        with self._lock:
            self._mem[k] = d
        return d

    def _put(self, project: str, ns: str, d: Dict[str, Any]) -> None:
        with self._lock:
            self._mem[(project, ns)] = d
        self.store.put_calib(project, ns, **d)

    def effective(self, project: str, ns: str) -> float:
        d = self.get(project, ns)
        return float(min(max(d["eff"], 0.0), self.ceiling()))

    # -- 校准 -------------------------------------------------------------
    def recalibrate(self, project: str, ns: str, vecs: Sequence[Sequence[float]],
                    qhashes: Optional[Sequence[str]] = None, force: bool = False,
                    reason: str = "auto") -> Dict[str, Any]:
        d = dict(self.get(project, ns))
        manual = bool(self.cfg.get("threshold"))
        n = len(vecs)

        mu = sigma = p99 = 0.0
        n_pairs = 0
        prior_floor = self.prior_floor()
        ceiling = self.ceiling()

        if n >= 6:
            # 用 (project, ns) 派生的固定种子，同一批数据每次算出同样的阈值，好复现
            rng = random.Random(int(sha("%s|%s" % (project, ns), 12), 16))
            sims = self._null_sims(vecs, qhashes, rng)
            if len(sims) >= 20:
                sims.sort()
                n_pairs = len(sims)
                p99 = sims[min(n_pairs - 1, int(n_pairs * 0.99))]
                # 掐掉尾部最相似的 10%：它们大概率是真重复，不是「背景噪声」。
                # 剩下的才是「两个不相关问题能有多像」，这才是阈值的安全基线。
                core = sims[: max(1, int(n_pairs * 0.9))]
                mu = sum(core) / len(core)
                var = sum((x - mu) ** 2 for x in core) / len(core)
                sigma = math.sqrt(var)

        lift = float(self.cfg.get("max_floor_lift") or 0.15)
        margin = float(self.cfg.get("calib_margin") or 0.02)
        if n_pairs >= 30 and sigma > 1e-6:
            z = float(self.cfg.get("calib_z", 3.0))
            floor_raw = mu + z * sigma
            # 语料越同质，背景噪声越高，阈值跟着抬高；
            # 但最多比先验高 lift，避免同质语料把缓存彻底关死。
            floor_raw = min(max(floor_raw, prior_floor), min(ceiling, prior_floor + lift))
            if str(d.get("status")) == "auto" and d.get("floor"):
                # 轻度平滑，避免小样本抖动导致阈值来回跳
                floor = 0.5 * float(d["floor"]) + 0.5 * floor_raw
            else:
                floor = floor_raw
            status = "auto"
        else:
            floor = float(d.get("floor") or prior_floor)
            status = "auto" if str(d.get("status")) == "auto" else "bootstrap"

        if manual:
            base = float(self.cfg["threshold"])
            status = "manual"
        elif status == "auto":
            # 校准到位后：工作阈值 = 安全下界 + 边际；再靠 bias 吸收误命中反馈
            base = floor + margin
        else:
            base = float(d.get("base") or self.bootstrap_base())
        base = min(max(base, floor), ceiling)

        # 偏置衰减（一轮没有新差评就松一点）
        bias = float(d.get("bias") or 0.0) * 0.85
        if abs(bias) < 0.002:
            bias = 0.0

        eff = min(max(base + bias, floor), ceiling)
        new = {
            "base": round(base, 6),
            "bias": round(bias, 6),
            "eff": round(eff, 6),
            "floor": round(floor, 6),
            "n_null": n_pairs,
            "mu": round(mu, 6),
            "sigma": round(sigma, 6),
            "p99": round(p99, 6),
            "lookups": int(d.get("lookups") or 0),
            "hits": int(d.get("hits") or 0),
            "bad": int(d.get("bad") or 0),
            "status": status,
        }
        old = float(d.get("eff") or 0.0)
        self._put(project, ns, new)
        if force or abs(new["eff"] - old) >= 0.002 or status != d.get("status"):
            self.store.log_calib(project, ns, old, new["eff"], new["floor"], mu, sigma,
                                 "%s/%s" % (reason, status))
        return new

    def _null_sims(self, vecs: Sequence[Sequence[float]],
                   qhashes: Optional[Sequence[str]],
                   rng: Optional[random.Random] = None) -> List[float]:
        """上三角两两相似度（排除同一条目 / 同 hash 的重复）。"""
        limit = int(self.cfg.get("calib_pairs", 4000))
        idx = list(range(len(vecs)))
        if len(idx) > 400:
            idx = (rng or random).sample(idx, 400)
        sims: List[float] = []
        if _np is not None:
            try:
                mat = _np.asarray([vecs[i] for i in idx], dtype="float32")
                full = matmul(mat, mat.T)
                iu = _np.triu_indices(len(idx), k=1)
                vals = full[iu]
                if qhashes is not None:
                    qs = [qhashes[i] for i in idx]
                    mask = _np.array([qs[a] != qs[b] for a, b in zip(iu[0].tolist(), iu[1].tolist())])
                    vals = vals[mask] if mask.any() else vals
                vals = vals[~_np.isnan(vals)]
                if len(vals) > limit:
                    vals = _np.random.default_rng(7).choice(vals, limit, replace=False)
                return [float(x) for x in vals]
            except Exception:
                sims = []
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                if qhashes is not None and qhashes[idx[a]] == qhashes[idx[b]]:
                    continue
                sims.append(cos_py(vecs[idx[a]], vecs[idx[b]]))
                if len(sims) >= limit:
                    return sims
        return sims

    # -- 反馈 -------------------------------------------------------------
    def feedback(self, project: str, ns: str, label: str, score: float = 0.0,
                 weight: float = 1.0, strong: bool = True) -> Dict[str, Any]:
        d = dict(self.get(project, ns))
        base = float(d.get("base") or self.bootstrap_base())
        bias = float(d.get("bias") or 0.0)
        if label == "bad":
            step = 0.02 * weight
            if strong and score > 0:
                # 明确差评：把阈值抬到「刚才那次误命中的分数」之上
                bump = max(step, (score + 0.02) - (base + bias))
            else:
                # 隐式信号（用户重问）：只做小幅微调，别把命名空间整死
                bump = min(step, 0.03)
            bias = min(bias + max(bump, 0.0), min(0.25, self.ceiling() - base))
            d["bad"] = int(d.get("bad") or 0) + 1
        elif label == "good":
            bias = max(-0.05, bias - 0.01 * weight)
        d["bias"] = round(bias, 6)
        d["eff"] = round(min(max(base + bias, float(d.get("floor") or 0.0)), self.ceiling()), 6)
        self._put(project, ns, d)
        self.store.log_calib(project, ns, float(d["eff"]), float(d["eff"]),
                             float(d.get("floor") or 0.0), 0.0, 0.0, "feedback:%s@%.3f" % (label, score))
        return d


# --------------------------------------------------------------------------- #
# 缓存引擎
# --------------------------------------------------------------------------- #

class _Index:
    __slots__ = ("ids", "qhashes", "vecs", "mats", "toks", "prompts", "hits", "pt", "ct",
                 "exp", "ents")

    def __init__(self) -> None:
        self.ids: List[int] = []
        self.qhashes: List[str] = []
        self.vecs: List[List[float]] = []
        self.mats: Any = None
        self.toks: List[set] = []
        self.prompts: List[str] = []
        self.hits: List[int] = []
        self.pt: List[int] = []
        self.ct: List[int] = []
        self.exp: List[Optional[float]] = []
        self.ents: List[Tuple[str, ...]] = []

    def __len__(self) -> int:
        return len(self.ids)


@dataclass
class Lookup:
    ok: bool
    kind: str = ""              # exact | semantic
    row: Optional[sqlite3.Row] = None
    score: float = 0.0
    threshold: float = 0.0
    near_score: float = 0.0
    near_prompt: str = ""
    reason: str = ""
    embed_cost: float = 0.0
    embed_cached: bool = False
    latency_ms: float = 0.0


class CacheEngine:
    def __init__(self, cfg: Dict[str, Any], store: Store, embedder: Embedder):
        self.cfg = cfg
        self.store = store
        self.embedder = embedder
        self.embed = EmbeddingService(embedder, store)
        self.calib = Calibrator(store, cfg, embedder.ident)
        self.prices = Prices(cfg.get("prices") or {})
        self._idx: Dict[Tuple[str, str], _Index] = {}
        self._idx_lock = threading.RLock()
        self._counters: Dict[Tuple[str, str], int] = {}
        self._recent: Dict[Tuple[str, str, str], Tuple[float, int, float, str]] = {}
        self._recent_lock = threading.Lock()
        self._inserts = 0
        self.stats = {"lookups": 0, "hits": 0, "misses": 0, "served": 0, "stored": 0}

    # -- 索引 -------------------------------------------------------------
    def _index(self, project: str, ns: str) -> _Index:
        k = (project, ns)
        rows = self.store.iter_ns(project, ns)
        with self._idx_lock:
            cur = self._idx.get(k)
            if cur is not None and len(cur) == len(rows):
                return cur
        idx = _Index()
        ts = now()
        for r in rows:
            exp = r["expires_at"]
            if exp is not None and exp < ts:
                continue
            v = blob_to_vec(r["vec"], r["dim"])
            if not v:
                continue
            idx.ids.append(int(r["id"]))
            idx.qhashes.append(r["qhash"])
            idx.vecs.append(v)
            idx.hits.append(int(r["hits"] or 0))
            idx.prompts.append(r["prompt"] or "")
            idx.pt.append(int(r["prompt_tokens"] or 0))
            idx.ct.append(int(r["completion_tokens"] or 0))
            idx.exp.append(exp)
            idx.toks.append(set(lex_tokens(r["prompt"] or "")))
            try:
                idx.ents.append(tuple(json.loads(r["ents"] or "[]")))
            except Exception:
                idx.ents.append(())
        if _np is not None and idx.vecs:
            try:
                idx.mats = _np.asarray(idx.vecs, dtype="float32")
                idx.vecs = idx.mats  # 只留一份连续内存，别把 Python list 也留着
            except Exception:
                idx.mats = None
        with self._idx_lock:
            self._idx[k] = idx
        return idx

    def _invalidate(self, project: str, ns: str) -> None:
        with self._idx_lock:
            self._idx.pop((project, ns), None)

    # -- 查询 -------------------------------------------------------------
    def lookup(self, prep: Prepared) -> Lookup:
        t0 = time.perf_counter()
        project, ns = prep.project, prep.ns
        th = None
        try:
            row = self.store.get_entry(project, ns, prep.qhash)
            if row is not None:
                if row["expires_at"] and row["expires_at"] < now():
                    self.store.delete_entry(row["id"])
                    self._invalidate(project, ns)
                    row = None
                elif prep.max_tokens and int(row["completion_tokens"] or 0) > prep.max_tokens:
                    row = None   # 缓存答案比这次允许的还长，不能直接回给客户端
            if row is not None:
                th = self.calib.effective(project, ns)
                return Lookup(True, "exact", row, 1.0, th, latency_ms=(time.perf_counter() - t0) * 1000)

            idx = self._index(project, ns)
            if len(idx) == 0:
                th = self.calib.effective(project, ns)
                return Lookup(False, reason="cold", threshold=th,
                              latency_ms=(time.perf_counter() - t0) * 1000)

            before = self.embed.embed_cost
            vec, was_cached = self.embed.embed_one(prep.text)
            embed_cost = self.embed.embed_cost - before
            th = self.calib.effective(project, ns)
            if not vec:
                return Lookup(False, reason="embed-failed", threshold=th, embed_cost=embed_cost,
                              latency_ms=(time.perf_counter() - t0) * 1000)

            # 实体必须完全一致（单号/金额/URL/ID 不同 → 绝不命中）
            q_ents = tuple(prep.entities)
            cand = [i for i, e in enumerate(idx.ents) if e == q_ents]
            if not cand:
                return Lookup(False, reason="entity-mismatch", threshold=th,
                              embed_cost=embed_cost, embed_cached=was_cached,
                              latency_ms=(time.perf_counter() - t0) * 1000)

            best_i, best = -1, -1.0
            if idx.mats is not None and _np is not None:
                q = _np.asarray(vec, dtype="float32")
                if len(cand) == len(idx.ids):
                    sims = matmul(idx.mats, q)
                    j = int(_np.argmax(sims))
                    best_i, best = j, float(sims[j])
                else:
                    sims = matmul(idx.mats[cand], q)
                    j = int(_np.argmax(sims))
                    best_i, best = cand[j], float(sims[j])
            else:
                for i in cand:
                    s = cos_py(idx.vecs[i], vec)
                    if s > best:
                        best_i, best = i, s

            # 本地向量是词面哈希，必须再过一道词面门槛，防哈希碰撞误命中
            if self.embedder.ident.startswith("local") and best_i >= 0:
                q_toks = set(lex_tokens(prep.display))
                cand_toks = idx.toks[best_i]
                gate = float(self.cfg.get("lexical_gate") or 0.6)
                cover = float(self.cfg.get("lexical_cover") or 0.8)
                jac = jaccard(q_toks, cand_toks)
                cov = overlap(q_toks, cand_toks)
                # 一字之差的改写：jac 会掉到 0.65，但包含度还在 0.85；
                # 「什么时候发货」vs「什么时候退款」：包含度只有 0.71，直接拦掉。
                if jac < gate or cov < cover:
                    return Lookup(False, score=best, threshold=th, near_score=best,
                                  near_prompt=idx.prompts[best_i], reason="lexical-gate",
                                  embed_cost=embed_cost, embed_cached=was_cached,
                                  latency_ms=(time.perf_counter() - t0) * 1000)

            if best >= th and best_i >= 0:
                if prep.max_tokens and idx.ct[best_i] > prep.max_tokens:
                    return Lookup(False, reason="max-tokens", score=best, threshold=th,
                                  near_score=best, near_prompt=idx.prompts[best_i],
                                  embed_cost=embed_cost, embed_cached=was_cached,
                                  latency_ms=(time.perf_counter() - t0) * 1000)
                row = self.store.get_entry_by_id(idx.ids[best_i])
                if row is not None:
                    return Lookup(True, "semantic", row, best, th, embed_cost=embed_cost,
                                  embed_cached=was_cached,
                                  latency_ms=(time.perf_counter() - t0) * 1000)

            near_score = best if best >= th - 0.10 else 0.0
            return Lookup(False, score=max(best, 0.0), threshold=th, near_score=near_score,
                          near_prompt=idx.prompts[best_i] if (near_score and best_i >= 0) else "",
                          reason="below-threshold", embed_cost=embed_cost,
                          embed_cached=was_cached, latency_ms=(time.perf_counter() - t0) * 1000)
        except Exception as e:  # 缓存出问题绝不能拖垮代理
            LOG.exception("lookup 异常: %s", e)
            return Lookup(False, reason="lookup-error:%s" % e,
                          latency_ms=(time.perf_counter() - t0) * 1000)

    # -- 写入 -------------------------------------------------------------
    def store_answer(self, prep: Prepared, resp: Dict[str, Any]) -> Optional[int]:
        ans = extract_answer(resp)
        if ans is None:
            return None
        try:
            before = self.embed.embed_cost
            vec, _ = self.embed.embed_one(prep.text)
            extra = self.embed.embed_cost - before
            if not vec:
                return None
            ttl = float(self.cfg.get("ttl") or 0)
            eid = self.store.insert_entry(
                project=prep.project,
                ns=prep.ns,
                qhash=prep.qhash,
                prompt=prep.display[:8000],
                answer=encode_answer(resp),
                vec=vec_to_blob(vec),
                dim=len(vec),
                model=prep.model,
                created_at=now(),
                last_hit_at=None,
                hits=0,
                prompt_tokens=ans.prompt_tokens or est_tokens(prep.full_text),
                completion_tokens=ans.completion_tokens or est_tokens(ans.text),
                finish_reason=ans.finish_reason,
                expires_at=(now() + ttl) if ttl > 0 else None,
                ents=json.dumps(list(prep.entities), ensure_ascii=False),
                meta=json.dumps({"temp": prep.temperature, "stream": prep.stream}),
            )
            self._invalidate(prep.project, prep.ns)
            self._inserts += 1
            if self._inserts % 500 == 0:
                self.store.evict(prep.project, int(self.cfg.get("max_entries") or 200000))
                self.store.purge_expired(now())
            self.stats["stored"] += 1
            return eid
        except Exception as e:
            LOG.exception("store 异常: %s", e)
            return None

    # -- 命中后组装响应 ---------------------------------------------------
    def materialize(self, row: sqlite3.Row, score: float, kind: str) -> Dict[str, Any]:
        obj = decode_answer(row["answer"])
        obj["id"] = "chatcmpl-cache-" + sha(str(row["id"]) + str(now()), 12)
        obj["created"] = int(now())
        obj["model"] = obj.get("model") or row["model"] or ""
        hits = int(row["hits"] or 0) + 1
        try:
            self.store.touch_entry(int(row["id"]), now(), hits)
            with self._idx_lock:
                idx = self._idx.get((row["project"], row["ns"]))
                if idx is not None:
                    for i, eid in enumerate(idx.ids):
                        if eid == int(row["id"]):
                            idx.hits[i] = hits
                            break
        except Exception:
            pass
        return obj

    # -- 事件记账 ---------------------------------------------------------
    def costs(self, model: str, pt: int, ct: int) -> float:
        return self.prices.cost(model, pt, ct)

    def log_event(self, kind: str, prep: Prepared, *, score: float = 0.0, detail: str = "",
                  entry_id: int = 0, pt: int = 0, ct: int = 0, latency_ms: float = 0.0,
                  embed_cost: float = 0.0, saved_usd: Optional[float] = None,
                  ts: Optional[float] = None) -> None:
        if saved_usd is None:
            saved_usd = self.costs(prep.model, pt, ct) if kind in ("hit", "shadow") else 0.0
        try:
            self.store.add_event(
                ts=float(ts or now()), project=prep.project, ns=prep.ns, model=prep.model, kind=kind,
                detail=detail, score=round(score, 6), prompt_tokens=int(pt),
                completion_tokens=int(ct), cost_saved=round(float(saved_usd), 8),
                cost_extra=round(float(embed_cost), 8), latency_ms=round(latency_ms, 3),
                prompt=prep.preview, entry_id=entry_id or None,
            )
        except Exception as e:
            LOG.warning("写入事件失败: %s", e)

    # -- 反馈 -------------------------------------------------------------
    def feedback(self, project: str, ns: str, label: str, score: float = 0.0,
                 weight: float = 1.0, detail: str = "explicit",
                 strong: bool = True) -> Dict[str, Any]:
        d = self.calib.feedback(project, ns, label, score=score, weight=weight, strong=strong)
        try:
            self.store.add_event(
                ts=now(), project=project, ns=ns, model="", kind="feedback",
                detail=detail + ":" + label, score=round(score, 6), prompt_tokens=0,
                completion_tokens=0, cost_saved=0.0, cost_extra=0.0, latency_ms=0.0,
                prompt="", entry_id=None,
            )
        except Exception:
            pass
        return d

    def note_served(self, prep: Prepared, score: float, entry_id: int, kind: str = "") -> None:
        with self._recent_lock:
            if len(self._recent) > 20000:
                self._recent.clear()
            self._recent[(prep.project, prep.ns, prep.qhash)] = (now(), entry_id, score, kind)

    def check_repeat(self, prep: Prepared) -> Optional[float]:
        """
        语义命中后短时间内同一问题又来 = 客户不满意，隐式差评。

        只对 semantic 命中生效：精确命中的重复更可能是网络重试/并发重放，
        拿它当差评会把缓存打死。
        """
        if not self.cfg.get("implicit_feedback"):
            return None
        with self._recent_lock:
            rec = self._recent.get((prep.project, prep.ns, prep.qhash))
        if not rec:
            return None
        ts, _eid, score, kind = rec
        if kind != "semantic":
            return None
        window = float(self.cfg.get("repeat_window") or 0)
        if 0 < now() - ts <= window:
            return score
        return None

    def tick(self, prep: Prepared, hit: bool, score: float) -> None:
        """计数并在合适的时候触发自动校准。"""
        k = (prep.project, prep.ns)
        n = self._counters.get(k, 0) + 1
        self._counters[k] = n
        d = dict(self.calib.get(prep.project, prep.ns))
        d["lookups"] = int(d.get("lookups") or 0) + 1
        if hit:
            d["hits"] = int(d.get("hits") or 0) + 1
        self.calib._mem[(prep.project, prep.ns)] = d
        every = int(self.cfg.get("calib_every") or 200)
        if every > 0 and n % every == 0:
            self.recalibrate(prep.project, prep.ns)

    def recalibrate(self, project: str, ns: str, force: bool = False) -> Dict[str, Any]:
        idx = self._index(project, ns)
        return self.calib.recalibrate(project, ns, idx.vecs, idx.qhashes, force=force,
                                      reason="force" if force else "auto")


# --------------------------------------------------------------------------- #
# 上游 HTTP 客户端（带连接复用）
# --------------------------------------------------------------------------- #

HOP_HEADERS = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length",
    "accept-encoding", "expect",
}


class Upstream:
    def __init__(self, base: str, timeout: float = 300.0):
        p = urlsplit(base)
        if not p.scheme:
            p = urlsplit("https://" + base)
        self.scheme = p.scheme
        self.host = p.hostname or "api.openai.com"
        self.port = p.port
        self.base_path = (p.path or "").rstrip("/")
        self.timeout = timeout
        self._pool: List[http.client.HTTPConnection] = []
        self._lock = threading.Lock()
        self._ssl = ssl.create_default_context()

    def _connect(self) -> http.client.HTTPConnection:
        if self.scheme == "https":
            return http.client.HTTPSConnection(self.host, self.port, timeout=self.timeout, context=self._ssl)
        return http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)

    def _get_conn(self) -> http.client.HTTPConnection:
        with self._lock:
            if self._pool:
                return self._pool.pop()
        return self._connect()

    def _put_conn(self, conn: http.client.HTTPConnection) -> None:
        with self._lock:
            if len(self._pool) < 32:
                self._pool.append(conn)
                return
        try:
            conn.close()
        except Exception:
            pass

    def path_for(self, path: str) -> str:
        bp = self.base_path
        if not bp:
            return path
        if bp.endswith("/v1") and path.startswith("/v1/"):
            return bp[:-3] + path
        if path.startswith(bp):
            return path
        return bp + path

    def open(self, method: str, path: str, headers: Dict[str, str],
             body: Optional[bytes]) -> Tuple[http.client.HTTPConnection, http.client.HTTPResponse]:
        last: Optional[Exception] = None
        for _attempt in range(2):
            conn = self._get_conn()
            try:
                conn.request(method, self.path_for(path), body=body, headers=headers)
                return conn, conn.getresponse()
            except (http.client.RemoteDisconnected, http.client.CannotSendRequest,
                    BrokenPipeError, ConnectionResetError, ConnectionAbortedError) as e:
                last = e
                try:
                    conn.close()
                except Exception:
                    pass
                continue
            except Exception as e:
                try:
                    conn.close()
                except Exception:
                    pass
                raise e
        raise last or RuntimeError("upstream connect failed")

    def release(self, conn: http.client.HTTPConnection, resp: http.client.HTTPResponse) -> None:
        try:
            resp.read()
        except Exception:
            try:
                conn.close()
            except Exception:
                pass
            return
        self._put_conn(conn)

    def close(self) -> None:
        with self._lock:
            pool, self._pool = self._pool, []
        for c in pool:
            try:
                c.close()
            except Exception:
                pass


# --------------------------------------------------------------------------- #
# 运行时
# --------------------------------------------------------------------------- #

@dataclass
class Runtime:
    cfg: Dict[str, Any]
    store: Store
    engine: CacheEngine
    upstream: Upstream
    started_at: float = field(default_factory=now)

    @property
    def prices(self) -> Prices:
        return self.engine.prices


def _header_text(v: str) -> str:
    """HTTP 头是 latin-1；客户端若塞了 UTF-8 中文项目名，这里还原回来。"""
    try:
        return v.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return v


def resolve_project(headers: Any, cfg: Dict[str, Any]) -> str:
    p = headers.get("X-Semcache-Project") or headers.get("x-semcache-project")
    if p:
        return _safe_project(_header_text(p))
    auth = headers.get("Authorization") or headers.get("authorization") or ""
    key = auth.split()[-1] if auth.strip() else ""
    for proj, prefix in (cfg.get("project_keys") or {}).items():
        if key and str(prefix) and key.startswith(str(prefix)):
            return _safe_project(proj)
    if key:
        return "key-" + sha(key, 8)
    return "default"


_PROJ_RE = re.compile(r"[^A-Za-z0-9_.\-\u4e00-\u9fff]")


def _safe_project(p: str) -> str:
    s = _PROJ_RE.sub("_", str(p).strip())[:64]
    return s or "default"


# --------------------------------------------------------------------------- #
# HTTP 处理
# --------------------------------------------------------------------------- #

def _miss_detail(near: Optional["Lookup"]) -> str:
    if near is None:
        return "none"
    if near.reason in ("lexical-gate", "entity-mismatch"):
        return "gate"
    if near.near_score <= 0:
        return "none"
    return "near"


def _json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "semcache/" + VERSION
    sys_version = ""

    # -- 公共 -------------------------------------------------------------
    @property
    def rt(self) -> Runtime:
        return self.server.runtime  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        LOG.debug("%s - %s", self.address_string(), fmt % args)

    def _send(self, status: int, body: bytes, ctype: str = "application/json",
              extra: Optional[Dict[str, str]] = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Semcache-Version", VERSION)
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(self, obj: Any, status: int = 200,
                   extra: Optional[Dict[str, str]] = None) -> None:
        self._send(status, _json_bytes(obj), "application/json; charset=utf-8", extra)

    def _read_body(self) -> bytes:
        maxb = int(self.rt.cfg.get("max_body") or 0) or 32 * 1024 * 1024
        te = (self.headers.get("Transfer-Encoding") or "").lower()
        if "chunked" in te:
            out = bytearray()
            while True:
                line = self.rfile.readline(65536).strip()
                if not line:
                    break
                try:
                    n = int(line.split(b";")[0], 16)
                except ValueError:
                    break
                if n == 0:
                    self.rfile.readline(65536)
                    break
                out += self.rfile.read(n)
                self.rfile.read(2)
                if len(out) > maxb:
                    break
            return bytes(out)
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = 0
        if n <= 0:
            return b""
        return self.rfile.read(min(n, maxb))

    # -- 路由 -------------------------------------------------------------
    def do_GET(self) -> None:
        self._route()

    def do_POST(self) -> None:
        self._route()

    def do_PUT(self) -> None:
        self._route()

    def do_PATCH(self) -> None:
        self._route()

    def do_DELETE(self) -> None:
        self._route()

    def do_HEAD(self) -> None:
        self._route()

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,PUT,PATCH,DELETE,OPTIONS")
        self.send_header("Access-Control-Max-Age", "86400")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _route(self) -> None:
        try:
            parts = urlsplit(self.path)
            path = parts.path
            if path in ("/healthz", "/health"):
                return self._healthz()
            if path.startswith("/admin/"):
                return self._admin(path, parts.query)
            if path in ("/", "/dashboard", "/dashboard/"):
                return self._dashboard()
            if self.command == "POST" and path.endswith("/chat/completions"):
                return self._proxy_chat(path)
            return self._passthrough(path)
        except (BrokenPipeError, ConnectionResetError):
            LOG.debug("客户端断开: %s", self.path)
        except Exception as e:
            LOG.exception("请求处理失败: %s", e)
            try:
                self._send_json({"error": {"message": "semcache internal error: %s" % e,
                                           "type": "proxy_error"}}, 500)
            except Exception:
                pass

    # -- 健康检查 ---------------------------------------------------------
    def _healthz(self) -> None:
        rtc = self.rt
        self._send_json({
            "ok": True,
            "version": VERSION,
            "uptime_s": round(now() - rtc.started_at, 1),
            "upstream": rtc.cfg.get("upstream"),
            "embedder": rtc.engine.embedder.ident,
            "dry_run": bool(rtc.cfg.get("dry_run")),
            "entries": rtc.store.count_entries(),
            "stats": rtc.engine.stats,
        })

    # -- 主路径：聊天补全 -------------------------------------------------
    def _proxy_chat(self, path: str) -> None:
        rtc = self.rt
        cfg = rtc.cfg
        engine = rtc.engine
        body = self._read_body()
        if not body:
            return self._send_json({"error": {"message": "empty body"}}, 400)
        try:
            payload = json.loads(body)
        except Exception:
            return self._passthrough(path, body)

        project = resolve_project(self.headers, cfg)
        prep = prepare(
            payload, project, engine.embedder.ident,
            max_temperature=float(cfg.get("max_temperature", 1.0)),
            mask_dynamic_on=bool(cfg.get("mask_dynamic", True)),
            max_embed_chars=int(cfg.get("max_embed_chars") or 4000),
        )

        if (self.headers.get("X-Semcache-No-Cache") or "").strip() in ("1", "true", "yes"):
            prep.ok = False
            prep.reason = "client-bypass"

        if not prep.ok:
            engine.log_event("bypass", prep, detail=prep.reason)
            return self._forward(path, body, prep, store_ok=False)

        if prep.stream and not cfg.get("cache_stream", True):
            engine.log_event("bypass", prep, detail="stream-no-cache")
            return self._forward(path, body, prep, store_ok=False)

        lk = engine.lookup(prep)

        if lk.ok:
            if cfg.get("dry_run"):
                engine.log_event("shadow", prep, score=lk.score, detail=lk.kind,
                                 entry_id=int(lk.row["id"]), pt=int(lk.row["prompt_tokens"] or 0),
                                 ct=int(lk.row["completion_tokens"] or 0),
                                 latency_ms=lk.latency_ms, embed_cost=lk.embed_cost)
                return self._forward(path, body, prep, store_ok=True)
            return self._serve_hit(prep, lk)

        # 未命中：交给 _forward 统一记一次 miss（避免重复计数）
        repeat_score = engine.check_repeat(prep)
        if repeat_score is not None:
            engine.feedback(prep.project, prep.ns, "bad", score=repeat_score, weight=0.5,
                            detail="implicit-repeat", strong=False)
        return self._forward(path, body, prep, store_ok=True, near=lk)

    def _serve_hit(self, prep: Prepared, lk: Lookup) -> None:
        rtc = self.rt
        engine = rtc.engine
        t0 = time.perf_counter()
        row = lk.row
        assert row is not None
        obj = engine.materialize(row, lk.score, lk.kind)
        pt = int(row["prompt_tokens"] or 0)
        ct = int(row["completion_tokens"] or 0)
        saved = engine.costs(prep.model, pt, ct)
        headers = {
            "X-Semcache": "hit",
            "X-Semcache-Kind": lk.kind,
            "X-Semcache-Score": "%.4f" % lk.score,
            "X-Semcache-Threshold": "%.4f" % lk.threshold,
            "X-Semcache-Tokens-Saved": str(pt + ct),
            "X-Semcache-Cost-Saved-USD": "%.8f" % saved,
            "X-Semcache-Age-S": "%.0f" % max(0.0, now() - float(row["created_at"] or now())),
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Expose-Headers": "*",
        }
        engine.log_event("hit", prep, score=lk.score, detail=lk.kind, entry_id=int(row["id"]),
                         pt=pt, ct=ct,
                         latency_ms=lk.latency_ms + (time.perf_counter() - t0) * 1000,
                         embed_cost=lk.embed_cost, saved_usd=saved)
        engine.tick(prep, hit=True, score=lk.score)
        engine.note_served(prep, lk.score, int(row["id"]), lk.kind)
        engine.stats["hits"] += 1
        engine.stats["served"] += 1

        if prep.stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            for k, v in headers.items():
                self.send_header(k, v)
            self.end_headers()
            for chunk in sse_replay(obj):
                self._write_chunk(chunk)
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        else:
            self._send(200, _json_bytes(obj), "application/json; charset=utf-8", headers)

    def _forward(self, path: str, body: bytes, prep: Prepared, store_ok: bool = True,
                 near: Optional[Lookup] = None) -> None:
        rtc = self.rt
        engine = rtc.engine
        headers = self._upstream_headers()
        t0 = time.perf_counter()
        try:
            conn, resp = rtc.upstream.open(self.command, path, headers, body)
        except Exception as e:
            engine.log_event("error", prep, detail="connect:%s" % trunc(str(e), 120))
            return self._send_json({"error": {"message": "semcache upstream error: %s" % e,
                                              "type": "upstream_error"}}, 502)
        status = resp.status
        resp_headers = resp.getheaders()
        if status != 200:
            data = resp.read()
            rtc.upstream.release(conn, resp)
            engine.log_event("error", prep, detail="upstream-%d" % status,
                             latency_ms=(time.perf_counter() - t0) * 1000)
            return self._send(status, data, self._ctype(resp_headers),
                              {"X-Semcache": "error"})

        if prep.stream:
            return self._relay_stream(prep, conn, resp, store_ok, t0, near)

        data = resp.read()
        rtc.upstream.release(conn, resp)
        latency = (time.perf_counter() - t0) * 1000
        pt = ct = 0
        obj = None
        try:
            obj = json.loads(data)
            usage = (obj or {}).get("usage") or {}
            pt = int(usage.get("prompt_tokens") or 0)
            ct = int(usage.get("completion_tokens") or 0)
        except Exception:
            obj = None
        if prep.ok:
            near_score = near.near_score if near else 0.0
            engine.log_event("miss", prep, score=near_score,
                             detail=_miss_detail(near),
                             pt=pt, ct=ct, latency_ms=latency,
                             embed_cost=(near.embed_cost if near else 0.0))
            engine.tick(prep, hit=False, score=near_score)
            if store_ok and obj is not None:
                engine.store_answer(prep, obj)
        self._send(200, data, self._ctype(resp_headers),
                   {"X-Semcache": "miss" if prep.ok else "bypass",
                    "X-Semcache-Near": "%.3f" % (near.near_score if near else 0.0)})

    def _relay_stream(self, prep: Prepared, conn: http.client.HTTPConnection,
                      resp: http.client.HTTPResponse, store_ok: bool, t0: float,
                      near: Optional[Lookup] = None) -> None:
        rtc = self.rt
        engine = rtc.engine
        self.send_response(resp.status)
        for k, v in resp.getheaders():
            if k.lower() in HOP_HEADERS or k.lower() == "content-length":
                continue
            self.send_header(k, v)
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("X-Semcache", "miss")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        buf = bytearray()
        cap = 8 * 1024 * 1024
        aborted = False
        try:
            while True:
                chunk = resp.read(2048)
                if not chunk:
                    break
                if len(buf) < cap:
                    buf += chunk
                try:
                    self._write_chunk(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    aborted = True
                    break
            if not aborted:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
        except Exception as e:
            LOG.debug("relay 异常: %s", e)
            aborted = True
        finally:
            if aborted:
                try:
                    conn.close()
                except Exception:
                    pass
            else:
                rtc.upstream.release(conn, resp)

        latency = (time.perf_counter() - t0) * 1000
        if prep.ok:
            obj = None
            try:
                obj = assemble_sse(bytes(buf), prep.model)
            except Exception:
                obj = None
            pt = ct = 0
            if obj:
                u = obj.get("usage") or {}
                pt, ct = int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)
            near_score = near.near_score if near else 0.0
            engine.log_event("miss", prep, score=near_score,
                             detail="stream/" + _miss_detail(near),
                             pt=pt, ct=ct, latency_ms=latency,
                             embed_cost=(near.embed_cost if near else 0.0))
            engine.tick(prep, hit=False, score=near_score)
            if store_ok and obj is not None:
                engine.store_answer(prep, obj)

    def _write_chunk(self, data: bytes) -> None:
        self.wfile.write(b"%x\r\n" % len(data))
        self.wfile.write(data)
        self.wfile.write(b"\r\n")
        self.wfile.flush()

    # -- 透传 -------------------------------------------------------------
    def _upstream_headers(self) -> Dict[str, str]:
        cfg = self.rt.cfg
        out: Dict[str, str] = {}
        for k, v in self.headers.items():
            if k.lower() in HOP_HEADERS:
                continue
            out[k] = v
        out["Accept-Encoding"] = "identity"
        if cfg.get("upstream_key"):
            out["Authorization"] = "Bearer " + str(cfg["upstream_key"])
        return out

    @staticmethod
    def _ctype(headers: Sequence[Tuple[str, str]]) -> str:
        for k, v in headers:
            if k.lower() == "content-type":
                return v
        return "application/json"

    def _passthrough(self, path: str, body: Optional[bytes] = None) -> None:
        rtc = self.rt
        if body is None:
            body = self._read_body() if self.command in ("POST", "PUT", "PATCH") else None
        headers = self._upstream_headers()
        try:
            conn, resp = rtc.upstream.open(self.command, path, headers, body)
        except Exception as e:
            return self._send_json({"error": {"message": "semcache upstream error: %s" % e}}, 502)
        status = resp.status
        resp_headers = resp.getheaders()
        if self.command == "HEAD":
            rtc.upstream.release(conn, resp)
            self.send_response(status)
            for k, v in resp_headers:
                if k.lower() not in HOP_HEADERS:
                    self.send_header(k, v)
            self.send_header("Content-Length", resp.getheader("Content-Length") or "0")
            self.end_headers()
            return
        data = resp.read()
        rtc.upstream.release(conn, resp)
        self._send(status, data, self._ctype(resp_headers), {"X-Semcache": "passthrough"})

    # -- 管理接口 ---------------------------------------------------------
    def _admin_ok(self, query: str) -> bool:
        token = str(self.rt.cfg.get("admin_token") or "")
        if not token:
            return True
        qs = parse_qs(query or "")
        if qs.get("token", [""])[0] == token:
            return True
        auth = self.headers.get("Authorization") or ""
        return auth.split()[-1] == token if auth.strip() else False

    def _admin(self, path: str, query: str) -> None:
        if not self._admin_ok(query):
            return self._send_json({"error": "unauthorized"}, 401)
        rtc = self.rt
        qs = parse_qs(query or "")
        if path == "/admin/ping":
            return self._send_json({"ok": True, "version": VERSION})
        if path == "/admin/stats":
            return self._send_json(self._stats_payload())
        if path == "/admin/calib":
            rows = rtc.store.calib_all(qs.get("project", [None])[0])
            return self._send_json({"calibrations": [dict(r) for r in rows],
                                    "embedder": rtc.engine.embedder.ident})
        if path == "/admin/report":
            fmt = qs.get("format", ["json"])[0]
            month = qs.get("month", [None])[0] or None
            project = qs.get("project", [None])[0] or None
            rep = build_report(rtc.store, rtc.cfg, month=month, project=project)
            if fmt == "md":
                return self._send(200, rep["markdown"].encode("utf-8"), "text/markdown; charset=utf-8")
            if fmt == "text":
                return self._send(200, rep["text"].encode("utf-8"), "text/plain; charset=utf-8")
            return self._send_json(rep["json"])
        if path == "/admin/calibrate" and self.command == "POST":
            body = self._read_body()
            try:
                req = json.loads(body) if body else {}
            except Exception:
                req = {}
            targets = []
            if req.get("project") and req.get("ns"):
                targets = [(req["project"], req["ns"])]
            else:
                targets = [(r["project"], r["ns"]) for r in rtc.store.all_namespaces()]
            out = []
            for p, n in targets:
                out.append(rtc.engine.recalibrate(p, n, force=True))
            return self._send_json({"recalibrated": len(out), "results": out})
        if path == "/admin/feedback" and self.command == "POST":
            body = self._read_body()
            try:
                req = json.loads(body)
            except Exception:
                return self._send_json({"error": "bad json"}, 400)
            label = str(req.get("label") or "bad")
            project = str(req.get("project") or "default")
            ns = str(req.get("ns") or "")
            if not ns and req.get("entry_id"):
                row = rtc.store.get_entry_by_id(int(req["entry_id"]))
                if row:
                    project, ns = row["project"], row["ns"]
            if not ns:
                return self._send_json({"error": "need ns or entry_id"}, 400)
            d = rtc.engine.feedback(project, ns, label, score=float(req.get("score") or 0.0),
                                    detail="api")
            return self._send_json({"ok": True, "calib": d})
        if path == "/admin/clear" and self.command == "POST":
            body = self._read_body()
            try:
                req = json.loads(body) if body else {}
            except Exception:
                req = {}
            n = 0
            for r in rtc.store.all_namespaces():
                if req.get("project") and r["project"] != req["project"]:
                    continue
                if req.get("ns") and r["ns"] != req["ns"]:
                    continue
                rtc.store._conn().execute(
                    "DELETE FROM entries WHERE project=? AND ns=?", (r["project"], r["ns"])
                )
                rtc.store._conn().commit()
                rtc.engine._invalidate(r["project"], r["ns"])
                n += 1
            return self._send_json({"ok": True, "namespaces_cleared": n})
        return self._send_json({"error": "not found", "path": path}, 404)

    def _stats_payload(self) -> Dict[str, Any]:
        rtc = self.rt
        month = None
        rep = build_report(rtc.store, rtc.cfg, month=month)
        return {
            "now": now(),
            "month": month,
            "version": VERSION,
            "uptime_s": round(now() - rtc.started_at, 1),
            "dry_run": bool(rtc.cfg.get("dry_run")),
            "embedder": rtc.engine.embedder.ident,
            "upstream": rtc.cfg.get("upstream"),
            "db": rtc.store.path,
            "corpus": rtc.store.corpus_stats(),
            "runtime": rtc.engine.stats,
            "report": rep["json"],
        }

    def _dashboard(self) -> None:
        if not self._admin_ok(""):
            return self._send(401, b"unauthorized", "text/plain; charset=utf-8")
        html = dashboard_html(self.rt.cfg)
        self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")


class ProxyServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr: Tuple[str, int], handler: Any, runtime: Runtime):
        self.runtime = runtime
        super().__init__(addr, handler)


# --------------------------------------------------------------------------- #
# 报告
# --------------------------------------------------------------------------- #

SPARK = "▁▂▃▄▅▆▇█"


def sparkline(values: Sequence[float], width: int = 40) -> str:
    if not values:
        return ""
    vals = list(values)
    if len(vals) > width:
        step = len(vals) / float(width)
        vals = [max(vals[int(i * step):max(int(i * step) + 1, int((i + 1) * step))]) for i in range(width)]
    hi = max(vals) or 1.0
    return "".join(SPARK[min(7, int(v / hi * 7.999))] for v in vals)


def render_table(headers: Sequence[str], rows: Sequence[Sequence[Any]],
                 aligns: Optional[Sequence[str]] = None, indent: str = " ") -> List[str]:
    aligns = aligns or ["left"] * len(headers)
    cols = [display_width(str(h)) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            cols[i] = max(cols[i], display_width(str(c)))
    out = [indent + "  ".join(pad(str(h), cols[i], aligns[i]) for i, h in enumerate(headers))]
    out.append(indent + "  ".join("─" * cols[i] for i in range(len(headers))))
    for r in rows:
        out.append(indent + "  ".join(pad(str(c), cols[i], aligns[i]) for i, c in enumerate(r)))
    return out


def build_report(store: Store, cfg: Dict[str, Any], month: Optional[str] = None,
                 project: Optional[str] = None) -> Dict[str, Any]:
    rate = float(cfg.get("usd_cny") or 7.2)
    rows = store.report_rows(month, project)
    daily = store.daily_rows(month, project)
    top = store.top_prompts(month, project, 10)
    near = store.near_misses(month, project)
    kinds = store.hit_kinds(month, project)

    exact_map: Dict[str, int] = {}
    sem_map: Dict[str, int] = {}
    for k in kinds:
        if k["detail"] == "exact":
            exact_map[k["project"]] = int(k["n"])
        else:
            sem_map[k["project"]] = sem_map.get(k["project"], 0) + int(k["n"])

    projects: List[Dict[str, Any]] = []
    for r in rows:
        calls = int(r["hits"] or 0) + int(r["misses"] or 0) + int(r["shadows"] or 0)
        cacheable = calls
        saved = float(r["saved_usd"] or 0)
        extra = float(r["extra_usd"] or 0)
        proj = {
            "project": r["project"],
            "calls": calls,
            "cacheable": cacheable,
            "hits": int(r["hits"] or 0),
            "exact_hits": exact_map.get(r["project"], 0),
            "semantic_hits": sem_map.get(r["project"], 0),
            "misses": int(r["misses"] or 0),
            "shadows": int(r["shadows"] or 0),
            "bypasses": int(r["bypasses"] or 0),
            "errors": int(r["errors"] or 0),
            "hit_rate": (int(r["hits"] or 0) / calls) if calls else 0.0,
            "saved_tokens_in": int(r["saved_in"] or 0),
            "saved_tokens_out": int(r["saved_out"] or 0),
            "saved_tokens": int(r["saved_tokens"] or 0),
            "saved_usd": round(saved, 6),
            "saved_cny": round(saved * rate, 4),
            "embed_cost_usd": round(extra, 8),
            "net_usd": round(saved - extra, 6),
            "net_cny": round((saved - extra) * rate, 4),
            "shadow_usd": round(float(r["shadow_usd"] or 0), 6),
            "avg_hit_latency_ms": round((float(r["hit_lat"] or 0) / int(r["hits"])) if r["hits"] else 0, 1),
            "avg_miss_latency_ms": round((float(r["miss_lat"] or 0) / int(r["misses"])) if r["misses"] else 0, 1),
            "first_day": r["first_day"],
            "last_day": r["last_day"],
        }
        projects.append(proj)

    t = {
        "calls": sum(p["calls"] for p in projects),
        "hits": sum(p["hits"] for p in projects),
        "exact_hits": sum(p["exact_hits"] for p in projects),
        "semantic_hits": sum(p["semantic_hits"] for p in projects),
        "misses": sum(p["misses"] for p in projects),
        "shadows": sum(p["shadows"] for p in projects),
        "bypasses": sum(p["bypasses"] for p in projects),
        "errors": sum(p["errors"] for p in projects),
        "saved_tokens_in": sum(p["saved_tokens_in"] for p in projects),
        "saved_tokens_out": sum(p["saved_tokens_out"] for p in projects),
        "saved_tokens": sum(p["saved_tokens"] for p in projects),
        "saved_usd": sum(p["saved_usd"] for p in projects),
        "embed_cost_usd": sum(p["embed_cost_usd"] for p in projects),
        "shadow_usd": sum(p["shadow_usd"] for p in projects),
    }
    t["net_usd"] = t["saved_usd"] - t["embed_cost_usd"]
    t["saved_cny"] = t["saved_usd"] * rate
    t["net_cny"] = t["net_usd"] * rate
    t["hit_rate"] = (t["hits"] / t["calls"]) if t["calls"] else 0.0
    avg_hit_usd = (t["saved_usd"] / t["hits"]) if t["hits"] else 0.0
    miss_detail = {r["detail"]: int(r["n"]) for r in store.miss_details(month, project)}
    near_n = int(near["n"] or 0) if near else 0
    t["potential_from_near_usd"] = round(near_n * avg_hit_usd, 6)
    t["scale_per_10k_usd"] = round(t["net_usd"] / t["calls"] * 10000, 4) if t["calls"] else 0.0

    payload = {
        "month": month,
        "generated_at": now(),
        "project_filter": project,
        "currency": {"usd_cny": rate, "symbol": cfg.get("currency", "CNY")},
        "totals": t,
        "projects": projects,
        "daily": [
            {"day": d["day"], "calls": int(d["calls"] or 0), "hits": int(d["hits"] or 0),
             "hit_rate": (int(d["hits"] or 0) / int(d["calls"])) if d["calls"] else 0.0,
             "usd": round(float(d["usd"] or 0), 6)}
            for d in daily
        ],
        "top_prompts": [
            {"prompt": trunc(p["prompt"] or "", 80), "model": p["model"],
             "count": int(p["n"]), "usd": round(float(p["usd"] or 0), 6),
             "best_score": round(float(p["best"] or 0), 4)}
            for p in top
        ],
        "miss_breakdown": miss_detail,
        "near_misses": {
            "count": near_n,
            "avg_score": round(float(near["avg_score"] or 0), 4) if near else 0.0,
            "max_score": round(float(near["max_score"] or 0), 4) if near else 0.0,
            "potential_usd": t["potential_from_near_usd"],
        },
        "calibration": [
            {"project": c["project"], "ns": c["ns"][:8], "eff": round(float(c["eff"]), 4),
             "floor": round(float(c["floor"]), 4), "bias": round(float(c["bias"]), 4),
             "mu": round(float(c["mu"]), 4), "sigma": round(float(c["sigma"]), 4),
             "lookups": int(c["lookups"]), "hits": int(c["hits"]), "bad": int(c["bad"]),
             "status": c["status"]}
            for c in store.calib_all(project)
        ],
    }
    payload["headline"] = _headline(payload, rate, bool(cfg.get("dry_run")))
    payload["text"] = _report_text(payload, store, cfg)
    payload["markdown"] = _report_md(payload)
    return {"json": payload, "text": payload["text"], "markdown": payload["markdown"]}


def _headline(p: Dict[str, Any], rate: float, dry_run: bool) -> str:
    t = p["totals"]
    if t["calls"] == 0:
        return "还没有数据：把客户端的 base_url 指到本代理，跑一天再回来看这张账单。"
    pct = t["hit_rate"] * 100
    base = ("本期 %d 次调用中，%d 次是重复提问（%.0f%%），已由缓存直接返回，省下 %s（%s）。"
            % (t["calls"], t["hits"] + t["shadows"], pct,
               money(t["saved_usd"] + t["shadow_usd"]),
               cny(t["saved_usd"] + t["shadow_usd"], rate)))
    if dry_run or (t["shadows"] and not t["hits"]):
        base = ("【影子模式】本期 %d 次调用中，%d 次可命中缓存（%.0f%%），"
                "上线后预计省 %s（%s）。当前未生效。"
                % (t["calls"], t["shadows"], pct, money(t["shadow_usd"]), cny(t["shadow_usd"], rate)))
    return base


def _report_text(p: Dict[str, Any], store: Store, cfg: Dict[str, Any]) -> str:
    rate = float(p["currency"]["usd_cny"])
    t = p["totals"]
    title = "语义缓存 · 重复率报告"
    month = p["month"] or "全部时间"
    W = 74
    L: List[str] = []
    L.append("═" * W)
    L.append(" " + pad(title, W - 2 - display_width(month)) + month + " ")
    L.append("═" * W)
    if not p["projects"]:
        L.append(" 还没有任何调用记录。")
        L.append(" 把客户端 base_url 改成 http://127.0.0.1:%s/v1 后重新跑一遍。" % cfg.get("port"))
        L.append("═" * W)
        return "\n".join(L)

    hit_rate = t["hit_rate"] * 100
    net = t["net_cny"]
    L.append(" 调用总数      %s 次" % pad(human_int(t["calls"]), 12, "right"))
    L.append(" 命中缓存      %s 次（精确 %s / 语义 %s）"
             % (pad(human_int(t["hits"]), 6, "right"), human_int(t["exact_hits"]), human_int(t["semantic_hits"])))
    L.append(" 重复率        %s" % pad("%.1f%%" % hit_rate, 12, "right"))
    if t["shadows"]:
        L.append(" 影子命中      %s 次（只记账未生效）" % pad(human_int(t["shadows"]), 6, "right"))
    L.append(" 省下 token    %s（输入 %s / 输出 %s）"
             % (pad(human_int(t["saved_tokens"]), 10, "right"),
                human_int(t["saved_tokens_in"]), human_int(t["saved_tokens_out"])))
    L.append(" 省下费用      %s = %s" % (pad(money(t["saved_usd"]), 10, "right"),
                                       cny(t["saved_usd"], rate)))
    if t["embed_cost_usd"] > 0:
        L.append(" 净收益        %s（已扣除 embedding 成本 %s）"
                 % (pad("¥%.2f" % net, 12, "right"), money(t["embed_cost_usd"])))
    L.append(" 平均延迟      命中 %.1fms ／ 未命中 %.1fms"
             % (_wavg(p["projects"], "avg_hit_latency_ms", "hits"),
                _wavg(p["projects"], "avg_miss_latency_ms", "misses")))
    L.append("─" * W)
    rows = []
    for pr in p["projects"]:
        rows.append([
            pr["project"],
            human_int(pr["calls"]),
            human_int(pr["hits"]),
            "%.1f%%" % (pr["hit_rate"] * 100),
            human_int(pr["saved_tokens"]),
            money_cny(pr["saved_cny"]),
        ])
    L.extend(render_table(
        ["项目", "调用", "命中", "命中率", "省下token", "省下"],
        rows, ["left", "right", "right", "right", "right", "right"]))
    L.append("─" * W)
    wrap_width = W - 2
    L.extend(_wrap(p["headline"], wrap_width, " 结论：", "       "))
    L.append("─" * W)

    if p["top_prompts"]:
        L.append(" 被问最多的重复问题")
        for i, tp in enumerate(p["top_prompts"][:8], 1):
            left = "  %2d. %s" % (i, trunc(tp["prompt"], 38))
            right = "×%d  %s" % (tp["count"], money_cny(tp["usd"] * rate))
            L.append(pad(left, W - display_width(right) - 1) + right)
        L.append("─" * W)

    nm = p["near_misses"]
    if nm["count"] > 0:
        L.extend(_wrap(
            "另有 %d 次提问相似度接近阈值但未命中（最高 %.2f）。放宽阈值预计每期多省 %s，"
            "建议先用 --dry-run 观察两天。"
            % (nm["count"], nm["max_score"], money_cny(nm["potential_usd"] * rate)),
            wrap_width, " 建议：", "      "))
    elif t["hits"] == 0:
        L.extend(_wrap(
            "还没有命中。检查：① 客户端真的走了本代理？② 阈值 %s 是否过高？"
            "③ --embedder local 只认近似重复，跨语义改写建议换 openai embedder。"
            % _threshold_hint(p), wrap_width, " 提示：", "      "))

    if t["calls"] >= 200:
        per10k = t["net_usd"] / t["calls"] * 10000
        L.append(" 规模外推      按当前重复率，每 1 万次调用约省 %s（%s）"
                 % (cny(per10k, rate), money(per10k)))
    gated = int((p.get("miss_breakdown") or {}).get("gate", 0))
    if gated:
        L.append(" 安全拦截      %s 次：提问相似，但单号/金额/ID 不同，未命中"
                 % human_int(gated))
    elif t["hits"]:
        L.append(" 安全拦截      0 次：没有出现「单号不同却命中」")
    if t["hits"] and str(cfg.get("embedder") or "local") == "local":
        sem_ratio = t["semantic_hits"] / float(t["hits"])
        if sem_ratio < 0.10:
            L.append(" 提示          命中里只有 %.0f%% 是语义命中——local 向量只能抓近似重复，"
                     "要抓改写请加 --embedder openai" % (sem_ratio * 100))
    if p["daily"]:
        L.append(" 每日命中 " + sparkline([d["hit_rate"] for d in p["daily"]]))
    cal = p["calibration"]
    if cal:
        effs = [c["eff"] for c in cal]
        floors = [c["floor"] for c in cal]
        bad = sum(c["bad"] for c in cal)
        L.append(" 阈值自动校准  %d 个命名空间 ／ 阈值 %.3f~%.3f ／ 安全下界 %.3f~%.3f%s"
                 % (len(cal), min(effs), max(effs), min(floors), max(floors),
                    (" ／ 差评修正 %d 次" % bad) if bad else ""))
    L.append("═" * W)
    L.append(" 数据库 %s" % display_path(store.path))
    L.append(" 生成于 %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    return "\n".join(L)


def _threshold_hint(p: Dict[str, Any]) -> str:
    cal = p.get("calibration") or []
    return "%.2f" % cal[0]["eff"] if cal else "0.72"


def _wavg(projects: Sequence[Dict[str, Any]], field: str, weight: str) -> float:
    tot = sum(float(p.get(weight) or 0) for p in projects)
    if not tot:
        return 0.0
    return sum(float(p.get(field) or 0) * float(p.get(weight) or 0) for p in projects) / tot


def _wrap(text: str, width: int, first_prefix: str, prefix: str) -> List[str]:
    out: List[str] = []
    lines = []
    cur = ""
    for ch in text:
        cur += ch
        if display_width(cur) >= width - display_width(first_prefix) and ch in "，。；：、 ":
            lines.append(cur)
            cur = ""
    if cur:
        lines.append(cur)
    for i, ln in enumerate(lines):
        out.append((first_prefix if i == 0 else prefix) + ln)
    return out


def _report_md(p: Dict[str, Any]) -> str:
    rate = float(p["currency"]["usd_cny"])
    t = p["totals"]
    month = p["month"] or "全部时间"
    L = ["# 语义缓存重复率报告（%s）" % month, ""]
    L.append("> %s" % p["headline"])
    L.append("")
    L.append("| 指标 | 数值 |")
    L.append("|---|---|")
    L.append("| 调用总数 | %d |" % t["calls"])
    L.append("| 命中缓存 | %d（精确 %d / 语义 %d） |" % (t["hits"], t["exact_hits"], t["semantic_hits"]))
    L.append("| 重复率 | %.1f%% |" % (t["hit_rate"] * 100))
    L.append("| 省下 token | %s |" % human_int(t["saved_tokens"]))
    L.append("| 省下费用 | %s / %s |" % (money(t["saved_usd"]), money_cny(t["saved_cny"])))
    if t["embed_cost_usd"] > 0:
        L.append("| 净收益 | %s（已扣 embedding %s） |"
                 % (money_cny(t["net_cny"]), money(t["embed_cost_usd"])))
    L.append("")
    if p["projects"]:
        L.append("## 分项目")
        L.append("")
        L.append("| 项目 | 调用 | 命中 | 命中率 | 省下 token | 省下 |")
        L.append("|---|---:|---:|---:|---:|---:|")
        for pr in p["projects"]:
            L.append("| %s | %d | %d | %.1f%% | %s | %s |"
                     % (pr["project"], pr["calls"], pr["hits"], pr["hit_rate"] * 100,
                        human_int(pr["saved_tokens"]), money_cny(pr["saved_cny"])))
        L.append("")
    if p["top_prompts"]:
        L.append("## 被问最多的重复问题")
        L.append("")
        L.append("| # | 问题 | 次数 | 省下 |")
        L.append("|---:|---|---:|---:|")
        for i, tp in enumerate(p["top_prompts"][:10], 1):
            L.append("| %d | %s | %d | %s |" % (i, tp["prompt"].replace("|", "\\|"),
                                               tp["count"], money_cny(tp["usd"] * rate)))
        L.append("")
    if t["calls"] >= 200:
        L.append("- 规模外推：按当前重复率，每 1 万次调用约省 %s"
                 % money_cny(t["scale_per_10k_usd"] * rate))
    gated = int((p.get("miss_breakdown") or {}).get("gate", 0))
    if gated:
        L.append("- 安全拦截：%d 次提问相似但单号/金额/ID 不同，未命中" % gated)
        L.append("")
    nm = p["near_misses"]
    if nm["count"]:
        L.append("## 差一点命中的")
        L.append("")
        L.append("- 近失 %d 次，最高相似度 %.2f，放宽阈值预计再省 ¥%.2f/期"
                 % (nm["count"], nm["max_score"], nm["potential_usd"] * rate))
        L.append("")
    if p["calibration"]:
        L.append("## 阈值自动校准")
        L.append("")
        L.append("| 项目 | 命名空间 | 阈值 | 安全下界 | 背景 μ | 背景 σ | 状态 |")
        L.append("|---|---|---:|---:|---:|---:|---|")
        for c in p["calibration"][:12]:
            L.append("| %s | `%s` | %.3f | %.3f | %.3f | %.3f | %s |"
                     % (c["project"], c["ns"], c["eff"], c["floor"], c["mu"], c["sigma"], c["status"]))
        L.append("")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# 仪表盘（内嵌单页，无外部依赖）
# --------------------------------------------------------------------------- #

DASHBOARD_HTML = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>semcache · 语义缓存看板</title>
<style>
:root{--bg:#0d1117;--panel:#161b22;--line:#242c37;--fg:#e6edf3;--dim:#8b949e;
      --acc:#3fb950;--acc2:#58a6ff;--warn:#d29922;--bad:#f85149}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
     font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif}
header{padding:22px 28px;border-bottom:1px solid var(--line);display:flex;
       align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px}
h1{font-size:17px;margin:0;font-weight:600;letter-spacing:.3px}
h1 span{color:var(--acc)}
.tag{font-size:12px;color:var(--dim);border:1px solid var(--line);border-radius:99px;
     padding:2px 10px;margin-left:8px}
main{padding:24px 28px;max-width:1180px;margin:0 auto}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px;margin-bottom:22px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px 18px}
.card .k{color:var(--dim);font-size:12px;letter-spacing:.4px}
.card .v{font-size:26px;font-weight:650;margin-top:6px;font-variant-numeric:tabular-nums}
.card .s{color:var(--dim);font-size:12px;margin-top:4px}
.card.hero .v{color:var(--acc)}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;
       padding:18px;margin-bottom:18px;overflow:auto}
.panel h2{font-size:13px;margin:0 0 14px;color:var(--dim);font-weight:600;
          letter-spacing:.6px;text-transform:uppercase}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:9px 10px;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--dim);font-weight:500;font-size:12px}
td.n,th.n{text-align:right}
tr:last-child td{border-bottom:0}
.bar{height:6px;border-radius:3px;background:var(--acc);min-width:2px;display:inline-block;
     vertical-align:middle;margin-right:8px}
.bars{display:flex;align-items:flex-end;gap:3px;height:90px}
.bars i{flex:1;background:linear-gradient(180deg,var(--acc2),#1f6feb);border-radius:2px 2px 0 0;
        min-height:2px;opacity:.85}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}
.dim{color:var(--dim)}
.pill{display:inline-block;padding:1px 8px;border-radius:99px;font-size:11px;
      border:1px solid var(--line);color:var(--dim)}
.pill.ok{color:var(--acc);border-color:#1f4325;background:#0f2413}
.pill.boot{color:var(--warn);border-color:#3d2f10;background:#221a06}
#msg{color:var(--bad);font-size:13px;padding:8px 0}
footer{color:var(--dim);font-size:12px;padding:6px 28px 40px;max-width:1180px;margin:0 auto}
</style></head><body>
<header>
  <h1>semcache <span>语义缓存看板</span><span class="tag" id="mode">…</span></h1>
  <div class="dim" id="where"></div>
</header>
<main>
  <div id="msg"></div>
  <div class="cards" id="cards"></div>
  <div class="panel"><h2>结论</h2><div id="headline" style="font-size:15px"></div></div>
  <div class="panel"><h2>分项目</h2><table id="proj"></table></div>
  <div class="panel"><h2>每日调用量（绿=命中 / 蓝=未命中）</h2><div class="bars" id="daily"></div>
       <div class="dim" id="dailycap" style="margin-top:8px;font-size:12px"></div></div>
  <div class="panel"><h2>被问最多的重复问题</h2><table id="top"></table></div>
  <div class="panel"><h2>阈值自动校准</h2><table id="calib"></table></div>
</main>
<footer id="foot"></footer>
<script>
const qs=new URLSearchParams(location.search);
const token=qs.get('token')||localStorage.getItem('semcache_token')||'';
function url(p){return p+(token?('?token='+encodeURIComponent(token)):'');}
const nf=new Intl.NumberFormat('zh-CN');
function h(n){n=+n||0;if(n>=1e6)return (n/1e6).toFixed(2)+'M';if(n>=1e4)return (n/1e3).toFixed(1)+'k';return nf.format(n);}
function y(n){return '¥'+(+n||0).toFixed(2);}
function pct(x){return ((x||0)*100).toFixed(1)+'%';}
function el(t,a,txt){const e=document.createElement(t);if(a)for(const k in a)e.setAttribute(k,a[k]);
  if(txt!==undefined)e.textContent=txt;return e;}
async function tick(){
  try{
    const r=await fetch(url('/admin/stats'));
    if(!r.ok){document.getElementById('msg').textContent='需要口令：在地址后加 ?token=xxx';return;}
    const s=await r.json(), rep=s.report, t=rep.totals;
    document.getElementById('mode').textContent =
      (s.dry_run?'影子模式':'在线') + ' · ' + (rep.month||'全部时间');
    document.getElementById('where').textContent = s.upstream + ' · ' + s.embedder + ' · 条目 ' + h(s.corpus.entries);
    const cards=[
      ['重复率（命中）',pct(t.hit_rate),h(t.hits)+' / '+h(t.calls)+' 次调用',true],
      ['已省费用',y(t.saved_cny),'$'+(t.saved_usd||0).toFixed(2),false],
      ['省下 token',h(t.saved_tokens),'输入 '+h(t.saved_tokens_in)+' / 输出 '+h(t.saved_tokens_out),false],
      ['规模外推',y(t.scale_per_10k_usd*rep.currency.usd_cny)+'/万次','按当前重复率折算',false],
      ['命中延迟','—','平均 '+(rep.totals.hits?0:0),false]
    ];
    const cc=document.getElementById('cards');cc.innerHTML='';
    for(const [k,v,sub,hero] of cards){
      const c=el('div',{class:'card'+(hero?' hero':'')});
      c.appendChild(el('div',{class:'k'},k));c.appendChild(el('div',{class:'v'},v));
      c.appendChild(el('div',{class:'s'},sub));cc.appendChild(c);
    }
    // 用分项目加权出延迟
    let sl=0,sw=0;
    rep.projects.forEach(p=>{sl+=(p.avg_hit_latency_ms||0)*p.hits;sw+=p.hits;});
    cc.children[4].querySelector('.v').textContent = sw? (sl/sw).toFixed(1)+'ms' : '—';
    cc.children[4].querySelector('.s').textContent = '对比未命中 ' +
      (rep.projects.length? (rep.projects.reduce((a,p)=>a+(p.avg_miss_latency_ms||0)*p.misses,0)/
        (rep.projects.reduce((a,p)=>a+p.misses,0)||1)).toFixed(1)+'ms' : '—');

    document.getElementById('headline').textContent = rep.headline;
    const pt=document.getElementById('proj');pt.innerHTML='';
    let hd=el('tr');
    ['项目','调用','命中','命中率','省下 token','省下'].forEach((x,i)=>
      hd.appendChild(el('th',{class:i?'n':''},x)));
    pt.appendChild(hd);
    rep.projects.forEach(p=>{
      const tr=el('tr');
      tr.appendChild(el('td',null,p.project));
      tr.appendChild(el('td',{class:'n'},h(p.calls)));
      tr.appendChild(el('td',{class:'n'},h(p.hits)));
      const td=el('td',{class:'n'});
      td.appendChild(el('span',{class:'bar',style:'width:'+Math.round(p.hit_rate*70)+'px'}));
      td.appendChild(document.createTextNode(pct(p.hit_rate)));
      tr.appendChild(td);
      tr.appendChild(el('td',{class:'n'},h(p.saved_tokens)));
      tr.appendChild(el('td',{class:'n'},y(p.saved_cny)));
      pt.appendChild(tr);
    });
    const dl=document.getElementById('daily');dl.innerHTML='';
    const mx=Math.max(1,...rep.daily.map(d=>d.calls));
    rep.daily.slice(-60).forEach(d=>{
      const hp = d.calls? (d.hits/d.calls*100):0;
      dl.appendChild(el('i',{style:'height:'+Math.max(4,Math.round(d.calls/mx*100))+'%;'+
        'background:linear-gradient(to top,var(--acc) '+hp.toFixed(1)+'%,#1f6feb '+hp.toFixed(1)+'%)',
        title:d.day+'  调用 '+d.calls+'  命中率 '+pct(d.hit_rate)}));
    });
    document.getElementById('dailycap').textContent = rep.daily.length?
      ('绿=命中 蓝=未命中　'+rep.daily[0].day+' → '+rep.daily[rep.daily.length-1].day):'暂无数据';

    const tt=document.getElementById('top');tt.innerHTML='';
    let th=el('tr');['#','问题','次数','省下'].forEach((x,i)=>th.appendChild(el('th',{class:i>1?'n':''},x)));
    tt.appendChild(th);
    rep.top_prompts.forEach((p,i)=>{
      const tr=el('tr');tr.appendChild(el('td',{class:'n dim'},i+1));
      tr.appendChild(el('td',{class:'mono'},p.prompt));
      tr.appendChild(el('td',{class:'n'},nf.format(p.count)));
      tr.appendChild(el('td',{class:'n'},y(p.usd*rep.currency.usd_cny)));
      tt.appendChild(tr);
    });

    const ct=document.getElementById('calib');ct.innerHTML='';
    let ch=el('tr');['项目','命名空间','阈值','下界','背景 μ/σ','状态'].forEach((x,i)=>
      ch.appendChild(el('th',{class:i>1?'n':''},x)));
    ct.appendChild(ch);
    rep.calibration.slice(0,12).forEach(c=>{
      const tr=el('tr');
      tr.appendChild(el('td',null,c.project));
      tr.appendChild(el('td',{class:'mono dim'},c.ns));
      tr.appendChild(el('td',{class:'n'},c.eff.toFixed(3)));
      tr.appendChild(el('td',{class:'n dim'},c.floor.toFixed(3)));
      tr.appendChild(el('td',{class:'n dim'},c.mu.toFixed(3)+' / '+c.sigma.toFixed(3)));
      const td=el('td');
      td.appendChild(el('span',{class:'pill '+(c.status==='auto'?'ok':'boot')},c.status));
      tr.appendChild(td);ct.appendChild(tr);
    });
    document.getElementById('foot').textContent='semcache '+s.version+' · '+
      new Date(s.now*1000).toLocaleString('zh-CN')+' · 缓存条目 '+h(s.corpus.entries)+
      ' · 事件 '+h(s.corpus.events);
  }catch(e){document.getElementById('msg').textContent='加载失败: '+e;}
}
tick();setInterval(tick,5000);
</script></body></html>
"""


def dashboard_html(cfg: Dict[str, Any]) -> str:
    return DASHBOARD_HTML


# --------------------------------------------------------------------------- #
# 配置装配 / 启动
# --------------------------------------------------------------------------- #

def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, str(level).upper(), logging.INFO),
        format="%(asctime)s %(levelname).1s %(message)s",
        datefmt="%H:%M:%S",
    )


TRUEISH = ("1", "true", "yes", "on", "y", "t")


def _coerce(value: Any, ref: Any) -> Any:
    if isinstance(ref, bool):
        return str(value).strip().lower() in TRUEISH
    if isinstance(ref, int) and not isinstance(ref, bool):
        return int(float(value))
    if isinstance(ref, float):
        return float(value)
    return value


def load_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg: Dict[str, Any] = dict(DEFAULTS)
    path = getattr(args, "config", None)
    if path:
        with open(os.path.expanduser(path), "r", encoding="utf-8") as f:
            file_cfg = json.load(f)
        for k, v in file_cfg.items():
            if k in ("prices", "project_keys") and isinstance(v, dict):
                merged = dict(cfg.get(k) or {})
                merged.update(v)
                cfg[k] = merged
            elif k in DEFAULTS:
                cfg[k] = v
            else:
                LOG.debug("忽略未知配置项 %s", k)
    for k in DEFAULTS:
        env = os.environ.get("SEMCACHE_" + k.upper())
        if env is not None and env != "":
            try:
                cfg[k] = _coerce(env, DEFAULTS[k])
            except Exception:
                LOG.warning("环境变量 SEMCACHE_%s 无法解析: %r", k.upper(), env)
    for k in DEFAULTS:
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v
    prices = dict(cfg.get("prices") or {})
    for spec in (getattr(args, "price", None) or []):
        if "=" in spec:
            name, val = spec.split("=", 1)
            prices[name.strip()] = val.strip()
    cfg["prices"] = prices
    if isinstance(cfg.get("project_keys"), str):
        pk = {}
        for part in cfg["project_keys"].split(","):
            if "=" in part:
                a, b = part.split("=", 1)
                pk[a.strip()] = b.strip()
        cfg["project_keys"] = pk
    cfg["db"] = os.path.expanduser(str(cfg["db"]))
    return cfg


def make_embedder(cfg: Dict[str, Any]) -> Embedder:
    kind = str(cfg.get("embedder") or "local").lower()
    if kind in ("openai", "remote", "api"):
        key = str(cfg.get("embed_key") or os.environ.get("SEMCACHE_OPENAI_KEY")
                  or os.environ.get("OPENAI_API_KEY") or cfg.get("upstream_key") or "")
        base = str(cfg.get("embed_base") or cfg.get("upstream") or "https://api.openai.com")
        dims = int(cfg.get("dim") or 0)
        return OpenAIEmbedder(base, key, str(cfg.get("embed_model") or "text-embedding-3-small"),
                              dims=dims, timeout=30.0)
    return LocalEmbedder(int(cfg.get("local_dim") or 2048))


def build_runtime(cfg: Dict[str, Any]) -> Runtime:
    store = Store(cfg["db"])
    embedder = make_embedder(cfg)
    engine = CacheEngine(cfg, store, embedder)
    upstream = Upstream(str(cfg["upstream"]), float(cfg.get("request_timeout") or 300))
    return Runtime(cfg=cfg, store=store, engine=engine, upstream=upstream)


BANNER = """\
┌──────────────────────────────────────────────────────────────┐
│  semcache {ver}  ·  单文件语义缓存代理                       │
└──────────────────────────────────────────────────────────────┘
  监听      http://{host}:{port}/v1
  上游      {upstream}
  向量      {embedder}{embed_note}
  数据库    {db}
  模式      {mode}
  阈值      {thr}
  看板      http://{host}:{port}/dashboard
  报告      python3 semcache.py report --md

  接入：OpenAI(base_url="http://{host}:{port}/v1", api_key="sk-...")
"""


def cmd_serve(cfg: Dict[str, Any]) -> int:
    setup_logging(cfg.get("log_level", "INFO"))
    rt = build_runtime(cfg)
    try:
        srv = ProxyServer((str(cfg["host"]), int(cfg["port"])), Handler, rt)
    except OSError as e:
        print("端口 %s 不可用：%s" % (cfg["port"], e), file=sys.stderr)
        print("换一个：python3 semcache.py serve --port 8788", file=sys.stderr)
        return 2
    embedder = rt.engine.embedder
    embed_note = ""
    if isinstance(embedder, LocalEmbedder):
        embed_note = "（离线、零成本；跨语义改写请用 --embedder openai）"
    else:
        embed_note = "（%s）" % getattr(embedder, "model", "")
    thr = cfg.get("threshold")
    thr_txt = ("人工固定 %.3f" % float(thr)) if thr else "自动校准（初始 %.2f，安全下界 %.2f）" % (
        rt.engine.calib.bootstrap_base(), rt.engine.calib.prior_floor())
    print(BANNER.format(
        ver=VERSION, host=cfg["host"], port=cfg["port"], upstream=cfg["upstream"],
        embedder=embedder.ident, embed_note=embed_note, db=rt.store.path,
        mode="影子模式（只记账，不命中）" if cfg.get("dry_run") else "在线",
        thr=thr_txt,
    ), flush=True)
    if not cfg.get("admin_token"):
        print("  提示：管理接口未设口令（--admin-token），本机使用没问题，暴露公网前请设一个。\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n收到 Ctrl-C，正在退出…")
    finally:
        srv.shutdown()
        srv.server_close()
        rt.upstream.close()
        print("已退出。看报告：python3 semcache.py report")
    return 0


def cmd_report(cfg: Dict[str, Any], args: argparse.Namespace) -> int:
    store = Store(cfg["db"])
    month = getattr(args, "month", None) or None
    project = getattr(args, "project", None) or None
    rep = build_report(store, cfg, month=month, project=project)
    if getattr(args, "json", False):
        text = json.dumps(rep["json"], ensure_ascii=False, indent=2)
    elif getattr(args, "md", False):
        text = rep["markdown"]
    else:
        text = rep["text"]
    out = getattr(args, "out", None)
    if out:
        with open(os.path.expanduser(out), "w", encoding="utf-8") as f:
            f.write(text)
        print("已写入 %s" % out)
    else:
        print(text)
    return 0


def cmd_calibrate(cfg: Dict[str, Any], args: argparse.Namespace) -> int:
    setup_logging(cfg.get("log_level", "WARNING"))
    store = Store(cfg["db"])
    engine = CacheEngine(cfg, store, make_embedder(cfg))
    targets = [(r["project"], r["ns"]) for r in store.all_namespaces()]
    if getattr(args, "project", None):
        targets = [t for t in targets if t[0] == args.project]
    if getattr(args, "ns", None):
        targets = [t for t in targets if t[1].startswith(args.ns)]
    if not targets:
        print("没有可校准的命名空间（缓存还是空的）。")
        return 0
    rows = []
    for p, n in targets:
        d = engine.recalibrate(p, n, force=True)
        rows.append([p, n[:8], "%.3f" % d["eff"], "%.3f" % d["floor"], "%.3f" % d["mu"],
                     "%.3f" % d["sigma"], str(d["n_null"]), d["status"]])
    print("\n".join(render_table(
        ["项目", "命名空间", "阈值", "安全下界", "背景μ", "背景σ", "样本对", "状态"],
        rows, ["left", "left", "right", "right", "right", "right", "right", "left"])))
    print("\n共校准 %d 个命名空间。" % len(rows))
    return 0


def cmd_stats(cfg: Dict[str, Any]) -> int:
    store = Store(cfg["db"])
    rep = build_report(store, cfg, month=None)["json"]
    t = rep["totals"]
    rate = float(cfg.get("usd_cny") or 7.2)
    print("调用 %d ｜ 命中 %d (%.1f%%) ｜ 省 token %s ｜ 省 %s（¥%.2f）"
          % (t["calls"], t["hits"], t["hit_rate"] * 100, human_int(t["saved_tokens"]),
             money(t["saved_usd"]), t["saved_usd"] * rate))
    return 0


# --------------------------------------------------------------------------- #
# 自检：起一个假上游 + 真代理，把关键行为全跑一遍
# --------------------------------------------------------------------------- #

_MOCK_EMB: Optional[LocalEmbedder] = None


class _MockHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a: Any) -> None:
        pass

    def _send_body(self, body: bytes, ctype: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        global _MOCK_EMB
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b"{}"
        try:
            req = json.loads(raw or b"{}")
        except Exception:
            req = {}
        if self.path.endswith("/embeddings"):
            if _MOCK_EMB is None:
                _MOCK_EMB = LocalEmbedder(64)
            inputs = req.get("input") or []
            if isinstance(inputs, str):
                inputs = [inputs]
            vecs = _MOCK_EMB.embed([str(t) for t in inputs])
            body = json.dumps({
                "object": "list",
                "model": req.get("model", "mock-embed"),
                "data": [{"object": "embedding", "index": i, "embedding": v}
                         for i, v in enumerate(vecs)],
            }).encode("utf-8")
            return self._send_body(body, "application/json")

        msgs = req.get("messages") or []
        q = ""
        for m in reversed(msgs):
            if isinstance(m, dict) and m.get("role") == "user":
                c = m.get("content")
                if isinstance(c, list):
                    q = " ".join(str(p.get("text", "")) for p in c if isinstance(p, dict))
                else:
                    q = str(c or "")
                break
        ans = "【模拟回答】关于「%s」的固定答案。" % trunc(q, 40)
        usage = {"prompt_tokens": est_tokens(q) + 12, "completion_tokens": est_tokens(ans),
                 "total_tokens": est_tokens(q) + 12 + est_tokens(ans)}
        mdl = req.get("model", "mock-model")
        if req.get("stream"):
            parts = []
            for i in range(0, len(ans), 6):
                ev = {"id": "chatcmpl-mock", "object": "chat.completion.chunk",
                      "created": int(now()), "model": mdl,
                      "choices": [{"index": 0, "delta": {"content": ans[i:i + 6]},
                                   "finish_reason": None}]}
                parts.append(b"data: " + json.dumps(ev, ensure_ascii=False).encode() + b"\n\n")
            ev = {"id": "chatcmpl-mock", "object": "chat.completion.chunk", "created": int(now()),
                  "model": mdl, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                  "usage": usage}
            parts.append(b"data: " + json.dumps(ev, ensure_ascii=False).encode() + b"\n\n")
            parts.append(b"data: [DONE]\n\n")
            return self._send_body(b"".join(parts), "text/event-stream")
        resp = {
            "id": "chatcmpl-mock", "object": "chat.completion", "created": int(now()),
            "model": mdl,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": ans},
                         "finish_reason": "stop"}],
            "usage": usage,
        }
        self._send_body(json.dumps(resp, ensure_ascii=False).encode(), "application/json")


class MockUpstream:
    def __init__(self) -> None:
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _MockHandler)
        self.httpd.daemon_threads = True
        self.port = int(self.httpd.server_address[1])
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self) -> "MockUpstream":
        self.thread.start()
        return self

    def __exit__(self, *a: Any) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def http_call(host: str, port: int, method: str, path: str, body: Optional[bytes] = None,
              headers: Optional[Dict[str, str]] = None, timeout: float = 30.0
              ) -> Tuple[int, Dict[str, str], bytes]:
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        conn.request(method, path, body=body, headers=headers or {})
        r = conn.getresponse()
        data = r.read()
        return r.status, {k.lower(): v for k, v in r.getheaders()}, data
    finally:
        conn.close()


def _sse_text(raw: bytes) -> str:
    parts = []
    for line in raw.split(b"\n"):
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if not data or data == b"[DONE]":
            continue
        try:
            obj = json.loads(data)
        except Exception:
            continue
        for ch in obj.get("choices") or []:
            c = (ch.get("delta") or {}).get("content")
            if isinstance(c, str):
                parts.append(c)
    return "".join(parts)


def cmd_selftest(cfg: Dict[str, Any], args: argparse.Namespace) -> int:
    import tempfile

    setup_logging("ERROR")
    tmp = tempfile.mkdtemp(prefix="semcache-selftest-")
    print("自检开始（临时目录 %s）\n" % tmp)
    fails: List[str] = []

    def check(name: str, cond: Any, info: str = "") -> None:
        ok = bool(cond)
        if not ok:
            fails.append(name)
        print("  %s %s%s" % ("✓" if ok else "✗", name, ("   " + info) if info else ""))

    with MockUpstream() as mock:
        cfg = dict(cfg)
        cfg.update({
            "db": os.path.join(tmp, "t.db"),
            "upstream": "http://127.0.0.1:%d" % mock.port,
            "embedder": "local", "local_dim": 512,
            "cache_stream": True, "calib_every": 0, "log_level": "ERROR",
            "admin_token": "", "dry_run": False,
        })
        rt = build_runtime(cfg)
        srv = ProxyServer(("127.0.0.1", 0), Handler, rt)
        th = threading.Thread(target=srv.serve_forever, daemon=True)
        th.start()
        port = int(srv.server_address[1])
        host = "127.0.0.1"

        def chat(prompt: str, project: Optional[str] = None, system: Optional[str] = None,
                 stream: bool = False, extra: Optional[Dict[str, str]] = None,
                 model: str = "gpt-4o-mini") -> Tuple[int, Dict[str, str], bytes]:
            msgs = []
            if system:
                msgs.append({"role": "system", "content": system})
            msgs.append({"role": "user", "content": prompt})
            payload: Dict[str, Any] = {"model": model, "messages": msgs, "temperature": 0.7}
            if stream:
                payload["stream"] = True
            h = {"Content-Type": "application/json"}
            if project:
                h["X-Semcache-Project"] = project.encode("utf-8").decode("latin-1")
            h.update(extra or {})
            return http_call(host, port, "POST", "/v1/chat/completions",
                             json.dumps(payload).encode(), h)

        try:
            st, _, body = http_call(host, port, "GET", "/healthz")
            check("健康检查", st == 200 and json.loads(body).get("ok") is True)

            q1 = "订单 12345 什么时候发货？"
            st, h1, b1 = chat(q1)
            check("首次请求未命中", st == 200 and h1.get("x-semcache") == "miss")
            st, h2, b2 = chat(q1)
            check("完全相同 → 精确命中",
                  h2.get("x-semcache") == "hit" and h2.get("x-semcache-kind") == "exact")
            check("命中省下 token", int(h2.get("x-semcache-tokens-saved", "0")) > 0,
                  "%s tokens" % h2.get("x-semcache-tokens-saved"))
            check("命中响应可用", json.loads(b2)["choices"][0]["message"]["content"] != "")

            st, h3, _ = chat("订单12345什么时候发货")
            check("归一化重复（空格/标点）→ 精确命中", h3.get("x-semcache-kind") == "exact")

            st, h4, _ = chat("订单 12345 什么时候能发货")
            check("近似改写 → 语义命中",
                  h4.get("x-semcache-kind") == "semantic",
                  "score=%s thr=%s" % (h4.get("x-semcache-score"), h4.get("x-semcache-threshold")))

            st, h5, _ = chat("你们的退款政策具体是什么")
            check("不相关问题 → 未命中", h5.get("x-semcache") == "miss")

            st, h6, _ = chat(q1, project="另一个项目")
            check("项目隔离（不同 project 不串味）", h6.get("x-semcache") == "miss")

            st, h7, _ = chat(q1, system="你是一个法务顾问")
            check("系统提示变化 → 新命名空间", h7.get("x-semcache") == "miss")

            st, s1, b_s1 = chat("流式：今天天气如何", stream=True)
            check("流式首次未命中", s1.get("x-semcache") == "miss")
            st, s2, b_s2 = chat("流式：今天天气如何", stream=True)
            check("流式重复 → 回放 SSE", s2.get("x-semcache") == "hit",
                  "content-type=%s" % s2.get("content-type", "")[:30])
            check("SSE 回放内容一致", _sse_text(b_s2) == _sse_text(b_s1),
                  _sse_text(b_s2)[:24])

            st, h8, _ = chat(q1, extra={"X-Semcache-No-Cache": "1"})
            check("客户端强制绕过", h8.get("x-semcache") == "bypass")

            for mdl in ("gpt-4o-mini", "deepseek-chat"):
                for i in range(6):
                    chat("校准样本问题 %s %d 的内容分别是什么" % (mdl, i), model=mdl)

            st, _, body = http_call(host, port, "POST", "/admin/calibrate", b"{}",
                                    {"Content-Type": "application/json"})
            cal = json.loads(body)
            check("强制校准执行", st == 200 and cal.get("recalibrated", 0) > 0,
                  "%d 个命名空间" % cal.get("recalibrated", 0))
            rows = json.loads(http_call(host, port, "GET", "/admin/calib")[2])["calibrations"]
            ok = all((0.0 <= r["eff"] <= 0.995) and r["floor"] >= 0.4 for r in rows)
            check("阈值落在安全区间", ok and len(rows) > 0,
                  "eff∈[%.3f,%.3f]" % (min(r["eff"] for r in rows),
                                       max(r["eff"] for r in rows)))

            ns_rows = [r for r in rt.store.all_namespaces() if r["project"] == "default"]
            target = ns_rows[0]
            before = {r["ns"]: r for r in rows}
            payload = {"project": target["project"], "ns": target["ns"],
                       "label": "bad", "score": 0.8}
            http_call(host, port, "POST", "/admin/feedback",
                      json.dumps(payload).encode(), {"Content-Type": "application/json"})
            rows2 = json.loads(http_call(host, port, "GET", "/admin/calib")[2])["calibrations"]
            hit = [r for r in rows2 if r["ns"] == target["ns"]][0]
            check("差评 → 阈值上调",
                  hit["bias"] > 0 and hit["eff"] >= before[hit["ns"]]["eff"],
                  "bias=%+.3f eff=%.3f" % (hit["bias"], hit["eff"]))

            st, _, body = http_call(host, port, "GET", "/admin/report?format=json")
            rep = json.loads(body)
            t = rep["totals"]
            check("报告：命中数统计", t["hits"] >= 4, "hits=%d" % t["hits"])
            check("报告：省下费用 > 0", t["saved_usd"] > 0, money(t["saved_usd"]))
            check("报告：重复率合理", 0 < t["hit_rate"] < 1, "%.1f%%" % (t["hit_rate"] * 100))
            st, _, body = http_call(host, port, "GET", "/admin/report?format=md")
            check("报告：Markdown 可导出", body.startswith("# 语义缓存".encode("utf-8")))

            st, _, body = http_call(host, port, "GET", "/dashboard")
            check("看板页面可访问", st == 200 and b"semcache" in body)

            emb = OpenAIEmbedder("http://127.0.0.1:%d" % mock.port, "test-key",
                                 "text-embedding-3-small", dims=64)
            vecs = emb.embed(["订单 12345 什么时候发货", "订单 12345 什么时候发货"])
            norm = math.sqrt(sum(x * x for x in vecs[0]))
            check("openai embedder 可用", len(vecs[0]) == 64 and abs(norm - 1) < 0.01,
                  "dim=%d |v|=%.3f" % (len(vecs[0]), norm))

            cfg2 = dict(cfg)
            cfg2.update({"db": os.path.join(tmp, "t2.db"), "embedder": "openai",
                         "embed_model": "text-embedding-3-small", "dim": 64,
                         "embed_base": "http://127.0.0.1:%d" % mock.port})
            rt2 = build_runtime(cfg2)
            srv2 = ProxyServer(("127.0.0.1", 0), Handler, rt2)
            threading.Thread(target=srv2.serve_forever, daemon=True).start()
            p2 = int(srv2.server_address[1])
            hh: Dict[str, str] = {}
            for _ in range(2):
                st, hh, _ = http_call(
                    host, p2, "POST", "/v1/chat/completions",
                    json.dumps({"model": "gpt-4o-mini",
                                "messages": [{"role": "user", "content": q1}]}).encode(),
                    {"Content-Type": "application/json"})
            check("openai embedder 端到端命中", hh.get("x-semcache") == "hit")
            srv2.shutdown()
            srv2.server_close()
        finally:
            srv.shutdown()
            srv.server_close()
            rt.upstream.close()

    print()
    if fails:
        print("失败 %d 项：" % len(fails))
        for f in fails:
            print("  - " + f)
        return 1
    print("全部通过 ✓  （临时库 %s）" % os.path.join(tmp, "t.db"))
    if not getattr(args, "keep", False):
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


# --------------------------------------------------------------------------- #
# 演示数据：跑真代码路径，生成一份像模像样的月度账单
# --------------------------------------------------------------------------- #

DEMO_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "客服-主站": {
        "model": "gpt-4o-mini", "share": 0.40, "repeat": 0.66, "temp": 0.7,
        "system": "你是电商客服助手，回答简洁，直接给结论。",
        "canon": [
            ("退款要多久到账？", ["退款到账要多久？", "退款要多久到账", "退款多长时间到账"]),
            ("怎么修改收货地址", ["收货地址怎么修改", "修改收货地址怎么操作", "如何修改收货地址？"]),
            ("优惠券为什么用不了", ["优惠券怎么用不了", "为什么我的优惠券不能用", "优惠券无法使用"]),
            ("订单 12345 什么时候发货", ["订单12345什么时候发货", "什么时候发货 订单 12345"]),
            ("怎么申请开发票", ["如何申请发票", "发票怎么开", "要开个发票怎么弄"]),
            ("支持七天无理由退货吗", ["七天无理由退货吗", "可以七天无理由退吗"]),
            ("密码忘了怎么办", ["忘记密码了怎么办", "我密码忘了怎么处理"]),
        ],
    },
    "内部-代码助手": {
        "model": "gpt-4o", "share": 0.22, "repeat": 0.34, "temp": 0.2,
        "system": "你是资深工程师，回答要给出可运行代码。",
        "canon": [
            ("如何在 Python 里读取大文件不爆内存",
             ["Python 读取大文件内存占用高怎么办", "python 怎么流式读取超大文件"]),
            ("SQL 慢查询怎么优化", ["数据库慢查询优化思路", "SQL 查询很慢如何优化"]),
            ("Docker 容器里时区不对怎么改", ["容器时区不正确如何设置", "docker 时区怎么改"]),
            ("React useEffect 重复执行的原因",
             ["useEffect 为什么会执行两次", "React useEffect 重复调用"]),
            ("怎么给 pandas DataFrame 去重", ["pandas 如何去重", "DataFrame 去重怎么写"]),
        ],
    },
    "文档问答": {
        "model": "deepseek-chat", "share": 0.24, "repeat": 0.48, "temp": 0.0,
        "system": "你是制度文档助手，只依据给定文档回答。",
        "canon": [
            ("年假有几天", ["公司年假有几天", "年假天数怎么算", "年假一共几天"]),
            ("报销流程是什么", ["报销流程是什么啊", "费用报销流程是什么"]),
            ("出差住宿标准多少", ["出差住宿费标准是多少", "住宿报销标准"]),
            ("试用期多久", ["试用期是多长时间", "试用期几个月"]),
            ("远程办公政策", ["公司远程办公怎么规定", "可以远程办公吗"]),
        ],
    },
    "个人-杂问": {
        "model": "gpt-4o-mini", "share": 0.14, "repeat": 0.18, "temp": 0.9,
        "system": "你是一个乐于助人的助手。",
        "canon": [
            ("帮我把这段话翻译成英文", ["翻译成英文", "把这段翻译成英语"]),
            ("写一首关于秋天的短诗", ["来一首秋天的诗", "写首秋天的小诗"]),
            ("解释一下什么是向量数据库", ["向量数据库是什么", "说说向量数据库"]),
        ],
    },
}

DEMO_CONTEXT: Dict[str, str] = {
    "客服-主站": (
        "【售后知识库 v3.2｜内部资料，请勿外传】"
        "退款规则：签收后七天内可申请无理由退货，商品需保持完好、吊牌齐全；"
        "定制类、生鲜类、拆封的个护类不支持无理由退货。退款到账时间：原路退回，"
        "微信/支付宝一般 1-3 个工作日，银行卡 3-7 个工作日，信用卡以银行入账为准。"
        "运费规则：质量问题由商家承担来回运费；非质量问题由买家承担寄回运费，"
        "满 99 元订单可用运费券抵扣。发票：支持电子普票与增值税专票，下单时勾选或"
        "在订单详情页补开，专票需提供公司名称、税号、开户行及账号。"
    ),
    "内部-代码助手": (
        "【团队工程规范 v1.8｜内部资料】"
        "代码风格：Python 使用 black + ruff，行宽 100；提交前必须跑 pre-commit。"
        "分支策略：main 受保护，功能分支命名 feat/<ticket>-<slug>，合并需 2 个 approve。"
        "日志规范：使用结构化日志，字段包含 trace_id、user_id、latency_ms；禁止打印手机号"
        "与身份证号。性能红线：P99 接口耗时不得超过 800ms，慢查询超过 200ms 必须加索引"
        "或在周会上说明。发布流程：灰度 5% → 观察 30 分钟 → 全量，回滚脚本必须随包发布。"
        "安全：密钥一律走 KMS，仓库中禁止出现明文 token；依赖升级需跑完安全扫描。"
    ),
    "文档问答": (
        "【员工手册 2025 版｜人力资源部】"
        "年假：入职满一年享有 5 天带薪年假，满三年 10 天，满五年 15 天；"
        "年假需提前三个工作日在系统提交，跨年最多结转 5 天，逾期作废。"
        "报销：差旅报销需在出差结束后 15 天内提交，附行程单与发票；"
        "住宿标准：一线城市 600 元/晚，二线 400 元/晚，其他 300 元/晚，超标部分自理。"
        "试用期：一般为 3 个月，特殊岗位 6 个月，试用期薪资不低于转正的 80%。"
        "远程办公：每周最多 2 天，需提前一天在系统报备，涉及客户信息的操作必须在"
        "公司设备与内网环境下进行。"
    ),
    "个人-杂问": (
        "【个人知识库｜常用背景】"
        "我在一家做企业服务的公司负责增长，日常关注 SaaS 定价、留存与获客成本；"
        "写作偏好简洁、少形容词、多用数字与对比；翻译偏好英式拼写。"
        "常用工具：Notion、Figma、Obsidian；常用语言：中文为主，英文可读。"
    ),
}

_UNIQUE_TEMPLATES = [
    "帮我查一下订单 {oid} 的物流进度",
    "客户 {name} 投诉说收到的商品有划痕，订单 {oid}，怎么回复比较好",
    "这段报错怎么排查：index out of range at line {n}",
    "把这段需求拆成 3 个开发任务：{oid} 号需求",
    "总结一下这篇文章的要点（文章 {n}）",
    "帮我写一封催款邮件，金额 {n} 元，客户 {name}",
    "这个 SQL 为什么慢：select * from orders where id={n}",
    "给我 5 个关于 {name} 活动的宣传文案，活动编号 {oid}",
]
_NAMES = ["张伟", "李娜", "王芳", "刘洋", "陈静", "赵磊", "孙悦", "周涛"]


_UNIQ_SEQ = [100000]


def _demo_call(rng: random.Random, proj: str, spec: Dict[str, Any],
               uniq_cache: Dict[str, List[str]]) -> str:
    """长尾问题：每次都带一个新编号，保证是「新问题」，不该命中。"""
    if rng.random() < float(spec["repeat"]):
        canon, variants = rng.choice(spec["canon"])
        return rng.choice(variants) if rng.random() < 0.8 else canon
    _UNIQ_SEQ[0] += 1
    seq = _UNIQ_SEQ[0]
    t = rng.choice(_UNIQUE_TEMPLATES)
    return t.format(oid=seq, name=rng.choice(_NAMES), n=seq)


def cmd_demo(cfg: Dict[str, Any], args: argparse.Namespace) -> int:
    import tempfile

    setup_logging("ERROR")
    calls = int(getattr(args, "calls", 12000) or 12000)
    days = int(getattr(args, "days", 30) or 30)
    keep = bool(getattr(args, "keep", False))
    default_db = os.path.expanduser(DEFAULTS["db"])
    if os.path.abspath(cfg["db"]) == os.path.abspath(default_db):
        cfg["db"] = os.path.join(tempfile.gettempdir(), "semcache-demo.db")
    dbp = cfg["db"]
    if os.path.exists(dbp) and not keep:
        for suffix in ("", "-wal", "-shm"):
            try:
                os.remove(dbp + suffix)
            except OSError:
                pass
    cfg.update({"embedder": "local", "local_dim": 1024, "dry_run": False})
    store = Store(dbp)
    engine = CacheEngine(cfg, store, make_embedder(cfg))
    rng = random.Random(20251001)
    t_end = now()
    t_start = t_end - days * 86400
    uniq_cache: Dict[str, List[str]] = {}
    projects = list(DEMO_SCENARIOS.keys())
    shares = [float(DEMO_SCENARIOS[p]["share"]) for p in projects]
    hits = missed = 0

    for i in range(calls):
        ts = t_start + (i / float(calls)) * (t_end - t_start) + rng.random() * 60
        proj = rng.choices(projects, weights=shares, k=1)[0]
        spec = DEMO_SCENARIOS[proj]
        prompt = _demo_call(rng, proj, spec, uniq_cache)
        system_text = spec["system"] + "\n\n" + DEMO_CONTEXT.get(proj, "") * 3
        payload = {
            "model": spec["model"],
            "temperature": spec["temp"],
            "messages": [
                {"role": "system", "content": system_text},
                {"role": "user", "content": prompt},
            ],
        }
        prep = prepare(payload, proj, engine.embedder.ident,
                       max_temperature=float(cfg["max_temperature"]),
                       mask_dynamic_on=bool(cfg["mask_dynamic"]),
                       max_embed_chars=int(cfg["max_embed_chars"]))
        lk = engine.lookup(prep)
        if lk.ok and lk.row is not None:
            row = lk.row
            pt = int(row["prompt_tokens"] or 0)
            ct = int(row["completion_tokens"] or 0)
            engine.log_event("hit", prep, score=lk.score, detail=lk.kind,
                             entry_id=int(row["id"]), pt=pt, ct=ct,
                             latency_ms=1.2 + rng.random() * 3, ts=ts)
            engine.tick(prep, hit=True, score=lk.score)
            hits += 1
        else:
            ans_tokens = int(180 + rng.random() * 420)
            pt = est_tokens(prompt) + est_tokens(system_text)
            ans = {"id": "chatcmpl-demo", "object": "chat.completion",
                   "created": int(ts), "model": spec["model"],
                   "choices": [{"index": 0,
                                "message": {"role": "assistant",
                                            "content": "（演示数据）关于「%s」的回答…" % trunc(prompt, 30)},
                                "finish_reason": "stop"}],
                   "usage": {"prompt_tokens": pt, "completion_tokens": ans_tokens,
                             "total_tokens": pt + ans_tokens}}
            engine.log_event("miss", prep, score=lk.score, detail=_miss_detail(lk),
                             pt=pt, ct=ans_tokens,
                             latency_ms=700 + rng.random() * 1500, ts=ts)
            engine.tick(prep, hit=False, score=lk.score)
            engine.store_answer(prep, ans)
            missed += 1

    for r in store.all_namespaces():
        engine.recalibrate(r["project"], r["ns"], force=True)

    rep = build_report(store, cfg, month=None)
    print("═" * 74)
    print(" 演示数据：%d 次调用 / %d 天 / 4 个项目（命中 %d 次，%.1f%%）"
          % (calls, days, hits, hits * 100.0 / max(1, calls)))
    print(" 数据库：%s" % dbp)
    print("═" * 74)
    print(rep["text"])
    print()
    print("上面这份就是可以发给客户的账。真实跑：")
    print("  python3 semcache.py serve --dry-run     # 先影子模式，只记账不生效")
    print("  python3 semcache.py report --md --out 报告.md")
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

EPILOG = """\
例子
  python3 semcache.py serve                                    # 本地起代理（零依赖）
  python3 semcache.py serve --dry-run                          # 影子模式：只记账不命中
  python3 semcache.py serve --embedder openai \\
      --embed-model text-embedding-3-small                     # 跨语义改写，质量更高
  python3 semcache.py serve --project-keys "客服=sk-a,内部=sk-b"
  python3 semcache.py report --md --out 本月账单.md
  python3 semcache.py calibrate --all
  python3 semcache.py selftest
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="semcache.py", description="面向 OpenAI 兼容 chat/completions 的语义缓存代理",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=EPILOG)
    p.add_argument("--version", action="version", version="semcache " + VERSION)
    sub = p.add_subparsers(dest="cmd")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--db", help="SQLite 路径（默认 ~/.semcache/semcache.db）")
    common.add_argument("--config", help="JSON 配置文件")
    common.add_argument("--log-level", dest="log_level",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    s = sub.add_parser("serve", parents=[common], help="启动代理（默认命令）")
    s.add_argument("--host")
    s.add_argument("--port", type=int)
    s.add_argument("--upstream", help="上游 base_url，如 https://api.openai.com")
    s.add_argument("--upstream-key", dest="upstream_key", help="代理自带上游 key")
    s.add_argument("--embedder", choices=["local", "openai"])
    s.add_argument("--embed-model", dest="embed_model")
    s.add_argument("--embed-base", dest="embed_base")
    s.add_argument("--embed-key", dest="embed_key")
    s.add_argument("--dim", dest="local_dim", type=int, help="local 维度 / openai 目标维度")
    s.add_argument("--threshold", type=float, help="固定阈值（设置后关闭自动校准）")
    s.add_argument("--min-threshold", dest="min_threshold", type=float, help="安全下界")
    s.add_argument("--ttl", type=int, help="缓存过期秒数，0=永不过期")
    s.add_argument("--max-entries", dest="max_entries", type=int)
    s.add_argument("--max-temperature", dest="max_temperature", type=float,
                   help="温度高于此值不缓存（默认 1.0 表示都缓存）")
    s.add_argument("--no-cache-stream", dest="cache_stream", action="store_false",
                   default=None, help="不缓存流式响应")
    s.add_argument("--dry-run", dest="dry_run", action="store_true", default=None,
                   help="影子模式：只统计能省多少，不真正命中")
    s.add_argument("--admin-token", dest="admin_token")
    s.add_argument("--project-keys", dest="project_keys", metavar="proj=sk-xxx,proj2=sk-yyy")
    s.add_argument("--price", action="append", metavar="MODEL=IN/OUT",
                   help="覆盖价格（$/1M tokens），可重复")
    s.add_argument("--usd-cny", dest="usd_cny", type=float)
    s.add_argument("--calib-every", dest="calib_every", type=int)
    s.add_argument("--repeat-window", dest="repeat_window", type=int)
    s.add_argument("--no-implicit-feedback", dest="implicit_feedback", action="store_false",
                   default=None, help="关闭「重问即差评」的隐式反馈")
    s.add_argument("--lexical-gate", dest="lexical_gate", type=float,
                   help="local 向量的词面 Jaccard 门槛（默认 0.6）")
    s.add_argument("--lexical-cover", dest="lexical_cover", type=float,
                   help="local 向量的包含度门槛（默认 0.8）")
    s.add_argument("--max-floor-lift", dest="max_floor_lift", type=float,
                   help="阈值相对先验最多抬高多少（默认 0.15）")

    r = sub.add_parser("report", parents=[common], help="重复率报告")
    r.add_argument("--month", help="YYYY-MM，默认全部")
    r.add_argument("--project")
    r.add_argument("--md", action="store_true", help="输出 Markdown")
    r.add_argument("--json", action="store_true", help="输出 JSON")
    r.add_argument("--out", help="写到文件")

    c = sub.add_parser("calibrate", parents=[common], help="立刻重算阈值")
    c.add_argument("--project")
    c.add_argument("--ns")
    c.add_argument("--all", action="store_true")

    sub.add_parser("stats", parents=[common], help="一行摘要")

    t = sub.add_parser("selftest", parents=[common], help="自检")
    t.add_argument("--keep", action="store_true", help="保留临时库")

    d = sub.add_parser("demo", parents=[common], help="生成演示数据并打印报告")
    d.add_argument("--calls", type=int, default=12000)
    d.add_argument("--days", type=int, default=30)
    d.add_argument("--keep", action="store_true")
    return p


SUBCOMMANDS = ("serve", "report", "calibrate", "stats", "selftest", "demo")


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # 不带子命令时默认 serve，方便单文件场景直接 `python semcache.py --port 9000`
    if not any(a in SUBCOMMANDS for a in argv) and not any(
            a in ("-h", "--help", "--version") for a in argv):
        argv.insert(0, "serve")
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args)
    except Exception as e:
        print("配置加载失败: %s" % e, file=sys.stderr)
        return 2
    cmd = getattr(args, "cmd", None) or "serve"
    if cmd == "serve":
        return cmd_serve(cfg)
    if cmd == "report":
        return cmd_report(cfg, args)
    if cmd == "calibrate":
        return cmd_calibrate(cfg, args)
    if cmd == "stats":
        return cmd_stats(cfg)
    if cmd == "selftest":
        return cmd_selftest(cfg, args)
    if cmd == "demo":
        return cmd_demo(cfg, args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())






