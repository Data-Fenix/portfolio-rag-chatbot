"""Structured logging and cost accounting for every model call."""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from dataclasses import dataclass, field

from .config import settings

# USD per million tokens. Keep in sync with https://platform.claude.com/docs pricing.
PRICING = {
    "claude-opus-5-5": {"input": 4.0, "output": 20.0, "cache_write": 5.0, "cache_read": 0.20},
    "claude-sonnet-5-5": {"input": 2.0, "output": 10.0, "cache_write": 2.5, "cache_read": 0.10},
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50, "cache_write": 0.125, "cache_read": 0.01},
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if hasattr(record, "data"):
            payload.update(record.data)  # type: ignore[attr-defined]
        return json.dumps(payload, default=str)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(JsonFormatter())
        logger.addHandler(h)
        logger.setLevel(settings.log_level)
        logger.propagate = False
    return logger


def estimate_cost(model: str, usage) -> float:
    p = PRICING.get(model)
    if not p or usage is None:
        return 0.0
    cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    return (
        (usage.input_tokens or 0) * p["input"]
        + (usage.output_tokens or 0) * p["output"]
        + cache_write * p["cache_write"]
        + cache_read * p["cache_read"]
    ) / 1_000_000


@dataclass
class RequestTrace:
    """Accumulates usage across the tool-use loop of one chat request."""

    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    started: float = field(default_factory=time.perf_counter)
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    tools_used: list[str] = field(default_factory=list)
    retrieved: list[str] = field(default_factory=list)
    stop_reason: str | None = None

    def add_usage(self, model: str, usage) -> None:
        if usage is None:
            return
        self.model_calls += 1
        self.input_tokens += usage.input_tokens or 0
        self.output_tokens += usage.output_tokens or 0
        self.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
        self.cache_write_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0
        self.cost_usd += estimate_cost(model, usage)

    def as_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "latency_ms": round((time.perf_counter() - self.started) * 1000),
            "model_calls": self.model_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "tools_used": self.tools_used,
            "retrieved": self.retrieved,
            "stop_reason": self.stop_reason,
        }
