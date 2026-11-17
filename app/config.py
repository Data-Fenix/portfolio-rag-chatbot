"""Runtime configuration loaded from environment variables.

Every knob that affects cost, safety or behaviour lives here so it can be
changed per environment without touching code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent


def _env_list(name: str, default: str) -> list[str]:
    return [x.strip() for x in os.getenv(name, default).split(",") if x.strip()]


@dataclass(frozen=True)
class Settings:
    # Model
    model: str = os.getenv("MODEL", "claude-opus-5-5")
    judge_model: str = os.getenv("JUDGE_MODEL", "claude-sonnet-5-5")
    effort: str = os.getenv("EFFORT", "low")  # chat does well at low effort
    max_output_tokens: int = int(os.getenv("MAX_OUTPUT_TOKENS", "2048"))

    # Retrieval
    knowledge_dir: Path = ROOT / "data" / "knowledge"
    projects_file: Path = ROOT / "data" / "projects.json"
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "900"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    top_k: int = int(os.getenv("TOP_K", "5"))
    retriever_backend: str = os.getenv("RETRIEVER_BACKEND", "bm25")  # bm25 | chroma | hybrid
    chroma_dir: Path = ROOT / ".chroma"

    # Guardrails
    max_question_chars: int = int(os.getenv("MAX_QUESTION_CHARS", "500"))
    max_history_turns: int = int(os.getenv("MAX_HISTORY_TURNS", "6"))

    # Server
    allowed_origins: list[str] = field(
        default_factory=lambda: _env_list("ALLOWED_ORIGINS", "http://localhost:8080,http://127.0.0.1:8080")
    )
    rate_limit_per_minute: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "20"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()
