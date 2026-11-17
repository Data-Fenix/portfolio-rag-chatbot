"""Load and chunk the markdown knowledge base.

Each markdown file in ``data/knowledge`` is a *source*. A source is split on
``##`` headings first (so chunks stay topically coherent), then long sections
are further split with overlap. Every chunk keeps its source title and URL so
the model can cite it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .config import settings

FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.S)


@dataclass(frozen=True)
class Chunk:
    id: str
    source_id: str
    title: str
    url: str
    section: str
    text: str

    @property
    def citation(self) -> dict:
        return {"id": self.id, "title": self.title, "url": self.url, "section": self.section}


def _parse_frontmatter(raw: str) -> tuple[dict, str]:
    meta: dict[str, str] = {}
    m = FRONTMATTER.match(raw)
    if not m:
        return meta, raw
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta, raw[m.end():]


def _split_long(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    if len(" ".join(words)) <= size:
        return [text.strip()]
    out, cur, cur_len = [], [], 0
    for w in words:
        cur.append(w)
        cur_len += len(w) + 1
        if cur_len >= size:
            out.append(" ".join(cur))
            # keep a tail of words as overlap
            tail: list[str] = []
            tail_len = 0
            for tw in reversed(cur):
                if tail_len + len(tw) + 1 > overlap:
                    break
                tail.insert(0, tw)
                tail_len += len(tw) + 1
            cur, cur_len = tail, tail_len
    if cur:
        out.append(" ".join(cur))
    return [c.strip() for c in out if c.strip()]


def load_chunks(knowledge_dir: Path | None = None) -> list[Chunk]:
    knowledge_dir = knowledge_dir or settings.knowledge_dir
    chunks: list[Chunk] = []
    for path in sorted(knowledge_dir.glob("*.md")):
        meta, body = _parse_frontmatter(path.read_text(encoding="utf-8"))
        source_id = path.stem
        title = meta.get("title", source_id.replace("-", " ").title())
        url = meta.get("url", "")
        sections = re.split(r"^##\s+", body, flags=re.M)
        for sec in sections:
            sec = sec.strip()
            if not sec:
                continue
            heading, _, content = sec.partition("\n")
            if not content.strip():  # heading-only or preamble
                heading, content = "Overview", heading
            heading = heading.strip().lstrip("# ").strip() or "Overview"
            for i, piece in enumerate(_split_long(content.strip(), settings.chunk_size, settings.chunk_overlap)):
                chunks.append(
                    Chunk(
                        id=f"{source_id}#{_slug(heading)}-{i}",
                        source_id=source_id,
                        title=title,
                        url=url,
                        section=heading,
                        text=piece,
                    )
                )
    return chunks


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40]
