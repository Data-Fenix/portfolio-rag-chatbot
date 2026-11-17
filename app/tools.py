"""Client-side tools the model can call for structured lookups.

Retrieval handles fuzzy questions; these tools handle exact ones ("list every
AWS project", "what's the URL of the parking repo"). Both are *read only* and
cannot fail in a way that leaks anything, which keeps the trust boundary
simple.
"""

from __future__ import annotations

import json
from functools import lru_cache

from .config import settings


@lru_cache(maxsize=1)
def load_projects() -> list[dict]:
    return json.loads(settings.projects_file.read_text(encoding="utf-8"))


TOOLS: list[dict] = [
    {
        "name": "list_projects",
        "description": (
            "List Lahiru's public projects with title, category, year, repository URL and tags. "
            "Optionally filter by category (genai, mlops, ml, data-eng, devops, analytics) or by a "
            "tag/technology keyword such as 'SageMaker'. Use when the visitor wants an exhaustive "
            "or filtered list."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "description": "Category id to filter by, or omit for all."},
                "keyword": {"type": "string", "description": "Case-insensitive keyword matched against title and tags."},
            },
            "additionalProperties": False,
        },
        "strict": True,
        "eager_input_streaming": True,
    },
    {
        "name": "get_project",
        "description": (
            "Get full details of one project by its title or repository name, including summary, "
            "highlights, tags and the GitHub URL."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "Project title or repo name (fuzzy match)."}},
            "required": ["name"],
            "additionalProperties": False,
        },
        "strict": True,
        "eager_input_streaming": True,
    },
]


def _public(p: dict) -> dict:
    return {
        "title": p["title"],
        "category": p["category"],
        "year": p.get("year"),
        "url": f"https://github.com/{p['repo']}",
        "tags": p.get("tags", []),
    }


def list_projects(category: str | None = None, keyword: str | None = None) -> dict:
    items = load_projects()
    if category:
        items = [p for p in items if p["category"] == category.lower().strip()]
    if keyword:
        kw = keyword.lower().strip()
        items = [p for p in items if kw in p["title"].lower() or any(kw in t.lower() for t in p.get("tags", []))]
    return {"count": len(items), "projects": [_public(p) for p in items]}


def get_project(name: str) -> dict:
    needle = (name or "").lower().strip()
    best = None
    for p in load_projects():
        hay = f"{p['title']} {p['repo']}".lower()
        if needle and needle in hay:
            best = p
            break
    if not best:
        return {"error": f"No project matches '{name}'. Try list_projects to see the available titles."}
    return {**_public(best), "summary": best.get("summary"), "highlights": best.get("highlights", [])}


HANDLERS = {"list_projects": list_projects, "get_project": get_project}


def run_tool(name: str, inputs: dict) -> tuple[str, bool]:
    """Execute a tool; returns (json_result, is_error)."""
    handler = HANDLERS.get(name)
    if not handler:
        return json.dumps({"error": f"unknown tool {name}"}), True
    try:
        return json.dumps(handler(**inputs)), False
    except TypeError as exc:  # bad arguments from a malformed/truncated tool input
        return json.dumps({"error": f"invalid arguments: {exc}"}), True
