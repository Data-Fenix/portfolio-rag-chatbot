"""The RAG engine: guardrails -> retrieval -> Claude (streaming, tools) -> cited answer.

Design notes
- The system prompt and tool list are byte-stable and carry a cache breakpoint,
  so repeated requests hit the prompt cache (``cache_read_input_tokens`` > 0).
- Streaming is used for every call so the UI sees tokens immediately.
- Tool use runs as a manual loop so we keep full control: tool inputs are
  validated, results are returned as ``tool_result`` blocks, and the full
  assistant content (including thinking blocks) is appended back verbatim.
- Every call reports token usage and estimated cost through ``RequestTrace``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass

import anthropic

from . import guardrails
from .config import settings
from .knowledge import Chunk
from .observability import RequestTrace, get_logger
from .prompts import SYSTEM_PROMPT, build_user_turn
from .retriever import Retriever, build_retriever
from .tools import TOOLS, run_tool

log = get_logger("rag.engine")

MAX_TOOL_ROUNDS = 4


@dataclass
class Event:
    type: str  # delta | sources | done | error
    text: str = ""
    sources: list[dict] | None = None
    trace: dict | None = None
    message: str = ""


class RagEngine:
    def __init__(self, client: anthropic.Anthropic | None = None, retriever: Retriever | None = None) -> None:
        self.client = client or anthropic.Anthropic()
        self.retriever = retriever or build_retriever()

    # ---- public API -------------------------------------------------------

    def chat(self, question: str, history: list[dict] | None = None) -> Iterator[Event]:
        trace = RequestTrace()
        check = guardrails.check_input(question)
        if not check.ok:
            log.info("guardrail_block", extra={"data": {"request_id": trace.request_id, "reason": check.reason}})
            yield Event("delta", text=check.reply)
            yield Event("done", trace=trace.as_dict())
            return

        chunks = self._retrieve(check.cleaned, history)
        trace.retrieved = [c.id for c in chunks]
        sources = _dedupe_sources(chunks)

        messages = guardrails.sanitize_history(history or [])
        messages.append({"role": "user", "content": build_user_turn(check.cleaned, chunks)})

        full_text = ""
        try:
            for round_no in range(MAX_TOOL_ROUNDS + 1):
                with self.client.messages.stream(
                    model=settings.model,
                    max_tokens=settings.max_output_tokens,
                    system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
                    tools=TOOLS,
                    output_config={"effort": settings.effort},
                    messages=messages,
                ) as stream:
                    for text in stream.text_stream:
                        full_text += text
                        yield Event("delta", text=guardrails.check_output(text))
                    response = stream.get_final_message()

                trace.add_usage(settings.model, response.usage)
                trace.stop_reason = response.stop_reason

                if response.stop_reason == "refusal":
                    yield Event("delta", text="\n\nI can't help with that request. Ask me about Lahiru's work instead.")
                    break
                if response.stop_reason == "max_tokens":
                    yield Event("delta", text="\n\n[answer truncated]")
                    break
                if response.stop_reason != "tool_use":
                    break

                # Tool round: append the assistant turn verbatim, run every tool, return all results at once.
                messages.append({"role": "assistant", "content": response.content})
                results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    trace.tools_used.append(block.name)
                    inputs = block.input if isinstance(block.input, dict) else {}
                    output, is_error = run_tool(block.name, inputs)
                    results.append(
                        {"type": "tool_result", "tool_use_id": block.id, "content": output, "is_error": is_error}
                    )
                messages.append({"role": "user", "content": results})
                if round_no == MAX_TOOL_ROUNDS:
                    yield Event("delta", text="\n\n[stopped after too many tool calls]")
        except anthropic.RateLimitError:
            yield Event("error", message="The assistant is busy right now. Please try again in a minute.")
        except anthropic.APIStatusError as exc:
            log.error("api_error",
                      extra={"data": {"request_id": trace.request_id, "status": exc.status_code, "msg": exc.message[:200]}})
            yield Event("error", message="The assistant hit a temporary problem. Please try again.")
        except anthropic.APIConnectionError:
            yield Event("error", message="Could not reach the model service. Please try again.")

        yield Event("sources", sources=sources)
        log.info("chat_complete", extra={"data": {**trace.as_dict(), "answer_chars": len(full_text)}})
        yield Event("done", trace=trace.as_dict())

    def answer(self, question: str, history: list[dict] | None = None) -> tuple[str, list[dict], dict]:
        """Non-streaming convenience wrapper used by the CLI and the evals."""
        text, sources, trace = [], [], {}
        for ev in self.chat(question, history):
            if ev.type == "delta":
                text.append(ev.text)
            elif ev.type == "sources":
                sources = ev.sources or []
            elif ev.type == "error":
                text.append(ev.message)
            elif ev.type == "done":
                trace = ev.trace or {}
        return "".join(text), sources, trace

    # ---- internals --------------------------------------------------------

    def _retrieve(self, question: str, history: list[dict] | None) -> list[Chunk]:
        # Light query expansion: include the previous user turn so follow-ups like
        # "and what metrics did it reach?" still retrieve the right project.
        query = question
        prev_user = [t["content"] for t in (history or []) if t.get("role") == "user"]
        if prev_user and len(question.split()) < 8:
            query = f"{prev_user[-1]} {question}"
        return [c for c, _ in self.retriever.search(query, settings.top_k)]


def _dedupe_sources(chunks: list[Chunk]) -> list[dict]:
    seen, out = set(), []
    for c in chunks:
        key = (c.title, c.url)
        if key in seen:
            continue
        seen.add(key)
        out.append({"title": c.title, "url": c.url})
    return out


def sse(event: Event) -> str:
    """Serialise an Event as a Server-Sent Events frame."""
    payload = {"type": event.type}
    if event.type == "delta":
        payload["text"] = event.text
    elif event.type == "sources":
        payload["sources"] = event.sources or []
    elif event.type == "done":
        payload["trace"] = event.trace or {}
    elif event.type == "error":
        payload["message"] = event.message
    return f"data: {json.dumps(payload)}\n\n"
