"""Terminal chat. Run: ``python -m app.cli``"""

from __future__ import annotations

import sys

from .engine import RagEngine


def main() -> None:
    engine = RagEngine()
    history: list[dict] = []
    print("Portfolio assistant. Ask about Lahiru's work. Ctrl+C or 'exit' to quit.\n")
    while True:
        try:
            question = input("you > ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        sys.stdout.write("bot > ")
        answer, sources, trace = [], [], {}
        for ev in engine.chat(question, history):
            if ev.type == "delta":
                sys.stdout.write(ev.text)
                sys.stdout.flush()
                answer.append(ev.text)
            elif ev.type == "sources":
                sources = ev.sources or []
            elif ev.type == "error":
                sys.stdout.write(ev.message)
            elif ev.type == "done":
                trace = ev.trace or {}
        print()
        if sources:
            print("      sources: " + "; ".join(s["title"] for s in sources))
        if trace:
            print(
                f"      [{trace.get('latency_ms')} ms, {trace.get('input_tokens')} in / "
                f"{trace.get('output_tokens')} out, cache read {trace.get('cache_read_tokens')}, "
                f"~${trace.get('cost_usd')}]"
            )
        history += [{"role": "user", "content": question}, {"role": "assistant", "content": "".join(answer)}]


if __name__ == "__main__":
    main()
