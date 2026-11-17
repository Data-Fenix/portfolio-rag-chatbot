# Portfolio RAG Assistant

A retrieval-augmented chatbot that answers questions about my work, grounded in
my CV, project READMEs and research notes, and cites its sources. It powers the
"Ask my AI assistant" widget on my portfolio site.

Built with the Anthropic Claude API, FastAPI and a dependency-free BM25
retriever (ChromaDB dense retrieval optional).

```
visitor question
   │
   ▼
guardrails ── length, control chars, prompt-injection patterns, topic scope
   │
   ▼
retrieval ─── BM25 | ChromaDB | hybrid (reciprocal rank fusion), top-k chunks
   │
   ▼
Claude ────── cached system prompt + tagged <context> + tools (list_projects, get_project)
   │              streaming, manual tool loop, refusal / max_tokens handling
   ▼
answer ─────── inline citations [n] + source list + usage/cost trace (JSON logs)
```

## GenAI practices demonstrated

| Practice | Where |
|---|---|
| Grounding and citations: answer only from retrieved passages, cite `[n]` | `app/prompts.py`, `app/engine.py` |
| Retrieved text treated as untrusted data (tagged blocks, explicit instruction) | `app/prompts.py` |
| Input guardrails before any token is spent; output redaction | `app/guardrails.py` |
| Prompt caching: byte-stable system prompt with a cache breakpoint, volatile content in the user turn | `app/prompts.py`, `app/engine.py` |
| Streaming for every call | `app/engine.py`, `app/server.py` (SSE) |
| Tool use with strict JSON schemas, all tool results returned in one message, errors returned as `is_error` | `app/tools.py`, `app/engine.py` |
| Stop-reason handling (`tool_use`, `refusal`, `max_tokens`) and typed error handling | `app/engine.py` |
| Observability: request id, latency, tokens, cache hits, cost per request | `app/observability.py` |
| Evals: deterministic retrieval gate in CI, LLM-as-judge faithfulness and relevance with structured outputs | `evals/` |
| Config via environment, secrets never in code, rate limiting and CORS at the edge | `app/config.py`, `app/server.py` |
| Swappable retriever backend behind one interface | `app/retriever.py` |

## Quick start

```bash
git clone https://github.com/Data-Fenix/portfolio-rag-chatbot
cd portfolio-rag-chatbot
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                             # add your ANTHROPIC_API_KEY

python -m app.cli                                # terminal chat
uvicorn app.server:app --reload --port 8000      # API for the website widget
```

Then open the portfolio site (`python -m http.server 8080` in the portfolio
folder) and the widget will detect the backend at `http://localhost:8000`.

Try with curl:

```bash
curl -N http://localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"message":"What did Lahiru do at Dialog Axiata?"}'
```

### Dense retrieval

```bash
pip install chromadb
RETRIEVER_BACKEND=hybrid python -m app.cli
```

The first run embeds the knowledge base into `.chroma/` using ChromaDB's
default local embedding model.

## Tests and evals

```bash
pytest                                      # unit tests, no network (fake client)
python -m evals.run_evals --retrieval-only  # hit-rate@k and MRR on the golden set
python -m evals.run_evals                   # + LLM-as-judge faithfulness/relevance (needs API key)
```

The golden set lives in `evals/golden_set.jsonl`: one question per line with
the expected source document and facts the answer must contain. Gates:
retrieval hit-rate >= 0.85, average faithfulness and relevance >= 4/5. A
failing gate returns a non-zero exit code so CI blocks the merge. The last
report is written to `evals/last_report.json`.

## Update the knowledge base

Add or edit markdown files in `data/knowledge/`. Each file starts with a
frontmatter block (`title`, `url`) and is chunked on `##` headings. Structured
project data used by the tools lives in `data/projects.json`. Add a golden
question for any new fact you want protected by the evals.

## Deployment

```bash
docker build -t portfolio-rag .
docker run -p 8000:8000 --env-file .env portfolio-rag
```

Set `ALLOWED_ORIGINS` to the portfolio site's origin. The in-memory rate
limiter is per instance; put a shared limiter at the gateway for multi-instance
deployments.

## Cost

With prompt caching the system prompt and tool definitions are served from
cache on repeated requests. A typical question costs well under one cent on
`claude-opus-5-5` at `low` effort; the exact figure is logged per request as
`cost_usd`. Switch `MODEL` to `claude-sonnet-5-5` or `claude-haiku-5-5` for
cheaper serving.
