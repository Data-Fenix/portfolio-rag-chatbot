---
title: GenAI projects
url: https://github.com/Data-Fenix/portfolio-rag-chatbot
---

## Portfolio RAG assistant (chatbot)

Repository: https://github.com/Data-Fenix/portfolio-rag-chatbot. A retrieval-augmented generation chatbot that answers questions about Lahiru's projects, experience and skills. Built in Python on the Anthropic Claude API with FastAPI. The assistant works in four steps. First, input guardrails apply length limits, screen for prompt injection and keep questions on topic. Second, it retrieves passages from a markdown knowledge base using BM25 by default, with ChromaDB dense retrieval or hybrid reciprocal-rank fusion as options. Third, Claude answers with a cached system prompt, streaming output and two read-only tools (list_projects and get_project). Finally, it returns a cited answer with its sources. Retrieved passages are passed inside tagged blocks as untrusted data so instructions inside documents are not followed. Every request logs token usage, cache hits and estimated cost as structured JSON. The eval suite measures retrieval hit rate on a golden question set and uses an LLM-as-judge to score faithfulness and relevance; GitHub Actions runs unit tests on every push. It is the assistant embedded in the portfolio website.

## LLM ticket triage pipeline

Repository: https://github.com/Data-Fenix/llm-ticket-triage. A production-style LLMOps pipeline that classifies customer support tickets and extracts structured fields (category, priority, sentiment, product, requested action, summary) using Claude structured outputs validated against Pydantic schemas, so every record is guaranteed valid JSON. Prompts live in a versioned registry; a new prompt version ships only if the evaluation accuracy on a labelled dataset stays above a gate enforced in CI. Bulk processing uses the Message Batches API for 50% lower cost, with per-run cost reports. Includes a Streamlit review UI, a labelled synthetic dataset and confusion-matrix evaluation.

## GenAI foundations

Repository: https://github.com/Data-Fenix/GenAI. Hands-on notebooks covering simple LLM calling, a blood-work health report analysis app built with Streamlit, and vector database basics with ChromaDB and LangChain.

## GenAI best practices Lahiru follows

Ground answers in retrieved data and cite sources; treat retrieved content as untrusted; keep system prompts stable to benefit from prompt caching; stream responses; enforce output schemas with structured outputs; version prompts and gate releases with evals; log tokens and cost per request; apply rate limits and input validation at the API boundary; keep secrets in environment variables, never in code.
