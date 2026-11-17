"""Input and output guardrails.

Defence in depth for a public-facing chatbot:

1. **Input validation** - length limits, empty input, control characters.
2. **Prompt-injection screening** - obvious "ignore previous instructions"
   patterns are refused before an API call is made. This is a cheap first
   filter; the real defence is that retrieved content is passed to the model
   as *data* inside tagged blocks and the system prompt tells the model to
   treat it as untrusted.
3. **Scope** - the assistant answers only portfolio questions. Out-of-scope
   requests get a polite, fixed redirect without spending tokens.
4. **Output screening** - the model's reply is checked for accidental leakage
   of the system prompt or of contact details that are not public.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .config import settings

INJECTION_PATTERNS = [
    r"ignore (all |the |any )?(previous|prior|above) (instructions|prompts?|rules)",
    r"disregard (all |the |your )?(previous|prior|system) (instructions|prompt)",
    r"you are now (dan|an? (unfiltered|unrestricted))",
    r"reveal (your|the) (system|hidden|secret) prompt",
    r"print (your|the) (system )?prompt",
    r"\bjailbreak\b",
    r"developer mode",
]
INJECTION_RE = re.compile("|".join(INJECTION_PATTERNS), re.I)

# Topics the assistant should not engage with at all (not a portfolio concern).
OFF_TOPIC_RE = re.compile(
    r"\b(write (me )?(a|an) (poem|story|essay)|medical advice|legal advice|stock tips|"
    r"crypto(currency)? (price|tip)|homework|translate this)\b",
    re.I,
)

PRIVATE_PATTERNS = [
    r"\+?\d[\d\s().-]{8,}\d",  # phone numbers
    r"<system>|</system>|BEGIN SYSTEM PROMPT",
]
PRIVATE_RE = re.compile("|".join(PRIVATE_PATTERNS), re.I)

REFUSAL_INJECTION = (
    "I can only help with questions about Lahiru's projects, experience and skills, "
    "and I don't change my instructions based on chat messages. What would you like to know about his work?"
)
REFUSAL_OFF_TOPIC = (
    "That's outside what I can help with. I answer questions about Lahiru's portfolio: "
    "his MLOps and data platform work, machine learning projects, research and GenAI applications."
)


@dataclass(frozen=True)
class InputCheck:
    ok: bool
    cleaned: str = ""
    reply: str = ""  # canned reply when not ok
    reason: str = ""


def check_input(text: str) -> InputCheck:
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text or "").strip()
    if not cleaned:
        return InputCheck(False, reply="Please type a question about Lahiru's work.", reason="empty")
    if len(cleaned) > settings.max_question_chars:
        return InputCheck(
            False,
            reply=f"Please keep questions under {settings.max_question_chars} characters.",
            reason="too_long",
        )
    if INJECTION_RE.search(cleaned):
        return InputCheck(False, reply=REFUSAL_INJECTION, reason="injection")
    if OFF_TOPIC_RE.search(cleaned):
        return InputCheck(False, reply=REFUSAL_OFF_TOPIC, reason="off_topic")
    return InputCheck(True, cleaned=cleaned)


def sanitize_history(history: list[dict]) -> list[dict]:
    """Keep only well-formed user/assistant string turns, capped in length."""
    out: list[dict] = []
    for turn in history or []:
        role = turn.get("role")
        content = turn.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            continue
        content = content.strip()[: settings.max_question_chars * 4]
        if content:
            out.append({"role": role, "content": content})
    # Keep the last N turns and make sure the sequence alternates and ends with assistant
    out = out[-(settings.max_history_turns * 2):]
    while out and out[0]["role"] != "user":
        out.pop(0)
    while out and out[-1]["role"] != "assistant":
        out.pop()
    return out


def check_output(text: str) -> str:
    """Redact anything that looks like private data or leaked prompt scaffolding."""
    return PRIVATE_RE.sub("[redacted]", text)
