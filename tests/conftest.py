"""Shared fixtures: a fake Anthropic client so tests never hit the network."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.knowledge import load_chunks
from app.retriever import BM25Retriever


class FakeUsage(SimpleNamespace):
    pass


def usage(inp=100, out=50, cache_read=0, cache_write=0):
    return FakeUsage(
        input_tokens=inp, output_tokens=out, cache_read_input_tokens=cache_read, cache_creation_input_tokens=cache_write
    )


class FakeStream:
    """Mimics the context manager returned by client.messages.stream()."""

    def __init__(self, final):
        self.final = final

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    @property
    def text_stream(self):
        for block in self.final.content:
            if block.type == "text":
                # emit in two pieces to exercise concatenation
                mid = len(block.text) // 2
                yield block.text[:mid]
                yield block.text[mid:]

    def get_final_message(self):
        return self.final


def text_block(text):
    return SimpleNamespace(type="text", text=text)


def tool_use_block(name, inputs, id_="toolu_1"):
    return SimpleNamespace(type="tool_use", name=name, input=inputs, id=id_)


def message(content, stop_reason="end_turn", u=None):
    return SimpleNamespace(content=content, stop_reason=stop_reason, usage=u or usage())


class FakeClient:
    """Returns scripted responses in order and records every request."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.messages = self

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("FakeClient ran out of scripted responses")
        return FakeStream(self.responses.pop(0))


@pytest.fixture(scope="session")
def chunks():
    return load_chunks()


@pytest.fixture(scope="session")
def retriever(chunks):
    return BM25Retriever(chunks)


@pytest.fixture
def make_engine(retriever):
    from app.engine import RagEngine

    def _make(responses):
        client = FakeClient(responses)
        return RagEngine(client=client, retriever=retriever), client

    return _make
