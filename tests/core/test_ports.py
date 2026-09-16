"""Tests that ports are structural (runtime-checkable) protocols."""

from porsuk.core.models import EmbedResult
from porsuk.core.ports import Embedder, LLMProvider, Parser, VectorStore


class _StructuralEmbedder:
    """Satisfies Embedder without inheriting from it."""

    dim = 3

    def embed_documents(self, texts: list[str]) -> EmbedResult:
        return EmbedResult(dense=tuple((0.0, 0.0, 0.0) for _ in texts))

    def embed_query(self, text: str) -> EmbedResult:
        return EmbedResult(dense=((0.0, 0.0, 0.0),))


class _NotAnEmbedder:
    dim = 3


def test_structural_conformance_needs_no_inheritance():
    embedder = _StructuralEmbedder()
    assert isinstance(embedder, Embedder)
    assert not issubclass(_StructuralEmbedder, type(Embedder))


def test_missing_methods_fail_the_protocol_check():
    assert not isinstance(_NotAnEmbedder(), Embedder)


def test_all_ports_are_runtime_checkable():
    for port in (LLMProvider, Embedder, VectorStore, Parser):
        assert getattr(port, "_is_runtime_protocol", False), port
