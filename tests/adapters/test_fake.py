import math

from porsuk.adapters.embedding.fake import FakeEmbedder
from porsuk.adapters.llm.fake import FakeLLM
from porsuk.core.ports import Embedder, LLMProvider
from porsuk.core.registry import build


def test_fake_llm_satisfies_the_port():
    assert isinstance(FakeLLM(model="m"), LLMProvider)


def test_fake_embedder_satisfies_the_port():
    assert isinstance(FakeEmbedder(dim=8), Embedder)


def test_fake_llm_returns_canned_responses_in_order():
    llm = FakeLLM(model="m", responses=["first", "second"])
    assert llm.complete("a") == "first"
    assert llm.complete("b") == "second"


def test_fake_llm_repeats_the_last_response_when_exhausted():
    llm = FakeLLM(model="m", responses=["only"])
    llm.complete("a")
    assert llm.complete("b") == "only"


def test_fake_llm_records_prompts():
    llm = FakeLLM(model="m")
    llm.complete("hello")
    assert llm.prompts == ["hello"]


def test_fake_embedder_is_deterministic():
    e = FakeEmbedder(dim=8)
    assert e.embed_query("merhaba") == e.embed_query("merhaba")


def test_fake_embedder_distinguishes_texts():
    e = FakeEmbedder(dim=8)
    assert e.embed_query("merhaba") != e.embed_query("hello")


def test_fake_embedder_returns_unit_vectors():
    e = FakeEmbedder(dim=8)
    (vector,) = e.embed_query("merhaba").dense
    assert math.isclose(sum(x * x for x in vector), 1.0, rel_tol=1e-9)


def test_fake_embedder_embeds_a_batch():
    e = FakeEmbedder(dim=4)
    result = e.embed_documents(["a", "b", "c"])
    assert len(result.dense) == 3
    assert all(len(v) == 4 for v in result.dense)


def test_fake_embedder_can_skip_normalisation():
    """The flag is part of the embedding fingerprint, so it must really change the output."""
    normalised = FakeEmbedder(dim=8)
    raw = FakeEmbedder(dim=8, normalize=False)
    (unit,) = normalised.embed_query("merhaba").dense
    (plain,) = raw.embed_query("merhaba").dense
    assert math.isclose(sum(x * x for x in unit), 1.0, rel_tol=1e-9)
    assert not math.isclose(sum(x * x for x in plain), 1.0, rel_tol=1e-9)


def test_fakes_are_registered():
    assert isinstance(build("llm", {"provider": "fake", "model": "m"}), FakeLLM)
    assert isinstance(build("embedder", {"provider": "fake", "dim": 8}), FakeEmbedder)
