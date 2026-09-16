"""Tests for the dependency-injection container (`porsuk.core.container`)."""

import pytest

from porsuk.adapters.embedding.fake import FakeEmbedder
from porsuk.adapters.llm.fake import FakeLLM
from porsuk.adapters.store.inmemory import FingerprintMismatchError, InMemoryStore
from porsuk.core.config import AgentConfig, load_config
from porsuk.core.container import (
    build_app,
    build_chat_model,
    build_pipeline,
    build_retriever,
    embedding_fingerprint,
)


def test_build_chat_model_accepts_every_agent_config_field():
    """Regression: `build_chat_model`'s denylist must be kept in sync with
    `AgentConfig`: a new field that isn't excluded gets forwarded straight
    into `LangGraphChatModel.__init__`'s strict kwargs and blows up at
    request time (not at config-load time, since `FakeChatModel` swallows
    unknown kwargs and every other test path uses that fake). This exercises
    the REAL `langgraph_chat` provider: `ChatOpenAI()` builds without any
    network call, so this stays network-free."""
    cfg = load_config("config/local.yaml")
    cfg = cfg.model_copy(
        update={
            "agent": AgentConfig(
                provider="langgraph_chat",
                model="m",
                base_url="http://localhost:1234/v1",
            )
        }
    )
    build_chat_model(cfg)  # must not raise TypeError for an unexpected kwarg


def test_build_app_wires_every_dependency_from_a_profile():
    app = build_app(load_config("config/local.yaml"))
    assert isinstance(app.llm, FakeLLM)
    assert isinstance(app.embedder, FakeEmbedder)
    assert isinstance(app.store, InMemoryStore)


def test_build_app_stamps_the_fingerprint_on_the_store():
    cfg = load_config("config/local.yaml")
    app = build_app(cfg)
    assert app.store.stored_fingerprint() == embedding_fingerprint(cfg)


def test_fingerprint_includes_provider_model_and_dim():
    cfg = load_config("config/local.yaml")
    fingerprint = embedding_fingerprint(cfg)
    assert "fake" in fingerprint
    assert "8" in fingerprint


def test_fingerprint_changes_when_the_dimension_changes(tmp_path):
    profile = tmp_path / "other.yaml"
    profile.write_text(
        "llm:\n  provider: fake\n  model: m\n"
        "embedder:\n  provider: fake\n  dim: 16\n"
        "store:\n  provider: inmemory\n",
        encoding="utf-8",
    )
    a = embedding_fingerprint(load_config("config/local.yaml"))
    b = embedding_fingerprint(load_config(profile))
    assert a != b


def test_fingerprint_includes_the_normalisation_flag():
    cfg = load_config("config/local.yaml")
    assert "norm=True" in embedding_fingerprint(cfg)


def test_fingerprint_folds_in_whether_sparse_is_configured():
    cfg = load_config("config/local.yaml")
    assert embedding_fingerprint(cfg).endswith(":sparse=False")


def test_fingerprint_does_not_depend_on_keyword_backend():
    # one index carries both the "sparse" and "bm25" vectors, so
    # switching keyword_backend must NOT force a re-index.
    base = load_config("config/local.yaml")
    bm = base.model_copy(
        update={"retrieval": base.retrieval.model_copy(update={"keyword_backend": "bm25"})}
    )
    sp = base.model_copy(
        update={"retrieval": base.retrieval.model_copy(update={"keyword_backend": "sparse"})}
    )
    assert embedding_fingerprint(bm) == embedding_fingerprint(sp)


def test_a_dense_only_index_opened_with_sparse_configured_raises(tmp_path):
    """An index built without a sparse endpoint has no sparse vectors;
    opening it with one configured must raise, not silently degrade."""
    dense_only = tmp_path / "dense.yaml"
    dense_only.write_text(
        "llm:\n  provider: fake\n  model: m\n"
        "embedder:\n  provider: fake\n  dim: 8\n"
        "store:\n  provider: inmemory\n",
        encoding="utf-8",
    )
    with_sparse = tmp_path / "sparse.yaml"
    with_sparse.write_text(
        "llm:\n  provider: fake\n  model: m\n"
        "embedder:\n  provider: fake\n  dim: 8\n"
        "  sparse_base_url: http://localhost:19000\n"
        "store:\n  provider: inmemory\n",
        encoding="utf-8",
    )
    store = InMemoryStore()
    store.ensure_collections(embedding_fingerprint=embedding_fingerprint(load_config(dense_only)))
    with pytest.raises(FingerprintMismatchError):
        store.ensure_collections(
            embedding_fingerprint=embedding_fingerprint(load_config(with_sparse))
        )


def test_build_pipeline_and_retriever_can_share_an_app(tmp_path):
    """index and search over an in-memory store only see the same data if they
    share one App (one Qdrant client). Both builders accept an optional `app`."""
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(
        "llm: {provider: fake, model: fake}\n"
        "embedder: {provider: fake, dim: 8}\n"
        "store: {provider: qdrant, url: ':memory:'}\n",
        encoding="utf-8",
    )
    cfg = load_config(cfg_file)
    app = build_app(cfg)
    pipeline = build_pipeline(cfg, str(tmp_path / "s.db"), app=app)
    retriever = build_retriever(cfg, app=app)
    assert pipeline._store is retriever._store
    assert pipeline._embedder is retriever._embedder


def test_build_pipeline_passes_job_id_through(tmp_path):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(
        "llm: {provider: fake, model: fake}\n"
        "embedder: {provider: fake, dim: 8}\n"
        "store: {provider: inmemory}\n",
        encoding="utf-8",
    )
    cfg = load_config(cfg_file)
    app = build_app(cfg)
    pipeline = build_pipeline(cfg, str(tmp_path / "s.db"), app=app, job_id="j1")
    assert pipeline._job_id == "j1"

    default = build_pipeline(cfg, str(tmp_path / "s2.db"), app=app)
    assert default._job_id is None


def test_build_agent_runner_returns_a_callable():
    from porsuk.core.container import build_agent_runner

    cfg = load_config("config/local.yaml")
    run = build_agent_runner(cfg)
    assert callable(run)


def test_build_reranker_rejects_a_rerank_block_with_no_base_url(tmp_path):
    from porsuk.core.config import ConfigError, load_config
    from porsuk.core.container import build_reranker

    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(
        "llm: {provider: fake, model: fake}\n"
        "embedder: {provider: fake, dim: 8}\n"
        "store: {provider: inmemory}\n"
        "retrieval:\n"
        "  strategy: hybrid\n"
        "  rerank: {provider: flag_embedding_http, base_url: '${RERANK_BASE_URL:-}'}\n"
    )
    cfg = load_config(cfg_file, env={})
    with pytest.raises(ConfigError, match="base_url"):
        build_reranker(cfg)


def test_fingerprint_changes_when_normalisation_changes(tmp_path):
    profile = tmp_path / "unnormalised.yaml"
    profile.write_text(
        "llm:\n  provider: fake\n  model: m\n"
        "embedder:\n  provider: fake\n  dim: 8\n  normalize: false\n"
        "store:\n  provider: inmemory\n",
        encoding="utf-8",
    )
    a = embedding_fingerprint(load_config("config/local.yaml"))
    b = embedding_fingerprint(load_config(profile))
    assert a != b
