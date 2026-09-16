"""The one place dependencies are assembled.

Importing the adapter packages here is what runs their @register decorators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from porsuk.core.config import Config
from porsuk.core.registry import build
from porsuk.ingestion.router import ParserRouter


def _load_adapters() -> None:
    """Import for registration side effects only."""
    from porsuk.adapters.embedding import fake as _embedding_fake  # noqa: F401
    from porsuk.adapters.embedding import openai_compatible as _embedding_openai  # noqa: F401
    from porsuk.adapters.llm import fake as _llm_fake  # noqa: F401
    from porsuk.adapters.llm import langgraph_chat as _llm_langgraph_chat  # noqa: F401
    from porsuk.adapters.llm import openai_compatible as _llm_openai  # noqa: F401

    # ocr and docling each register either the real parser or, without the
    # `parsers` extra, the stub from unavailable.py: one registration site
    # per name, so the order ruff sorts these imports into cannot decide
    # which one wins.
    from porsuk.adapters.parsers import docling as _parser_docling  # noqa: F401
    from porsuk.adapters.parsers import docx as _parser_docx  # noqa: F401
    from porsuk.adapters.parsers import ocr as _parser_ocr  # noqa: F401
    from porsuk.adapters.parsers import pptx as _parser_pptx  # noqa: F401
    from porsuk.adapters.parsers import pymupdf as _parser_pymupdf  # noqa: F401
    from porsuk.adapters.parsers import text as _parser_text  # noqa: F401
    from porsuk.adapters.parsers import xlsx as _parser_xlsx  # noqa: F401
    from porsuk.adapters.rerank import fake as _rerank_fake  # noqa: F401
    from porsuk.adapters.rerank import flag_embedding_http as _rerank_http  # noqa: F401
    from porsuk.adapters.rerank import openrouter as _rerank_openrouter  # noqa: F401
    from porsuk.adapters.store import inmemory as _store_inmemory  # noqa: F401
    from porsuk.adapters.store import qdrant as _store_qdrant  # noqa: F401


def embedding_fingerprint(cfg: Config) -> str:
    """Identity of the vectors in an index.

    Vectors from different models are not comparable, so this string is
    written into the store and checked on every open. Normalisation is part
    of the identity, not a detail: the same model with normalisation on and
    off produces vectors that cannot be compared, and a store scoring by dot
    product would rank them differently while reporting no error.

    Whether sparse (lexical) weights were produced is part of the identity
    too: an index built without them has no sparse vectors to search, so
    opening it with sparse configured must raise rather than silently return
    only dense hits from the keyword route.

    `keyword_backend` is deliberately NOT part of the identity:
    `upsert_chunks` writes both the bge-m3 "sparse" vector (when the embedder
    produced it) and the "bm25" term-frequency vector on every point, so one
    index serves both keyword routes and switching `keyword_backend` needs no
    re-index. This is what lets a production A/B run compare them on the same
    index.
    """
    return (
        f"{cfg.embedder.provider}:{cfg.embedder.model or 'default'}"
        f":{cfg.embedder.dim}:norm={cfg.embedder.normalize}"
        f":sparse={cfg.embedder.sparse_base_url is not None}"
    )


@dataclass(frozen=True)
class App:
    config: Config
    llm: Any
    embedder: Any
    store: Any
    router: Any


def build_router(cfg: Config) -> ParserRouter:
    """Assemble the parser chain named in config.

    The chain comes from config rather than from a hardcoded list so that this
    stays free of provider branching: the registry resolves each
    name, and the router sorts by declared cost.
    """
    _load_adapters()
    parsers = [build("parser", {"provider": name}) for name in cfg.parsing.parsers]
    return ParserRouter(parsers, cfg.quality)


def build_app(cfg: Config) -> App:
    _load_adapters()
    llm = build("llm", cfg.llm.model_dump(exclude_none=True, exclude={"api_key_env"}))
    embedder = build(
        "embedder", cfg.embedder.model_dump(exclude_none=True, exclude={"api_key_env"})
    )
    store = build("store", _store_params(cfg))
    store.ensure_collections(embedding_fingerprint=embedding_fingerprint(cfg))
    return App(config=cfg, llm=llm, embedder=embedder, store=store, router=build_router(cfg))


def _store_params(cfg: Config) -> dict[str, Any]:
    """Store config plus the embedder's dimension.

    A vector store's dimension is the embedder's, not an independent value -
    the fingerprint guards that they agree. It is passed through
    `dim`, which qdrant takes and inmemory ignores.
    """
    params = cfg.store.model_dump(exclude_none=True)
    if cfg.embedder.dim is not None:
        params.setdefault("dim", cfg.embedder.dim)
    return params


def build_reranker(cfg: Config):
    """Build the reranker named in config, or None when reranking is off.

    Returned type is the `Reranker` port; the import is lazy to match
    `build_pipeline` and to keep this module's top level off `ports`.
    """
    if cfg.retrieval.rerank is None:
        return None
    params = cfg.retrieval.rerank.model_dump(exclude_none=True, exclude={"api_key_env"})
    if "base_url" not in params:
        from porsuk.core.config import ConfigError

        raise ConfigError(
            "retrieval.rerank names a provider but no base_url resolved "
            "(RERANK_BASE_URL unset?); set it or remove the rerank: block"
        )
    _load_adapters()
    return build("reranker", params)


def build_chat_model(cfg: Config):
    """The LangChain chat model behind the agent, via the
    langgraph_chat adapter. Returns the adapter object; `.chat` is the model
    `create_agent` binds tools to.

    `AgentConfig` also carries the loop parameters (`max_tool_calls`, the
    `expand_*` window, `max_context_chars`); those belong to `build_tools`
    and the loop, not the chat model, so they are excluded from the dump.
    `api_key_env` is resolved into `api_key` by `load_config`.
    """
    _load_adapters()
    params = cfg.agent.model_dump(
        exclude_none=True,
        # denylist: a new AgentConfig field must be added here or accepted
        # by the adapter (FakeChatModel swallows unknown kwargs).
        exclude={
            "api_key_env",
            "max_tool_calls",
            "max_context_chars",
            "expand_before_chars",
            "expand_after_chars",
            "context_trim_tokens",
        },
    )
    return build("llm", params)


def build_retriever(cfg: Config, *, app: App | None = None):
    """Assemble the retrieval engine on top of a built app.

    The import is lazy to match `build_pipeline` and keep this module's top
    level off `retrieval/`. Pass `app` to share one App, and thus one
    `:memory:` Qdrant client, with `build_pipeline`; separate CLI processes
    with a real Qdrant URL do not need this (the server persists).
    """
    from porsuk.retrieval.search import Retriever

    app = app or build_app(cfg)
    return Retriever(
        embedder=app.embedder,
        store=app.store,
        reranker=build_reranker(cfg),
        cfg=cfg.retrieval,
    )


def build_agent_runner(cfg: Config, *, app: App | None = None):
    """The agent entry point on top of a built app. Lazy import
    keeps this module's top level off langgraph."""
    from porsuk.agent.run import run_agent

    shared = app or build_app(cfg)

    def run(
        question: str,
        *,
        lang: str | None = None,
        verbose: bool = False,
        on_event=None,
        stop=None,
        scope: str | None = None,
        history: list[dict[str, str]] | None = None,
    ):
        return run_agent(
            question,
            cfg=cfg,
            lang=lang,
            verbose=verbose,
            on_event=on_event,
            stop=stop,
            app=shared,
            scope=scope,
            history=history,
        )

    return run


def build_pipeline(
    cfg: Config, state_path: str, *, app: App | None = None, job_id: str | None = None
):
    """Assemble the ingestion pipeline on top of a built app.

    Pass `app` to share one App, and thus one `:memory:` Qdrant client,
    with `build_retriever`; separate CLI processes with a real Qdrant URL do
    not need this (the server persists).
    """
    from porsuk.core.state import StateStore
    from porsuk.ingestion.pipeline import Pipeline

    app = app or build_app(cfg)
    # The profiler can name its own LLM (a cheap
    # model for summaries, not the agent model). If it does, use it;
    # otherwise the profiler shares the agent LLM.
    profile_llm = app.llm
    if cfg.profile.llm is not None:
        profile_llm = build(
            "llm", cfg.profile.llm.model_dump(exclude_none=True, exclude={"api_key_env"})
        )
    return Pipeline(
        router=app.router,
        embedder=app.embedder,
        store=app.store,
        llm=profile_llm,
        state=StateStore(state_path),
        chunking_cfg=cfg.chunking,
        profile_cfg=cfg.profile,
        parse_workers=cfg.pipeline.parse_workers,
        embed_batch_size=cfg.pipeline.embed_batch_size,
        job_id=job_id,
    )
