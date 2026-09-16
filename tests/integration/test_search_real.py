"""End-to-end search against a live bge-m3 dense + sparse endpoint.

Skipped unless EMBEDDING_BASE_URL and EMBEDDING_SPARSE_BASE_URL are set
(infra/vast_provision.py up writes them). Uses the in-memory Qdrant, so no
Docker is needed - index and search share one App (one client) so the search
sees what the pipeline wrote. No reranker: rerank_enabled is off by default
and the FlagEmbedding box only serves /embed today (see infra/README.md).
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.needs_network

_BASE = os.environ.get("EMBEDDING_BASE_URL")
_SPARSE = os.environ.get("EMBEDDING_SPARSE_BASE_URL")


@pytest.mark.skipif(not (_BASE and _SPARSE), reason="no live bge-m3 dense+sparse endpoint")
def test_all_three_strategies_return_hits(tmp_path):
    from porsuk.core.config import load_config
    from porsuk.core.container import build_app, build_pipeline, build_retriever
    from porsuk.core.models import Filters

    # a tiny corpus of Turkish + English text files, ~4-5 sentences each so
    # the chunk-level language heuristic tags them reliably.
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "tedarik.txt").write_text(
        "Tedarik sözleşmesi bu belgeyle taraflar arasında akdedilmiştir. "
        "Ödeme koşulları şu şekildedir: fatura tarihinden itibaren otuz gün "
        "içinde ödeme yapılır. Gecikme faizi aylık yüzde iki olarak "
        "uygulanır ve her ayın sonunda hesaplanır. Teslimat İstanbul "
        "depolarına yapılır ve taşıma masrafları satıcıya aittir. "
        "Sözleşmenin süresi bir yıldır ve taraflarca yazılı olarak "
        "uzatılabilir.",
        encoding="utf-8",
    )
    (corpus / "kira.txt").write_text(
        "Kira sözleşmesi konut amaçlı olarak düzenlenmiştir. Depozito "
        "tutarı iki aylık kira bedeline eşittir ve sözleşme sonunda iade "
        "edilir. Tahliye taahhüdü noter huzurunda ayrıca verilecektir. "
        "Kiracı aidat ve elektrik giderlerinden sorumludur. Kira artışı "
        "her yıl tüketici fiyat endeksine göre yapılır.",
        encoding="utf-8",
    )
    (corpus / "payment.txt").write_text(
        "Payment terms for this supply agreement are net thirty days from "
        "the invoice date. Late payment interest accrues at two percent per "
        "month and is compounded at the end of each month. Delivery is made "
        "to the buyer's warehouse and freight is paid by the seller. The "
        "agreement runs for one year and may be extended in writing by both "
        "parties.",
        encoding="utf-8",
    )

    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(
        "llm: {provider: fake, model: fake}\n"
        "embedder:\n"
        "  provider: openai_compatible\n"
        "  model: bge-m3\n"
        "  dim: 1024\n"
        f"  base_url: {_BASE}\n"
        f"  sparse_base_url: {_SPARSE}\n"
        "store: {provider: qdrant, url: ':memory:'}\n"
        "parsing: {parsers: [text]}\n"
        "profile: {llm_enabled: false}\n"
        "retrieval: {strategy: hybrid}\n"
    )
    cfg = load_config(cfg_file)

    app = build_app(cfg)  # ONE app -> ONE :memory: Qdrant client
    pipeline = build_pipeline(cfg, str(tmp_path / "state.db"), app=app)
    for _ in pipeline.run(str(corpus)):
        pass

    retriever = build_retriever(cfg, app=app)  # SAME app -> sees indexed data
    q = "ödeme koşulları ve gecikme faizi"

    semantic = retriever.search(q, k=5, filters=Filters(), strategy="semantic")
    keyword = retriever.search(q, k=5, filters=Filters(), strategy="keyword")
    hybrid = retriever.search(q, k=5, filters=Filters(), strategy="hybrid")

    assert semantic, "semantic search returned nothing"
    assert keyword, "keyword search returned nothing"
    assert hybrid, "hybrid search returned nothing"
    assert all(h.retrieval_method == "semantic" for h in semantic)
    assert all(h.retrieval_method == "keyword" for h in keyword)
    assert all(h.retrieval_method == "hybrid" for h in hybrid)
    # the Turkish supply doc (the one that talks about "gecikme faizi") should
    # surface for a Turkish payment query. document_id is a content UUID, not
    # the filename, so match on the chunk text.
    assert any("gecikme faizi" in h.chunk.text.lower() for h in hybrid)

    # language filter actually excludes the English doc
    tr_only = retriever.search(q, k=5, filters=Filters(languages=("tr",)), strategy="semantic")
    assert tr_only
    assert all(h.chunk.language == "tr" for h in tr_only)
