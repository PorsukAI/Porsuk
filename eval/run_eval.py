"""Run the retrieval strategies over the golden sets: keyword / semantic /
hybrid are measured separately and broken down by question type, since the
answer "which strategy" differs for specific-term vs. conceptual questions.
That table feeds two decisions, the reranker default and whether hybrid
stays in the agent's tool set. It is eval-only, `porsuk/` never imports it.
Every goldset is indexed into its own fresh `:memory:` Qdrant (see
`_isolated`), so RAGTurk's CC-BY-NC-SA chunks, the XQuAD English contexts and
the mevzuat corpus never touch the product store or each other.
"""

from __future__ import annotations

import argparse
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from eval.goldsets import (
    GoldQuestion,
    load_manual,
    load_ragturk,
    load_ragturk_corpus,
    load_xquad_en_corpus,
    load_xquad_tr,
    resolve_manual_doc_ids,
)
from eval.metrics import mrr, ndcg_at_k, recall_at_k
from porsuk.core.models import Filters


@dataclass(frozen=True)
class Result:
    goldset: str
    strategy: str
    question_type: str
    n: int
    recall_at_5: float
    mrr: float
    ndcg_at_5: float
    params: dict = field(default_factory=dict)


def _isolated(cfg):
    """A copy of `cfg` whose store is a fresh in-process Qdrant.

    The embedder / llm / reranker sections are untouched, the real bge-m3
    endpoint is still used. Only the vector store is swapped, so an eval run
    never writes into the product corpus. Call it once per
    goldset: three separate `:memory:` stores keep the three eval corpora
    isolated from each other too.
    """
    return cfg.model_copy(
        update={"store": cfg.store.model_copy(update={"url": ":memory:", "path": None})}
    )


def _ids(hits, level: str) -> list[str]:
    """The ranked id list the metrics score, deduped, first occurrence wins.

    At `level="doc"` the top-k chunks routinely share a document, so an
    un-deduped list repeats the same doc id: `recall_at_k` would then count
    one relevant doc several times and sail past 1.0, and `ndcg` would
    double-count it. `dict.fromkeys` keeps the first-seen order.
    """
    return list(
        dict.fromkeys(h.chunk.document_id if level == "doc" else h.chunk.chunk_id for h in hits)
    )


def evaluate(
    retriever,
    questions: list[GoldQuestion],
    *,
    strategy: str,
    k: int,
    level: str,
    filter_language: bool = True,
    params: dict | None = None,
) -> list[Result]:
    """Run each question through `retriever.search(..., strategy=strategy)`,
    map hits to `level` ("doc" -> document_id, "chunk" -> chunk_id), compute
    the three metrics per question, aggregate by `question_type` plus an
    "all" row.

    `filter_language`: normally the search is filtered to the question's
    language. The XQuAD cross-lingual probe asks a Turkish question
    against English contexts, so `main()` passes `filter_language=False` there:
    a `languages=("tr",)` filter would exclude every English context and the
    probe would always score 0.

    `params`: the config knobs that produced these numbers (chunk_k, rerank,
    neighbour expansion, stemmer, strategy), stamped onto every `Result` so a
    later run can be compared to this one. `k` must be at least 5: the @5
    metrics need at least that many candidates to be meaningful.
    """
    assert k >= 5, f"k={k} < 5: the @5 metrics would see fewer candidates than the cutoff"
    params = dict(params or {})
    # search wider than the metric cutoff: doc-level dedup collapses many
    # chunks to few docs, so k chunks can yield fewer than 5 distinct docs.
    search_k = max(k, 20)
    buckets: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for q in questions:
        relevant = q.relevant_doc_ids if level == "doc" else q.relevant_chunk_ids
        filters = Filters(languages=(q.language,)) if filter_language else Filters()
        hits = retriever.search(q.question, k=search_k, filters=filters, strategy=strategy)
        ranked = _ids(hits, level)
        triple = (
            recall_at_k(ranked, set(relevant), 5),
            mrr(ranked, set(relevant)),
            ndcg_at_k(ranked, set(relevant), 5),
        )
        buckets["all"].append(triple)
        buckets[q.question_type].append(triple)

    source = questions[0].source if questions else "?"
    out: list[Result] = []
    for qtype, triples in buckets.items():
        n = len(triples)
        out.append(
            Result(
                goldset=source,
                strategy=strategy,
                question_type=qtype,
                n=n,
                recall_at_5=sum(t[0] for t in triples) / n,
                mrr=sum(t[1] for t in triples) / n,
                ndcg_at_5=sum(t[2] for t in triples) / n,
                params=params,
            )
        )
    return out


def _index_tuples_and_build_app(
    cfg, tuples: list[tuple[str, str, str]], *, collection: str, language: str
):
    """Index already-chunked corpus tuples (doc_id, chunk_id, text) into a
    store isolated from the product corpus and return the built
    `App`. RAGTurk / XQuAD arrive pre-chunked, no pipeline parse. The caller
    turns the App into per-config `Retriever`s (the keyword sweep reuses
    one index for four keyword configs).

    Isolation: the caller passes a `cfg` from `_isolated(...)`, whose
    `store.url` is `":memory:"`, so `build_app` creates a fresh in-process
    Qdrant per call that never touches the product corpus. The `collection`
    param is kept because it documents intent; it is only used in
    the log line here. If a later change wants a named collection on a real
    Qdrant instead, vary `cfg.store` before the call, the retriever it
    returns does not care.
    """
    from porsuk.core.container import build_app
    from porsuk.core.models import Chunk, EmbedResult

    app = build_app(cfg)  # cfg.store.url == ":memory:" -> fresh isolated Qdrant
    chunks, texts = [], []
    for doc_id, chunk_id, text in tuples:
        chunks.append(Chunk(chunk_id, doc_id, text, None, language, None, None, (0, len(text))))
        texts.append(text)

    # embed in batches to stay under the endpoint's limits
    BATCH = 64
    dense, sparse = [], []
    for i in range(0, len(texts), BATCH):
        res = app.embedder.embed_documents(texts[i : i + BATCH])
        dense.extend(res.dense)
        if res.sparse is not None:
            sparse.extend(res.sparse)
    vectors = EmbedResult(dense=tuple(dense), sparse=tuple(sparse) if sparse else None)
    app.store.upsert_chunks(chunks, vectors)
    print(f"  indexed {len(chunks)} chunks into isolated store ({collection})")
    return app


def _params_of(cfg, strategy: str) -> dict[str, object]:
    r = cfg.retrieval
    return {
        "chunk_k": r.chunk_k,
        "rerank_enabled": r.rerank_enabled,
        "neighbor_expansion": r.neighbor_expansion,
        "stemmer": r.stemmer_enabled,
        "keyword_backend": r.keyword_backend,  # text | bm25 | sparse
        "strategy": strategy,
    }


# The keyword strategy is swept over its engine: plain
# word membership (`text`), classic BM25, and the bge-m3 learned sparse vector,
# and the Turkish stemmer. semantic / hybrid run once, with whatever
# keyword_backend the config carries.
_KEYWORD_SWEEP: tuple[tuple[str, bool], ...] = (
    ("text", False),
    ("text", True),
    ("bm25", False),
    ("bm25", True),
    ("sparse", False),
    ("sparse", True),
)


def _retriever_over(app, cfg):
    """A `Retriever` reading `cfg.retrieval` over an already-built `app`'s index."""
    from porsuk.core.container import build_reranker
    from porsuk.retrieval.search import Retriever

    return Retriever(
        embedder=app.embedder,
        store=app.store,
        reranker=build_reranker(cfg),
        cfg=cfg.retrieval,
    )


def _run_strategies(app, cfg, questions, *, k, level, filter_language=True):
    """Evaluate semantic + hybrid once and keyword across its engine sweep.

    One built index (`app`), four keyword configs plus semantic/hybrid. The
    keyword sweep varies `keyword_backend` (BM25 vs the shelved bge-m3 sparse)
    and the Turkish stemmer; semantic/hybrid do not touch either.
    """
    out: list[Result] = []
    base = _retriever_over(app, cfg)
    for strategy in ("semantic", "hybrid"):
        out += evaluate(
            base, questions, strategy=strategy, k=k, level=level,
            filter_language=filter_language, params=_params_of(cfg, strategy),
        )
    for backend, stem in _KEYWORD_SWEEP:
        variant = cfg.model_copy(
            update={
                "retrieval": cfg.retrieval.model_copy(
                    update={"keyword_backend": backend, "stemmer_enabled": stem}
                )
            }
        )
        try:
            out += evaluate(
                _retriever_over(app, variant), questions, strategy="keyword", k=k,
                level=level, filter_language=filter_language,
                params=_params_of(variant, "keyword"),
            )
        except Exception as exc:  # noqa: BLE001 - a sparse-less index skips that half
            print(f"keyword sweep {backend}/stem={stem} skipped: {exc}")
    return out


def _write_table(results: list[Result], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y-%m-%d")
    path = out_dir / f"{stamp}-retrieval.md"
    lines = [f"# Retrieval evaluation: {stamp}\n"]
    by_set: dict[str, list[Result]] = defaultdict(list)
    for r in results:
        by_set[r.goldset].append(r)
    # Params that are the same across every row are printed once above the
    # table; the ones that vary (keyword_backend and stemmer are swept)
    # become table columns.
    _SWEEP_KEYS = ("keyword_backend", "stemmer")
    for goldset, rows in by_set.items():
        lines.append(f"\n## {goldset}\n")
        all_params = [r.params for r in rows if r.params]
        if all_params:
            fixed = {
                k: v
                for k, v in sorted(all_params[0].items())
                if k not in {"strategy", *_SWEEP_KEYS}
                and all(p.get(k) == v for p in all_params)
            }
            lines.append(
                "**params:** " + ", ".join(f"{k}={v}" for k, v in fixed.items()) + "\n"
            )
        lines.append(
            "| strategy | backend | stemmer | question type | n | recall@5 | MRR | nDCG@5 |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in sorted(
            rows,
            key=lambda x: (
                x.strategy,
                str(x.params.get("keyword_backend", "")),
                str(x.params.get("stemmer", "")),
                x.question_type,
            ),
        ):
            backend = r.params.get("keyword_backend", "-") if r.strategy == "keyword" else "-"
            stemmer = r.params.get("stemmer", "-") if r.strategy == "keyword" else "-"
            lines.append(
                f"| {r.strategy} | {backend} | {stemmer} | {r.question_type} | {r.n} | "
                f"{r.recall_at_5:.3f} | {r.mrr:.3f} | {r.ndcg_at_5:.3f} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/vllm.yaml")
    ap.add_argument("--manual-corpus", default="tests/fixtures/downloaded/mevzuat")
    ap.add_argument("--ragturk-limit", type=int, default=300)
    ap.add_argument("--xquad-limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("docs/eval"))
    args = ap.parse_args()

    from porsuk.core.config import load_config
    from porsuk.core.container import build_app, build_pipeline

    cfg = load_config(args.config)
    results: list[Result] = []
    ran: list[str] = []

    # --- manual set: real parse of the mevzuat PDFs ----------------------
    # Its own isolated :memory: store. The pipeline and the retriever share
    # ONE App so the retriever searches the same index the pipeline built.
    # A fresh state db per run: the
    # pipeline skips docs whose content_hash is unchanged, so a stale db
    # would embed nothing and zero the manual metrics with no error.
    try:
        mem = _isolated(cfg)
        state_path = str(Path(tempfile.mkdtemp(prefix="porsuk_eval_")) / "manual.db")
        app = build_app(mem)
        pipe = build_pipeline(mem, state_path, app=app)
        for _ in pipe.run(args.manual_corpus):
            pass
        # The manual YAML keys gold docs by filename; document_id_for hashes
        # the resolved path, so resolve against the SAME dir just indexed.
        manual_qs = resolve_manual_doc_ids(load_manual(), args.manual_corpus)
        results += _run_strategies(app, mem, manual_qs, k=cfg.retrieval.chunk_k, level="doc")
        ran.append("manual")
    except Exception as exc:  # noqa: BLE001 - one goldset failing must not sink the rest
        print(f"manual goldset failed, skipping: {exc}")

    # --- RAGTurk: its own corpus, its own isolated store ------------------
    # load_ragturk_corpus(limit) yields (doc_id, chunk_id, text) for the SAME
    # articles load_ragturk(limit) draws questions from. Evaluate at doc level
    # (the article) AND chunk level (the gold related_chunk_ids).
    try:
        rag_qs = load_ragturk(limit=args.ragturk_limit)
        rag_corpus = load_ragturk_corpus(limit=args.ragturk_limit)
        rag_app = _index_tuples_and_build_app(
            _isolated(cfg), rag_corpus, collection="ragturk", language="tr"
        )
        for level in ("doc", "chunk"):
            results += _run_strategies(
                rag_app, cfg, rag_qs, k=cfg.retrieval.chunk_k, level=level
            )
        ran.append("ragturk")
    except Exception as exc:  # noqa: BLE001 - 429s must not sink the manual numbers
        print(f"ragturk goldset failed, skipping: {exc}")

    # --- XQuAD-tr cross-lingual probe -------------------------------------
    # Index the ENGLISH contexts, ask the TURKISH questions; a hit is the
    # parallel context (same id). filter_language=False: the question is tr,
    # the target document en, so a language filter would zero the probe.
    try:
        xq_qs = load_xquad_tr(limit=args.xquad_limit)
        xq_corpus = load_xquad_en_corpus(limit=args.xquad_limit)
        xq_app = _index_tuples_and_build_app(
            _isolated(cfg), xq_corpus, collection="xquad_en", language="en"
        )
        results += _run_strategies(
            xq_app, cfg, xq_qs, k=cfg.retrieval.chunk_k, level="doc", filter_language=False
        )
        ran.append("xquad_tr")
    except Exception as exc:  # noqa: BLE001 - XQuAD unreachable must not sink the rest
        print(f"xquad goldset failed, skipping: {exc}")

    print(f"goldsets that ran: {', '.join(ran) or 'none'}")
    print(f"wrote {_write_table(results, args.out)}")


if __name__ == "__main__":
    main()
