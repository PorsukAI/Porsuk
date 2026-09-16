"""Tests for the retrieval-strategy evaluation runner."""

from eval.goldsets import GoldQuestion
from eval.run_eval import Result, _ids, _isolated, _params_of, _write_table, evaluate


class _StubRetriever:
    """Returns a fixed ranking regardless of query, for metric plumbing tests."""

    def __init__(self, ranking):
        self._ranking = ranking

    def search(self, query, *, k, filters, strategy=None):
        from porsuk.core.models import Chunk, ChunkHit

        return [
            ChunkHit(
                chunk=Chunk(cid, cid.split("::")[0], "t", 1, "tr", None, None, (0, 1)),
                score=1.0 - i * 0.1,
                retrieval_method=strategy or "semantic",
            )
            for i, cid in enumerate(self._ranking[:k])
        ]


def test_evaluate_computes_doc_level_metrics():
    q = GoldQuestion(
        question="x",
        relevant_doc_ids=frozenset({"d1"}),
        relevant_chunk_ids=frozenset({"d1::c0"}),
        question_type="specific",
        language="tr",
        source="manual",
    )
    r = _StubRetriever(["d2::c0", "d1::c0", "d3::c0"])
    (all_row, type_row) = sorted(
        evaluate(r, [q], strategy="semantic", k=5, level="doc"),
        key=lambda x: x.question_type,
    )
    # d1 is at rank 2 -> MRR 0.5, recall@5 1.0
    assert all_row.mrr == 0.5
    assert all_row.recall_at_5 == 1.0


def test_evaluate_breaks_down_by_question_type():
    qs = [
        GoldQuestion("a", frozenset({"d1"}), frozenset(), "specific", "tr", "manual"),
        GoldQuestion("b", frozenset({"d2"}), frozenset(), "conceptual", "tr", "manual"),
    ]
    r = _StubRetriever(["d1::c0", "d2::c0"])
    rows = evaluate(r, qs, strategy="semantic", k=5, level="doc")
    types = {row.question_type for row in rows}
    assert types == {"all", "specific", "conceptual"}


def test_ids_dedups_preserving_rank_order():
    from porsuk.core.models import Chunk, ChunkHit

    def _hit(cid):
        return ChunkHit(
            chunk=Chunk(cid, cid.split("::")[0], "t", 1, "tr", None, None, (0, 1)),
            score=1.0,
            retrieval_method="semantic",
        )

    hits = [_hit(c) for c in ["d1::c0", "d1::c1", "d2::c0", "d1::c2", "d3::c0"]]
    assert _ids(hits, "doc") == ["d1", "d2", "d3"]
    assert _ids(hits, "chunk") == ["d1::c0", "d1::c1", "d2::c0", "d1::c2", "d3::c0"]


def test_evaluate_recall_never_exceeds_one():
    q = GoldQuestion("a", frozenset({"d1"}), frozenset(), "specific", "tr", "manual")
    # 10 chunks, all from d1: an un-deduped ranked list would count d1 ten times
    r = _StubRetriever([f"d1::c{i}" for i in range(10)])
    rows = evaluate(r, [q], strategy="semantic", k=5, level="doc")
    all_row = next(row for row in rows if row.question_type == "all")
    assert all_row.recall_at_5 == 1.0
    assert all_row.mrr == 1.0


def test_evaluate_chunk_level():
    q = GoldQuestion("a", frozenset({"d1"}), frozenset({"d1::c0"}), "specific", "tr", "ragturk")
    r = _StubRetriever(["d1::c1", "d1::c0"])
    rows = evaluate(r, [q], strategy="semantic", k=5, level="chunk")
    all_row = next(row for row in rows if row.question_type == "all")
    # gold chunk d1::c0 is at rank 2 -> MRR 0.5
    assert all_row.mrr == 0.5


class _RecordingRetriever:
    """Records the `filters` it was last called with."""

    def __init__(self):
        self.last_filters = None

    def search(self, query, *, k, filters, strategy=None):
        self.last_filters = filters
        return []


def test_evaluate_can_disable_language_filter():
    q = GoldQuestion("a", frozenset({"d1"}), frozenset(), "crosslingual", "tr", "xquad_tr")
    r = _RecordingRetriever()
    evaluate(r, [q], strategy="semantic", k=5, level="doc", filter_language=False)
    assert r.last_filters.languages == ()


def test_evaluate_language_filter_on_by_default():
    q = GoldQuestion("a", frozenset({"d1"}), frozenset(), "specific", "tr", "manual")
    r = _RecordingRetriever()
    evaluate(r, [q], strategy="semantic", k=5, level="doc")
    assert r.last_filters.languages == ("tr",)


def test_evaluate_stamps_params_on_results():
    q = GoldQuestion("a", frozenset({"d1"}), frozenset(), "specific", "tr", "manual")
    r = _RecordingRetriever()
    rows = evaluate(
        r,
        [q],
        strategy="hybrid",
        k=5,
        level="doc",
        params={"chunk_k": 10, "strategy": "hybrid"},
    )
    assert all(row.params == {"chunk_k": 10, "strategy": "hybrid"} for row in rows)


def test_evaluate_rejects_k_below_metric_cutoff():
    import pytest

    q = GoldQuestion("a", frozenset({"d1"}), frozenset(), "specific", "tr", "manual")
    with pytest.raises(AssertionError):
        evaluate(_RecordingRetriever(), [q], strategy="semantic", k=3, level="doc")


def test_isolated_swaps_only_the_store(tmp_path):
    from porsuk.core.config import load_config

    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(
        "llm: {provider: fake, model: fake}\n"
        "embedder: {provider: fake, dim: 8}\n"
        "store: {provider: qdrant, url: 'http://prod:6333', path: /var/db}\n",
        encoding="utf-8",
    )
    cfg = load_config(cfg_file)
    iso = _isolated(cfg)
    assert iso.store.url == ":memory:"
    assert iso.store.path is None
    # everything else is untouched: the real bge-m3 endpoint still used
    assert iso.embedder == cfg.embedder
    assert iso.llm == cfg.llm
    assert iso.retrieval == cfg.retrieval
    # the original config is not mutated
    assert cfg.store.url != ":memory:"


def test_main_uses_an_isolated_store(tmp_path, monkeypatch, capsys):
    """main() must hand every builder a :memory: store, never the configured
    product store: RAGTurk/XQuAD/mevzuat corpora must not reach the live
    server. Capture the cfg build_app is called with."""
    import eval.run_eval as mod
    from porsuk.core import container

    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(
        "llm: {provider: fake, model: fake}\n"
        "embedder: {provider: fake, dim: 8}\n"
        "store: {provider: qdrant, url: 'http://prod:6333'}\n",
        encoding="utf-8",
    )
    corpus = tmp_path / "corpus"
    corpus.mkdir()

    seen_urls = []

    def _spy_build_app(cfg):
        seen_urls.append(cfg.store.url)
        raise RuntimeError("stop here: we only wanted the cfg")

    monkeypatch.setattr(container, "build_app", _spy_build_app)
    # load_ragturk / load_xquad_tr would hit the network; stub them empty so
    # each goldset block reaches its build_app call and then the spy raises.
    monkeypatch.setattr(mod, "load_ragturk", lambda **kw: [])
    monkeypatch.setattr(mod, "load_ragturk_corpus", lambda **kw: [])
    monkeypatch.setattr(mod, "load_xquad_tr", lambda **kw: [])
    monkeypatch.setattr(mod, "load_xquad_en_corpus", lambda **kw: [])
    monkeypatch.setattr(mod, "load_manual", lambda *a, **kw: [])
    monkeypatch.setattr(mod, "resolve_manual_doc_ids", lambda qs, d: qs)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_eval",
            "--config",
            str(cfg_file),
            "--manual-corpus",
            str(corpus),
            "--out",
            str(tmp_path / "out"),
        ],
    )

    mod.main()

    assert seen_urls, "build_app was never called"
    assert all(u == ":memory:" for u in seen_urls), seen_urls


def test_write_table_groups_by_goldset(tmp_path):
    from datetime import UTC, datetime

    results = [
        Result("manual", "semantic", "all", 3, 0.5, 0.5, 0.5),
        Result("ragturk", "hybrid", "all", 7, 0.9, 0.8, 0.85),
    ]
    path = _write_table(results, tmp_path)
    assert path.exists()
    stamp = datetime.now(UTC).strftime("%Y-%m-%d")
    assert path.name == f"{stamp}-retrieval.md"
    body = path.read_text(encoding="utf-8")
    assert "## manual" in body
    assert "## ragturk" in body


def test_params_of_stamps_keyword_backend_and_stemmer():
    from porsuk.core.config import load_config

    cfg = load_config("config/local.yaml")
    p = _params_of(cfg, "keyword")
    assert p["keyword_backend"] == cfg.retrieval.keyword_backend
    assert p["stemmer"] == cfg.retrieval.stemmer_enabled


def test_write_table_shows_the_keyword_sweep_as_columns(tmp_path):
    rows = [
        Result("manual", "keyword", "all", 5, 0.6, 0.5, 0.55,
                params={"keyword_backend": "bm25", "stemmer": False, "strategy": "keyword"}),
        Result("manual", "keyword", "all", 5, 0.4, 0.3, 0.35,
                params={"keyword_backend": "sparse", "stemmer": False, "strategy": "keyword"}),
        Result("manual", "semantic", "all", 5, 0.9, 0.8, 0.85,
                params={"keyword_backend": "bm25", "stemmer": False, "strategy": "semantic"}),
    ]
    body = _write_table(rows, tmp_path).read_text(encoding="utf-8")
    assert "| backend |" in body
    assert "| keyword | bm25 |" in body
    assert "| keyword | sparse |" in body
    # semantic rows do not carry a backend
    assert "| semantic | - |" in body
