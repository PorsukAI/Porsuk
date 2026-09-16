"""Offline unit tests for the end-to-end agent eval runner.

No network, no GPU: `run_agent` is monkeypatched to a canned `AgentAnswer`,
the judge is a stub, and `app` is an ignored sentinel.
"""

from __future__ import annotations

from pathlib import Path

import eval.run_agent_eval as rae
from eval.agent_metrics import aggregate
from eval.goldsets import GoldQuestion
from eval.judge import Verdict
from porsuk.agent.models import AgentAnswer, Source


def _q(
    question: str, *, ref: str = "ref", qtype: str = "specific", source: str = "ragturk"
) -> GoldQuestion:
    return GoldQuestion(
        question=question,
        relevant_doc_ids=frozenset({"d1"}),
        relevant_chunk_ids=frozenset({"d1::c0"}),
        question_type=qtype,
        language="tr",
        source=source,
        reference_answer=ref,
    )


def _canned_answer() -> AgentAnswer:
    return AgentAnswer(
        text="cevap",
        sources=[Source(file="d1", page=None, section=None, quote="q", chunk_id="d1::c0")],
        tool_calls=3,
        truncated=False,
    )


class _StubJudge:
    def __init__(self, verdict: Verdict | None) -> None:
        self._verdict = verdict

    def judge_or_skip(self, q, answer_text: str) -> Verdict | None:  # noqa: ARG002
        if not q.reference_answer:
            return None
        return self._verdict


def test_run_agent_eval_scores_each_question(monkeypatch):
    monkeypatch.setattr(rae, "run_agent", lambda *a, **k: _canned_answer())
    params = {"chunk_k": 10, "model": "m"}
    results = rae.run_agent_eval(
        object(),
        [_q("a"), _q("b")],
        object(),
        goldset="ragturk",
        judge=_StubJudge(Verdict(True, "ok", "raw")),
        params=params,
    )
    assert len(results) == 2
    assert all(r.answer_correct is True for r in results)
    assert all(r.params == params for r in results)


def test_run_agent_eval_skips_judge_for_citation_only(monkeypatch):
    monkeypatch.setattr(rae, "run_agent", lambda *a, **k: _canned_answer())
    results = rae.run_agent_eval(
        object(),
        [_q("a", ref="")],
        object(),
        goldset="ragturk",
        judge=_StubJudge(None),
        params={},
    )
    assert len(results) == 1
    assert results[0].answer_correct is None


def test_run_agent_eval_respects_limit(monkeypatch):
    monkeypatch.setattr(rae, "run_agent", lambda *a, **k: _canned_answer())
    results = rae.run_agent_eval(
        object(),
        [_q(str(i)) for i in range(5)],
        object(),
        goldset="ragturk",
        judge=_StubJudge(Verdict(True, "ok", "raw")),
        params={},
        limit=2,
    )
    assert len(results) == 2


def test_write_agent_table_renders_none_as_dash(tmp_path: Path):
    from eval.agent_metrics import AgentResult

    # A manual-set result with no gold chunk ids -> aggregate emits None for
    # the chunk columns.
    raw = [
        AgentResult(
            goldset="manual",
            question="q",
            question_type="specific",
            answer_correct=True,
            judge_reason="ok",
            judge_raw="EVET\nok",
            cite_doc_precision=1.0,
            cite_doc_recall=1.0,
            cite_chunk_precision=0.0,
            cite_chunk_recall=0.0,
            tool_calls=2,
            truncated=False,
            chunk_scored=False,
            doc_scored=True,
            params={"note": "n=12 gösterge"},
        )
    ]
    summary = aggregate(raw)
    # sanity: the group's chunk columns really are None (not 0.0)
    all_row = next(r for r in summary if r["question_type"] == "all")
    assert all_row["cite_chunk_recall"] is None
    assert all_row["cite_chunk_precision"] is None

    md_path, raw_path = rae._write_agent_table(summary, raw, tmp_path)
    text = md_path.read_text(encoding="utf-8")
    # locate the "all" data row and check its chunk-recall / -precision cells
    row_line = next(
        ln for ln in text.splitlines() if ln.startswith("| all |") or ln.startswith("| all|")
    )
    cells = [c.strip() for c in row_line.strip("|").split("|")]
    # columns: question_type n answer_accuracy n_judged cite_doc_recall
    #          cite_doc_precision cite_chunk_recall cite_chunk_precision ...
    assert cells[6] == "—"  # cite_chunk_recall
    assert cells[7] == "—"  # cite_chunk_precision
    assert "0.000" not in (cells[6], cells[7])
    assert raw_path.exists()
    assert "gösterge" in text
    raw_text = raw_path.read_text(encoding="utf-8")
    assert "=== JUDGE PROMPT ===" in raw_text
    assert "judge_raw: EVET" in raw_text


def test_index_agent_app_creates_searchable_profiles():
    """The box run showed `search_documents` returning `[]` on the pre-chunked
    RAGTurk / XQuAD stores because `_index_agent_app` indexed chunks but no
    profiles. It must now build one synthetic profile per distinct doc_id,
    with `filename == doc_id`, whose summary carries real chunk text."""
    from porsuk.core.config import load_config
    from porsuk.core.models import Filters

    cfg = load_config("config/local.yaml")  # fake embedder (dim 8), inmemory store
    tuples = [
        ("d1", "d1::c0", "alpha content about payment"),
        ("d1", "d1::c1", "more d1"),
        ("d2", "d2::c0", "beta content about weather"),
    ]
    app = rae._index_agent_app(cfg, tuples, language="tr")

    rows = app.store.list_documents(Filters(), limit=10)
    assert len(rows) == 2
    assert {r.document_id for r in rows} == {"d1", "d2"}
    assert rows[0].filename == rows[0].document_id  # filename = doc_id decision

    hits = app.store.search_profiles(app.embedder.embed_query("payment"), k=2, filters=Filters())
    # fake embedder ordering is only roughly topical: assert both docs come
    # back rather than a strict rank.
    assert len(hits) == 2
    assert {h.profile.document_id for h in hits} == {"d1", "d2"}


def test_agent_params_shape():
    from porsuk.core.config import load_config

    cfg = load_config("config/local.yaml")
    p = rae._agent_params(cfg)
    assert set(p) == {"chunk_k", "neighbor_expansion", "max_tool_calls", "expand_budget", "model"}


def test_run_agent_eval_manual_set_runs_unfiltered(monkeypatch):
    """The mono-lingual mevzuat manual set must call run_agent with lang=None:
    a `tr` filter can silently drop short / mixed-language chunks (F1)."""
    seen: list[object] = []

    def _spy(question, *, cfg, lang=None, app=None, verbose=False):  # noqa: ARG001
        seen.append(lang)
        return _canned_answer()

    monkeypatch.setattr(rae, "run_agent", _spy)
    qs = [
        GoldQuestion(
            question=f"q{i}",
            relevant_doc_ids=frozenset({"d1"}),
            relevant_chunk_ids=frozenset(),
            question_type="specific",
            language="tr",
            source="manual",
            reference_answer="ref",
        )
        for i in range(2)
    ]
    rae.run_agent_eval(
        object(),
        qs,
        object(),
        goldset="manual",
        judge=_StubJudge(Verdict(True, "ok", "raw")),
        params={},
    )
    assert seen == [None, None]


def test_main_manual_smoke(monkeypatch, tmp_path):
    """The manual-branch wiring (mem/app/doc_id_of_file/run_agent_eval) is
    exercised once offline before the box run, including the manual `## manual`
    section and the `n=12 gösterge` blockquote (F5)."""
    from datetime import UTC, datetime

    from porsuk.core.config import load_config

    monkeypatch.setattr(rae, "run_agent", lambda *a, **k: _canned_answer())
    monkeypatch.setattr(rae, "_build_judge", lambda cfg: _StubJudge(Verdict(True, "ok", "raw")))
    _local = load_config("config/local.yaml")
    monkeypatch.setattr("porsuk.core.config.load_config", lambda *_a, **_k: _local)
    monkeypatch.setattr(
        rae, "load_manual", lambda: [_q("manual q", qtype="specific", source="manual")]
    )
    monkeypatch.setattr(rae, "resolve_manual_doc_ids", lambda qs, corpus: qs)
    monkeypatch.setattr(rae, "_isolated", lambda cfg: cfg)
    monkeypatch.setattr("porsuk.core.container.build_app", lambda *a, **k: object())

    class _FakePipe:
        def run(self, folder):  # noqa: ARG002
            return iter(())

    monkeypatch.setattr("porsuk.core.container.build_pipeline", lambda *a, **k: _FakePipe())
    argv = [
        "run_agent_eval",
        "--goldset",
        "manual",
        "--manual-corpus",
        str(tmp_path),
        "--out",
        str(tmp_path),
    ]
    monkeypatch.setattr("sys.argv", argv)

    rae.main()

    stamp = datetime.now(UTC).strftime("%Y-%m-%d")
    md_path = tmp_path / f"{stamp}-agent.md"
    assert md_path.exists()
    assert (tmp_path / f"{stamp}-agent-raw.txt").exists()
    md_text = md_path.read_text(encoding="utf-8")
    assert "## manual" in md_text
    assert "> n=12" in md_text and "gösterge" in md_text
