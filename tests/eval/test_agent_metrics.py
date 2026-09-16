from eval.agent_metrics import AgentResult, aggregate, citation_scores, score_one
from eval.goldsets import GoldQuestion
from eval.judge import Verdict
from porsuk.agent.models import AgentAnswer, Source


def _q(**kw) -> GoldQuestion:
    base = dict(
        question="Kira bedeli nedir?",
        relevant_doc_ids=frozenset({"d1"}),
        relevant_chunk_ids=frozenset({"d1::c1"}),
        question_type="specific",
        language="tr",
        source="manual",
        reference_answer="42",
    )
    base.update(kw)
    return GoldQuestion(**base)


def test_citation_scores_precision_recall():
    p, r, f = citation_scores({"a", "b", "x"}, {"a", "b", "c"})
    assert p == 2 / 3
    assert r == 2 / 3
    assert f == 2 / 3


def test_citation_scores_empty_gold():
    assert citation_scores(set(), {"a"}) == (0.0, 0.0, 0.0)
    assert citation_scores({"a"}, set()) == (0.0, 0.0, 0.0)
    assert citation_scores(set(), set()) == (0.0, 0.0, 0.0)


def test_citation_scores_perfect():
    assert citation_scores({"a", "b"}, {"a", "b"}) == (1.0, 1.0, 1.0)


def test_score_one_full_match():
    q = _q()
    answer = AgentAnswer(
        text="42",
        sources=[Source(file="d1", page=1, section=None, quote="q", chunk_id="d1::c1")],
    )
    res = score_one(q, answer, Verdict(True, "ok", "raw"))
    assert res.answer_correct is True
    assert res.judge_reason == "ok"
    assert res.judge_raw == "raw"
    assert res.cite_doc_recall == 1.0
    assert res.cite_doc_precision == 1.0
    assert res.cite_chunk_recall == 1.0
    assert res.cite_chunk_precision == 1.0
    assert res.chunk_scored is True
    assert res.doc_scored is True


def test_score_one_chunk_scored_false_when_no_gold_chunks():
    q = _q(relevant_chunk_ids=frozenset())
    answer = AgentAnswer(text="42", sources=[])
    res = score_one(q, answer, Verdict(True, "ok", "raw"))
    assert res.chunk_scored is False
    assert res.doc_scored is True


def test_score_one_citation_only():
    q = _q(reference_answer="")
    answer = AgentAnswer(text="something", sources=[])
    res = score_one(q, answer, None)
    assert res.answer_correct is None
    assert res.judge_reason == ""
    assert res.judge_raw == ""


def test_score_one_doc_id_of_file():
    q = _q()
    answer = AgentAnswer(
        text="42",
        sources=[Source(file="Sozlesme.pdf", page=1, section=None, quote="q")],
    )
    res = score_one(q, answer, Verdict(True, "ok", "raw"), doc_id_of_file={"Sozlesme.pdf": "d1"})
    assert res.cite_doc_recall == 1.0
    assert res.cite_doc_precision == 1.0


def test_score_one_stamps_params():
    q = _q()
    answer = AgentAnswer(text="42", sources=[])
    res = score_one(q, answer, Verdict(True, "ok", "raw"), params={"note": "hi"})
    assert res.params == {"note": "hi"}


def _r(correct, **kw) -> AgentResult:
    base = dict(
        goldset="manual",
        question="q",
        question_type="specific",
        answer_correct=correct,
        judge_reason="",
        judge_raw="",
        cite_doc_precision=0.5,
        cite_doc_recall=0.5,
        cite_chunk_precision=0.5,
        cite_chunk_recall=0.5,
        tool_calls=2,
        truncated=False,
        chunk_scored=True,
        doc_scored=True,
    )
    base.update(kw)
    return AgentResult(**base)


def test_aggregate_answer_accuracy_ignores_none():
    results = [_r(True), _r(False), _r(None)]
    rows = aggregate(results)
    all_row = next(r for r in rows if r["question_type"] == "all")
    assert all_row["n"] == 3
    assert all_row["n_judged"] == 2
    assert all_row["answer_accuracy"] == 0.5


def test_aggregate_answer_accuracy_none_when_nothing_judged():
    rows = aggregate([_r(None), _r(None)])
    all_row = next(r for r in rows if r["question_type"] == "all")
    assert all_row["answer_accuracy"] is None
    assert all_row["n_judged"] == 0
    assert all_row["n"] == 2


def test_aggregate_chunk_columns_none_for_manual_style_group():
    # manual set: doc ids present, chunk ids absent
    results = [_r(True, chunk_scored=False), _r(False, chunk_scored=False)]
    all_row = next(r for r in aggregate(results) if r["question_type"] == "all")
    assert all_row["cite_chunk_recall"] is None
    assert all_row["cite_chunk_precision"] is None
    assert isinstance(all_row["cite_doc_recall"], float)


def test_aggregate_chunk_columns_average_only_scored():
    results = [
        _r(True, chunk_scored=True, cite_chunk_recall=1.0),
        _r(True, chunk_scored=False, cite_chunk_recall=0.0),
    ]
    all_row = next(r for r in aggregate(results) if r["question_type"] == "all")
    assert all_row["cite_chunk_recall"] == 1.0  # only the scored one


def test_aggregate_groups_by_question_type():
    results = [_r(True, question_type="a"), _r(False, question_type="b")]
    rows = aggregate(results)
    types = {r["question_type"] for r in rows}
    assert types == {"a", "b", "all"}
