"""Offline unit tests for the agent parameter sweep.

No network, no GPU: `run_agent_eval` returns a canned `[AgentResult]`,
`_index_agent_app` / `_build_judge` are no-op stubs, `load_config` is
monkeypatched to a real offline config.
"""

from __future__ import annotations

import pytest

import eval.agent_sweep as asw
from eval.agent_metrics import AgentResult


def _result(*, chunk_scored: bool = True) -> AgentResult:
    return AgentResult(
        goldset="ragturk",
        question="q",
        question_type="specific",
        answer_correct=True,
        judge_reason="ok",
        judge_raw="EVET\nok",
        cite_doc_precision=1.0,
        cite_doc_recall=1.0,
        cite_chunk_precision=1.0,
        cite_chunk_recall=1.0,
        tool_calls=3,
        truncated=False,
        chunk_scored=chunk_scored,
        doc_scored=True,
    )


def test_grid_shape():
    keys = [k for k, _ in asw.AGENT_SWEEP_GRID]
    assert keys == [
        "retrieval.chunk_k",
        "retrieval.neighbor_expansion",
        "agent.max_tool_calls",
    ]
    grid = dict(asw.AGENT_SWEEP_GRID)
    assert grid["retrieval.chunk_k"] == [5, 10, 20]
    assert grid["retrieval.neighbor_expansion"] == [0, 1, 2]
    assert grid["agent.max_tool_calls"] == [6, 8, 12]


def test_agent_sweep_row_fields():
    row = asw.AgentSweepRow(
        parameter="retrieval.chunk_k",
        value=10,
        goldset="ragturk",
        question_type="all",
        n=30,
        answer_accuracy=0.7,
        cite_doc_recall=0.6,
        cite_chunk_recall=0.5,
        mean_tool_calls=4.2,
    )
    assert row.parameter == "retrieval.chunk_k"
    assert row.value == 10
    assert row.n == 30
    assert row.answer_accuracy == 0.7
    assert row.note == ""


def test_agent_sweep_calls_run_per_cell(monkeypatch, tmp_path):
    from porsuk.core.config import load_config

    cfg = load_config("config/local.yaml")
    monkeypatch.setattr(asw, "load_config", lambda *_a, **_k: cfg)
    monkeypatch.setattr(asw, "_build_judge", lambda _cfg: object())
    monkeypatch.setattr(asw, "_index_agent_app", lambda *a, **k: object())
    monkeypatch.setattr(asw, "load_ragturk", lambda _limit: [])
    monkeypatch.setattr(asw, "load_ragturk_corpus", lambda _limit: [])

    calls: list[dict] = []

    def _fake_run(patched, questions, app, **kwargs):  # noqa: ARG001
        calls.append(kwargs)
        return [_result()]

    monkeypatch.setattr(asw, "run_agent_eval", _fake_run)

    rows = asw.agent_sweep("config/local.yaml", ragturk_limit=5, out_dir=tmp_path)

    assert len(calls) == 9  # 3 + 3 + 3 grid cells
    assert len(rows) == 9
    assert all(r.question_type == "all" for r in rows)
    assert all(r.goldset == "ragturk" for r in rows)
    params = {(r.parameter, r.value) for r in rows}
    assert ("retrieval.chunk_k", 5) in params
    assert ("agent.max_tool_calls", 12) in params


def test_write_agent_sweep_table_renders_none(tmp_path):
    rows = [
        asw.AgentSweepRow(
            parameter="retrieval.chunk_k",
            value=5,
            goldset="ragturk",
            question_type="all",
            n=30,
            answer_accuracy=None,
            cite_doc_recall=0.6,
            cite_chunk_recall=None,
            mean_tool_calls=4.0,
        )
    ]
    path = asw._write_agent_sweep_table(rows, tmp_path)
    text = path.read_text(encoding="utf-8")
    assert "—" in text
    assert "0.600" in text


def test_agent_sweep_rejects_non_ragturk():
    with pytest.raises(NotImplementedError):
        asw.agent_sweep("config/local.yaml", goldset="manual")
