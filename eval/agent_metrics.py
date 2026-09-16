"""Agent-eval scoring: citation accuracy and per-run aggregation.

Scores the agent's cited chunk/document ids against the golden set's
`relevant_doc_ids` / `relevant_chunk_ids` (precision/recall), and rolls up
the LLM-as-judge `Verdict` (`eval/judge.py`) into an answer-accuracy rate.
`score_one` produces one `AgentResult` per question; `aggregate` rolls a list
of them up by `(goldset, question_type)` with an extra `"all"` row per goldset.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from statistics import fmean

from eval.goldsets import GoldQuestion
from eval.judge import Verdict
from porsuk.agent.models import AgentAnswer


def citation_scores(cited: set[str], gold: set[str]) -> tuple[float, float, float]:
    """(precision, recall, f1) of `cited` against `gold`.

    Empty `gold` (the question is not scored at this level) → all zeros; the
    caller decides whether to fold that row into a mean. Empty `cited` with a
    non-empty `gold` → all zeros (the agent cited nothing relevant).
    """
    if not gold or not cited:
        return (0.0, 0.0, 0.0)
    hits = len(cited & gold)
    precision = hits / len(cited)
    recall = hits / len(gold)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return (precision, recall, f1)


@dataclass(frozen=True)
class AgentResult:
    goldset: str
    question: str
    question_type: str
    answer_correct: bool | None
    judge_reason: str
    judge_raw: str
    cite_doc_precision: float
    cite_doc_recall: float
    cite_chunk_precision: float
    cite_chunk_recall: float
    tool_calls: int
    truncated: bool
    chunk_scored: bool = False  # q.relevant_chunk_ids was non-empty
    doc_scored: bool = False  # q.relevant_doc_ids was non-empty
    params: dict = field(default_factory=dict)


def score_one(
    q: GoldQuestion,
    answer: AgentAnswer,
    verdict: Verdict | None,
    *,
    doc_id_of_file: dict[str, str] | None = None,
    params: dict | None = None,
) -> AgentResult:
    """Score one agent run against one gold question.

    `answer.sources` carries the agent's citations. `Source.file` is a
    filename for the manual set and the doc-id string itself for the isolated
    RAGTurk/XQuAD stores (where `run_agent` has no `doc_id → filename` map and
    `_source` falls back to the id). Pass `doc_id_of_file` for the manual set
    to map filenames back to store ids before scoring. `Source.chunk_id` is
    the id `run.py` carries; `None` entries (e.g. from a
    `documents_seen`-derived Source) are skipped.
    """
    doc_id_of_file = doc_id_of_file or {}
    cited_docs = {doc_id_of_file.get(s.file, s.file) for s in answer.sources}
    cited_chunks = {s.chunk_id for s in answer.sources if s.chunk_id}

    cite_doc_precision, cite_doc_recall, _ = citation_scores(cited_docs, set(q.relevant_doc_ids))
    cite_chunk_precision, cite_chunk_recall, _ = citation_scores(
        cited_chunks, set(q.relevant_chunk_ids)
    )

    return AgentResult(
        goldset=q.source,
        question=q.question,
        question_type=q.question_type,
        answer_correct=verdict.correct if verdict is not None else None,
        judge_reason=verdict.reason if verdict is not None else "",
        judge_raw=verdict.raw if verdict is not None else "",
        cite_doc_precision=cite_doc_precision,
        cite_doc_recall=cite_doc_recall,
        cite_chunk_precision=cite_chunk_precision,
        cite_chunk_recall=cite_chunk_recall,
        tool_calls=answer.tool_calls,
        truncated=answer.truncated,
        chunk_scored=bool(q.relevant_chunk_ids),
        doc_scored=bool(q.relevant_doc_ids),
        params=params or {},
    )


def aggregate(results: list[AgentResult]) -> list[dict]:
    """Roll `results` up by `(goldset, question_type)` with an `"all"` row per goldset.

    A metric is `None`, not `0.0`, when its group had nothing to score:
    `answer_accuracy` when no question in the group carried a reference answer,
    the chunk columns when no question had gold chunk ids (the manual set),
    the doc columns when no question had gold doc ids. `0.0` means "scored, and
    the agent got it wrong"; `None` means "not measured for this group".
    `n`, `mean_tool_calls` and `truncation_rate` are always over the full group.
    """
    groups: dict[tuple[str, str], list[AgentResult]] = defaultdict(list)
    for r in results:
        groups[(r.goldset, r.question_type)].append(r)
        groups[(r.goldset, "all")].append(r)

    rows: list[dict] = []
    for (goldset, question_type), group in groups.items():
        judged = [r for r in group if r.answer_correct is not None]
        doc_group = [r for r in group if r.doc_scored]
        chunk_group = [r for r in group if r.chunk_scored]
        notes = [r.params.get("note", "") for r in group if r.params.get("note", "")]
        rows.append(
            {
                "goldset": goldset,
                "question_type": question_type,
                "n": len(group),
                "answer_accuracy": (
                    fmean(1.0 if r.answer_correct else 0.0 for r in judged) if judged else None
                ),
                "n_judged": len(judged),
                "cite_doc_recall": (
                    fmean(r.cite_doc_recall for r in doc_group) if doc_group else None
                ),
                "cite_doc_precision": (
                    fmean(r.cite_doc_precision for r in doc_group) if doc_group else None
                ),
                "cite_chunk_recall": (
                    fmean(r.cite_chunk_recall for r in chunk_group) if chunk_group else None
                ),
                "cite_chunk_precision": (
                    fmean(r.cite_chunk_precision for r in chunk_group) if chunk_group else None
                ),
                "mean_tool_calls": fmean(r.tool_calls for r in group),
                "truncation_rate": fmean(1.0 if r.truncated else 0.0 for r in group),
                "note": notes[0] if notes else "",
            }
        )
    return rows
