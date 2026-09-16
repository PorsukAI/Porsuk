"""Run the agent over each golden set, judge its answers, write the results
table. This is the end-to-end counterpart of `eval/run_eval.py` (which
measures the retrieval layer alone): the whole agent loop runs, for each gold
question the agent searches, reads, expands and answers; an LLM-as-judge
(`eval/judge.py`, the same endpoint the agent uses) checks the answer against
the reference; `eval/agent_metrics.py` scores citation accuracy against the
golden set's `relevant_doc_ids` / `relevant_chunk_ids`. Every goldset is
indexed into its own fresh `:memory:` store via `_isolated(cfg)`, so
RAGTurk's CC-BY-NC-SA chunks, the XQuAD English contexts and the mevzuat
corpus never touch the product store or each other, and one goldset failing
(a 429, an unreachable endpoint) is caught so the rest still run. RAGTurk /
XQuAD are pre-chunked and indexed as bare `(doc_id, chunk_id, text)` tuples,
no pipeline; `_index_agent_app` also builds one minimal synthetic document
profile per distinct `doc_id` so the agent's `search_documents` tool returns
real hits instead of `[]`, with `filename == path == doc_id` so `Source.file`
stays the doc-id string that `score_one` expects for these sets. The manual
set runs through `build_pipeline` instead, so `doc_id_of_file` is passed to
`score_one` to map the agent's filename citations back to store ids.
"""

from __future__ import annotations

import argparse
import tempfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from eval.agent_metrics import AgentResult, aggregate, score_one
from eval.goldsets import (
    GoldQuestion,
    load_manual,
    load_ragturk,
    load_ragturk_corpus,
    load_xquad_en_corpus,
    load_xquad_tr,
    resolve_manual_doc_ids,
)
from eval.judge import JUDGE_PROMPT, Judge
from eval.run_eval import _isolated
from porsuk.agent.run import run_agent


def _index_agent_app(cfg, tuples: list[tuple[str, str, str]], *, language: str):
    """Index pre-chunked `(doc_id, chunk_id, text)` tuples into an isolated
    `:memory:` store and return the `App`. `run_agent(..., app=)`
    needs the whole App (it does `build_retriever(cfg, app=app)` and queries
    `app.store` from its tools), not just a `Retriever`.

    The caller passes a `cfg` from `_isolated(...)`, so `build_app` creates a
    fresh in-process Qdrant that never touches the product corpus.
    """
    # indexing body copied from eval/run_eval._index_tuples_and_build_app;
    # that one returns a Retriever, this returns the App run_agent needs.
    from porsuk.core.container import build_app
    from porsuk.core.models import Chunk, DocumentProfile, EmbedResult

    app = build_app(cfg)  # cfg.store.url == ":memory:" -> fresh isolated Qdrant
    chunks, texts = [], []
    for doc_id, chunk_id, text in tuples:
        chunks.append(Chunk(chunk_id, doc_id, text, None, language, None, None, (0, len(text))))
        texts.append(text)

    BATCH = 64
    dense, sparse = [], []
    for i in range(0, len(texts), BATCH):
        res = app.embedder.embed_documents(texts[i : i + BATCH])
        dense.extend(res.dense)
        if res.sparse is not None:
            sparse.extend(res.sparse)
    vectors = EmbedResult(dense=tuple(dense), sparse=tuple(sparse) if sparse else None)
    app.store.upsert_chunks(chunks, vectors)
    print(f"  indexed {len(chunks)} chunks into isolated store ({language})")

    # A minimal synthetic document profile per distinct doc_id. Without this
    # the agent's `search_documents` tool, which the system prompt tells the
    # agent to call *first*, always returns `[]` on these pre-chunked stores,
    # and a small model then flails instead of falling through to
    # `semantic_search`. `filename == path == doc_id` on purpose: `score_one`
    # (eval/agent_metrics.py) compares the agent's cited docs against
    # `q.relevant_doc_ids`, which for RAGTurk / XQuAD ARE the doc_id strings,
    # and `run_agent` maps `Source.file` through a `doc_id -> filename` map,
    # so keeping `filename == doc_id` makes that map an identity, and the
    # existing comparison keeps working with no change to `score_one`. The
    # summary carries real chunk text so a topical query retrieves the right
    # article.
    by_doc: dict[str, list[str]] = defaultdict(list)
    for doc_id, _chunk_id, text in tuples:
        by_doc[doc_id].append(text)
    profiles, summaries = [], []
    for doc_id, doc_texts in by_doc.items():
        summary = " ".join(doc_texts)[:600]
        summaries.append(summary)
        profiles.append(
            DocumentProfile(
                document_id=doc_id,
                path=doc_id,
                filename=doc_id,
                doc_type="text",
                language=language,
                created_at=None,
                modified_at=None,
                size=0,
                page_count=0,
                summary=summary,
                topics=(),
                entities=(),
                profile_level="cheap",
                profile_source=frozenset(),
                parse_quality=1.0,
                parse_quality_components=None,
                parser_used="eval",
                image_heavy=False,
                content_hash=doc_id,
            )
        )
    pdense = []
    for i in range(0, len(summaries), BATCH):
        pdense.extend(app.embedder.embed_documents(summaries[i : i + BATCH]).dense)
    app.store.upsert_profiles(profiles, EmbedResult(dense=tuple(pdense)))
    print(f"  indexed {len(profiles)} profiles into isolated store ({language})")
    return app


def _build_judge(cfg) -> Judge:
    """The LLM-as-judge, on the same endpoint the agent uses."""
    from porsuk.core.container import _load_adapters
    from porsuk.core.registry import build

    _load_adapters()
    llm = build("llm", cfg.llm.model_dump(exclude_none=True, exclude={"api_key_env"}))
    return Judge(llm)


def _agent_params(cfg) -> dict:
    """The config knobs that produced a run's numbers, stamped onto every
    `AgentResult` so a later run can be compared to this one."""
    return {
        "chunk_k": cfg.retrieval.chunk_k,
        "neighbor_expansion": cfg.retrieval.neighbor_expansion,
        "max_tool_calls": cfg.agent.max_tool_calls,
        "expand_budget": cfg.agent.expand_before_chars,
        "model": cfg.agent.model,
    }


def run_agent_eval(
    cfg,
    questions: list[GoldQuestion],
    app,
    *,
    goldset: str,
    judge: Judge,
    params: dict,
    doc_id_of_file: dict[str, str] | None = None,
    limit: int | None = None,
) -> list[AgentResult]:
    """Drive the agent over `questions`, judge, score. One `AgentResult` per
    question.

    Language filter: the XQuAD probe and the mono-lingual mevzuat manual set
    both run unfiltered (`lang=None`): for XQuAD because the target is
    English (a `tr` filter would exclude every candidate), for the
    manual set because a `tr` filter can silently drop short or
    mixed-language chunks (fastText returns `None` for a header or a
    definitions clause, `"en"` for a stray English phrase) from a corpus that
    is entirely Turkish, zeroing that question's `cite_doc_recall`. RAGTurk
    uses the question's own language.
    """
    qs = questions[:limit] if limit is not None else questions
    n = len(qs)
    results: list[AgentResult] = []
    for i, q in enumerate(qs, start=1):
        lang = None if goldset in ("xquad_tr", "manual") else q.language
        answer = run_agent(q.question, cfg=cfg, lang=lang, app=app)
        verdict = judge.judge_or_skip(q, answer.text)
        result = score_one(q, answer, verdict, doc_id_of_file=doc_id_of_file, params=params)
        results.append(result)
        print(
            f"  [{i}/{n}] {goldset} correct={result.answer_correct} "
            f"cite_doc_r={result.cite_doc_recall:.2f}"
        )
    return results


def _fmt(value: object) -> str:
    """`None` -> dash (not measured), a float -> `.3f`, anything else as-is."""
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


_COLUMNS = [
    ("question_type", "question_type"),
    ("n", "n"),
    ("answer_accuracy", "answer_accuracy"),
    ("n_judged", "n_judged"),
    ("cite_doc_recall", "cite_doc_recall"),
    ("cite_doc_precision", "cite_doc_precision"),
    ("cite_chunk_recall", "cite_chunk_recall"),
    ("cite_chunk_precision", "cite_chunk_precision"),
    ("mean_tool_calls", "mean_tool_calls"),
    ("truncation_rate", "truncation_rate"),
    ("note", "note"),
]


def _write_agent_table(
    summary: list[dict], raw: list[AgentResult], out_dir: Path
) -> tuple[Path, Path]:
    """`<out_dir>/<date>-agent.md` (the summary table, one section per goldset)
    and `<out_dir>/<date>-agent-raw.txt` (one block per `AgentResult`, for
    audit, git-ignored)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y-%m-%d")
    md_path = out_dir / f"{stamp}-agent.md"
    raw_path = out_dir / f"{stamp}-agent-raw.txt"

    header = "| " + " | ".join(label for label, _ in _COLUMNS) + " |"
    sep = "|" + "|".join("---" for _ in _COLUMNS) + "|"

    by_set: dict[str, list[dict]] = defaultdict(list)
    for row in summary:
        by_set[row["goldset"]].append(row)

    lines = [f"# Agent evaluation: {stamp}\n"]
    for goldset, rows in by_set.items():
        lines.append(f"\n## {goldset}\n")
        if goldset == "manual":
            lines.append("> n=12, gösterge: kullanıcı 25-30'a çıkaracak\n")
        lines.append(header)
        lines.append(sep)
        for row in sorted(rows, key=lambda r: (r["question_type"] != "all", r["question_type"])):
            lines.append("| " + " | ".join(_fmt(row[key]) for _, key in _COLUMNS) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    blocks = []
    for r in raw:
        blocks.append(
            "\n".join(
                [
                    f"goldset: {r.goldset}",
                    f"question: {r.question}",
                    f"answer_correct: {r.answer_correct}",
                    f"judge_reason: {r.judge_reason}",
                    f"judge_raw: {r.judge_raw}",
                    f"cite_doc_recall: {r.cite_doc_recall:.3f}",
                    f"cite_chunk_recall: {r.cite_chunk_recall:.3f}",
                ]
            )
        )
    raw_header = "=== JUDGE PROMPT ===\n" + JUDGE_PROMPT + "\n\n=== VERDICTS ===\n\n"
    raw_path.write_text(raw_header + "\n\n".join(blocks) + "\n", encoding="utf-8")
    return md_path, raw_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/vllm.yaml")
    ap.add_argument(
        "--ragturk-limit",
        type=int,
        default=25,
        help="RAGTurk articles to sample (~2 questions each, so ~50 questions)",
    )
    ap.add_argument("--manual-corpus", default="tests/fixtures/downloaded/mevzuat")
    ap.add_argument("--xquad-limit", type=int, default=100)
    ap.add_argument("--out", type=Path, default=Path("docs/eval"))
    ap.add_argument("--goldset", choices=("all", "ragturk", "manual", "xquad"), default="all")
    args = ap.parse_args()

    from porsuk.core.config import load_config
    from porsuk.core.container import build_app, build_pipeline

    cfg = load_config(args.config)
    judge = _build_judge(cfg)

    all_results: list[AgentResult] = []
    ran: list[str] = []
    want = {"ragturk", "manual", "xquad"} if args.goldset == "all" else {args.goldset}

    if "ragturk" in want:
        try:
            cfg_iso = _isolated(cfg)
            app = _index_agent_app(cfg_iso, load_ragturk_corpus(args.ragturk_limit), language="tr")
            all_results += run_agent_eval(
                cfg_iso,
                load_ragturk(args.ragturk_limit),
                app,
                goldset="ragturk",
                judge=judge,
                params=_agent_params(cfg),
            )
            ran.append("ragturk")
        except Exception as exc:  # noqa: BLE001 - one goldset failing must not sink the rest
            print(f"ragturk goldset failed, skipping: {exc}")

    if "manual" in want:
        try:
            mem = _isolated(cfg)
            state_path = str(Path(tempfile.mkdtemp(prefix="porsuk_agent_eval_")) / "manual.db")
            app = build_app(mem)
            pipe = build_pipeline(mem, state_path, app=app)
            for _ in pipe.run(args.manual_corpus):
                pass
            qs = resolve_manual_doc_ids(load_manual(), args.manual_corpus)
            base = Path(args.manual_corpus)
            from porsuk.ingestion.pipeline import document_id_for

            doc_id_of_file = {
                pdf.name: document_id_for(str(pdf))
                for pdf in sorted(base.iterdir())
                if pdf.is_file()
            }
            all_results += run_agent_eval(
                mem,
                qs,
                app,
                goldset="manual",
                judge=judge,
                params={**_agent_params(cfg), "note": "n=12 gösterge"},
                doc_id_of_file=doc_id_of_file,
            )
            ran.append("manual")
        except Exception as exc:  # noqa: BLE001 - one goldset failing must not sink the rest
            print(f"manual goldset failed, skipping: {exc}")

    if "xquad" in want:
        try:
            cfg_iso = _isolated(cfg)
            app = _index_agent_app(cfg_iso, load_xquad_en_corpus(args.xquad_limit), language="en")
            all_results += run_agent_eval(
                cfg_iso,
                load_xquad_tr(args.xquad_limit),
                app,
                goldset="xquad_tr",
                judge=judge,
                params=_agent_params(cfg),
                limit=args.xquad_limit,
            )
            ran.append("xquad")
        except Exception as exc:  # noqa: BLE001 - one goldset failing must not sink the rest
            print(f"xquad goldset failed, skipping: {exc}")

    print(f"goldsets that ran: {', '.join(ran) or 'none'}")
    summary = aggregate(all_results)
    md_path, raw_path = _write_agent_table(summary, all_results, args.out)
    print(f"wrote {md_path}")
    print(f"wrote {raw_path}")


if __name__ == "__main__":
    main()
