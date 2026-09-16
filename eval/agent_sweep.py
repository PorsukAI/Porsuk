"""The agent-facing parameter sweep: reruns the end-to-end agent eval
(`eval/run_agent_eval.py`) with each open agent-facing knob varied over its
candidate values, one at a time, the others held at the config default
(near-configs are within golden-set noise at this sample size, the written
table states `n` and that caveat). One index, N eval runs: none of the swept
parameters re-index, since `retrieval.chunk_k` and
`retrieval.neighbor_expansion` are retrieval-time, `agent.max_tool_calls` is
agent-time, and RAGTurk arrives pre-chunked anyway; each `run_agent_eval`
call passes the *patched* cfg into `run_agent`, which rebuilds the
retriever/reranker from it per call, so those knobs all take effect per cell
without re-indexing.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from eval.agent_metrics import aggregate
from eval.goldsets import load_ragturk, load_ragturk_corpus
from eval.run_agent_eval import (
    _agent_params,
    _build_judge,
    _fmt,
    _index_agent_app,
    run_agent_eval,
)
from eval.run_eval import _isolated
from eval.sweep import _patch
from porsuk.core.config import load_config

# grid: list of (dotted_key, [candidate values]); _patch handles agent.* since
# eval/sweep.py::_DOTTED_SECTION gained "agent": "agent".
AGENT_SWEEP_GRID: list[tuple[str, list[object]]] = [
    ("retrieval.chunk_k", [5, 10, 20]),
    ("retrieval.neighbor_expansion", [0, 1, 2]),
    ("agent.max_tool_calls", [6, 8, 12]),
]


@dataclass(frozen=True)
class AgentSweepRow:
    parameter: str
    value: object
    goldset: str
    question_type: str
    n: int
    answer_accuracy: float | None
    cite_doc_recall: float | None
    cite_chunk_recall: float | None
    mean_tool_calls: float
    note: str = ""


def agent_sweep(
    cfg_path: str,
    *,
    goldset: str = "ragturk",
    ragturk_limit: int = 15,
    out_dir: Path = Path("docs/eval"),
) -> list[AgentSweepRow]:
    """Run the agent sweep, write the table, return every row.

    `goldset` default `"ragturk"`, mirrors `sweep.py`'s guard; only ragturk is
    wired (the manual set is small and unverified, XQuAD only probes
    cross-lingual).
    """
    if goldset != "ragturk":
        raise NotImplementedError(
            f"agent_sweep goldset {goldset!r} not wired yet, only 'ragturk'."
        )

    base_cfg = load_config(cfg_path)
    judge = _build_judge(base_cfg)

    questions = load_ragturk(ragturk_limit)
    corpus = load_ragturk_corpus(ragturk_limit)

    # Index ONCE: no swept parameter re-indexes (see the module docstring).
    app = _index_agent_app(_isolated(base_cfg), corpus, language="tr")

    rows: list[AgentSweepRow] = []
    for key, values in AGENT_SWEEP_GRID:
        for value in values:
            patched = _patch(base_cfg, {key: value})
            results = run_agent_eval(
                patched,
                questions,
                app,
                goldset="ragturk",
                judge=judge,
                params=_agent_params(patched),
            )
            summary = aggregate(results)
            row = next(r for r in summary if r["question_type"] == "all")
            rows.append(
                AgentSweepRow(
                    parameter=key,
                    value=value,
                    goldset="ragturk",
                    question_type="all",
                    n=row["n"],
                    answer_accuracy=row["answer_accuracy"],
                    cite_doc_recall=row["cite_doc_recall"],
                    cite_chunk_recall=row["cite_chunk_recall"],
                    mean_tool_calls=row["mean_tool_calls"],
                )
            )

    _write_agent_sweep_table(rows, out_dir, ragturk_limit=ragturk_limit)
    return rows


_METRIC_COLS = [
    ("n", "n"),
    ("answer_accuracy", "answer_accuracy"),
    ("cite_doc_recall", "cite_doc_recall"),
    ("cite_chunk_recall", "cite_chunk_recall"),
    ("mean_tool_calls", "mean_tool_calls"),
]


def _write_agent_sweep_table(
    rows: list[AgentSweepRow], out_dir: Path, *, ragturk_limit: int = 15
) -> Path:
    """`<out_dir>/<YYYY-MM-DD>-agent-sweep.md`: one section per parameter,
    a table `value | n | answer_accuracy | cite_doc_recall | cite_chunk_recall
    | mean_tool_calls`. `None` renders as a dash, floats as `.3f` (via
    `run_agent_eval._fmt`)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y-%m-%d")
    path = out_dir / f"{stamp}-agent-sweep.md"

    lines = [
        f"# Agent parameter sweep: {stamp}\n",
        f"RAGTurk, one index, {ragturk_limit} articles. One parameter at a "
        "time, others at the config default. Near-configs are "
        "within noise at this sample size.",
        "",
    ]

    header_cells = ["value"] + [label for label, _ in _METRIC_COLS]

    by_param: dict[str, list[AgentSweepRow]] = defaultdict(list)
    for r in rows:
        by_param[r.parameter].append(r)

    for key, _values in AGENT_SWEEP_GRID:
        prows = by_param.get(key, [])
        lines.append(f"\n## {key}\n")
        lines.append("| " + " | ".join(header_cells) + " |")
        lines.append("|" + "---|" * len(header_cells))
        for r in prows:
            cells = [str(r.value)] + [_fmt(getattr(r, attr)) for _, attr in _METRIC_COLS]
            lines.append("| " + " | ".join(cells) + " |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/vllm.yaml")
    ap.add_argument("--goldset", default="ragturk")
    ap.add_argument("--limit", type=int, default=15, help="RAGTurk articles to sample")
    ap.add_argument("--out", type=Path, default=Path("docs/eval"))
    args = ap.parse_args()

    rows = agent_sweep(
        args.config,
        goldset=args.goldset,
        ragturk_limit=args.limit,
        out_dir=args.out,
    )
    stamp = datetime.now(UTC).strftime("%Y-%m-%d")
    print(f"{len(rows)} agent-sweep rows -> {args.out / f'{stamp}-agent-sweep.md'}")


if __name__ == "__main__":
    main()
