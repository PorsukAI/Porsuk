"""The retrieval parameter sweep: each open retrieval parameter is swept over
its candidate values one at a time, others held at the current config
default, since a full cross product would be hundreds of index+eval runs
(near-configs are within noise at this golden-set size). Re-index params
(chunk.*) force a fresh pipeline run per value; retrieval-only params (k,
neighbours, rerank, stemmer) reuse one index and just rebuild the Retriever.
RAGTurk/XQuAD arrive pre-chunked (the dataset fixes the chunk boundaries), so
`chunk.target_chars` / `chunk.overlap_ratio` cannot change anything there and
those rows are emitted as N/A for a pre-chunked goldset; chunk size only
sweeps meaningfully against a goldset that indexes through `build_pipeline`,
i.e. the manual set (a real PDF parse) once it is verified and large enough.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from eval.goldsets import load_ragturk, load_ragturk_corpus
from eval.run_eval import (
    Result,
    _index_tuples_and_build_app,
    _isolated,
    _retriever_over,
    evaluate,
)

# grid: list of (dotted_key, [candidate values])
SWEEP_GRID: list[tuple[str, list[object]]] = [
    ("chunk.target_chars", [1200, 1500, 1800, 2200]),
    ("chunk.overlap_ratio", [0.10, 0.15, 0.20]),
    ("retrieval.chunk_k", [5, 10, 20]),
    ("retrieval.neighbor_expansion", [0, 1, 2]),
    ("retrieval.rerank_enabled", [False, True]),
    ("retrieval.stemmer_enabled", [False, True]),
    ("retrieval.keyword_backend", ["text", "bm25", "sparse"]),
]

_REINDEX_KEYS = {"chunk.target_chars", "chunk.overlap_ratio"}

_DOTTED_SECTION = {
    "chunk": "chunking",
    "retrieval": "retrieval",
    "agent": "agent",
}  # dotted prefix -> Config attr

# Pre-chunked goldsets: chunk.* cannot re-chunk them, so those rows are N/A.
_PRE_CHUNKED = {"ragturk", "xquad"}


@dataclass(frozen=True)
class SweepRow:
    parameter: str
    value: object
    goldset: str
    strategy: str
    question_type: str
    n: int
    recall_at_5: float
    mrr: float
    ndcg_at_5: float
    note: str = ""


def _cells_for(key: str, values: list[object], *, base: dict) -> Iterator[dict]:
    """Yield a one-key override dict per candidate value.

    `base` is accepted for signature compatibility but is not merged into the
    yielded dict: each cell carries ONLY the swept key (see
    `test_cells_hold_others_at_default`, which asserts every cell has exactly
    one key). Holding the other params at default is `_patch`'s job: it
    starts from the loaded base config, which already holds them.
    """
    for v in values:
        yield {key: v}


def _patch(cfg, overrides: dict[str, object]):
    """Return a copy of `cfg` with dotted-key `overrides` applied.

    `"chunk.target_chars"` -> section attr `"chunking"` (via `_DOTTED_SECTION`),
    field `"target_chars"`. Overrides are grouped by section so each nested
    model is `model_copy`'d once, then the top config once. The original `cfg`
    is untouched (pydantic `model_copy` is a shallow copy; we never mutate a
    shared nested model).
    """
    by_section: dict[str, dict[str, object]] = defaultdict(dict)
    for dotted, value in overrides.items():
        prefix, field = dotted.split(".", 1)
        section = _DOTTED_SECTION[prefix]
        by_section[section][field] = value

    section_updates = {
        section: getattr(cfg, section).model_copy(update=fields)
        for section, fields in by_section.items()
    }
    return cfg.model_copy(update=section_updates)


def _rows_from_results(
    results: list[Result], *, parameter: str, value: object, goldset: str, note: str = ""
) -> list[SweepRow]:
    """The `"all"` Result per strategy -> a SweepRow. `evaluate` may return
    several rows (one per question_type plus "all"); the sweep table compares
    the aggregate."""
    out: list[SweepRow] = []
    for r in results:
        if r.question_type != "all":
            continue
        out.append(
            SweepRow(
                parameter=parameter,
                value=value,
                goldset=goldset,
                strategy=r.strategy,
                question_type=r.question_type,
                n=r.n,
                recall_at_5=r.recall_at_5,
                mrr=r.mrr,
                ndcg_at_5=r.ndcg_at_5,
                note=note,
            )
        )
    return out


def _na_rows(
    key: str, values: list[object], *, goldset: str, strategies: tuple[str, ...]
) -> list[SweepRow]:
    note = (
        f"N/A: {goldset} corpus is pre-chunked; chunk size only sweeps against "
        "the manual set (real PDF parse)"
    )
    return [
        SweepRow(
            parameter=key,
            value=v,
            goldset=goldset,
            strategy=strategy,
            question_type="all",
            n=0,
            recall_at_5=float("nan"),
            mrr=float("nan"),
            ndcg_at_5=float("nan"),
            note=note,
        )
        for v in values
        for strategy in strategies
    ]


def sweep(
    base_config_path: str,
    *,
    goldset: str = "ragturk",
    strategies: tuple[str, ...] = ("semantic", "keyword", "hybrid"),
    ragturk_limit: int = 300,
    out_dir: Path = Path("docs/eval"),
) -> list[SweepRow]:
    """Run the one-at-a-time sweep, write the table, return every row.

    `goldset` default `"ragturk"`: it is the largest, most meaningful set
    (~600 questions at limit 300). The manual set is 12 unverified entries
    (too small / noisy for a sweep); XQuAD only probes cross-lingual. The
    param is exposed so the sweep can later point at a verified manual set
    (that path also makes the chunk.* rows real, since the manual set
    indexes through `build_pipeline`).
    """
    from porsuk.core.config import load_config

    if goldset != "ragturk":
        raise NotImplementedError(
            f"sweep goldset {goldset!r} not wired yet, only 'ragturk'. "
            "The verified-manual-set path (and with it the real chunk.* sweep) is not wired yet."
        )

    base_cfg = load_config(base_config_path)

    # --- load the goldset once up front --------------------------------------
    questions = load_ragturk(limit=ragturk_limit)
    corpus = load_ragturk_corpus(limit=ragturk_limit)

    rows: list[SweepRow] = []

    # --- retrieval-only params: index ONCE at default chunking --------------
    app = _index_tuples_and_build_app(
        _isolated(load_config(base_config_path)),
        corpus,
        collection=goldset,
        language="tr",
    )
    for key, values in SWEEP_GRID:
        if key in _REINDEX_KEYS:
            continue
        for value in values:
            patched = _patch(base_cfg, {key: value})
            try:
                patched_retriever = _retriever_over(app, patched)
            except Exception as exc:  # noqa: BLE001 - absent reranker is fine unless rerank is on
                if patched.retrieval.rerank_enabled:
                    raise
                print(f"  {key}={value}: {exc}; rerank stays off")
                patched_retriever = _retriever_over(
                    app, _patch(patched, {"retrieval.rerank_enabled": False})
                )
            for strategy in strategies:
                results = evaluate(
                    patched_retriever,
                    questions,
                    strategy=strategy,
                    k=max(patched.retrieval.chunk_k, 5),
                    level="doc",
                )
                rows += _rows_from_results(results, parameter=key, value=value, goldset=goldset)

    # --- re-index params (chunk.*) ----------------------------------------
    for key, values in SWEEP_GRID:
        if key not in _REINDEX_KEYS:
            continue
        if goldset in _PRE_CHUNKED:
            rows += _na_rows(key, values, goldset=goldset, strategies=strategies)
            continue
        # The real re-index path (manual set): re-run build_pipeline per value.
        # Not reachable while goldset == "ragturk"; the manual set is not wired here yet.
        for value in values:  # pragma: no cover - no non-pre-chunked goldset yet
            patched = _patch(base_cfg, {key: value})
            raise NotImplementedError(
                f"chunk.* re-index for goldset {goldset!r} needs the build_pipeline "
                f"path: patched chunking = {patched.chunking}"
            )

    _write_sweep_table(rows, out_dir, goldset=goldset, ragturk_limit=ragturk_limit)
    return rows


def _write_sweep_table(
    rows: list[SweepRow], out_dir: Path, *, goldset: str, ragturk_limit: int
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y-%m-%d")
    path = out_dir / f"{stamp}-parameter-sweep.md"

    strategies = sorted({r.strategy for r in rows})
    lines = [
        f"# Parameter sweep: {stamp}\n",
        f"Goldset: **{goldset}** (ragturk_limit={ragturk_limit}).",
        "",
        "One parameter swept at a time; the others held at the current config "
        "default. Rows are treated as roughly independent: a full "
        "cross product would be 432 index+eval runs, this is 17. Near-configs "
        "are within golden-set noise at this size.",
        "",
    ]

    # column headers: <metric> (<strategy>) for each strategy
    metric_cols = [
        (label, attr)
        for label, attr in (
            ("recall@5", "recall_at_5"),
            ("MRR", "mrr"),
            ("nDCG@5", "ndcg_at_5"),
        )
    ]
    header_cells = ["value"] + [f"{label} ({s})" for s in strategies for label, _ in metric_cols]

    by_param: dict[str, list[SweepRow]] = defaultdict(list)
    for r in rows:
        by_param[r.parameter].append(r)

    for key, _values in SWEEP_GRID:
        prows = by_param.get(key, [])
        lines.append(f"\n## {key}\n")
        note = next((r.note for r in prows if r.note), "")
        if note:
            lines.append(f"_{note}_\n")

        # rows: one per candidate value, in grid order
        seen_values: list[object] = []
        for r in prows:
            if r.value not in seen_values:
                seen_values.append(r.value)

        # best value per (strategy, metric) column, for bolding
        best: dict[tuple[str, str], object] = {}
        for s in strategies:
            for _label, attr in metric_cols:
                cands = [
                    (getattr(r, attr), r.value)
                    for r in prows
                    if r.strategy == s and r.value in seen_values
                ]
                cands = [(m, v) for m, v in cands if m == m]  # drop NaN
                if cands:
                    best[(s, attr)] = max(cands)[1]

        lines.append("| " + " | ".join(header_cells) + " |")
        lines.append("|" + "---|" * len(header_cells))
        for v in seen_values:
            cells = [str(v)]
            for s in strategies:
                row = next((r for r in prows if r.strategy == s and r.value == v), None)
                for _label, attr in metric_cols:
                    if row is None or getattr(row, attr) != getattr(row, attr):
                        cells.append("-")
                        continue
                    val = f"{getattr(row, attr):.3f}"
                    if best.get((s, attr)) == v:
                        val = f"**{val}**"
                    cells.append(val)
            lines.append("| " + " | ".join(cells) + " |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/vllm.yaml")
    ap.add_argument("--goldset", default="ragturk")
    ap.add_argument("--ragturk-limit", type=int, default=300)
    ap.add_argument("--out", type=Path, default=Path("docs/eval"))
    args = ap.parse_args()

    rows = sweep(
        args.config,
        goldset=args.goldset,
        ragturk_limit=args.ragturk_limit,
        out_dir=args.out,
    )
    stamp = datetime.now(UTC).strftime("%Y-%m-%d")
    print(f"{len(rows)} sweep rows -> {args.out / f'{stamp}-parameter-sweep.md'}")


if __name__ == "__main__":
    main()
