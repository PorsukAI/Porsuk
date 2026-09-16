"""The three golden sets (altın set), loaded into one shape.

They answer different questions and are reported separately:
  - RAGTurk: is Turkish retrieval broadly adequate? (Wikipedia corpus,
    CC-BY-NC-SA, evaluation only, never in the product corpus)
  - manual: does it work on files like the boss's? (our mevzuat + fixtures)
  - XQuAD-tr: does a Turkish question actually find an English document?
    (bge-m3 embeds meaning, not language)

Each set also has a matching `_corpus` loader. RAGTurk and XQuAD bring their
own corpora, indexed into throwaway Qdrant collections kept apart from the
product corpus. The gold ids a question carries are the ids its own
`_corpus` loader emits, so the two must be drawn from the same sample, hence
the shared iteration helpers below.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path

import yaml

RAGTURK_REPO = "metunlp/ragturk"
XQUAD_REPO = "google/xquad"


@dataclass(frozen=True)
class GoldQuestion:
    question: str
    relevant_doc_ids: frozenset[str]
    relevant_chunk_ids: frozenset[str]
    question_type: str
    language: str
    source: str
    # Manual set only: the gold documents named by bare filename. The store's
    # document id is `document_id_for(resolved_path)` - a UUID keyed
    # to an absolute path, so it cannot be baked into a committed YAML that
    # must resolve on any checkout or on the box. The manual YAML carries
    # filenames; `resolve_manual_doc_ids` turns them into ids once the eval knows
    # the corpus directory. RAGTurk and XQuAD set `relevant_doc_ids` directly
    # (their corpora carry their own ids) and leave this empty.
    relevant_doc_files: frozenset[str] = frozenset()
    # The gold answer, for the LLM-as-judge answer-accuracy
    # metric. RAGTurk and XQuAD carry one; the manual YAML's is optional (the
    # user fills them when verifying the set). "" means "no reference - this
    # question is scored on citation accuracy only".
    reference_answer: str = ""


def load_manual(path: str = "eval/manual_goldset.yaml") -> list[GoldQuestion]:
    """Load the manual set. `relevant_doc_ids` is left empty - the gold docs

    are carried as `relevant_doc_files` (bare filenames); call
    `resolve_manual_doc_ids(qs, corpus_dir)` to fill `relevant_doc_ids`.
    """
    rows = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or []
    return [
        GoldQuestion(
            question=r["question"],
            relevant_doc_ids=frozenset(r.get("relevant_doc_ids", [])),
            relevant_chunk_ids=frozenset(r.get("relevant_chunk_ids", [])),
            question_type=r.get("question_type", "unknown"),
            language=r.get("language", "tr"),
            source="manual",
            relevant_doc_files=frozenset(r.get("relevant_doc_files", [])),
            reference_answer=r.get("reference_answer", ""),
        )
        for r in rows
    ]


def resolve_manual_doc_ids(questions: list[GoldQuestion], corpus_dir: str) -> list[GoldQuestion]:
    """Fill `relevant_doc_ids` from `relevant_doc_files` against `corpus_dir`.

    `document_id_for` hashes the resolved absolute path, so the id a question's
    gold document gets depends on where the corpus lives. Task 4 indexes the
    same directory it passes here, so the ids line up. Questions that already
    carry `relevant_doc_ids` (or no filenames) pass through unchanged.
    """
    from porsuk.ingestion.pipeline import document_id_for

    base = Path(corpus_dir)
    out: list[GoldQuestion] = []
    for q in questions:
        if q.relevant_doc_ids or not q.relevant_doc_files:
            out.append(q)
            continue
        ids = frozenset(document_id_for(str(base / fname)) for fname in q.relevant_doc_files)
        out.append(replace(q, relevant_doc_ids=ids))
    return out


# --- RAGTurk -----------------------------------------------------------------
#
# `metunlp/ragturk` is a git repo of 24k files, one JSON per Wikipedia
# article, NOT a `load_dataset()` dataset: `load_dataset` pulls every file
# and gets HTTP 429. We list the split's article files, sort for
# determinism, take the first `limit`, and download those one by one.
# Each article JSON holds `article` (with a global `id`), `chunks` (ids are
# article-local, e.g. `c0002`), and `questions.items` with `related_chunk_ids`
# and a `category`. The global chunk id is `f"{article_id}::{chunk_id}"`.


def _ragturk_article_files(limit: int | None, split: str) -> list[str]:
    from huggingface_hub import HfApi

    api = HfApi()
    prefix = f"{split}/dataset/json/"
    files = sorted(
        f
        for f in api.list_repo_files(RAGTURK_REPO, repo_type="dataset")
        if f.startswith(prefix) and f.endswith(".json")
    )
    return files if limit is None else files[:limit]


def _ragturk_articles(limit: int | None, split: str) -> Iterator[dict]:
    """Yield the parsed JSON of each sampled article, with backoff on 429."""
    from huggingface_hub import hf_hub_download

    for rel in _ragturk_article_files(limit, split):
        last_exc: Exception | None = None
        for attempt in range(4):
            try:
                path = hf_hub_download(RAGTURK_REPO, rel, repo_type="dataset")
                break
            except Exception as exc:  # noqa: BLE001 - HF raises several 429 types
                last_exc = exc
                time.sleep(2**attempt)
        else:
            raise RuntimeError(f"RAGTurk download kept failing for {rel}") from last_exc
        yield json.loads(Path(path).read_text(encoding="utf-8"))


def load_ragturk(limit: int | None = None, *, split: str = "formal_5k") -> list[GoldQuestion]:
    """`limit` counts ARTICLES, each of which carries ~2 questions."""
    out: list[GoldQuestion] = []
    for d in _ragturk_articles(limit, split):
        art_id = d["article"]["id"]
        for item in d["questions"]["items"]:
            chunk_ids = frozenset(f"{art_id}::{c}" for c in item["related_chunk_ids"])
            out.append(
                GoldQuestion(
                    question=item["question"],
                    relevant_doc_ids=frozenset({art_id}),
                    relevant_chunk_ids=chunk_ids,
                    question_type=item.get("category", "unknown"),
                    language="tr",
                    source="ragturk",
                    reference_answer=item["answer"],
                )
            )
    return out


def load_ragturk_corpus(
    limit: int | None = None, *, split: str = "formal_5k"
) -> list[tuple[str, str, str]]:
    """(doc_id, global_chunk_id, text) for every chunk of the sampled articles.

    Draws the SAME articles as `load_ragturk(limit)`, so every gold chunk id a
    question references resolves to a row here. Task 4 indexes these into a
    separate collection.
    """
    out: list[tuple[str, str, str]] = []
    for d in _ragturk_articles(limit, split):
        art_id = d["article"]["id"]
        for chunk in d["chunks"]:
            out.append((art_id, f"{art_id}::{chunk['id']}", chunk["content"]))
    return out


# --- XQuAD cross-lingual probe ---------------------------------------------
#
# `google/xquad` has `xquad.tr` and `xquad.en` as PARALLEL configs: row N of
# each is the same paragraph+question, translated, sharing one `id`. The
# probe indexes the ENGLISH contexts, asks the TURKISH questions, and counts
# a hit when the parallel English context (same id) comes back. It is small
# (~1 MB, 1190 rows) and loads through `load_dataset` fine (it may 429 once,
# then `datasets` retries on its own).


def load_xquad_tr(limit: int | None = None) -> list[GoldQuestion]:
    from datasets import load_dataset

    split = "validation" if limit is None else f"validation[:{limit}]"
    rows = load_dataset(XQUAD_REPO, "xquad.tr", split=split)
    return [
        GoldQuestion(
            question=row["question"],  # Turkish
            relevant_doc_ids=frozenset({row["id"]}),  # the parallel English context
            relevant_chunk_ids=frozenset({row["id"]}),
            question_type="crosslingual",
            language="tr",  # query language
            source="xquad_tr",
            reference_answer=(row["answers"]["text"][0] if row["answers"]["text"] else ""),
        )
        for row in rows
    ]


def load_xquad_en_corpus(limit: int | None = None) -> list[tuple[str, str, str]]:
    """(doc_id, chunk_id, english_context) for the eval to index.

    Consecutive xquad rows repeat a context, so each distinct `id` is emitted
    once; the result may hold fewer than `limit` rows.
    """
    from datasets import load_dataset

    split = "validation" if limit is None else f"validation[:{limit}]"
    rows = load_dataset(XQUAD_REPO, "xquad.en", split=split)
    seen: dict[str, str] = {}
    for row in rows:
        seen.setdefault(row["id"], row["context"])
    return [(cid, cid, ctx) for cid, ctx in seen.items()]
