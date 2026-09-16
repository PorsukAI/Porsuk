"""Tests for the golden-set loaders (manual, RAGTurk, XQuAD)."""

import pytest

from eval.goldsets import (
    GoldQuestion,
    load_manual,
    load_ragturk,
    load_ragturk_corpus,
    load_xquad_en_corpus,
    load_xquad_tr,
    resolve_manual_doc_ids,
)


def test_gold_question_shape():
    q = GoldQuestion(
        question="İş Kanunu'nda fazla mesai ücreti nasıl hesaplanır?",
        relevant_doc_ids=frozenset({"mevzuat_4857"}),
        relevant_chunk_ids=frozenset({"mevzuat_4857::c41"}),
        question_type="specific",
        language="tr",
        source="manual",
    )
    assert q.language == "tr"


def test_load_manual_reads_the_yaml(tmp_path):
    p = tmp_path / "g.yaml"
    p.write_text(
        "- question: Fazla mesai ücreti nasıl hesaplanır?\n"
        "  relevant_doc_files: [mevzuat_4857.pdf]\n"
        "  relevant_chunk_ids: []\n"
        "  question_type: specific\n"
        "  language: tr\n"
    )
    (q,) = load_manual(str(p))
    assert q.source == "manual"
    assert "mevzuat_4857.pdf" in q.relevant_doc_files
    assert q.relevant_doc_ids == frozenset()  # resolved at eval time


def test_resolve_manual_doc_ids(tmp_path):
    from porsuk.ingestion.pipeline import document_id_for

    (tmp_path / "x.pdf").write_bytes(b"%PDF-1.4\n")
    q = GoldQuestion(
        question="?",
        relevant_doc_ids=frozenset(),
        relevant_chunk_ids=frozenset(),
        question_type="specific",
        language="tr",
        source="manual",
        relevant_doc_files=frozenset({"x.pdf"}),
    )
    (resolved,) = resolve_manual_doc_ids([q], str(tmp_path))
    assert resolved.relevant_doc_ids == frozenset({document_id_for(str(tmp_path / "x.pdf"))})


def test_committed_manual_set_is_well_formed():
    from eval.goldsets import load_manual

    qs = load_manual()  # the real eval/manual_goldset.yaml
    assert qs, "manual_goldset.yaml must have at least a starter set"
    # The set was expanded from 12 to ~40 so chunk size / overlap /
    # neighbor / stemmer can be swept against it.
    assert len(qs) >= 38
    types = {q.question_type for q in qs}
    assert types <= {"specific", "conceptual"}
    assert "specific" in types and "conceptual" in types
    assert all(q.relevant_doc_files for q in qs), "every question needs a gold doc"
    n_specific = sum(1 for q in qs if q.question_type == "specific")
    n_conceptual = sum(1 for q in qs if q.question_type == "conceptual")
    assert n_specific >= 15 and n_conceptual >= 15


def test_manual_set_covers_the_definition_regression():
    from eval.goldsets import load_manual

    qs = load_manual()
    # the demo regression class: a definition question over mevzuat_2547
    assert any(
        "mevzuat_2547.pdf" in q.relevant_doc_files
        and ("tanım" in q.question.lower() or "kimdir" in q.question.lower())
        for q in qs
    )


def test_manual_reference_answer_is_optional():
    qs = load_manual()  # the real eval/manual_goldset.yaml
    assert qs, "manual_goldset.yaml must load"
    assert all(isinstance(q.reference_answer, str) for q in qs)


def test_ragturk_carries_reference_answer(monkeypatch):
    from eval import goldsets

    article = {
        "article": {"id": "a0001"},
        "questions": {
            "items": [
                {
                    "question": "Soru?",
                    "answer": "Altın referans cevap.",
                    "related_chunk_ids": ["c0001"],
                    "category": "specific",
                }
            ]
        },
    }
    monkeypatch.setattr(goldsets, "_ragturk_articles", lambda limit, split: iter([article]))
    (q,) = load_ragturk(limit=1)
    assert q.reference_answer == "Altın referans cevap."
    assert q.reference_answer


@pytest.mark.needs_network
def test_load_ragturk_shape():
    qs = load_ragturk(limit=10)  # 10 articles, ~2 questions each
    assert qs
    assert all(isinstance(q, GoldQuestion) for q in qs)
    assert all(q.relevant_chunk_ids for q in qs)
    assert all("::" in next(iter(q.relevant_chunk_ids)) for q in qs)  # global id


@pytest.mark.needs_network
def test_ragturk_corpus_covers_the_gold_chunks():
    qs = load_ragturk(limit=5)
    corpus = load_ragturk_corpus(limit=5)
    corpus_chunk_ids = {cid for _, cid, _ in corpus}
    gold = set().union(*(q.relevant_chunk_ids for q in qs))
    assert gold <= corpus_chunk_ids


@pytest.mark.needs_network
def test_load_xquad_tr_shape():
    qs = load_xquad_tr(limit=20)
    assert len(qs) == 20
    assert all(isinstance(q, GoldQuestion) for q in qs)
    assert all(q.question_type == "crosslingual" for q in qs)
    assert all(q.source == "xquad_tr" for q in qs)
    assert all(q.language == "tr" for q in qs)


@pytest.mark.needs_network
def test_xquad_en_corpus_is_english():
    corpus = load_xquad_en_corpus(limit=20)
    assert corpus
    corpus_ids = {cid for cid, _, _ in corpus}
    gold_ids = set().union(*(q.relevant_doc_ids for q in load_xquad_tr(limit=20)))
    assert gold_ids <= corpus_ids
