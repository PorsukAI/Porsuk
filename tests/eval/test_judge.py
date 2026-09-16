"""Tests for the LLM-as-judge answer-accuracy scorer."""

from eval.judge import JUDGE_PROMPT, Judge
from porsuk.adapters.llm.fake import FakeLLM


def test_judge_parses_evet():
    j = Judge(FakeLLM(responses=["EVET\nReferansla aynı bilgiyi veriyor."]))
    v = j.judge("soru?", "30 gün", "Ödeme 30 gün içinde yapılır.")
    assert v.correct is True
    assert "aynı bilgi" in v.reason


def test_judge_parses_hayir():
    j = Judge(FakeLLM(responses=["HAYIR\nYanlış süre söylüyor."]))
    v = j.judge("soru?", "30 gün", "60 gün.")
    assert v.correct is False


def test_judge_unparseable_is_incorrect():
    j = Judge(FakeLLM(responses=["Belki de öyledir"]))
    v = j.judge("q", "r", "a")
    assert v.correct is False
    assert "unparseable" in v.reason


def test_judge_or_skip_returns_none_without_reference():
    from eval.goldsets import GoldQuestion

    j = Judge(FakeLLM(responses=["EVET\nx"]))
    q = GoldQuestion(
        question="q",
        relevant_doc_ids=frozenset(),
        relevant_chunk_ids=frozenset(),
        question_type="specific",
        language="tr",
        source="manual",
    )  # reference_answer defaults to ""
    assert j.judge_or_skip(q, "some answer") is None


def test_judge_or_skip_judges_with_reference():
    from eval.goldsets import GoldQuestion

    j = Judge(FakeLLM(responses=["EVET\ndoğru"]))
    q = GoldQuestion(
        question="q",
        relevant_doc_ids=frozenset(),
        relevant_chunk_ids=frozenset(),
        question_type="specific",
        language="tr",
        source="ragturk",
        reference_answer="42",
    )
    v = j.judge_or_skip(q, "cevap 42")
    assert v is not None and v.correct is True


def test_prompt_has_the_three_placeholders():
    assert "{question}" in JUDGE_PROMPT
    assert "{reference}" in JUDGE_PROMPT
    assert "{answer}" in JUDGE_PROMPT


def test_prompt_tells_the_judge_to_ignore_citation_marks():
    # ⟦chunk_id⟧ marks appear in answer.text; the judge
    # scores correctness, not format, and must not penalise them.
    assert "⟦" in JUDGE_PROMPT and "yok say" in JUDGE_PROMPT


def test_judge_records_raw():
    j = Judge(FakeLLM(responses=["EVET\ngerekçe burada"]))
    v = j.judge("q", "r", "a")
    assert v.raw == "EVET\ngerekçe burada"


def test_prompt_formats_without_keyerror():
    filled = JUDGE_PROMPT.format(question="q", reference="r", answer="a")
    assert "q" in filled
    assert "r" in filled
    assert "a" in filled
