"""The system prompt names the two prompted searches, not hybrid, and asks
for inline chunk_id citation marks."""

from __future__ import annotations

from porsuk.agent.prompts import SYSTEM_PROMPT


def test_prompt_names_the_two_prompted_searches_not_hybrid():
    assert "keyword_search" in SYSTEM_PROMPT
    assert "semantic_search" in SYSTEM_PROMPT
    assert "expand_context" in SYSTEM_PROMPT
    assert "hybrid_search" not in SYSTEM_PROMPT


def test_prompt_asks_for_inline_chunk_id_marks():
    assert "⟦" in SYSTEM_PROMPT
    assert "⟧" in SYSTEM_PROMPT
    assert "gösterilir" in SYSTEM_PROMPT  # the marks are shown to the user


def test_prompt_has_the_cross_lingual_note():
    assert "İngilizce" in SYSTEM_PROMPT


def test_prompt_has_a_stop_rule_for_repeated_empty_searches():
    assert "BULAMADIM" in SYSTEM_PROMPT
    assert "DURMA KURALI" in SYSTEM_PROMPT


def test_mark_regex_matches_a_real_chunk_id():
    from porsuk.agent.run import _MARK

    real_id = "6310c32d-dca8-5ee8-8ea7-ea62a9ff29ac:00007"
    assert _MARK.findall(f"cümle ⟦{real_id}⟧.") == [real_id]


def test_system_prompt_uses_inline_chunk_id_marks():
    assert "⟦chunk_id⟧" in SYSTEM_PROMPT
    assert "⟦c3f9a1⟧⟦b2e8d4⟧" in SYSTEM_PROMPT  # the multi-source example
    # the old scheme is gone from the instruction
    assert "KULLANILAN:" not in SYSTEM_PROMPT
    assert "[Kaynak:" not in SYSTEM_PROMPT
