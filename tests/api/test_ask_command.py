"""Tests for the `porsuk ask` CLI command."""

from __future__ import annotations

import json

from porsuk.api.cli import main


def _answer(*, sources=None, documents_seen=None, truncated=False):
    from porsuk.agent.models import AgentAnswer, Source

    if sources is None:
        sources = [Source("Sozlesme.pdf", 12, "3.2 Ödeme", "Ödemeler ... 30 gün ...")]
    if documents_seen is None:
        documents_seen = [Source("Sozlesme.pdf", 8, None, "")]
    return AgentAnswer(
        text="Ödeme 30 gün içinde yapılır.",
        sources=sources,
        documents_seen=documents_seen,
        tool_calls=2,
        truncated=truncated,
    )


def _patch_runner(monkeypatch, answer=None, *, spy=None):
    """Replace build_agent_runner with one whose inner run returns `answer`
    (default `_answer()`) and records its call kwargs into `spy` if given."""
    ans = answer if answer is not None else _answer()

    def fake_build(cfg, **_kw):
        def run(question, *, lang=None, verbose=False):
            if spy is not None:
                spy["question"] = question
                spy["lang"] = lang
                spy["verbose"] = verbose
            return ans

        return run

    monkeypatch.setattr("porsuk.core.container.build_agent_runner", fake_build)


def test_ask_prints_both_blocks(monkeypatch, capsys):
    _patch_runner(monkeypatch)
    code = main(["ask", "ödeme koşulu ne?"])
    out = capsys.readouterr().out
    assert code == 0
    assert "CEVAP: Ödeme 30 gün içinde yapılır." in out
    assert "KAYNAKLAR (agent'ın atıf verdiği):" in out
    assert "BAKILAN DOKÜMANLAR (döngüde dokunulan):" in out
    assert "Sozlesme.pdf" in out
    assert 's.12 · "3.2 Ödeme"' in out
    assert '"Ödemeler ... 30 gün ..."' in out


def test_ask_json(monkeypatch, capsys):
    _patch_runner(monkeypatch)
    code = main(["ask", "x", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["text"].startswith("Ödeme")
    assert payload["sources"][0]["file"] == "Sozlesme.pdf"
    assert payload["truncated"] is False


def test_ask_warns_on_inmemory_store(monkeypatch, capsys):
    _patch_runner(monkeypatch)
    main(["ask", "x"])
    err = capsys.readouterr().err
    assert "does not" in err and "persist" in err
    assert "porsuk ask" in err


def test_ask_empty_sources_prints_yok(monkeypatch, capsys):
    _patch_runner(monkeypatch, _answer(sources=[]))
    main(["ask", "x"])
    out = capsys.readouterr().out
    _, _, rest = out.partition("KAYNAKLAR (agent'ın atıf verdiği):\n")
    assert rest.startswith("  (yok)")


def test_ask_truncated_note_on_stderr(monkeypatch, capsys):
    _patch_runner(monkeypatch, _answer(truncated=True))
    main(["ask", "x"])
    captured = capsys.readouterr()
    assert "[adım limiti aşıldı — cevap eldeki bilgiyle verildi]" in captured.err
    assert "adım limiti" not in captured.out


def test_ask_threads_lang_and_verbose_through_to_the_runner(monkeypatch, capsys):
    spy: dict = {}
    _patch_runner(monkeypatch, spy=spy)
    main(["ask", "soru", "--lang", "tr", "--verbose"])
    assert spy["question"] == "soru"
    assert spy["lang"] == "tr"
    assert spy["verbose"] is True


def test_ask_config_error_returns_2(monkeypatch, capsys):
    _patch_runner(monkeypatch)
    code = main(["ask", "x", "--config", "config/does-not-exist.yaml"])
    assert code == 2
    assert "config error" in capsys.readouterr().err


def test_ask_human_output_strips_citation_marks(monkeypatch, capsys):
    """porsuk ask (no --json) prints the answer without ⟦…⟧ marks; the
    --json path keeps them."""
    from porsuk.agent.models import AgentAnswer, Source

    fake = AgentAnswer(
        text="Ödeme 30 gün ⟦c1⟧. Faiz %2 ⟦c2⟧.",
        sources=[Source("a.pdf", 1, None, "q", "c1"), Source("b.pdf", 2, None, "q", "c2")],
        documents_seen=[],
        tool_calls=1,
    )
    _patch_runner(monkeypatch, answer=fake)
    code = main(["ask", "ödeme?"])
    out = capsys.readouterr().out
    assert code == 0
    assert "⟦" not in out and "⟧" not in out
    # The strip removes ' ⟦c1⟧' and ' ⟦c2⟧' including preceding whitespace
    assert "Ödeme 30 gün. Faiz %2." in out


def test_ask_json_output_preserves_citation_marks(monkeypatch, capsys):
    """--json output includes ⟦…⟧ marks in the raw AgentAnswer.text."""
    from porsuk.agent.models import AgentAnswer, Source

    fake = AgentAnswer(
        text="Ödeme 30 gün ⟦c1⟧. Faiz %2 ⟦c2⟧.",
        sources=[Source("a.pdf", 1, None, "q", "c1"), Source("b.pdf", 2, None, "q", "c2")],
        documents_seen=[],
        tool_calls=1,
    )
    _patch_runner(monkeypatch, answer=fake)
    code = main(["ask", "ödeme?", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    # JSON output must preserve the marks
    assert "⟦c1⟧" in payload["text"]
    assert "⟦c2⟧" in payload["text"]
