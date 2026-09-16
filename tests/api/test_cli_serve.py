"""Tests for the `porsuk serve` CLI command."""

from __future__ import annotations

import os

from porsuk.api.cli import main


def test_serve_sets_config_env_and_calls_uvicorn(monkeypatch):
    monkeypatch.delenv("PORSUK_CONFIG", raising=False)
    monkeypatch.setenv("PORSUK_API_KEY", "test-key")
    calls: dict = {}
    monkeypatch.setattr("uvicorn.run", lambda app, **kw: calls.update(app=app, **kw))

    code = main(["serve", "--config", "config/vllm.yaml", "--host", "0.0.0.0", "--port", "9999"])

    assert code == 0
    assert calls["app"] == "porsuk.api.server:app"
    assert calls["host"] == "0.0.0.0"
    assert calls["port"] == 9999
    assert os.environ["PORSUK_CONFIG"] == "config/vllm.yaml"


def test_serve_defaults(monkeypatch):
    monkeypatch.delenv("PORSUK_CONFIG", raising=False)
    monkeypatch.setenv("PORSUK_API_KEY", "test-key")
    calls: dict = {}
    monkeypatch.setattr("uvicorn.run", lambda app, **kw: calls.update(**kw))

    main(["serve"])

    assert calls["host"] == "127.0.0.1"
    assert calls["port"] == 8080
    assert os.environ["PORSUK_CONFIG"] == "config/vllm.yaml"


def test_serve_without_api_key_returns_2(monkeypatch):
    """`serve` refuses to start without PORSUK_API_KEY, consistent with the
    lifespan check: a one-line stderr message and exit code 2, no uvicorn."""
    monkeypatch.delenv("PORSUK_API_KEY", raising=False)
    called = {"v": False}
    monkeypatch.setattr("uvicorn.run", lambda *a, **kw: called.update(v=True))

    code = main(["serve"])

    assert code == 2
    assert called["v"] is False


def test_serve_help_mentions_indexing():
    import argparse

    from porsuk.api.cli import _build_parser

    parser = _build_parser()
    # find the serve subparser's help text from _choices_actions
    serve_action = next(
        a for a in parser._subparsers._group_actions if isinstance(a, argparse._SubParsersAction)
    )
    serve_choice = next(c for c in serve_action._choices_actions if c.dest == "serve")
    assert "index" in serve_choice.help.lower()
