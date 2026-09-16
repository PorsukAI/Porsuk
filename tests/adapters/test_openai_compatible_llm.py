import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from porsuk.adapters.llm.openai_compatible import OpenAICompatibleLLM
from porsuk.core.ports import LLMProvider


def test_satisfies_the_port():
    assert isinstance(OpenAICompatibleLLM(model="m", base_url="http://x"), LLMProvider)


class _LLMStub(BaseHTTPRequestHandler):
    fail_times = 0
    _seen = 0
    last_body: dict | None = None

    def log_message(self, *_a):
        pass

    def do_POST(self):
        _LLMStub._seen += 1
        length = int(self.headers["Content-Length"])
        _LLMStub.last_body = json.loads(self.rfile.read(length))
        if _LLMStub._seen <= _LLMStub.fail_times:
            self.send_response(503)
            self.end_headers()
            return
        out = json.dumps({"choices": [{"message": {"content": "merhaba"}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


@pytest.fixture
def stub_url():
    _LLMStub.fail_times = 0
    _LLMStub._seen = 0
    _LLMStub.last_body = None
    srv = HTTPServer(("127.0.0.1", 0), _LLMStub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1"
    srv.shutdown()


def test_complete_posts_and_extracts_content(stub_url):
    llm = OpenAICompatibleLLM(model="qwen3-4b", base_url=stub_url)
    assert llm.complete("selam") == "merhaba"
    assert _LLMStub.last_body["messages"] == [{"role": "user", "content": "selam"}]
    assert _LLMStub.last_body["max_tokens"] == 512
    assert _LLMStub.last_body["model"] == "qwen3-4b"


def test_complete_retries_on_5xx(stub_url, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_a: None)
    _LLMStub.fail_times = 2
    llm = OpenAICompatibleLLM(model="m", base_url=stub_url)
    assert llm.complete("x") == "merhaba"
    assert _LLMStub._seen == 3


def test_complete_raises_after_retries(stub_url, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_a: None)
    _LLMStub.fail_times = 999
    llm = OpenAICompatibleLLM(model="m", base_url=stub_url)
    with pytest.raises(RuntimeError, match="failed after 5 tries"):
        llm.complete("x")


def test_complete_defaults_to_thinking_enabled(stub_url):
    llm = OpenAICompatibleLLM(model="m", base_url=stub_url)
    llm.complete("x")
    assert "chat_template_kwargs" not in _LLMStub.last_body


def test_complete_can_disable_thinking(stub_url):
    llm = OpenAICompatibleLLM(model="m", base_url=stub_url)
    llm.complete("x", enable_thinking=False)
    assert _LLMStub.last_body["chat_template_kwargs"] == {"enable_thinking": False}


def test_registered():
    from porsuk.core.container import _load_adapters
    from porsuk.core.registry import build

    _load_adapters()
    llm = build("llm", {"provider": "openai_compatible", "model": "m", "base_url": "http://x"})
    assert llm.model == "m"
