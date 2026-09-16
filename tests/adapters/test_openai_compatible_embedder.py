import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from porsuk.adapters.embedding.openai_compatible import OpenAICompatibleEmbedder
from porsuk.core.ports import Embedder


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_a):
        pass

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        if self.path.endswith("/embed"):  # sparse service
            n = len(body["texts"])
            payload = {"sparse": [{"7": 0.5, "42": 0.9} for _ in range(n)], "dim": 250002}
        else:  # dense /v1/embeddings
            inp = body["input"]
            n = len(inp) if isinstance(inp, list) else 1
            payload = {"data": [{"embedding": [0.1, 0.2, 0.3, 0.4]} for _ in range(n)]}
        out = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


@pytest.fixture
def stub_url():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1"
    srv.shutdown()


@pytest.fixture
def stub_with_sparse():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    yield base + "/v1", base
    srv.shutdown()


def test_satisfies_the_port(stub_url):
    emb = OpenAICompatibleEmbedder(model="bge-m3", dim=4, base_url=stub_url)
    assert isinstance(emb, Embedder)


def test_embeds_documents_in_batches(stub_url):
    emb = OpenAICompatibleEmbedder(
        model="bge-m3", dim=4, base_url=stub_url, batch_size=2, normalize=False
    )
    result = emb.embed_documents(["a", "b", "c"])
    assert len(result.dense) == 3
    assert all(len(v) == 4 for v in result.dense)
    assert result.sparse is None


def test_embed_query_returns_one_vector(stub_url):
    emb = OpenAICompatibleEmbedder(model="bge-m3", dim=4, base_url=stub_url, normalize=False)
    result = emb.embed_query("soru")
    assert len(result.dense) == 1
    assert result.dense[0] == (0.1, 0.2, 0.3, 0.4)


def test_normalize_makes_unit_vectors(stub_url):
    emb = OpenAICompatibleEmbedder(model="bge-m3", dim=4, base_url=stub_url, normalize=True)
    (v,) = emb.embed_query("x").dense
    assert abs(sum(c * c for c in v) ** 0.5 - 1.0) < 1e-6


def test_trailing_slash_in_base_url_is_tolerated(stub_url):
    emb = OpenAICompatibleEmbedder(model="bge-m3", dim=4, base_url=stub_url + "/", normalize=False)
    assert len(emb.embed_query("x").dense) == 1


def test_empty_document_list_makes_no_request(stub_url):
    emb = OpenAICompatibleEmbedder(model="bge-m3", dim=4, base_url=stub_url)
    result = emb.embed_documents([])
    assert result.dense == ()


def test_no_sparse_when_sparse_base_url_unset(stub_url):
    emb = OpenAICompatibleEmbedder(model="bge-m3", dim=4, base_url=stub_url)
    result = emb.embed_documents(["a", "b"])
    assert result.sparse is None


def test_sparse_populated_when_sparse_base_url_set(stub_with_sparse):
    dense_url, sparse_base = stub_with_sparse
    emb = OpenAICompatibleEmbedder(
        model="bge-m3", dim=4, base_url=dense_url, sparse_base_url=sparse_base
    )
    result = emb.embed_documents(["a", "b"])
    assert len(result.dense) == 2
    assert result.sparse is not None
    assert len(result.sparse) == 2
    assert result.sparse[0] == {7: 0.5, 42: 0.9}


def test_query_gets_sparse_too(stub_with_sparse):
    dense_url, sparse_base = stub_with_sparse
    emb = OpenAICompatibleEmbedder(
        model="bge-m3", dim=4, base_url=dense_url, sparse_base_url=sparse_base
    )
    result = emb.embed_query("soru")
    assert len(result.dense) == 1
    assert result.sparse is not None
    assert result.sparse[0] == {7: 0.5, 42: 0.9}


class _FlakyHandler(BaseHTTPRequestHandler):
    fail_first = 2
    _seen = 0

    def log_message(self, *_a):
        pass

    def do_POST(self):
        _FlakyHandler._seen += 1
        length = int(self.headers["Content-Length"])
        self.rfile.read(length)
        if _FlakyHandler._seen <= _FlakyHandler.fail_first:
            self.send_response(530)
            self.end_headers()
            return
        out = json.dumps({"data": [{"embedding": [0.1, 0.2, 0.3, 0.4]}]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


def test_transient_5xx_is_retried(monkeypatch):
    monkeypatch.setattr("porsuk.adapters.embedding.openai_compatible._BACKOFF_BASE_S", 0.0)
    _FlakyHandler._seen = 0
    srv = HTTPServer(("127.0.0.1", 0), _FlakyHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        emb = OpenAICompatibleEmbedder(
            model="bge-m3",
            dim=4,
            base_url=f"http://127.0.0.1:{srv.server_address[1]}/v1",
            normalize=False,
        )
        result = emb.embed_query("x")
        assert len(result.dense) == 1
        assert _FlakyHandler._seen == 3  # 2 failures then success
    finally:
        srv.shutdown()


def test_wrong_dimension_from_endpoint_raises(stub_url):
    # the stub returns 4-dim vectors; configuring dim=8 must fail fast
    emb = OpenAICompatibleEmbedder(model="bge-m3", dim=8, base_url=stub_url)
    with pytest.raises(ValueError, match="returned dim 4, config says 8"):
        emb.embed_query("x")
