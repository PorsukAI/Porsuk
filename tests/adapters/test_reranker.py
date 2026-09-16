import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from porsuk.adapters.rerank.fake import FakeReranker
from porsuk.adapters.rerank.flag_embedding_http import FlagEmbeddingReranker
from porsuk.adapters.rerank.openrouter import OpenRouterReranker
from porsuk.core.models import Chunk, ChunkHit
from porsuk.core.ports import Reranker


def _hit(cid, text, score):
    return ChunkHit(
        chunk=Chunk(cid, "d1", text, 1, "tr", None, None, (0, len(text))),
        score=score,
        retrieval_method="semantic",
    )


def test_fake_satisfies_the_port():
    assert isinstance(FakeReranker(), Reranker)


def test_fake_reverses_and_caps():
    hits = [_hit("a", "x", 0.9), _hit("b", "y", 0.8), _hit("c", "z", 0.7)]
    out = FakeReranker().rerank("q", hits, top_n=2)
    assert [h.chunk.chunk_id for h in out] == ["b", "a"]
    assert out[0].rerank_score is not None


class _RerankStub(BaseHTTPRequestHandler):
    def log_message(self, *_a):
        pass

    def do_POST(self):
        n = len(json.loads(self.rfile.read(int(self.headers["Content-Length"])))["texts"])
        body = json.dumps({"scores": list(range(n, 0, -1))}).encode()  # first text scores highest
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def rerank_url():
    srv = HTTPServer(("127.0.0.1", 0), _RerankStub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_http_reranker_sorts_by_score(rerank_url):
    r = FlagEmbeddingReranker(base_url=rerank_url)
    hits = [_hit("a", "x", 0.1), _hit("b", "y", 0.2), _hit("c", "z", 0.3)]
    out = r.rerank("q", hits, top_n=3)
    # stub scores first text highest -> order preserved, rerank_score descending
    assert [h.chunk.chunk_id for h in out] == ["a", "b", "c"]
    assert out[0].rerank_score > out[1].rerank_score


class _OpenRouterRerankStub(BaseHTTPRequestHandler):
    def log_message(self, *_a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        n = len(body["documents"])
        # OpenRouter's shape: results carry the original index + a score,
        # not pre-sorted by this stub (real endpoint sorts; we don't rely on it).
        results = [{"index": i, "relevance_score": 1.0 / (i + 1)} for i in range(n)]
        resp = json.dumps({"results": results}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)


@pytest.fixture
def openrouter_rerank_url():
    srv = HTTPServer(("127.0.0.1", 0), _OpenRouterRerankStub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_openrouter_reranker_sorts_by_relevance_score(openrouter_rerank_url):
    r = OpenRouterReranker(model="qwen/qwen3-reranker-8b", base_url=openrouter_rerank_url)
    hits = [_hit("a", "x", 0.1), _hit("b", "y", 0.2), _hit("c", "z", 0.3)]
    out = r.rerank("q", hits, top_n=3)
    # stub gives index 0 the highest score -> "a" first, descending after
    assert [h.chunk.chunk_id for h in out] == ["a", "b", "c"]
    assert out[0].rerank_score > out[1].rerank_score > out[2].rerank_score


def test_openrouter_reranker_satisfies_the_port():
    assert isinstance(OpenRouterReranker(model="m", base_url="http://x"), Reranker)
