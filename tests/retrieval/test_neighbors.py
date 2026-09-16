from porsuk.adapters.store.inmemory import InMemoryStore
from porsuk.core.models import Chunk, ChunkHit, EmbedResult
from porsuk.retrieval.neighbors import expand_neighbours


def _mk_store():
    store = InMemoryStore()
    store.ensure_collections(embedding_fingerprint="x")
    chunks = [
        Chunk(f"c{i}", "d1", f"chunk {i} text", 1, "tr", None, None, (i * 10, i * 10 + 9))
        for i in range(5)
    ]
    store.upsert_chunks(chunks, EmbedResult(dense=tuple((0.0,) * 8 for _ in chunks)))
    return store, chunks


def test_width_zero_is_a_noop():
    store, chunks = _mk_store()
    hit = ChunkHit(chunk=chunks[2], score=1.0, retrieval_method="semantic")
    assert expand_neighbours([hit], store, width=0) == [hit]


def test_width_one_pulls_the_chunk_before_and_after():
    store, chunks = _mk_store()
    hit = ChunkHit(chunk=chunks[2], score=1.0, retrieval_method="semantic")
    (expanded,) = expand_neighbours([hit], store, width=1)
    assert "chunk 1 text" in expanded.chunk.text
    assert "chunk 2 text" in expanded.chunk.text
    assert "chunk 3 text" in expanded.chunk.text


def test_adjacent_hits_do_not_double_absorb():
    store, chunks = _mk_store()
    hits = [
        ChunkHit(chunk=chunks[1], score=1.0, retrieval_method="semantic"),
        ChunkHit(chunk=chunks[3], score=0.9, retrieval_method="semantic"),
    ]
    out = expand_neighbours(hits, store, width=1)
    # chunk 2 sits between them; it should appear in exactly one expansion
    joined = " ".join(h.chunk.text for h in out)
    assert joined.count("chunk 2 text") == 1
