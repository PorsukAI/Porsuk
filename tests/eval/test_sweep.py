import textwrap

from eval.sweep import _REINDEX_KEYS, SWEEP_GRID, SweepRow, _cells_for, _patch
from porsuk.core.config import load_config


def test_grid_has_the_spec_14_rows():
    keys = {k for k, _ in SWEEP_GRID}
    assert "chunk.target_chars" in keys
    assert "retrieval.rerank_enabled" in keys
    assert "retrieval.neighbor_expansion" in keys


def test_cells_hold_others_at_default():
    cells = list(_cells_for("retrieval.chunk_k", [5, 10, 20], base={"retrieval.chunk_k": 10}))
    assert len(cells) == 3
    assert all(len(c) == 1 for c in cells)  # only the swept key varies


_BASE = textwrap.dedent(
    """
    llm:
      provider: fake
      model: test-model
    embedder:
      provider: fake
      dim: 8
    store:
      provider: inmemory
    """
)


def _cfg(tmp_path):
    p = tmp_path / "profile.yaml"
    p.write_text(_BASE, encoding="utf-8")
    return load_config(p, env={})


def test_patch_config_dotted_key(tmp_path):
    cfg = _cfg(tmp_path)
    orig_target = cfg.chunking.target_chars
    orig_k = cfg.retrieval.chunk_k

    patched = _patch(cfg, {"chunk.target_chars": 1600, "retrieval.chunk_k": 20})

    assert patched.chunking.target_chars == 1600
    assert patched.retrieval.chunk_k == 20
    # original untouched
    assert cfg.chunking.target_chars == orig_target
    assert cfg.retrieval.chunk_k == orig_k
    # other fields intact
    assert patched.chunking.overlap_ratio == cfg.chunking.overlap_ratio
    assert patched.retrieval.strategy == cfg.retrieval.strategy
    assert patched.llm.model == cfg.llm.model


def test_reindex_vs_retrieval_only_split():
    assert "chunk.target_chars" in _REINDEX_KEYS
    assert "chunk.overlap_ratio" in _REINDEX_KEYS
    assert "retrieval.chunk_k" not in _REINDEX_KEYS
    assert "retrieval.rerank_enabled" not in _REINDEX_KEYS


def test_patch_handles_agent_keys():
    from eval.sweep import _patch
    from porsuk.core.config import load_config

    cfg = load_config("config/local.yaml")
    patched = _patch(cfg, {"agent.max_tool_calls": 12})
    assert patched.agent.max_tool_calls == 12
    assert cfg.agent.max_tool_calls != 12  # original untouched


def test_sweep_row_shape():
    row = SweepRow(
        parameter="retrieval.chunk_k",
        value=10,
        goldset="ragturk",
        strategy="semantic",
        question_type="all",
        n=600,
        recall_at_5=0.7,
        mrr=0.5,
        ndcg_at_5=0.6,
    )
    assert row.parameter == "retrieval.chunk_k"
    assert row.value == 10
    assert row.n == 600
