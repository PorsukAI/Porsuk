"""Tests for config loading, validation, and env-var interpolation."""

import textwrap
from pathlib import Path

import pytest

from porsuk.core.config import ConfigError, load_config


def _write(tmp_path, body: str):
    p = tmp_path / "profile.yaml"
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return p


BASE = """
    llm:
      provider: fake
      model: test-model
    embedder:
      provider: fake
      dim: 8
    store:
      provider: inmemory
    """


def test_loads_defaults_for_open_parameters(tmp_path):
    cfg = load_config(_write(tmp_path, BASE))
    assert cfg.chunking.target_chars == 1200
    assert cfg.chunking.overlap_ratio == 0.15
    assert cfg.retrieval.rerank_enabled is False
    assert cfg.retrieval.neighbor_expansion == 0
    assert cfg.retrieval.profile_k == 15
    assert cfg.profile.yake_top_n == 15
    assert cfg.agent.max_tool_calls == 8
    assert cfg.serving.context_length == 16384
    assert cfg.serving.kv_cache_dtype == "fp8"  # measured
    assert cfg.serving.gpu_memory_utilization == 0.75
    assert cfg.profile.llm is None


def test_pipeline_defaults(tmp_path):
    cfg = load_config(_write(tmp_path, BASE))
    assert cfg.pipeline.parse_workers == 0
    assert cfg.pipeline.embed_batch_size == 32
    assert cfg.profile.llm_enabled is True


def test_pipeline_overrides(tmp_path):
    body = (
        BASE
        + """
    pipeline:
      parse_workers: 4
      embed_batch_size: 64
    profile:
      llm_enabled: false
    """
    )
    cfg = load_config(_write(tmp_path, body))
    assert cfg.pipeline.parse_workers == 4
    assert cfg.pipeline.embed_batch_size == 64
    assert cfg.profile.llm_enabled is False


def test_unknown_key_is_rejected(tmp_path):
    body = (
        BASE
        + """
    chunking:
      taget_chars: 900
    """
    )
    with pytest.raises(ConfigError) as exc:
        load_config(_write(tmp_path, body))
    assert "taget_chars" in str(exc.value)


def test_api_key_is_resolved_from_named_env_var(tmp_path):
    body = """
    llm:
      provider: fake
      model: test-model
      api_key_env: MY_SECRET
    embedder:
      provider: fake
      dim: 8
    store:
      provider: inmemory
    """
    cfg = load_config(_write(tmp_path, body), env={"MY_SECRET": "s3cret"})
    assert cfg.llm.api_key == "s3cret"


def test_missing_env_var_fails_at_load_time(tmp_path):
    body = """
    llm:
      provider: fake
      model: test-model
      api_key_env: ABSENT_VAR
    embedder:
      provider: fake
      dim: 8
    store:
      provider: inmemory
    """
    with pytest.raises(ConfigError) as exc:
        load_config(_write(tmp_path, body), env={})
    assert "ABSENT_VAR" in str(exc.value)


def test_env_interpolation_in_string_values(tmp_path):
    body = """
    llm:
      provider: fake
      model: test-model
      base_url: ${LLM_HOST}/v1
    embedder:
      provider: fake
      dim: 8
    store:
      provider: inmemory
    """
    cfg = load_config(_write(tmp_path, body), env={"LLM_HOST": "http://x"})
    assert cfg.llm.base_url == "http://x/v1"


def test_overlap_ratio_out_of_range_is_rejected(tmp_path):
    body = (
        BASE
        + """
    chunking:
      overlap_ratio: 1.5
    """
    )
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, body))


def test_shipped_profiles_load(tmp_path):
    local = load_config("config/local.yaml", env={"LLM_API_KEY": "x"})
    assert local.store.provider == "inmemory"

    vllm = load_config(
        "config/vllm.yaml",
        env={
            "LLM_API_KEY": "x",
            "VLLM_BASE_URL": "http://localhost:8000/v1",
            "VLLM_MODEL": "test-model",
            "EMBEDDING_BASE_URL": "http://localhost:18000/v1",
            "EMBEDDING_MODEL": "bge-m3",
            "RERANK_BASE_URL": "http://localhost:18002",
        },
    )
    assert vllm.store.provider == "qdrant"
    assert vllm.llm.base_url == "http://localhost:8000/v1"
    assert vllm.embedder.provider == "openai_compatible"
    assert vllm.embedder.base_url == "http://localhost:18000/v1"
    # Reranking is on by default after measurement; RERANK_BASE_URL
    # is a required var there (docs/eval/2026-09-08-parameter-sweep.md).
    assert vllm.retrieval.rerank_enabled is True
    assert vllm.retrieval.rerank.base_url == "http://localhost:18002"


def test_serving_rejects_an_unknown_kv_cache_dtype(tmp_path):
    body = (
        BASE
        + """
    serving:
      kv_cache_dtype: int4
    """
    )
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, body))


def test_profile_can_name_a_separate_llm(tmp_path):
    body = (
        BASE
        + """
    profile:
      llm:
        provider: fake
        model: small-profiler
    """
    )
    cfg = load_config(_write(tmp_path, body))
    assert cfg.profile.llm is not None
    assert cfg.profile.llm.model == "small-profiler"


def test_nested_profile_llm_secret_is_resolved(tmp_path):
    body = (
        BASE
        + """
    profile:
      llm:
        provider: fake
        model: small-profiler
        api_key_env: PROFILE_LLM_KEY
    """
    )
    cfg = load_config(_write(tmp_path, body), env={"PROFILE_LLM_KEY": "p-s3cret"})
    assert cfg.profile.llm is not None
    assert cfg.profile.llm.api_key == "p-s3cret"


def test_embedder_normalisation_defaults_to_true(tmp_path):
    cfg = load_config(_write(tmp_path, BASE))
    assert cfg.embedder.normalize is True


def test_gpu_memory_utilization_rejects_values_outside_the_usable_band(tmp_path):
    body = (
        BASE
        + """
    serving:
      gpu_memory_utilization: 0.05
    """
    )
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, body))


def test_missing_nested_env_var_fails_at_load_time(tmp_path):
    body = (
        BASE
        + """
    profile:
      llm:
        provider: fake
        model: small-profiler
        api_key_env: ABSENT_PROFILE_KEY
    """
    )
    with pytest.raises(ConfigError) as exc:
        load_config(_write(tmp_path, body), env={})
    assert "ABSENT_PROFILE_KEY" in str(exc.value)


def test_quality_config_exposes_normalisation_bounds_and_weights(tmp_path):
    """Components are normalised to 0-1 before weighting.

    Without normalisation the largest-unit component (chars_per_page, in the
    hundreds) alone determines the score and the weights are
    decorative.
    """
    path = tmp_path / "profile.yaml"
    path.write_text(
        "llm: {provider: fake, model: m}\n"
        "embedder: {provider: fake, dim: 8}\n"
        "store: {provider: inmemory}\n",
        encoding="utf-8",
    )
    cfg = load_config(path, env={})
    q = cfg.quality

    for name in (
        "chars_per_page",
        "bad_char_ratio",
        "text_plausibility",
        "empty_table_blocks",
    ):
        assert hasattr(q, f"{name}_good"), f"missing normalisation bound {name}_good"
        assert hasattr(q, f"{name}_bad"), f"missing normalisation bound {name}_bad"
        assert hasattr(q, f"weight_{name}"), f"missing weight_{name}"

    weights = (
        q.weight_chars_per_page
        + q.weight_bad_char_ratio
        + q.weight_text_plausibility
        + q.weight_empty_table_blocks
    )
    assert abs(weights - 1.0) < 1e-9, f"weights must sum to 1.0, got {weights}"
    assert not hasattr(q, "weight_dict_hit_rate")


def test_malformed_yaml_raises_config_error(tmp_path):
    """A hand-edited profile with a syntax error must not leak a raw yaml.YAMLError.

    The CLI only catches (ConfigError, OSError); a bare yaml.YAMLError would
    escape as an unhandled exception and print a traceback.
    """
    path = tmp_path / "profile.yaml"
    path.write_text("llm: {provider: fake\n  broken: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError) as exc:
        load_config(path, env={})
    assert "profile.yaml" in str(exc.value)


def test_parsing_config_lists_the_chain_in_cost_order(tmp_path):
    """The router is built from a config-declared parser list.

    Naming the chain in config is what keeps `build_router` free of provider
    branching.
    """
    path = tmp_path / "profile.yaml"
    path.write_text(
        "llm: {provider: fake, model: m}\n"
        "embedder: {provider: fake, dim: 8}\n"
        "store: {provider: inmemory}\n"
        "parsing: {parsers: [pymupdf, text]}\n",
        encoding="utf-8",
    )
    cfg = load_config(path, env={})
    assert cfg.parsing.parsers == ("pymupdf", "text")


def test_negative_weights_are_rejected(tmp_path):
    """An invalid profile loaded happily and produced a silently wrong score:
    a negative weight, or all four at zero, made every document score 0.0,
    indistinguishable from "every document is broken"."""
    profile = tmp_path / "bad.yaml"
    profile.write_text(
        "llm: {provider: fake, model: m}\n"
        "embedder: {provider: fake, dim: 8}\n"
        "store: {provider: inmemory}\n"
        "quality: {weight_chars_per_page: -1.0}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        load_config(str(profile), env={})


def test_weights_that_do_not_sum_to_one_are_rejected(tmp_path):
    """`config/local.yaml` states "Weights sum to 1.0" as though it were
    enforced. It was not: `score` divides by the total, so a profile summing to
    0.5 quietly changes what every threshold means."""
    profile = tmp_path / "bad.yaml"
    profile.write_text(
        "llm: {provider: fake, model: m}\n"
        "embedder: {provider: fake, dim: 8}\n"
        "store: {provider: inmemory}\n"
        "quality:\n"
        "  weight_chars_per_page: 0.1\n"
        "  weight_bad_char_ratio: 0.1\n"
        "  weight_text_plausibility: 0.1\n"
        "  weight_empty_table_blocks: 0.0\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        load_config(str(profile), env={})


def test_a_docling_threshold_of_zero_is_rejected(tmp_path):
    """At 0 every document escalates to Docling, including clean ones."""
    profile = tmp_path / "bad.yaml"
    profile.write_text(
        "llm: {provider: fake, model: m}\n"
        "embedder: {provider: fake, dim: 8}\n"
        "store: {provider: inmemory}\n"
        "quality: {docling_min_empty_table_blocks: 0}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        load_config(str(profile), env={})


def test_the_shipped_profile_matches_the_calibrated_defaults():
    """`config/local.yaml` restates the quality block rather than inheriting
    it, and its own comment concedes the file is what CI and the demo actually
    run on. Nothing checked the two had not drifted."""
    import yaml

    from porsuk.core.config import QualityConfig

    raw = yaml.safe_load(Path("config/local.yaml").read_text(encoding="utf-8"))
    defaults = QualityConfig()
    for key, value in raw["quality"].items():
        assert getattr(defaults, key) == value, f"{key} drifted from the calibrated default"


def test_env_var_optional_with_default(tmp_path):
    body = (
        BASE
        + """
    embedder:
      provider: fake
      dim: 8
      sparse_base_url: ${MISSING_SPARSE:-}
    """
    )
    cfg = load_config(_write(tmp_path, body), env={})
    assert cfg.embedder.sparse_base_url in (None, "")


def test_env_var_optional_default_used_when_empty(tmp_path):
    body = (
        BASE
        + """
    llm:
      provider: fake
      model: m
      base_url: ${LLM_HOST:-http://fallback/v1}
    """
    )
    cfg = load_config(_write(tmp_path, body), env={"LLM_HOST": ""})
    assert cfg.llm.base_url == "http://fallback/v1"


def test_retrieval_strategy_default_and_override(tmp_path):
    cfg = load_config(_write(tmp_path, BASE))
    assert cfg.retrieval.strategy == "semantic"
    assert cfg.retrieval.rerank is None

    body = (
        BASE
        + """
    retrieval:
      strategy: hybrid
      rerank:
        provider: flag_embedding_http
        base_url: http://localhost:18002
    """
    )
    cfg = load_config(_write(tmp_path, body))
    assert cfg.retrieval.strategy == "hybrid"
    assert cfg.retrieval.rerank.provider == "flag_embedding_http"
    assert cfg.retrieval.rerank.base_url == "http://localhost:18002"


def test_retrieval_strategy_rejects_unknown(tmp_path):
    body = BASE + "\n    retrieval:\n      strategy: fuzzy\n"
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, body))


def test_rerank_enabled_without_a_rerank_block_is_rejected(tmp_path):
    body = BASE + "\n    retrieval:\n      rerank_enabled: true\n"
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, body))


def test_agent_config_defaults():
    from porsuk.core.config import AgentConfig

    a = AgentConfig()
    assert a.provider == "fake_chat"
    assert a.max_tool_calls == 8
    assert a.max_context_chars == 12_000
    assert a.expand_before_chars == 500 and a.expand_after_chars == 500
    assert a.temperature == 0.0


def test_agent_config_rejects_bad_values():
    import pytest
    from pydantic import ValidationError

    from porsuk.core.config import AgentConfig

    with pytest.raises(ValidationError):
        AgentConfig(max_tool_calls=0)
    with pytest.raises(ValidationError):
        AgentConfig(temperature=-1.0)


def test_ingest_config_defaults():
    from porsuk.core.config import IngestConfig

    cfg = IngestConfig()
    assert cfg.corpus_dir == ".porsuk/corpus"


def test_config_has_default_ingest_block():
    from porsuk.core.config import load_config

    cfg = load_config("config/local.yaml")
    # local.yaml overrides corpus_dir; the field must exist and be a str.
    assert isinstance(cfg.ingest.corpus_dir, str)
