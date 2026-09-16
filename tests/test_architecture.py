"""Guards for the two rules that keep the architecture from rotting."""

import ast
from pathlib import Path

import pytest

# Real providers - a service or a heavy model behind a port, swappable for
# another. These belong under adapters/, never in core/ingestion/retrieval.
PROVIDER_LIBRARIES = {
    "fitz",
    "pymupdf",
    "docling",
    "qdrant_client",
    "sentence_transformers",
    "openai",
    "anthropic",
    "langchain",
    "langchain_core",
    "langchain_openai",
    "langgraph",
    "docx",
    "openpyxl",
    "pptx",
}

# core/ takes no third-party library at all. ingestion/ and retrieval/ are
# checked separately (below): still no real provider, but their own in-process
# utilities are fine - YAKE and fastText for ingestion (keywords.py,
# entities.py and language.py live there by name), PyStemmer for
# retrieval (Turkish stemming for keyword search).
PROVIDER_FREE_PACKAGES = ("core",)

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _python_files(*packages: str) -> list[Path]:
    files: list[Path] = []
    for package in packages:
        root = _REPO_ROOT / "porsuk" / package
        if root.exists():
            files.extend(root.rglob("*.py"))
    return files


# A guard that scans nothing would report as passing while verifying nothing.
assert _python_files(*PROVIDER_FREE_PACKAGES), "architecture guard found no files to scan"


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize("path", _python_files(*PROVIDER_FREE_PACKAGES), ids=str)
def test_core_imports_no_provider_library(path):
    offending = _imported_roots(path) & PROVIDER_LIBRARIES
    assert not offending, (
        f"{path} imports {sorted(offending)}; core packages must "
        f"depend only on ports"
    )


@pytest.mark.parametrize("path", _python_files("retrieval"), ids=str)
def test_retrieval_imports_no_provider_but_may_use_its_own_utilities(path):
    offending = _imported_roots(path) & PROVIDER_LIBRARIES
    assert not offending, (
        f"{path} imports {sorted(offending)}; a real provider belongs in adapters/, "
        f"not retrieval/. PyStemmer is fine here."
    )


@pytest.mark.parametrize("path", _python_files("ingestion"), ids=str)
def test_ingestion_imports_no_provider_but_may_use_its_own_utilities(path):
    """ingestion/ still must not reach a real provider (pymupdf, docling,
    qdrant, an LLM SDK) - those live behind ports in adapters/. It is allowed
    its own in-process utilities (YAKE, fastText)."""
    offending = _imported_roots(path) & PROVIDER_LIBRARIES
    assert not offending, (
        f"{path} imports {sorted(offending)}; a real provider belongs in adapters/, "
        f"not ingestion/. YAKE / fastText are fine here."
    )


# agent/ is allowed the agent *framework* (langchain, langgraph, langchain_core),
# those are orchestration, not a swappable provider.
# `agent/graph.py` builds the loop with `langchain.agents.create_agent`, and
# `agent/tools.py` builds tools with `langchain_core.tools`. The real chat model
# lives behind adapters/llm/langgraph_chat.py, so langchain_openai and openai
# stay banned here, same as everywhere outside adapters/.
_AGENT_FRAMEWORK = {"langchain", "langgraph", "langchain_core"}


@pytest.mark.parametrize("path", _python_files("agent"), ids=str)
def test_agent_imports_only_the_framework(path):
    offending = (_imported_roots(path) & PROVIDER_LIBRARIES) - _AGENT_FRAMEWORK
    assert not offending, (
        f"{path} imports {sorted(offending)}; agent/ may use langchain, langgraph "
        f"and langchain_core (the framework) but a real provider belongs in "
        f"adapters/"
    )


# api/ is a presentation layer: it may import the web framework
# (fastapi, starlette, sse_starlette, uvicorn) and stdlib, and it calls
# core.container's composition roots. A real provider (qdrant_client,
# langchain_openai, openai, langchain*) belongs behind those roots, never here.
@pytest.mark.parametrize("path", _python_files("api"), ids=str)
def test_api_imports_only_the_web_framework(path):
    offending = _imported_roots(path) & PROVIDER_LIBRARIES
    assert not offending, (
        f"{path} imports {sorted(offending)}; api/ is a presentation layer: a "
        f"real provider belongs behind core.container"
    )


@pytest.mark.parametrize(
    "path",
    _python_files("core", "ingestion", "retrieval", "agent", "api"),
    ids=str,
)
def test_no_provider_branching_outside_the_registry(path):
    if path.name == "registry.py":
        pytest.skip("registry.py is the one place provider branching is permitted")
    source = path.read_text(encoding="utf-8")
    assert "provider ==" not in source, (
        f"{path} branches on a provider name; that is only allowed in core/registry.py"
    )
