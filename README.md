# 🦡 Porsuk

Agentic search over a document archive: ask a question in Turkish or
English, get a sourced answer. Porsuk is a ReAct agent that decides which
tools to call (semantic search, keyword search, document browsing), reads
what it finds, and cites the exact chunks its answer relies on. Built by two
engineers.

## 🏗️ Architecture

Hexagonal (ports and adapters): every external dependency, the LLM, the
embedder, the vector store, the reranker, sits behind a port, so swapping
vLLM for OpenRouter or Qdrant for an in-memory store is a config change, not
a code change. Layer isolation and dependency direction are enforced
mechanically in CI, not just by convention: a dedicated test walks the
import graph and fails the build if a layer reaches somewhere it shouldn't.

The whole thing is TDD'd: 850+ tests, adapters tested against real service
doubles where it matters (a real in-memory Qdrant client, not a mock).

### 📄 Document ingestion

Parsing is multi-stage, not single-tool: PyMuPDF handles clean PDFs, but
falls back automatically to RapidOCR or Docling when a quality scorer
detects PyMuPDF's output is bad (scanned pages, broken encoding, garbled
Turkish characters). Chunking runs a 3-layer strategy that preserves each
document's own "table of contents" structure, so a chunk keeps its section
context instead of being a blind character-count slice.

Text quality scoring is tuned for Turkish specifically: a Latin-character-
focused OCR heuristic that catches mojibake and script-mismatch garbage a
generic English-tuned scorer would miss, improving parse quality on Turkish
documents where the standard tools fall short.

### 🔍 Retrieval and the agent

Two retrieval strategies (semantic, via dense embeddings; keyword, via
BM25/sparse) plus a cross-encoder reranker, all exposed as tools to a
LangChain ReAct agent that decides at runtime which to use and when, rather
than a hardcoded pipeline choosing for it. Every strategy decision was
validated empirically against the RAGTurk and XQuAD-tr datasets, not argued
in the abstract.

## 🧠 Choose how it talks to a model

`config/local.yaml` is for trying the pipeline with no external dependency,
it uses a scripted fake model so the whole thing runs end to end on your
machine. For real answers, pick one of these and pass it as `--config`:

### ☁️ OpenRouter (what the demo runs on)

No GPU needed, one API for hundreds of hosted models. The live demo runs on
`Qwen3.8-27B` served through OpenRouter, with `BAAI/bge-m3` for embeddings
and `Qwen/Qwen3-Reranker-8B` for reranking, indexed into Qdrant. Set
`OPENROUTER_API_KEY` in `.env` (copy `.env.example`), then:

```bash
uv run porsuk ask "..." --config config/openrouter.yaml
```

### 🖥️ Self-hosted vLLM (your own GPU, or a rented one)

An alternative to OpenRouter if you'd rather run the model yourself: a
rented RTX 4090/5060Ti box on vast.ai works well, serving a model like
`ISTA-DASLab/Qwen3.8-27B-3Bit-GSQ` through vLLM. Set `VLLM_BASE_URL`,
`VLLM_MODEL`, `LLM_API_KEY`, `EMBEDDING_BASE_URL`, `EMBEDDING_MODEL`,
`RERANK_BASE_URL` in `.env`, then use `config/vllm.yaml`. `infra/README.md`
covers renting a card, including the exact vLLM flags and gotchas.

### 🍎 Self-hosted MLX (Apple Silicon, no GPU rental)

Run `mlx_lm.server` locally for the LLM and the embedder, each on its own
port, then use `config/mlx.yaml`. Reranking still needs a separate
FlagEmbedding HTTP process alongside them (see `infra/README.md`). Quality
depends on how large a model your machine can run, a genuine tradeoff on a
laptop, not a fixed limitation of MLX itself.

## 🚀 Quick start

```bash
uv sync
uv run porsuk index ./my-documents --config config/local.yaml
uv run porsuk ask "what does this say about X?" --config config/local.yaml
```

`porsuk index` parses and embeds every file in the folder into a vector
store; `porsuk ask` runs the agent over the index and prints a sourced
answer. `porsuk search <query>` returns ranked chunks directly, without the
agent loop, if you just want retrieval.

First indexing a folder takes a while (parsing, chunking, embedding every
document); every question after that is answered in seconds. Budget a
one-time setup, then it is instant.

## 🌐 The HTTP API

```bash
PORSUK_API_KEY=<your-key> uv run porsuk serve --config config/vllm.yaml
```

Every route except `/health` requires an `X-API-Key` header matching
`PORSUK_API_KEY`.

- `POST /ask` and `GET /ask/stream?q=...`: ask a question, get a sourced
  answer. The streaming version emits the agent's reasoning, tool calls,
  and answer text as Server-Sent Events.
- `POST /search`: retrieval only, no agent loop.
- `GET /health`: unauthenticated liveness check.

Uploading and indexing a folder over HTTP is a four-step flow: open a job
(`POST /index`), upload files to it (`POST /index/{job_id}/files`), start
indexing (`POST /index/{job_id}/start`), then poll or stream progress
(`GET /index/{job_id}`, `GET /index/{job_id}/stream`). Set
`PORSUK_CORS_ORIGINS` to a comma-separated list of allowed frontend origins
if the API is called from a browser on another origin. Full request/response
schemas: `/docs` once the server is running.

## 🐳 Docker

```bash
docker compose up
```

Bundles Porsuk and an embedded Qdrant in one container, with a persistent
volume so the index survives restarts. Copy `.env.example` to `.env` and
fill in whichever mode you're using (OpenRouter, vLLM, or MLX) before
starting; `PORSUK_CONFIG` in `.env` picks which config profile the container
loads.

## ✅ Running tests

```bash
uv run pytest
```
