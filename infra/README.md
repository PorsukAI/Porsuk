# infra/, the GPU (rented or your own)

This project was developed and measured against a rented **vast.ai** card
(RTX 4090, 24 GB). This directory is how we bring one up and get an
OpenAI-compatible **BAAI/bge-m3** embedding endpoint out of it. Nothing here
is imported by the `porsuk` package: it is operator tooling.

If you already own a GPU (or an Apple Silicon Mac), skip straight to
whichever mode matches your hardware: the manual path below for any CUDA
card, or the MLX section further down for Apple Silicon with no rental at
all.

The only thing you supply is a **vast.ai API key**. Everything else is
scripted or written down below.

```
.env                 VAST_API_KEY=...   (git-ignored, never commit)
vast_provision.py    up / down / status / tunnel / ssh / logs
gpu_ssh.sh           open a shell on the tracked instance
state/               tracked instance id + tunnel pid   (git-ignored)
```

---

## First-time setup

1. **API key.** Get it from <https://vast.ai/console/account>, then:
   ```
   printf 'VAST_API_KEY=<your key>\n' >> .env
   ```
   `.env` is git-ignored. Do not paste the key anywhere else.

2. **SSH key.** vast.ai injects every public key registered on the account
   into every instance. Use your **own** key, never share a private key.

   ```
   ssh-keygen -t ed25519 -f ~/.ssh/id_gpu -C "<you>-porsuk"
   ```

   Send the **public** half (`~/.ssh/id_gpu.pub`) to whoever owns the vast.ai
   account; they add it at <https://vast.ai/console/account> under *SSH Keys*.
   The scripts default to `~/.ssh/id_gpu`; override with `GPU_SSH_KEY=/path`.

3. **CLI.** `uv sync` installs the `vastai` CLI (it is a dev dependency). The
   scripts always call it as `uv run vastai`, so no global install is needed.

---

## The one-command path

```
uv run python infra/vast_provision.py up      # rent + serve + tunnel + write .env
# ... work ...
uv run python infra/vast_provision.py down     # destroy + close tunnel + clean .env
```

`up` does, in order:

1. **picks an offer**: RTX 4090, >=24 GB, verified, reliable, with real
   bandwidth (egress is billed per TB and the image pull is large). It does
   **not** take the absolute cheapest offer: the price floor is usually a
   mediocre host, so it takes the best host (disk + download speed) within
   **$0.03/h of the floor**. Confirms before renting unless you pass `--yes`.
2. **rents it** with the image `vastai/vllm:v0.27.1-cuda-13.0` and the env
   that selects the model (see *Why the env, not an onstart script* below).
3. **waits for boot**: up to 10 min for `actual_status = running`
   (a cold image pull is slow). A host that never gets there is a dead host
   with nothing to keep, so `up` destroys it, this is the **only** case
   where it destroys anything on its own.
4. **waits for bge-m3**: up to 8 min for `/v1/models` to answer inside the
   container. If the box is running but bge-m3 does not come up, `up`
   **leaves the instance running** (you paid for a working card) and prints
   how to inspect it with `logs` / `ssh` or destroy it with `down`. It never
   destroys a running instance without you asking.
5. **opens an SSH tunnel**: `localhost:18000` to the container's vLLM.
6. **writes `.env`**:
   ```
   EMBEDDING_BASE_URL=http://localhost:18000/v1
   EMBEDDING_MODEL=bge-m3
   ```

Then `config/vllm.yaml` (which reads `${EMBEDDING_BASE_URL}` etc.) points at a
live bge-m3.

Other subcommands:

| command | what it does |
|---|---|
| `status` | tracked instance, ssh target, tunnel state, whether bge-m3 answers, spend so far |
| `tunnel` | (re)open just the tunnel, e.g. after your laptop slept |
| `ssh` | shell on the instance, or `... ssh -- nvidia-smi` to run one command |
| `logs` | tail the vLLM logs on the instance |
| `down` | destroy, close tunnel, remove the two keys from `.env` |

---

## The manual path (what the tool automates)

If you would rather drive it yourself, or the tool breaks, this is the whole
procedure. It is also what a coding agent should do rather than fight the tool.

### 1. Rent

Use the vast.ai dashboard or CLI. **Image:** `vastai/vllm:v0.27.1-cuda-13.0`.
Runtype `ssh_direct` + `jupyter_direct`. Disk >= 40 GB. Add these env vars:

```
VLLM_MODEL   = BAAI/bge-m3
MODEL_NAME   = BAAI/bge-m3
VLLM_ARGS    = --runner pooling --served-model-name bge-m3
PORTAL_CONFIG= localhost:1111:11111:/:Instance Portal|localhost:18000:18000:/:vLLM API
```

### 2. Connect

The vast.ai **SSH proxy** (`sshN.vast.ai`) is blocked from some networks: it
fails at `kex_exchange_identification: Connection closed`. Use the **direct**
port instead: `public_ipaddr` + the host port mapped to container `22`
(shown in `vastai show instance <id>` under `ports."22/tcp"`).

```
ssh -o StrictHostKeyChecking=no -i ~/.ssh/id_gpu -p <direct-22-port> root@<public_ipaddr>
```

`infra/gpu_ssh.sh` resolves that automatically for the tracked instance.

### 3. Serve bge-m3 (if the env above did not already)

Inside the instance:

```
source /venv/main/bin/activate
vllm serve BAAI/bge-m3 \
  --runner pooling \
  --port 18000 --host 127.0.0.1 \
  --served-model-name bge-m3 \
  --gpu-memory-utilization 0.30
```

- `--runner pooling`, **not** `--task embed`: vLLM 0.27 removed the old flag.
  bge-m3 is an `XLMRobertaModel`; vLLM finds its sentence-transformers pooling
  config on its own, so no `--hf-overrides` is needed.
- `--host 127.0.0.1`: the endpoint has no auth, so it must not listen on
  `0.0.0.0`. We reach it only through the SSH tunnel.
- `--gpu-memory-utilization 0.30`: bge-m3 needs ~2 GB. The default 0.9
  reserves the whole card and leaves nothing for a reranker or the agent LLM
  later. It also fails outright if a previous vLLM did not fully release the
  GPU, check `nvidia-smi`, `kill -9` any leftover `vllm` process first.

Ready in ~30-40 s once weights are cached. Verify:

```
curl -s http://localhost:18000/v1/models
curl -s http://localhost:18000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"bge-m3","input":["Türkçe test","english test"]}'
```

bge-m3 returns **1024-dim** vectors (matches `dim: 1024` in `config/vllm.yaml`).

### 4. Tunnel

```
ssh -o StrictHostKeyChecking=no -i ~/.ssh/id_gpu -N \
  -L 18000:localhost:18000 -p <direct-22-port> root@<public_ipaddr>
```

Then on your machine `http://localhost:18000/v1` is bge-m3. Put that in `.env`
as `EMBEDDING_BASE_URL`.

### 5. Tear down

`vastai destroy instance <id>`: **recycle/destroy wipes the container**, so
anything not on a mounted volume is gone. That is fine here; the model
re-downloads in seconds on the next rent.

---

## The reranker (and keyword search)

The cross-encoder reranker (**bge-reranker-v2-m3**) runs in the **same
FlagEmbedding process that already serves the sparse (lexical) head** on
`localhost:18002`, not a separate vLLM/TEI instance. It adds a `/rerank`
endpoint alongside `/embed`:

> `keyword_search`'s lexical engine is `retrieval.keyword_backend`: `text`
> (plain word membership, Qdrant `MatchText`, **the default**), `bm25`
> (Qdrant `Modifier.IDF`), or `sparse` (bge-m3's learned weights). `text`
> and `bm25` need no service at all; only `sparse` needs
> `EMBEDDING_SPARSE_BASE_URL`. So with the shipped config the sparse
> `/embed` head is **not** required: `sparse_server.py` on `:18002` is
> there for `/rerank`. `RERANK_BASE_URL` stays required (reranking on by
> default).

```
POST http://localhost:18002/rerank
  {"query": "...", "texts": ["...", "..."]}
->  {"scores": [0.91, 0.12, ...]}
```

The serving script (`sparse_server.py` on the box) loads both heads:

```text
_sparse   = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, devices="cuda:0")
_reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True, devices="cuda:0")
# /embed  -> _sparse.encode(..., return_sparse=True)["lexical_weights"]
# /rerank -> _reranker.compute_score([[query, t] for t in texts], normalize=True)
```

Run it under `tmux` (a bare `nohup`/`setsid` from a one-shot SSH tends to die
with the connection):

```
tmux new-session -d -s sparse \
  'HF_TOKEN=... HF_HOME=/workspace/.hf_home \
   python3 -m uvicorn sparse_server:app --host 127.0.0.1 --port 18002 --app-dir /workspace'
```

Porsuk's HTTP reranker adapter (`porsuk/adapters/rerank/flag_embedding_http.py`)
talks to it unchanged. Reranking is **on by default** in `config/vllm.yaml`,
measured to help every golden set and hurt none, and `RERANK_BASE_URL` is a
**required** env var there, set it before an index or search run:

```
# .env
RERANK_BASE_URL=http://localhost:18002
```

```yaml
# config/vllm.yaml, retrieval section (shipped default, shown for reference)
retrieval:
  strategy: semantic
  rerank_enabled: true
  rerank:
    provider: flag_embedding_http
    base_url: ${RERANK_BASE_URL}
```

Budget: bge-reranker-v2-m3 is ~2.3 GB VRAM on top of the dense + sparse
heads, measured ~4.6 GB for sparse + reranker together, ~6.6 GB free on an
RTX 3090 alongside a dense bge-m3 at `gpu-memory-utilization 0.30`.

---

## MLX: Apple Silicon, no rental needed

`config/mlx.yaml` is a working alternative to renting a card at all, if you
have an Apple Silicon Mac. `mlx_lm.server` serves an OpenAI-compatible
endpoint the same way vLLM does, so no new adapter code is needed, just a
different `base_url`.

Run two `mlx_lm.server` instances, one for the LLM and one for the
embedder, each on its own port:

```bash
mlx_lm.server --model <your-llm-model> --port 8080
mlx_lm.server --model <your-embedding-model> --port 8081
```

Then point `config/mlx.yaml`'s env vars at them (or rely on the config's
own localhost defaults):

```
# .env
MLX_LLM_BASE_URL=http://localhost:8080/v1
MLX_LLM_MODEL=<your-llm-model>
MLX_EMBEDDING_BASE_URL=http://localhost:8081/v1
MLX_EMBEDDING_MODEL=<your-embedding-model>
```

The reranker is not MLX-native: `bge-reranker-v2-m3` still needs its own
FlagEmbedding HTTP process, the same `sparse_server.py` mechanism described
above for the reranker section, running alongside the two `mlx_lm.server`
instances rather than on the rented card. Point `RERANK_BASE_URL` at it,
same as `config/vllm.yaml`.

Quality on this path depends entirely on how large a model your Mac can
run. This is a real tradeoff, not a fixed ceiling: it scales with your
hardware, but on a laptop it will generally lag a rented 24 GB card running
a bigger model.

---

## The agent LLM

`config/vllm.yaml` carries an `agent:` block:

```yaml
# config/vllm.yaml, agent section (shipped default, shown for reference)
agent:
  provider: langgraph_chat
  base_url: ${VLLM_BASE_URL}
  model: ${VLLM_MODEL}
  api_key_env: LLM_API_KEY
```

`langgraph_chat` is a `ChatOpenAI` client. It points `base_url` / `model` at
the **same vLLM endpoint the profiler LLM uses**, one served model, reused
for both, so bringing the agent up is nothing beyond serving that model on
the box (`localhost:8001` in the eval config below). **qwen3-4b** was the
measured working candidate; a larger model is a straightforward config
change if a bigger card is available.

`config/local.yaml` sets `agent: {provider: fake_chat}`, the scripted fake,
no endpoint, which is what the test suite runs against.

---

## Running the agent eval

`eval/run_agent_eval.py` indexes the manual golden set (mevzuat PDFs) and a
RAGTurk subset into isolated `:memory:` stores, runs the agent over each
question, and scores **answer accuracy** (LLM-as-judge) and **citation
accuracy** (the agent's `KAYNAKLAR` vs the golden set's relevant
docs/chunks).

### Box services

Four processes, all local on one RTX 3090:

| service | port | command |
|---|---|---|
| bge-m3 dense | 8000 | `vllm serve BAAI/bge-m3 --runner pooling --served-model-name bge-m3 --gpu-memory-utilization 0.09` |
| reranker (+ optional sparse head) | 18002 | `python3 -m uvicorn sparse_server:app --host 127.0.0.1 --port 18002 --app-dir /workspace`: `/rerank` (required). `/embed` (sparse lexical head) only if `keyword_backend: sparse`; the default `text` and `bm25` need no service |
| **qwen3-4b (agent)** | 8001 | `vllm serve Qwen/Qwen3-4B-Instruct-2507 --served-model-name qwen3-4b --max-model-len 16384 --kv-cache-dtype fp8 --enable-auto-tool-choice --tool-call-parser hermes --gpu-memory-utilization <computed>` |
| **judge** | (none, external API) | DeepSeek `deepseek-chat` API (`DEEPSEEK_API_KEY` in `.env`); the local 4B model was measured to be an unreliable judge, so a stronger external judge is used instead |

`infra/serve.sh` brings up dense + qwen with the right numbers (sparse
assumed already running). `infra/box_run.sh` is a full run including the
model startup.

**Gotchas, agent eval measurement, RTX 3090:**

- **`--enable-auto-tool-choice --tool-call-parser hermes`** on qwen3-4b, or
  every agent tool call is a `400 "auto" tool choice requires ...`.
- **`--kv-cache-dtype fp8 --max-model-len 16384`**, not fp16/8k. An agent
  loop over Turkish content (system prompt + ~10 accumulated tool results at
  `chunk_k=10`) exceeds 8192 tokens: RAGTurk failed at question 21, and
  dropping `chunk_k` to 5 did not help. fp8 KV halves KV memory
  (`Available KV cache: 7.85 GiB` at 16k), which is the only way the 16k
  context fits alongside the embedder + reranker on 24 GB.
- **VRAM: the full stack is 23.1 / 24.6 GB, 94% of the card.** dense ~2.0
  (at util 0.09), sparse + reranker ~3.0 (resident after the first warm
  request), qwen3-4b ~18 (weights 8 + fp8 KV 7.85 + overhead). Compute
  qwen's util *after* dense and sparse are up: `(free_MiB - 1500) / 24576`,
  capped ~0.80. No room for a 14B/27B agent, that needs 2x GPU.
- **`tmux` does not survive the SSH disconnect on this image**, and worse:
  `tmux kill-session` / `pkill -f 'Qwen'` frequently leave a
  `VLLM::EngineCore` process holding ~18 GB that only dies to
  `kill -9 <gpu-pid>` (find it with
  `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`). Sparse
  (plain `uvicorn`) is the only robust one. **Use `setsid nohup` one-shot
  scripts** that write a `.done` flag, and a separate one for each eval run.
- **The box's PyPI is slow.** `uv sync --extra dev` stalls on `borb`. Use
  `uv sync` (base) then
  `uv run --with datasets --with numpy --with huggingface_hub python -m eval.run_agent_eval ...`.
- **`HF_TOKEN`** (read-only) for the RAGTurk per-article downloads,
  **`DEEPSEEK_API_KEY`** for the judge, both exported in `box_run.sh`.
- **`--out <dir>` per goldset run**: `run_agent_eval.main` overwrites
  `<date>-agent.md` for whatever goldsets it ran, so a `--goldset ragturk`
  run clobbers an earlier `--goldset xquad` table. Point them at different
  dirs and merge by hand.

The box-local config is `eval/eval_box.yaml` (git-ignored): `llm:` uses the
DeepSeek judge, `agent:` uses local qwen3-4b, `embedder:` uses local bge-m3 +
sparse, `rerank:` uses local, `store: {:memory:}`.

```
# on the box:
setsid nohup bash infra/serve.sh >/dev/null 2>&1 &   # dense + qwen up
# then, once /workspace/serve.done exists:
setsid nohup env HF_TOKEN=... DEEPSEEK_API_KEY=... \
  uv run --with datasets --with numpy --with huggingface_hub \
  python -m eval.run_agent_eval --config eval/eval_box.yaml --goldset all >/workspace/eval.log 2>&1 &
#   -> docs/eval/<date>-agent.md
#   -> vram_idle.txt / vram_peak.txt
```

---

## The HTTP API

For a UI demo, run the API where the retrieval stack is reachable: the box
itself, or a laptop with the tunnel up:

```bash
PORSUK_API_KEY=<shared-with-the-UI> \
  uv run porsuk serve --config config/vllm.yaml --host 0.0.0.0 --port 8080
```

- `PORSUK_API_KEY` must be set or the server refuses to start; the UI sends
  it as `X-API-Key`. `/health` is the one unauthenticated route.
- `--config` becomes `PORSUK_CONFIG`, read once at startup. The API builds
  **one** `App`, one Qdrant connection, one agent chat client, so it needs
  the persistent qdrant store from `config/vllm.yaml`, not `:memory:`.
- `--host 0.0.0.0` only if the UI is on another machine and the box's port
  is exposed; otherwise `127.0.0.1` + an SSH tunnel.
- Endpoints and schema: `/docs`.

### Query API: `/ask/stream` streaming events

`GET /ask/stream?q=<query>` streams the agent's reasoning and answer as
Server-Sent Events. The client receives events in this order:

1. `event: thought`: the agent's reasoning before a tool call (e.g. "let me search for X")
2. `event: tool_call`: the tool invoked (e.g. `{"name": "search", "args": {...}}`)
3. `event: tool_result`: the tool's result (e.g. `{"name": "search", "summary": "..."}`)
4. `event: chunk`: the answer text as it streams, one `chunk` per model
   turn (per-token flushing is not implemented). The UI concatenates the
   `chunk` payloads for a live answer preview.
5. `event: answer`: the final `AgentAnswer` with the complete answer text and
   sources. Take the cleaned answer text and the sources from this event, not
   from the concatenated chunks.

Multiple `thought` / `tool_call` / `tool_result` events interleave for a
multi-step run: `thought* -> tool_call -> tool_result -> ... -> chunk* -> answer`.

**Inline citations.** The answer text contains `⟦chunk_id⟧` marks, one
after each sentence that rests on a source. The client matches each mark
against `answer.sources[].chunk_id` and renders it as a clickable source
link. A mark that doesn't resolve is dropped. `porsuk ask` (the CLI,
non-`--json`) strips the marks; `--json` keeps them raw.

**Per-chat document scope.** `GET /ask/stream?q=...&scope=<job_id>`,
`POST /ask` / `POST /search` with a `scope` field, restricts every
retrieval (and the browse tools `list_documents` / `get_document`) to the
documents from one index job. A `doc_id` from another job reports
not-found; nothing leaks across scopes. Omit `scope` to search the whole
corpus. `porsuk index <folder>` documents carry no `job_id` and are only
visible to unscoped queries.

**Conversation history.** Stateless: the server stores nothing, the client
resends every prior turn on every call. `POST /ask` takes
`history: [{"role": "user"|"assistant", "content": str}, ...]` in the body;
`GET /ask/stream` takes the same list JSON-encoded in a `?history=` query
param (a GET has no body). Omit it (or send `[]`) for a fresh conversation.
Malformed `history` JSON on `/ask/stream` is a `422`, not a silent drop.

### Index API

Upload a folder of documents over HTTP and index them, one job at a time.
`ingest.corpus_dir` in the config must point at real disk, uploads are kept.

```
# 1. open a job
curl -sX POST localhost:8080/index -H "X-API-Key: $KEY"
#   -> {"job_id": "..."}

# 2. upload files in batches (repeat; filename may be a relative path)
curl -sX POST localhost:8080/index/$JOB/files -H "X-API-Key: $KEY" \
     -F "files=@report.pdf" -F "files=@sub/notes.docx"
#   -> {"received": 2, "total": 2}

# 3. start indexing
curl -sX POST localhost:8080/index/$JOB/start -H "X-API-Key: $KEY"

# 4. poll or stream progress
curl -s localhost:8080/index/$JOB -H "X-API-Key: $KEY"
curl -sN localhost:8080/index/$JOB/stream -H "X-API-Key: $KEY"
```

A second `start` while one job runs returns 409, wait and retry. The store
must be persistent (`qdrant`); `:memory:` indexes into nothing.

**Limits and operator upkeep.** `/index/{job_id}/files` does not cap upload
size or file count: an authenticated client can fill the disk or spike
memory with one large file. This is acceptable for the single-user,
API-key-gated demo; do not expose the endpoint more widely without adding
limits. `ingest.corpus_dir` also grows without bound as jobs accumulate
(uploads are kept on purpose). The operator prunes it by hand: delete old
`<corpus_dir>/<job_id>/` directories once their index is no longer needed.

---

## Why the env, not an onstart script

The first version of `vast_provision.py` passed a bash script as
`--onstart-cmd`. That was wrong: **overriding onstart skips the image's own
`entrypoint.sh`**, so supervisor, Caddy, the SSH daemon and the portal never
start, the box shows `running` but is hollow (`ssh: command not found`
looping in the logs).

The `vastai/vllm` image already runs vLLM as a supervised service. You only
have to tell it *which* model: `VLLM_MODEL` / `MODEL_NAME`, plus `VLLM_ARGS`
for the pooling runner, plus a `PORTAL_CONFIG` line that includes the vLLM
API, without that line the portal logs `Skipping vllm startup (not in
/etc/portal.yaml)` and never starts it.

---

## Running the retrieval eval

The eval measures retrieval quality: it must run **on the box**, with the
embedder, sparse head and reranker local (a tunnel inflated earlier numbers
~10x, since network latency dominated the measurement). Rsync the repo to
`/workspace/porsuk-src`, `uv sync --extra dev`, then a box-local config
wires the three local endpoints (git-ignored, not committed):

```yaml
# eval/eval_box.yaml
llm:      {provider: openai_compatible, model: qwen3-4b, base_url: http://localhost:8001/v1}
embedder: {provider: openai_compatible, model: bge-m3, dim: 1024,
           base_url: http://localhost:8000/v1, sparse_base_url: http://localhost:18002}
store:    {provider: qdrant, url: ":memory:"}
parsing:  {parsers: [pymupdf, docx, xlsx, pptx, text]}
profile:  {llm_enabled: false}
retrieval: {strategy: hybrid, rerank: {provider: flag_embedding_http, base_url: http://localhost:18002}}
```

`qwen3-4b` on `:8001` is only needed to *draft* the manual golden set
(`eval.draft_manual_questions`); the eval itself never calls the LLM.

```
# strategy comparison: recall@5 / MRR / nDCG@5 per goldset x strategy x question type
HF_TOKEN=... uv run python -m eval.run_eval --config eval/eval_box.yaml \
  --ragturk-limit 300 --xquad-limit 400
#   -> docs/eval/<date>-retrieval.md

# parameter sweep (one parameter at a time, RAGTurk)
HF_TOKEN=... uv run python -m eval.sweep --config eval/eval_box.yaml --ragturk-limit 300
#   -> docs/eval/<date>-parameter-sweep.md
```

Each goldset indexes into its **own** `:memory:` Qdrant, RAGTurk's
CC-BY-NC-SA corpus never touches the product index. `HF_TOKEN` (read-only,
in `.env`) lifts the anonymous rate limit on the ~300 per-article RAGTurk
downloads.

---

## Costs & gotchas

- **RTX 4090 ~ $0.30-0.35/h.** A rent -> provision -> destroy cycle is a few
  cents. `up` auto-destroys **only** a host that never reached `running`;
  a running instance is always yours to destroy with `down`.
- **Egress is billed per TB** (~$3-40/TB depending on host). The image pull
  and the ~2.3 GB model download are the only large transfers.
- **`static_ip` / direct ports**: most hosts include them free; a few charge
  for a static IP. `_pick_offer` requires `direct_port_count >= 2`.
- **Rotate the API key** if it has been pasted into a chat or a log. It has
  full account control (it can rent and destroy).
- **The tunnel dies when your laptop sleeps.** Re-run `... tunnel`.
