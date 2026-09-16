# Porsuk backend + Qdrant in one container (no separate Qdrant container on
# the VPS). Qdrant's own binary is copied out of its official image rather
# than reimplementing its install; Porsuk always talks to it over
# localhost:6333 either way, the app never assumes an in-process store.
FROM qdrant/qdrant:v1.19.1 AS qdrant

FROM python:3.12-slim AS app

# curl: uv's installer / the healthcheck. libgomp1: pymupdf's runtime
# dependency on Debian slim. libunwind8: the Qdrant binary (built on Debian
# 13) links against it and python:3.12-slim (Debian 12) doesn't ship it.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl libgomp1 libunwind8 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /uvx /usr/local/bin/

WORKDIR /app

# Dependencies first (own layer, cache survives source-only changes).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY porsuk ./porsuk
COPY config ./config
RUN uv sync --frozen --no-dev

# Qdrant's binary + its own entrypoint script, straight from the official
# image — not reinstalled, not rebuilt. Its entrypoint.sh calls `./qdrant`
# (relative), so it needs to actually run from /qdrant, not just find the
# binary on PATH.
RUN mkdir -p /qdrant
COPY --from=qdrant /qdrant/qdrant /qdrant/qdrant
COPY --from=qdrant /qdrant/entrypoint.sh /qdrant/entrypoint.sh
RUN chmod +x /qdrant/qdrant /qdrant/entrypoint.sh

# /data holds both Qdrant's storage and the ingest corpus_dir, so one named
# volume (see docker-compose.yml) makes the whole container's state
# persistent across `docker compose restart` / image rebuilds.
RUN mkdir -p /data/qdrant/storage /data/corpus

ENV QDRANT__STORAGE__STORAGE_PATH=/data/qdrant/storage \
    QDRANT__SERVICE__HTTP_PORT=6333 \
    QDRANT__SERVICE__GRPC_PORT=6334 \
    PORSUK_CONFIG=config/production.yaml \
    PATH="/app/.venv/bin:$PATH"

COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/docker-entrypoint.sh"]
