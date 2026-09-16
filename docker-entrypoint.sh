#!/bin/sh
# Starts Qdrant in the background, waits for it to answer, then execs the
# Porsuk API in the foreground. Porsuk becomes PID 1, so `docker stop` /
# a crash signal it directly and `restart: unless-stopped` (compose) reacts
# to *its* exit, not Qdrant's.
set -e

(cd /qdrant && ./entrypoint.sh) &
QDRANT_PID=$!

echo "waiting for qdrant..."
until curl -sf http://localhost:6333/healthz >/dev/null 2>&1; do
    if ! kill -0 "$QDRANT_PID" 2>/dev/null; then
        echo "qdrant exited before becoming healthy" >&2
        exit 1
    fi
    sleep 1
done
echo "qdrant is up"

exec python3 -m porsuk.api.cli serve --config "$PORSUK_CONFIG" --host 0.0.0.0 --port 8000
