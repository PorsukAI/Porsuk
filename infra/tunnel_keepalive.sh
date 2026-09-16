#!/usr/bin/env bash
# Auto-reconnecting SSH tunnel to the vast.ai box (direct IP; the proxy is
# blocked from some networks). Restarts the tunnel whenever it drops.
#   dense  localhost:18000 -> box:8000   (vLLM bge-m3)
#   sparse localhost:18002 -> box:18002  (FlagEmbedding lexical head)
set -u
KEY="${GPU_SSH_KEY:-$HOME/.ssh/id_gpu}"
HOST="${GPU_SSH_HOST:-181.91.124.48}"
PORT="${GPU_SSH_PORT:-34183}"
while true; do
  ssh -N \
    -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -o ExitOnForwardFailure=yes -o IdentitiesOnly=yes \
    -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
    -o ConnectTimeout=20 -o TCPKeepAlive=yes \
    -i "$KEY" \
    -L 18000:localhost:8000 -L 18002:localhost:18002 \
    -p "$PORT" "root@$HOST"
  echo "[$(date +%H:%M:%S)] tunnel dropped (rc $?), reconnecting in 3s" >&2
  sleep 3
done
