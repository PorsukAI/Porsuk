#!/bin/bash
# Shell into the current vast.ai instance over its direct SSH port.
# The vast.ai SSH proxy (ssh*.vast.ai) is blocked from some networks;
# the direct port (public_ipaddr : the '22/tcp' host port) is not.
#
# Usage:
#   infra/gpu_ssh.sh                 # interactive shell
#   infra/gpu_ssh.sh 'command ...'   # run one command
#
# Reads host/port from `vastai show instance` for the tracked instance.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE="$REPO/infra/state/instance.json"
KEY="${GPU_SSH_KEY:-$HOME/.ssh/id_gpu}"

if [[ ! -f "$STATE" ]]; then
    echo "no tracked instance ($STATE missing)" >&2
    exit 1
fi

IID="$(python3 -c "import json;print(json.load(open('$STATE'))['instance_id'])")"
read -r IP PORT < <(
    cd "$REPO" && uv run vastai show instance "$IID" --raw 2>/dev/null | python3 -c "
import json, sys
d = json.load(sys.stdin)
ports = d.get('ports') or {}
m = (ports.get('22/tcp') or [{}])[0]
print(d.get('public_ipaddr', ''), m.get('HostPort', ''))
"
)

if [[ -z "$IP" || -z "$PORT" ]]; then
    echo "could not resolve direct SSH host:port for instance $IID" >&2
    exit 1
fi

exec ssh \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -o ConnectTimeout=15 \
    -o LogLevel=ERROR \
    -o IdentitiesOnly=yes \
    -i "$KEY" \
    -p "$PORT" "root@$IP" "$@"
