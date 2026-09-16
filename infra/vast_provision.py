#!/usr/bin/env python3
"""Rent a vast.ai GPU, serve BAAI/bge-m3, tunnel it to localhost.

Development and measurement run on a rented vast.ai card. This tool brings
a card up, gets an OpenAI-compatible embedding endpoint out of it, and
tears it down when done.

Not part of the Porsuk package - it is an operator tool. It shells out to
the `vastai` CLI (a dev dependency) and keeps its state in infra/state/
(git-ignored).

How it works (learned the hard way against the real image):

  * The `vastai/vllm` image already runs vLLM as a supervised service. You
    do NOT pass an --onstart script - overriding onstart skips the image's
    own entrypoint and you get a half-booted box. Instead the model is
    chosen through env: VLLM_MODEL / MODEL_NAME, plus VLLM_ARGS for the
    embedding runner, plus a PORTAL_CONFIG line so the portal actually
    starts the vllm service.
  * The vast.ai SSH *proxy* (ssh*.vast.ai) is blocked from some networks.
    The *direct* port (public_ipaddr : the container's mapped :22) is not,
    so everything here - health checks, the tunnel - goes direct.
  * vLLM listens on 127.0.0.1:18000 inside the container. `up` opens an
    SSH -L tunnel so http://localhost:18000/v1 on this machine is bge-m3,
    and writes that into .env.

Commands:
    up      rent the cheapest card that fits, wait for bge-m3, open the
            tunnel, write EMBEDDING_BASE_URL / EMBEDDING_MODEL to .env
    tunnel  (re)open just the SSH tunnel to the tracked instance
    status  show the tracked instance and whether bge-m3 answers
    logs    tail the instance's vLLM log
    ssh     open a shell on the tracked instance (or run one command)
    down    destroy the tracked instance, close the tunnel, clean .env

The rent step costs real money. `up` prints the exact offer and asks first
unless --yes is given.

Auto-destroy is deliberately narrow: `up` destroys an instance ONLY when the
host never reached "running" (a dead host - nothing to keep, it would only
bill). If the box IS running but bge-m3 did not come up in time, `up` leaves
it alone and tells you how to inspect or destroy it - you paid for a working
card and you decide. Nothing else in this tool ever destroys without `down`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
INFRA = REPO / "infra"
STATE_DIR = INFRA / "state"
STATE_FILE = STATE_DIR / "instance.json"
TUNNEL_PID_FILE = STATE_DIR / "tunnel.pid"
ENV_FILE = REPO / ".env"
SSH_KEY = Path(os.environ.get("GPU_SSH_KEY", str(Path.home() / ".ssh" / "id_gpu")))

IMAGE = "vastai/vllm:v0.27.1-cuda-13.0"
HF_MODEL = "BAAI/bge-m3"
SERVED_NAME = "bge-m3"
CONTAINER_PORT = 18000  # where vLLM listens inside the image
LOCAL_PORT = 18000  # where the tunnel exposes it on this machine
DISK_GB = 40

# What "fits": one 24 GB consumer card, reliable, enough disk for
# the image + model. Egress is billed per TB and the image pull is large, so
# require real bandwidth. The cheapest offer is usually a mediocre host -
# skip the price floor and take one ~$0.03/h above it.
OFFER_QUERY = (
    "rentable=true verified=true rented=false "
    "gpu_name=RTX_4090 num_gpus=1 gpu_ram>=24 "
    "disk_space>=60 inet_down>=400 inet_up>=200 "
    "reliability>=0.98 dph_total<=0.60"
)
PRICE_FLOOR_DELTA = 0.03

# Env handed to the image at creation. PORTAL_CONFIG must list the vLLM API
# or the portal refuses to start the vllm service ("not in /etc/portal.yaml").
# VLLM_ARGS selects the embedding runner - bge-m3 is a pooling model and
# vLLM 0.27 dropped `--task embed` in favour of `--runner pooling`.
INSTANCE_ENV = (
    f"-e VLLM_MODEL={HF_MODEL} "
    f"-e MODEL_NAME={HF_MODEL} "
    f"-e VLLM_ARGS=--runner pooling --served-model-name {SERVED_NAME} "
    f"-e PORTAL_CONFIG=localhost:1111:11111:/:Instance Portal"
    f"|localhost:{CONTAINER_PORT}:{CONTAINER_PORT}:/:vLLM API"
)

# A dead host never reaches "running" - bail on that. A live host still
# needs to pull the image, then pull and load bge-m3.
BOOT_TIMEOUT_S = 10 * 60
MODEL_TIMEOUT_S = 8 * 60
POLL_S = 15

SSH_OPTS = [
    "-o",
    "StrictHostKeyChecking=no",
    "-o",
    "UserKnownHostsFile=/dev/null",
    "-o",
    "ConnectTimeout=15",
    "-o",
    "LogLevel=ERROR",
    "-o",
    "IdentitiesOnly=yes",
    "-i",
    str(SSH_KEY),
]


# --------------------------------------------------------------- vastai CLI


def _run(
    args: list[str], *, check: bool = True, capture: bool = True
) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        ["uv", "run", "vastai", *args], cwd=REPO, text=True, capture_output=capture
    )
    if check and proc.returncode != 0:
        sys.exit(f"vastai {' '.join(args)} failed:\n{proc.stderr or proc.stdout}")
    return proc


def _vastai_json(args: list[str]) -> object:
    proc = _run([*args, "--raw"])
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        sys.exit(f"vastai {' '.join(args)} did not return JSON:\n{proc.stdout[:500]}")


# ------------------------------------------------------------------- state


def _load_state() -> dict | None:
    return json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else None


def _save_state(state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def _clear_state() -> None:
    STATE_FILE.unlink(missing_ok=True)


def _require_state() -> dict:
    state = _load_state()
    if not state:
        sys.exit("no instance tracked - run `up` first")
    return state


# --------------------------------------------------------------------- env


def _write_env(pairs: dict[str, str]) -> None:
    lines: list[str] = []
    seen: set[str] = set()
    for line in ENV_FILE.read_text().splitlines() if ENV_FILE.exists() else []:
        m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*=", line)
        if m and m.group(1) in pairs:
            lines.append(f"{m.group(1)}={pairs[m.group(1)]}")
            seen.add(m.group(1))
        else:
            lines.append(line)
    for key, value in pairs.items():
        if key not in seen:
            lines.append(f"{key}={value}")
    ENV_FILE.write_text("\n".join(lines) + "\n")


def _remove_env(keys: set[str]) -> None:
    if not ENV_FILE.exists():
        return
    kept = [
        line
        for line in ENV_FILE.read_text().splitlines()
        if not any(re.match(rf"\s*{re.escape(k)}\s*=", line) for k in keys)
    ]
    ENV_FILE.write_text("\n".join(kept) + "\n")


# ------------------------------------------------------------ instance info


def _instance(instance_id: int) -> dict:
    data = _vastai_json(["show", "instance", str(instance_id)])
    return data if isinstance(data, dict) else {}


def _ssh_target(instance_id: int) -> tuple[str, str] | None:
    """(ip, host_port) for the container's mapped :22 - the direct route."""
    d = _instance(instance_id)
    ip = d.get("public_ipaddr")
    mapped = (d.get("ports") or {}).get("22/tcp") or [{}]
    port = mapped[0].get("HostPort")
    return (ip.strip(), port) if ip and port else None


def _ssh_base(instance_id: int) -> list[str] | None:
    """Full ssh argv prefix: ssh <opts> -p <port> root@<ip>."""
    target = _ssh_target(instance_id)
    if not target:
        return None
    ip, port = target
    return ["ssh", *SSH_OPTS, "-p", port, f"root@{ip}"]


def _model_healthy(instance_id: int) -> bool:
    base = _ssh_base(instance_id)
    if not base:
        return False
    proc = subprocess.run(
        [*base, f"curl -sf http://localhost:{CONTAINER_PORT}/v1/models"],
        text=True,
        capture_output=True,
        timeout=40,
    )
    return proc.returncode == 0 and '"object"' in proc.stdout


# ------------------------------------------------------------------ tunnel


def _tunnel_running() -> bool:
    if not TUNNEL_PID_FILE.exists():
        return False
    try:
        os.kill(int(TUNNEL_PID_FILE.read_text().strip()), 0)
        return True
    except (ValueError, ProcessLookupError, PermissionError):
        return False


def _open_tunnel(instance_id: int) -> None:
    if _tunnel_running():
        print(f"tunnel already up on localhost:{LOCAL_PORT}")
        return
    target = _ssh_target(instance_id)
    if not target:
        sys.exit("cannot resolve SSH target for the tunnel")
    ip, port = target
    cmd = [
        "ssh",
        *SSH_OPTS,
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=3",
        "-o",
        "ExitOnForwardFailure=yes",
        "-N",
        "-L",
        f"{LOCAL_PORT}:localhost:{CONTAINER_PORT}",
        "-p",
        port,
        f"root@{ip}",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    TUNNEL_PID_FILE.write_text(str(proc.pid))
    time.sleep(4)
    if proc.poll() is not None:
        TUNNEL_PID_FILE.unlink(missing_ok=True)
        sys.exit("tunnel exited immediately - check `status`")
    print(f"tunnel up: localhost:{LOCAL_PORT} -> instance:{CONTAINER_PORT} (pid {proc.pid})")


def _close_tunnel() -> None:
    if not TUNNEL_PID_FILE.exists():
        return
    try:
        os.kill(int(TUNNEL_PID_FILE.read_text().strip()), signal.SIGTERM)
    except (ValueError, ProcessLookupError, PermissionError):
        pass
    TUNNEL_PID_FILE.unlink(missing_ok=True)


# ------------------------------------------------------------------ offers


def _pick_offer() -> dict:
    offers = _vastai_json(["search", "offers", OFFER_QUERY, "-o", "dph_total"])
    if not isinstance(offers, list) or not offers:
        sys.exit("no offers matched - loosen OFFER_QUERY in infra/vast_provision.py")
    usable = [o for o in offers if o.get("direct_port_count", 0) >= 2]
    usable = usable or offers
    floor = min(o.get("dph_total", 1e9) for o in usable)
    # Prefer the best host within a small band above the floor, not the
    # rock-bottom price. "Best" = disk bandwidth then download speed.
    band = [o for o in usable if o.get("dph_total", 1e9) <= floor + PRICE_FLOOR_DELTA]
    band.sort(key=lambda o: (o.get("disk_bw", 0), o.get("inet_down", 0)), reverse=True)
    return band[0]


def _offer_line(o: dict) -> str:
    return (
        f"  offer {o['id']}  {o.get('gpu_name')}  {round(o.get('gpu_ram', 0) / 1024)}GB  "
        f"${o.get('dph_total', 0):.3f}/h  disk {round(o.get('disk_space', 0))}GB "
        f"@ {round(o.get('disk_bw', 0))}MB/s  "
        f"net {round(o.get('inet_down', 0))}/{round(o.get('inet_up', 0))}Mbps  "
        f"egress ${o.get('internet_down_cost_per_tb', 0):.0f}/TB  "
        f"rel {o.get('reliability', 0) * 100:.1f}%  {o.get('geolocation', '?')}"
    )


# ---------------------------------------------------------------------- up


def cmd_up(args: argparse.Namespace) -> None:
    if _load_state():
        sys.exit("an instance is already tracked - `down` first, or `status` to inspect")
    if not SSH_KEY.exists():
        sys.exit(f"SSH key {SSH_KEY} not found (set GPU_SSH_KEY, or add it to vast.ai)")

    offer = _pick_offer()
    print("selected offer (best host within $0.03 of the price floor):")
    print(_offer_line(offer))
    print(f"  image {IMAGE}, disk {DISK_GB}GB, serving {HF_MODEL} as '{SERVED_NAME}'")
    if not args.yes and input("rent this? [y/N] ").strip().lower() != "y":
        print("aborted, nothing rented")
        return

    result = _vastai_json(
        [
            "create",
            "instance",
            str(offer["id"]),
            "--image",
            IMAGE,
            "--disk",
            str(DISK_GB),
            "--ssh",
            "--direct",
            "--env",
            INSTANCE_ENV,
        ]
    )
    if not (isinstance(result, dict) and result.get("success")):
        sys.exit(f"rent failed: {result}")

    instance_id = int(result["new_contract"])
    _save_state(
        {"instance_id": instance_id, "dph_total": offer.get("dph_total"), "rented_at": time.time()}
    )
    print(f"rented instance {instance_id} (${offer.get('dph_total', 0):.3f}/h)")

    def destroy_dead_host(reason: str) -> None:
        # Only ever called when the host NEVER reached 'running' - there is
        # nothing to keep and it would just bill. A box that IS running is
        # never destroyed automatically (see keep_running below).
        print(f"\n{reason} - destroying (never came up, nothing to keep)")
        _run(["destroy", "instance", str(instance_id)], check=False)
        _clear_state()
        sys.exit("provisioning failed; dead host destroyed")

    def keep_running(reason: str) -> None:
        # The box is up but bge-m3 did not come up in time. Do NOT destroy -
        # the user paid for a working card and may want to debug it. Leave
        # it, tell them how to look and how to kill it themselves.
        print(f"\n{reason}")
        print(f"the instance ({instance_id}) is RUNNING and still billing - it was NOT destroyed.")
        print("inspect it:   uv run python infra/vast_provision.py logs")
        print("              uv run python infra/vast_provision.py ssh")
        print("destroy it:   uv run python infra/vast_provision.py down")
        sys.exit("bge-m3 not healthy; instance left running for you to decide")

    print(f"waiting up to {BOOT_TIMEOUT_S // 60} min for the box to reach 'running'...")
    boot_deadline = time.time() + BOOT_TIMEOUT_S
    while time.time() < boot_deadline:
        time.sleep(POLL_S)
        status = _instance(instance_id).get("actual_status")
        print(f"  status={status} ({int(boot_deadline - time.time())}s left)")
        if status == "running":
            break
    else:
        destroy_dead_host("host never reached 'running' within the boot timeout")

    print(f"box is up; waiting up to {MODEL_TIMEOUT_S // 60} min for bge-m3 to serve...")
    model_deadline = time.time() + MODEL_TIMEOUT_S
    while time.time() < model_deadline:
        time.sleep(POLL_S)
        healthy = _model_healthy(instance_id)
        print(f"  bge-m3 healthy={healthy} ({int(model_deadline - time.time())}s left)")
        if healthy:
            break
    else:
        keep_running("bge-m3 did not answer within the model timeout")

    _open_tunnel(instance_id)
    _write_env(
        {
            "EMBEDDING_BASE_URL": f"http://localhost:{LOCAL_PORT}/v1",
            "EMBEDDING_MODEL": SERVED_NAME,
        }
    )
    print("\nbge-m3 is serving through the tunnel.")
    print(f"  EMBEDDING_BASE_URL=http://localhost:{LOCAL_PORT}/v1")
    print(f"  EMBEDDING_MODEL={SERVED_NAME}")
    print("written to .env. Run `infra/vast_provision.py down` when finished.")


# ------------------------------------------------------------ tunnel / ssh


def cmd_tunnel(_: argparse.Namespace) -> None:
    _open_tunnel(_require_state()["instance_id"])


def cmd_ssh(args: argparse.Namespace) -> None:
    base = _ssh_base(_require_state()["instance_id"])
    if not base:
        sys.exit("cannot resolve SSH target")
    os.execvp(base[0], [*base, *(["--", *args.command] if args.command else [])])


# ------------------------------------------------------------------ status


def cmd_status(_: argparse.Namespace) -> None:
    state = _require_state()
    instance_id = state["instance_id"]
    d = _instance(instance_id)
    target = _ssh_target(instance_id)
    age_min = (time.time() - state.get("rented_at", time.time())) / 60
    rate = state.get("dph_total") or 0
    ssh_line = f"root@{target[0]} -p {target[1]}" if target else "(not resolved)"
    tunnel_line = f"up on localhost:{LOCAL_PORT}" if _tunnel_running() else "down"
    print(f"instance   {instance_id}")
    print(f"status     {d.get('actual_status')}")
    print(f"ssh        {ssh_line}")
    print(f"tunnel     {tunnel_line}")
    print(f"bge-m3     {'healthy' if _model_healthy(instance_id) else 'not answering'}")
    print(f"uptime     {age_min:.0f} min  (~${age_min / 60 * rate:.2f} at ${rate:.3f}/h)")


def cmd_logs(_: argparse.Namespace) -> None:
    base = _ssh_base(_require_state()["instance_id"])
    if not base:
        sys.exit("cannot resolve SSH target")
    remote = "tail -n 60 -f /var/log/portal/vllm.log /var/log/bge_m3.log 2>/dev/null"
    subprocess.run([*base, remote])


# ------------------------------------------------------------------- down


def cmd_down(args: argparse.Namespace) -> None:
    state = _load_state()
    _close_tunnel()
    _remove_env({"EMBEDDING_BASE_URL", "EMBEDDING_MODEL"})
    if not state:
        print("no instance tracked - closed tunnel and cleaned .env")
        return
    instance_id = state["instance_id"]
    if not args.yes and input(f"destroy instance {instance_id}? [y/N] ").strip().lower() != "y":
        print("kept (tunnel closed, .env cleaned)")
        return
    _run(["destroy", "instance", str(instance_id)], check=False)
    _clear_state()
    print(f"destroyed {instance_id}, closed tunnel, cleaned .env")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_up = sub.add_parser("up", help="rent a card, serve bge-m3, open the tunnel")
    p_up.add_argument("--yes", action="store_true", help="skip the rent confirmation")
    p_up.set_defaults(func=cmd_up)

    sub.add_parser("tunnel", help="(re)open the SSH tunnel").set_defaults(func=cmd_tunnel)
    sub.add_parser("status", help="show the tracked instance").set_defaults(func=cmd_status)
    sub.add_parser("logs", help="tail the instance's vLLM log").set_defaults(func=cmd_logs)

    p_ssh = sub.add_parser("ssh", help="shell on the instance, or run one command")
    p_ssh.add_argument("command", nargs=argparse.REMAINDER)
    p_ssh.set_defaults(func=cmd_ssh)

    p_down = sub.add_parser("down", help="destroy the instance, close tunnel, clean .env")
    p_down.add_argument("--yes", action="store_true", help="skip the destroy confirmation")
    p_down.set_defaults(func=cmd_down)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
