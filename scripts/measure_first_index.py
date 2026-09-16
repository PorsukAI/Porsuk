"""Measure the first-index time and VRAM: runs the ingestion pipeline end to
end, times it, and samples nvidia-smi so the "one-time setup, instant
afterwards" estimate (20-60 minutes) and the VRAM budget become real
measurements against a corpus the size of the real thing. Run it on the
vast.ai box (infra/README.md) with a config whose embedder is the real
bge-m3; writes a dated record to docs/measurements/.
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass
class Measurement:
    files: int
    seconds: float
    failed: int
    peak_vram_mib: int | None
    embed_calls: int = 0
    embed_texts: int = 0
    embed_seconds: float = 0.0


class _TimedEmbedder:
    """Wraps the real embedder, counting calls, texts and time spent in it.

    The pipeline embeds chunk batches and one profile text per document, all
    on the main thread, so this measures the serial embedding cost directly -
    the number that tells whether the first-index time is bound by the model,
    the network, or the pipeline's own serial structure.
    """

    def __init__(self, inner) -> None:
        self._inner = inner
        self.dim = inner.dim
        self.calls = 0
        self.texts = 0
        self.seconds = 0.0

    def _timed(self, fn, texts):
        self.calls += 1
        self.texts += len(texts) if isinstance(texts, list) else 1
        start = time.monotonic()
        try:
            return fn(texts)
        finally:
            self.seconds += time.monotonic() - start

    def embed_documents(self, texts):
        return self._timed(self._inner.embed_documents, texts)

    def embed_query(self, text):
        return self._timed(self._inner.embed_query, text)


def _sample_vram(stop: threading.Event, out: list[int]) -> None:
    while not stop.is_set():
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if res.returncode == 0:
                out.append(max(int(x) for x in res.stdout.split()))
        except (FileNotFoundError, ValueError, subprocess.TimeoutExpired):
            pass
        stop.wait(2.0)


def measure(folder: str, config_path: str, *, sample_gpu: bool) -> Measurement:
    from porsuk.core.config import load_config
    from porsuk.core.container import build_pipeline

    cfg = load_config(config_path)
    # A fresh state db outside the corpus - inside it, the WAL side-files
    # would be discovered as documents to index.
    state_dir = Path(tempfile.mkdtemp(prefix="porsuk-measure-"))
    pipeline = build_pipeline(cfg, str(state_dir / "state.db"))

    timed = _TimedEmbedder(pipeline._embedder)
    pipeline._embedder = timed

    samples: list[int] = []
    stop = threading.Event()
    sampler: threading.Thread | None = None
    if sample_gpu:
        sampler = threading.Thread(target=_sample_vram, args=(stop, samples), daemon=True)
        sampler.start()

    start = time.monotonic()
    last = None
    for event in pipeline.run(folder):
        last = event
    elapsed = time.monotonic() - start

    stop.set()
    if sampler is not None:
        sampler.join(timeout=5)

    return Measurement(
        files=last.total if last else 0,
        seconds=elapsed,
        failed=last.failed if last else 0,
        peak_vram_mib=max(samples) if samples else None,
        embed_calls=timed.calls,
        embed_texts=timed.texts,
        embed_seconds=timed.seconds,
    )


def write_record(m: Measurement, out_dir: Path, *, corpus_name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y-%m-%d")
    path = out_dir / f"{stamp}-first-index.md"
    rate = m.files / m.seconds * 60 if m.seconds else 0.0
    extrapolated = 5000 / rate if rate else 0.0
    vram = f"{m.peak_vram_mib} MiB" if m.peak_vram_mib is not None else "(not sampled)"
    embed_pct = (m.embed_seconds / m.seconds * 100) if m.seconds else 0.0
    per_call = (m.embed_seconds / m.embed_calls * 1000) if m.embed_calls else 0.0
    path.write_text(
        f"# First-index measurement: {stamp}\n\n"
        f"Corpus: {corpus_name}\n\n"
        f"| metric | value |\n"
        f"|---|---|\n"
        f"| files | {m.files} |\n"
        f"| failed | {m.failed} |\n"
        f"| wall time | {m.seconds:.1f} s ({m.seconds / 60:.1f} min) |\n"
        f"| throughput | {rate:.1f} files/min |\n"
        f"| peak VRAM | {vram} |\n"
        f"| embed calls | {m.embed_calls} |\n"
        f"| embed texts | {m.embed_texts} |\n"
        f"| time in embed | {m.embed_seconds:.1f} s ({embed_pct:.0f}% of wall) |\n"
        f"| avg per embed call | {per_call:.0f} ms |\n\n"
        f"Extrapolated to 5000 files: **{extrapolated:.0f} min** "
        f"(estimate: 20-60 min).\n",
        encoding="utf-8",
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure first-index time and VRAM.")
    parser.add_argument("folder")
    parser.add_argument("--config", default="config/vllm.yaml")
    parser.add_argument("--out", type=Path, default=Path("docs/measurements"))
    parser.add_argument("--no-gpu", action="store_true", help="skip nvidia-smi sampling")
    args = parser.parse_args()

    m = measure(args.folder, args.config, sample_gpu=not args.no_gpu)
    record = write_record(m, args.out, corpus_name=Path(args.folder).name)
    vram = f"{m.peak_vram_mib} MiB" if m.peak_vram_mib is not None else "not sampled"
    print(f"{m.files} files in {m.seconds / 60:.1f} min, peak VRAM {vram}")
    print(f"record: {record}")


if __name__ == "__main__":
    main()
