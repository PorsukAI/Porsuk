"""Draft manual golden-set questions from a parsed corpus.

An LLM proposes questions from sampled chunks; a human must verify each
before it enters eval/manual_goldset.yaml before moving survivors in
(stripping the `_` keys) - skipping that measures the question generator,
not retrieval. Writes eval/manual_goldset.draft.yaml with a `_VERIFY` marker
and the source file on every entry.

The gold key is a bare filename, not the store's document id, because that
id is a UUID derived from the resolved absolute path and would differ
between checkouts; `resolve_manual_doc_ids` turns filenames into ids against
whatever corpus directory is indexed at eval time.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import httpx
import yaml

# Qwen3-4B-Instruct: no <think> block, but it still likes to number lines,
# add a lead-in ("İşte iki soru:"), wrap in markdown, or echo "SPESİFİK:".
# Ask for a strict JSON array so parsing does not fight prose; fall back to
# line parsing if it answers in prose anyway.
_PROMPT = (
    "Aşağıda bir Türk mevzuat belgesinden bir parça var. Bu parçaya dayanarak, "
    "birisi arama motoruna yazacakmış gibi, KENDİ BAŞINA ANLAŞILIR iki Türkçe "
    "soru üret. Her soru; konuyu, kurumu veya kanunu adıyla anmalı - "
    '"bu", "burada", "söz konusu" gibi parçaya gönderme yapan sözcükler '
    "KULLANMA. Evet/hayır soruları yazma; tam bir bilgi soran sorular yaz.\n"
    '1. "specific": cevabı parçada birebir geçen belirli bir sayı, tarih, '
    "tutar, oran veya isim olan soru.\n"
    '2. "conceptual": cevabı parçada açıklanan ama sorunun sözcükleri parçada '
    "aynen geçmeyen, bir kuralı veya kavramı soran soru.\n\n"
    "SADECE şu biçimde bir JSON dizisi döndür, başka hiçbir şey yazma:\n"
    '["birinci soru?", "ikinci soru?"]\n\n'
    "BELGE PARÇASI:\n{chunk}"
)

_NOISE_PREFIX = re.compile(
    r"^\s*(?:[-*•]\s*|\d+[.)]\s*|(?:soru|question|q)\s*\d*\s*[:.\-]\s*"
    r"|(?:spesif\w*|kavramsal|specific|conceptual)\s*[:.\-]\s*)+",
    re.IGNORECASE,
)


def _clean(line: str) -> str:
    line = line.strip().strip("`").strip()
    line = _NOISE_PREFIX.sub("", line)
    return line.strip(" \"'").strip()


def _looks_like_question(line: str) -> bool:
    return len(line) >= 12 and ("?" in line or line.lower().startswith(("ne", "kaç", "hangi")))


def _parse_questions(raw: str) -> list[str]:
    """Pull exactly the question strings out of whatever the model returned.

    Tries JSON first (what the prompt asks for), then falls back to taking the
    first two question-looking lines after stripping numbering/markdown/labels.
    """
    text = raw.strip()
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            arr = json.loads(match.group(0))
            qs = [_clean(str(x)) for x in arr if str(x).strip()]
            qs = [q for q in qs if _looks_like_question(q)]
            if len(qs) >= 2:
                return qs[:2]
        except (json.JSONDecodeError, TypeError):
            pass
    lines = [_clean(ln) for ln in text.splitlines()]
    lines = [ln for ln in lines if _looks_like_question(ln)]
    return lines[:2]


def _complete(base_url: str, model: str, prompt: str, *, timeout: float = 120.0) -> str:
    resp = httpx.post(
        f"{base_url.rstrip('/')}/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.4,
            "max_tokens": 400,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def draft(corpus_dir: str, config_path: str, n_chunks: int, out: Path) -> None:
    from porsuk.core.config import load_config
    from porsuk.core.container import build_router
    from porsuk.ingestion.chunking import chunk as chunk_document
    from porsuk.ingestion.pipeline import document_id_for

    cfg = load_config(config_path)
    router = build_router(cfg)
    base_url, model = cfg.llm.base_url, cfg.llm.model
    if not base_url:
        raise SystemExit("config llm.base_url is not set")

    entries: list[dict] = []
    paths = [str(p) for p in sorted(Path(corpus_dir).rglob("*")) if p.is_file()]
    random.shuffle(paths)

    for path in paths:
        if len(entries) >= n_chunks * 2:
            break
        outcome = router.parse(path)
        if outcome.document is None:
            print(f"skip (parse failed): {path}")
            continue
        doc_id = document_id_for(path)
        chunks = chunk_document(outcome.document, doc_id, cfg.chunking)
        # Only chunks with enough substance to ask two questions about.
        candidates = [c for c in chunks if len(c.text) >= 300]
        if not candidates:
            continue
        c = random.choice(candidates)
        header = f"[{Path(path).stem}"
        if c.section_path or c.section_title:
            header += f" / {c.section_path or c.section_title}"
        header += "]\n"
        raw = _complete(base_url, model, _PROMPT.format(chunk=header + c.text[:1500]))
        questions = _parse_questions(raw)
        if len(questions) < 2:
            print(f"skip (LLM gave {len(questions)} usable lines): {Path(path).name}")
            continue
        for i, q in enumerate(questions):
            entries.append(
                {
                    "question": q,
                    "relevant_doc_files": [Path(path).name],
                    "relevant_chunk_ids": [],
                    "question_type": "specific" if i == 0 else "conceptual",
                    "language": "tr",
                    "_source_path": path,
                    "_source_filename": Path(path).name,
                    "_box_doc_id": doc_id,
                    "_box_chunk_id": c.chunk_id,
                    "_VERIFY": "check the question is answerable from this file "
                    "and the question_type is right",
                }
            )
        print(f"drafted 2 from {Path(path).name} (chunk {c.chunk_id})")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(entries, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"\n{len(entries)} draft questions -> {out}")
    print("verify each, then move survivors to eval/manual_goldset.yaml")


def main() -> None:
    ap = argparse.ArgumentParser(description="Draft manual golden-set questions.")
    ap.add_argument("corpus_dir")
    ap.add_argument("--config", default="config/vllm.yaml")
    ap.add_argument("--n", type=int, default=15, help="chunks to sample; drafts 2 questions each")
    ap.add_argument("--out", type=Path, default=Path("eval/manual_goldset.draft.yaml"))
    args = ap.parse_args()
    draft(args.corpus_dir, args.config, args.n, args.out)


if __name__ == "__main__":
    main()
