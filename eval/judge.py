"""LLM-as-judge for the agent eval: scores whether an answer carries the same
information as a reference answer. `Judge` takes any chat LLM behind the
`LLMProvider` port; a small (4B) local model was measured to be an unreliable
judge, so the judge is DeepSeek `deepseek-chat` (V3) via API, which
`eval_box.yaml`'s `llm:` block points to.
"""

from __future__ import annotations

from dataclasses import dataclass

JUDGE_PROMPT = """Bir soru-cevap sisteminin cevabını değerlendiriyorsun.

SORU: {question}

REFERANS CEVAP (doğru kabul edilen): {reference}

SİSTEMİN CEVABI: {answer}

Sistemin cevabı, referans cevabın taşıdığı bilgiyi veriyor mu? Farklı
kelimelerle söylemesi ya da fazladan doğru ayrıntı vermesi sorun değil.
Eksik bir anahtar bilgi, yanlış bilgi ya da çelişki varsa HAYIR.

Cevaptaki ⟦...⟧ işaretlerini yok say, onlar kaynak bağlantısıdır ve
cevabın doğruluğunu etkilemez.

İlk satıra tam olarak EVET ya da HAYIR yaz. İkinci satıra tek cümlelik
gerekçe."""


@dataclass(frozen=True)
class Verdict:
    correct: bool
    reason: str
    raw: str


class Judge:
    def __init__(self, llm) -> None:
        self._llm = llm

    def judge(self, question: str, reference: str, answer: str) -> Verdict:
        prompt = JUDGE_PROMPT.format(question=question, reference=reference, answer=answer)
        raw = self._llm.complete(prompt, max_tokens=200)
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        head = lines[0].upper() if lines else ""
        reason = lines[1] if len(lines) > 1 else ""
        if head.startswith("EVET"):
            return Verdict(True, reason, raw)
        if head.startswith("HAYIR"):
            return Verdict(False, reason, raw)
        return Verdict(False, f"unparseable: {lines[0] if lines else '<empty>'}", raw)

    def judge_or_skip(self, q, answer: str) -> Verdict | None:
        if not q.reference_answer:
            return None
        return self.judge(q.question, q.reference_answer, answer)
