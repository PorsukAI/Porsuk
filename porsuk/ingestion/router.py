"""Parser chain of responsibility: cheapest parser that can handle the file
goes first, the quality gate decides whether to climb, the best outcome by
quality is kept (never merely the last one), and no parser failure escapes as
an exception.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from porsuk.core.config import QualityConfig
from porsuk.core.models import ParseOutcome
from porsuk.core.ports import Parser
from porsuk.ingestion.quality import (
    GateDecision,
    decide,
    document_script_validity,
    measure,
    score,
)


@dataclass
class CanHandleFailure:
    """How many times, and with what, one parser's can_handle has thrown."""

    count: int
    first_error: str


class ParserRouter:
    def __init__(self, parsers: Sequence[Parser], quality: QualityConfig) -> None:
        self._parsers = tuple(sorted(parsers, key=lambda p: p.cost))
        self._quality = quality
        # Aggregated by parser name, not appended per file: a parser whose
        # can_handle always throws is a broken parser, not a broken file, and
        # that only becomes visible across many calls (thousands of files,
        # not one). Keying by name and storing a count plus the
        # first message keeps this O(number of registered parsers) rather
        # than O(files parsed), a handful of entries for the life of the
        # router, whether it sees 1 file or 5000, and safe to read after a
        # run without having retained one string per failure.
        self.can_handle_failures: dict[str, CanHandleFailure] = {}

    @property
    def parser_names(self) -> tuple[str, ...]:
        """The chain's parser names in cost order, what a worker process
        needs to rebuild an equivalent router."""
        return tuple(p.name for p in self._parsers)

    def parse(self, path: str) -> ParseOutcome:
        candidates: list[Parser] = []
        broken: list[str] = []
        for parser in self._parsers:
            handles, error = self._safe_can_handle(parser, path)
            if error is not None:
                broken.append(error)
                self._record_can_handle_failure(parser.name, error)
            if handles:
                candidates.append(parser)

        if not candidates:
            detail = f" (can_handle raised for: {'; '.join(broken)})" if broken else ""
            return ParseOutcome(
                document=None,
                status="failed",
                quality=0.0,
                parser_used="router",
                error=f"no parser can handle {path!r}{detail}",
            )

        tried: set[str] = set()
        # The decision travels WITH the outcome it was made about. Keeping a
        # single `decision` rebound per parse meant `_finalise` could report
        # the last parser's verdict against a different parser's document.
        best: tuple[ParseOutcome, GateDecision] | None = None
        first_failure: ParseOutcome | None = None
        current: Parser | None = candidates[0]
        escalating = False
        escalation_error: str | None = None

        while current is not None:
            tried.add(current.name)
            outcome = self._safe_parse(current, path)
            scored = self._safe_score(outcome) if outcome.document is not None else None

            if scored is None:
                first_failure = first_failure or outcome
                if escalating:
                    escalation_error = outcome.error
                    # The gate named this parser and it failed. Falling
                    # through to whatever is next would run a parser nobody
                    # asked for, e.g. a torch-backed Docling pass on
                    # every scanned page.
                    break
                current = self._next_untried(candidates, tried)
                continue

            best = self._better(best, scored)

            if scored[1].escalate_to is None:
                break
            current = self._escalation_target(candidates, scored[1].escalate_to, tried)
            escalating = current is not None

        if best is None:
            return first_failure or ParseOutcome(
                document=None,
                status="failed",
                quality=0.0,
                parser_used="router",
                error=f"every parser failed on {path!r}",
            )
        return self._finalise(*best, escalation_error)

    def _record_can_handle_failure(self, parser_name: str, error: str) -> None:
        existing = self.can_handle_failures.get(parser_name)
        if existing is None:
            self.can_handle_failures[parser_name] = CanHandleFailure(count=1, first_error=error)
        else:
            existing.count += 1

    @staticmethod
    def _safe_can_handle(parser: Parser, path: str) -> tuple[bool, str | None]:
        """can_handle is part of the Parser protocol too.

        A parser that throws while merely being asked "is this yours?" must
        not remove every other parser from consideration: that would let one
        broken parser's can_handle stop the whole 5000-file run, exactly the
        failure mode _safe_parse already guards against for parse() itself.
        Treated as "cannot handle this file" and recorded rather than
        swallowed outright, so a parser whose can_handle always throws does
        not look identical to one that simply never matches.
        """
        try:
            return bool(parser.can_handle(path)), None
        except Exception as exc:  # noqa: BLE001
            return False, f"{parser.name}: {type(exc).__name__}: {exc}"

    @staticmethod
    def _safe_parse(parser: Parser, path: str) -> ParseOutcome:
        """Never raises, and never returns anything but a ParseOutcome.

        Catching exceptions is only half the guarantee: a parser that returns
        None, or a string, breaks the router just as thoroughly a line later.
        No parser in the tree does that today, which is exactly why it is
        checked: the guarantee exists to survive the parser nobody has
        written yet.
        """
        try:
            result = parser.parse(path)
        except Exception as exc:  # noqa: BLE001
            return ParseOutcome(
                document=None,
                status="failed",
                quality=0.0,
                parser_used=parser.name,
                error=f"{type(exc).__name__}: {exc}",
            )
        if not isinstance(result, ParseOutcome):
            return ParseOutcome(
                document=None,
                status="failed",
                quality=0.0,
                parser_used=parser.name,
                error=f"{parser.name} returned {type(result).__name__}, not a ParseOutcome",
            )
        return result

    def _safe_score(self, outcome: ParseOutcome) -> tuple[ParseOutcome, GateDecision] | None:
        """Scoring is inside the guarantee too: `measure` and `decide` run on
        text a parser produced, and pathological text must not escape as an
        exception any more than a pathological file may."""
        try:
            return self._score(outcome)
        except Exception:  # noqa: BLE001
            return None

    def _score(self, outcome: ParseOutcome) -> tuple[ParseOutcome, GateDecision]:
        assert outcome.document is not None
        components = measure(outcome.document)
        decision = decide(
            components,
            self._quality,
            image_count=outcome.document.image_count,
            script_validity=document_script_validity(outcome.document),
        )
        scored = replace(
            outcome,
            quality=score(components, self._quality),
            components=components,
            image_heavy=decision.image_heavy,
        )
        return scored, decision

    @staticmethod
    def _better(
        best: tuple[ParseOutcome, GateDecision] | None,
        candidate: tuple[ParseOutcome, GateDecision],
    ) -> tuple[ParseOutcome, GateDecision]:
        if best is None:
            return candidate
        return candidate if candidate[0].quality > best[0].quality else best

    @staticmethod
    def _next_untried(candidates: list[Parser], tried: set[str]) -> Parser | None:
        return next((p for p in candidates if p.name not in tried), None)

    @staticmethod
    def _escalation_target(candidates: list[Parser], wanted: str, tried: set[str]) -> Parser | None:
        """The named parser, if it is in the chain and has not been tried.

        Falling back to 'any pricier parser' would silently route a document
        the gate wanted OCR'd into Docling. If the named target is unavailable
        the honest answer is that the escalation could not be fulfilled.
        """
        return next((p for p in candidates if p.name == wanted and p.name not in tried), None)

    @staticmethod
    def _finalise(
        best: ParseOutcome, decision: GateDecision, escalation_error: str | None = None
    ) -> ParseOutcome:
        """Downgrade when the parse is known to be incomplete or damaged.

        Two shapes of that. The gate wanted an escalation that never happened:
        the document parsed, but the gate believes it needs a parser we do
        not have. Or the gate found damage no parser can undo, in which case
        there was nothing to escalate to in the first place. Reporting either
        as `ok` would hide a known-bad parse.
        """
        if decision.unrecoverable:
            return replace(best, status="degraded", error=decision.reason)
        if decision.escalate_to is None:
            return best
        if best.parser_used == decision.escalate_to:
            return best
        # The escalation target's own message says what is missing and when
        # it lands (see adapters/parsers/unavailable.py). Dropping it left the
        # user unable to tell "not built yet" from "broken".
        detail = f": {escalation_error}" if escalation_error else ""
        return replace(
            best,
            status="degraded",
            error=(
                f"gate requested escalation to {decision.escalate_to!r} "
                f"({decision.reason}) but it did not produce a better result{detail}"
            ),
        )
