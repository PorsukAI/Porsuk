"""Every fixture's expected gate decision, declared in the manifest.

Asserting against a declared answer rather than a snapshot is what makes this
a calibration check instead of a change detector: a snapshot would happily
record a wrong decision as correct.
"""

import pytest
import yaml

from porsuk.core.config import load_config
from porsuk.core.container import build_router

MANIFEST = "tests/fixtures/manifest.yaml"


def _stubbed_parsers() -> frozenset[str]:
    """Escalation targets that are the `unavailable.py` stub on this install.

    The manifest declares what the gate should conclude *and* which parser
    should answer. Without the `parsers` extra the answer is a stub, so those
    rows are skipped rather than failed: the gate's decision is still checked
    on every install, only the fulfilment is not.
    """
    from porsuk.adapters.parsers.unavailable import UnavailableParser
    from porsuk.core.container import _load_adapters
    from porsuk.core.registry import build

    _load_adapters()
    stubbed = set()
    for name in ("ocr", "docling"):
        try:
            parser = build("parser", {"provider": name})
        except Exception:  # noqa: BLE001 - an unregistered slot is not a stub
            continue
        if isinstance(parser, UnavailableParser):
            stubbed.add(name)
    return frozenset(stubbed)


_STUBBED_PARSERS = _stubbed_parsers()


def _expectations():
    with open(MANIFEST, encoding="utf-8") as handle:
        categories = yaml.safe_load(handle)["categories"]
    for name, spec in categories.items():
        for filename, expected in (spec.get("expected") or {}).items():
            yield name, filename, expected


@pytest.fixture(scope="module")
def router():
    return build_router(load_config("config/local.yaml", env={}))


@pytest.mark.parametrize(
    ("category", "filename", "expected"),
    list(_expectations()),
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_the_gate_reaches_the_declared_decision(router, fixtures_dir, category, filename, expected):
    outcome = router.parse(str(fixtures_dir / category / filename))

    # When the manifest expects an escalation parser that is only a stub on
    # this install, the declared outcome describes the *fulfilled* climb -
    # status `ok`, the real parser named. The stub cannot fulfil it, so the
    # gate still requests the escalation but the router keeps the cheap
    # result and marks it `degraded`. Check that the gate made the right
    # call, then skip the fulfilment assertions. CI runs without the
    # `parsers` extra, so this is the default path.
    if expected.get("parser_used") in _STUBBED_PARSERS:
        assert expected["parser_used"] in (outcome.error or ""), (
            f"{category}/{filename}: gate should still request "
            f"{expected['parser_used']}, error was {outcome.error!r}"
        )
        pytest.skip(f"{expected['parser_used']} is the bare-install stub")
    assert outcome.status == expected["status"], (
        f"{category}/{filename}: expected status {expected['status']}, got "
        f"{outcome.status} (error: {outcome.error})"
    )
    if expected.get("escalate_to") is None:
        assert outcome.error is None or "escalation" not in outcome.error
    else:
        assert expected["escalate_to"] in (outcome.error or "")

    # The trigger, not just the fact of an escalation. Which component fired is
    # the calibrated claim, and without this two thresholds could be swapped
    # and every test still pass: the finding that scan_with_garbage_layer.pdf
    # is caught by text_plausibility rather than chars_per_page had no
    # regression guard.
    if expected.get("trigger") is not None:
        assert expected["trigger"] in (outcome.error or ""), (
            f"{category}/{filename}: expected trigger {expected['trigger']}, "
            f"got error {outcome.error!r}"
        )

    # Which parser the outcome came from, where the manifest declares it.
    # A fulfilled escalation is only observable here: status `ok` alone cannot
    # tell "the cheap parser was good enough" from "the climb worked".
    if expected.get("parser_used") is not None:
        assert outcome.parser_used == expected["parser_used"], (
            f"{category}/{filename}: expected parser {expected['parser_used']}, "
            f"got {outcome.parser_used} (error: {outcome.error})"
        )

    if "image_heavy" in expected:
        assert outcome.image_heavy is expected["image_heavy"], (
            f"{category}/{filename}: expected image_heavy={expected['image_heavy']}"
        )


def test_every_non_broken_category_declares_expectations():
    """A category with no declared expectation is a category the calibration
    never checked."""
    with open(MANIFEST, encoding="utf-8") as handle:
        categories = yaml.safe_load(handle)["categories"]
    missing = [name for name, spec in categories.items() if not spec.get("expected")]
    assert not missing, f"categories without expected gate decisions: {missing}"
