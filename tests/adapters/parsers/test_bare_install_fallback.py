"""Both escalation slots stay filled without the `parsers` extra.

Runs on every install, with or without the extra: the missing library is
simulated, so this is the bare-install path exercised on a machine that has
RapidOCR and Docling sitting right there.

What it guards is narrow: `ocr.py` and `unavailable.py` both registering
"ocr" made the winner depend on the order of the imports in `_load_adapters`,
and ruff sorts those alphabetically with `unavailable` last, so the stub
would have overwritten the real parser and every scanned PDF would have come
back "not available" with nothing failing. One registration site per name is
what makes that unrepresentable.
"""

from __future__ import annotations

import builtins
import importlib
import sys

import pytest

from porsuk.core import registry

_MODULE = "porsuk.adapters.parsers.ocr"
# Both packages, because ocr.py prefers the newer one and falls back to the
# older: hiding only one leaves the other satisfying the import and the stub
# never gets registered.
_BLOCKED = {"rapidocr", "rapidocr_onnxruntime"}


@pytest.fixture
def without_rapidocr(monkeypatch):
    """Reimport the OCR module with RapidOCR unimportable, then reimport it
    for real so the session is left holding the real parser again.

    Teardown re-imports rather than restoring a snapshot of the registry.
    Restoring a snapshot looks equivalent and is not: registration happens as
    an import side effect, so putting back a dict taken before
    `unavailable.py` was first imported silently dropped `docling` for the
    rest of the session, the module stayed in `sys.modules`, so nothing ever
    registered it again, and eleven CLI tests failed a long way from here.
    """
    real_import = builtins.__import__

    def blocking_import(name, *args, **kwargs):
        if name in _BLOCKED:
            raise ImportError(f"simulated missing dependency: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocking_import)
    sys.modules.pop(_MODULE, None)
    yield importlib.import_module(_MODULE)

    monkeypatch.undo()
    sys.modules.pop(_MODULE, None)
    importlib.import_module(_MODULE)


def test_the_ocr_slot_is_still_registered(without_rapidocr):
    assert "ocr" in registry.registered("parser")


def test_the_registered_parser_is_the_stub(without_rapidocr):
    parser = registry.build("parser", {"provider": "ocr"})
    assert parser.name == "ocr"
    assert parser.cost == 100
    assert parser.can_handle("/x/scan.pdf") is True


def test_the_stub_fails_with_a_reason_naming_what_is_missing(without_rapidocr):
    parser = registry.build("parser", {"provider": "ocr"})
    outcome = parser.parse("/x/scan.pdf")
    assert outcome.status == "failed"
    assert outcome.document is None
    assert "parsers" in (outcome.error or "")
