"""Scanned-PDF parser (escalation target when a PDF's text layer is missing or
garbage): rasterises every page via PyMuPDF's `get_pixmap` and runs RapidOCR,
preferring the `rapidocr` 3.x package over `rapidocr-onnxruntime` 1.x because
only the former's Latin recogniser can emit Turkish letters (ş ğ ı İ ö ü ç) at
all. `use_cls` (the angle classifier) is deliberately off: it measurably hurt
Turkish recognition, at the cost of no longer auto-correcting a page rotated
180 degrees. Dotless ı is still misrecognised even with the Latin model, a
known gap requiring a Turkish fine-tune. Weights are fetched once into
`model_dir` (`models/` by default, git-ignored) rather than rapidocr's
site-packages default, so a venv rebuild does not re-download them. Requires
the `parsers` extra; falls back to the `unavailable.py` stub without it.
"""

from __future__ import annotations

import os
from pathlib import Path

from porsuk.core.registry import register

_NAME = "ocr"
_COST = 100

# Rasterise at 200 DPI. Below ~150 the detector starts dropping small type;
# above 300 the pixmaps get large enough to matter across 5000 files and
# recognition does not measurably improve.
_DPI = 200

_DEFAULT_MODEL_DIR = os.environ.get("PORSUK_OCR_MODEL_DIR", "models")

try:
    import numpy as np
    import pymupdf

    _RASTER_AVAILABLE = True
except ImportError:  # pragma: no cover - pymupdf is a base dependency
    _RASTER_AVAILABLE = False


def _build_engine(model_dir: str, use_angle_classifier: bool):
    """The best RapidOCR available, or None.

    Order is the point: the modern package's Latin recogniser is the whole
    reason Turkish works, so a machine with both installed must get that one.
    """
    try:
        from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR

        engine = RapidOCR(
            params={
                "Global.model_root_dir": str(Path(model_dir)),
                "Global.use_cls": use_angle_classifier,
                "Rec.lang_type": LangRec.LATIN,
                "Rec.ocr_version": OCRVersion.PPOCRV5,
                "Rec.model_type": ModelType.MOBILE,
            }
        )
        return _ModernEngine(engine)
    except ImportError:
        pass

    try:
        from rapidocr_onnxruntime import RapidOCR as LegacyRapidOCR
    except ImportError:
        return None
    return _LegacyEngine(LegacyRapidOCR(), use_angle_classifier)


class _ModernEngine:
    """`rapidocr` 3.x. Returns a result object with a `txts` tuple."""

    name = "rapidocr"

    def __init__(self, engine) -> None:
        self._engine = engine

    def read(self, image) -> list[str]:
        result = self._engine(image)
        return list(getattr(result, "txts", None) or [])


class _LegacyEngine:
    """`rapidocr-onnxruntime` 1.x. Returns `(regions, elapsed)`, where each
    region is `[box, text, confidence]` - and `None` rather than an empty list
    when the detector found nothing."""

    name = "rapidocr-onnxruntime"

    def __init__(self, engine, use_angle_classifier: bool) -> None:
        self._engine = engine
        self._use_cls = use_angle_classifier

    def read(self, image) -> list[str]:
        regions, _elapsed = self._engine(image, use_cls=self._use_cls)
        return [text for _box, text, _confidence in regions or []]


if _RASTER_AVAILABLE and _build_engine(_DEFAULT_MODEL_DIR, False) is not None:
    from porsuk.core.models import Block, Outline, ParsedDocument, ParseOutcome

    def _failed(error: str) -> ParseOutcome:
        return ParseOutcome(
            document=None, status="failed", quality=0.0, parser_used=_NAME, error=error
        )

    def _page_image(page):
        """One page as an RGB array.

        `colorspace=csRGB` is not a default worth trusting - a CMYK or
        greyscale source would otherwise give a pixmap with a different `n`,
        and the reshape below would either throw or silently mis-shape the
        image.
        """
        pixmap = page.get_pixmap(dpi=_DPI, colorspace=pymupdf.csRGB)
        return np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
            pixmap.height, pixmap.width, pixmap.n
        )

    @register("parser", _NAME)
    class OCRParser:
        name = _NAME
        cost = _COST

        def __init__(
            self,
            model_dir: str = _DEFAULT_MODEL_DIR,
            use_angle_classifier: bool = False,
        ) -> None:
            # Loads the ONNX models once - never construct this per document.
            # The chain is assembled through the registry, which passes no
            # arguments today, so these defaults are what production uses;
            # they are parameters so a rotated corpus or a relocated model
            # directory can be tested without editing the module.
            self._engine = _build_engine(model_dir, use_angle_classifier)
            self.engine_name = self._engine.name

        def can_handle(self, path: str) -> bool:
            return Path(path).suffix.lower() == ".pdf"

        def parse(self, path: str) -> ParseOutcome:
            try:
                blocks, page_count = self._recognise(path)
            except Exception as exc:  # noqa: BLE001 - a Result, not an exception
                return _failed(f"{type(exc).__name__}: {exc}")

            if not blocks:
                # Not an exception either: a scan of a blank page, or a page
                # of handwriting, is a document OCR legitimately cannot read.
                return _failed("OCR produced no text")

            document = ParsedDocument(
                path=path,
                doc_type="pdf",
                blocks=tuple(blocks),
                # OCR recovers text, never structure. Claiming a heading
                # hierarchy here would be inventing one.
                outline=Outline(source="none"),
                page_count=page_count,
                # Every page was an image - that is why we are in this parser
                # at all. The gate reads this together with chars_per_page to
                # decide image_heavy.
                image_count=page_count,
            )
            # `quality` is deliberately 0.0: the router re-runs the gate on
            # whatever document a parser returns and overwrites this. A parser
            # scoring its own output would let it out-rank a better cheap
            # result on nothing but self-assessment.
            return ParseOutcome(
                document=document, status="ok", quality=0.0, parser_used=_NAME, error=None
            )

        def _recognise(self, path: str) -> tuple[list[Block], int]:
            blocks: list[Block] = []
            with pymupdf.open(path) as doc:
                if doc.needs_pass:
                    raise ValueError("PDF is password protected")
                for page_no, page in enumerate(doc, start=1):
                    for text in self._engine.read(_page_image(page)):
                        stripped = text.strip()
                        if stripped:
                            blocks.append(Block(text=stripped, page_no=page_no))
                return blocks, doc.page_count

else:  # pragma: no cover - exercised only without the `parsers` extra
    # One registration site per parser name, real or stub. Letting both this
    # module and `unavailable.py` register "ocr" would make which one wins
    # depend on import order in `_load_adapters` - and ruff sorts those
    # imports alphabetically, putting `unavailable` last, so the stub would
    # quietly overwrite the real parser and every scan would come back
    # "not available".
    from porsuk.adapters.parsers.unavailable import OCRParser as _OCRStub

    OCRParser = register("parser", _NAME)(_OCRStub)
