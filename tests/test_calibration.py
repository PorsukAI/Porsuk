"""Tests for the parse quality calibration script and corpus handling."""

from pathlib import Path


def test_no_real_documents_were_committed():
    """mevzuat and CUAD are fetched, never redistributed."""
    downloaded = Path("tests/fixtures/downloaded")
    if not downloaded.exists():
        return
    import subprocess

    tracked = subprocess.run(
        ["git", "ls-files", str(downloaded)], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert not tracked, f"downloaded corpus must not be tracked: {tracked}"


def test_measure_corpus_reports_every_component_per_file(tmp_path):
    from porsuk.core.config import load_config
    from porsuk.core.container import build_router
    from scripts.calibrate_quality import measure_corpus
    from scripts.fixtures.generate import generate_fixtures

    generate_fixtures(tmp_path)
    cfg = load_config("config/local.yaml", env={})
    rows = measure_corpus(build_router(cfg), tmp_path, cfg)
    assert rows
    for row in rows:
        assert set(row) >= {
            "path",
            "category",
            "parser_used",
            "status",
            "quality",
            "chars_per_page",
            "bad_char_ratio",
            "text_plausibility",
            "empty_table_blocks",
            "escalate_to",
            "trigger",
        }


def test_measure_corpus_also_reports_chunking(tmp_path):
    """Almost every generated fixture is one chunk, so the chunking half of
    this phase has very little evidence behind it. Reporting chunk count and
    the section paths alongside the gate components is what lets the real
    mevzuat PDFs - the first multi-chunk documents in this project - say
    something about tier 2 as well as about the thresholds.
    """
    from porsuk.core.config import load_config
    from porsuk.core.container import build_router
    from scripts.calibrate_quality import measure_corpus
    from scripts.fixtures.generate import generate_fixtures

    generate_fixtures(tmp_path)
    cfg = load_config("config/local.yaml", env={})
    rows = measure_corpus(build_router(cfg), tmp_path, cfg)
    parsed = [r for r in rows if r["status"] != "failed"]
    assert parsed
    for row in parsed:
        assert isinstance(row["chunk_count"], int)
        assert isinstance(row["section_paths"], tuple)

    madde = next(r for r in rows if r["path"].endswith("madde_numbering.pdf"))
    # The heading marker is fused to its article body in this fixture
    # ("MADDE 1 - Isbu sozlesme, ..."), so this also pins the title trimming
    # from 1a5ae24: the path carries the marker, not the whole paragraph.
    assert "BİRİNCİ BÖLÜM > MADDE 1" in madde["section_paths"], madde["section_paths"]
