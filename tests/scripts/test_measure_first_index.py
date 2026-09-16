from scripts.measure_first_index import Measurement, measure, write_record

_CFG = (
    "llm: {provider: fake, model: fake}\n"
    "embedder: {provider: fake, dim: 8}\n"
    "store: {provider: inmemory}\n"
    "parsing: {parsers: [text]}\n"
    "profile: {llm_enabled: false}\n"
    "pipeline: {parse_workers: 1}\n"
)


def test_measure_reports_timing_and_counts(tmp_path):
    corpus = tmp_path / "c"
    corpus.mkdir()
    for i in range(5):
        (corpus / f"{i}.txt").write_text("Belge metni burada duruyor. " * 40)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(_CFG)

    m = measure(str(corpus), str(cfg), sample_gpu=False)
    assert m.files == 5
    assert m.failed == 0
    assert m.seconds > 0
    assert m.peak_vram_mib is None  # sampling off


def test_measure_counts_failures_without_stopping(tmp_path):
    corpus = tmp_path / "c"
    corpus.mkdir()
    (corpus / "good.txt").write_text("Geçerli metin burada. " * 30)
    (corpus / "bad.bin").write_bytes(b"\x00\x00\x00")
    cfg = tmp_path / "c.yaml"
    cfg.write_text(_CFG)

    m = measure(str(corpus), str(cfg), sample_gpu=False)
    assert m.files == 2
    assert m.failed == 1


def test_write_record_produces_a_dated_markdown_file(tmp_path):
    m = Measurement(files=42, seconds=120.0, failed=1, peak_vram_mib=3500)
    out = write_record(m, tmp_path / "measurements", corpus_name="mevzuat-80")
    assert out.exists()
    body = out.read_text()
    assert "mevzuat-80" in body
    assert "42" in body
    assert "3500" in body
    assert out.name.endswith("-first-index.md")
