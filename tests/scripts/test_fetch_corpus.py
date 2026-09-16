"""Bulk corpus fetch against a stub server. No network: the real
mevzuat.gov.tr walk is a --mevzuat run, not a test.
"""

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest


class _MevzuatStub(BaseHTTPRequestHandler):
    """Odd numbers 404, multiples of ten answer 200 with an HTML error page,
    the rest serve a PDF. The real site is far sparser than that and does use
    both failure shapes."""

    def log_message(self, *args):
        pass

    def do_GET(self):
        number = int(self.path.rsplit(".", 2)[-2])
        if number % 2:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if number % 10 == 0:
            body, content_type = b"<html>Aradiginiz sayfa bulunamadi</html>", "text/html"
        else:
            body, content_type = b"%PDF-1.4\n%%EOF\n", "application/pdf"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def stub(monkeypatch):
    server = HTTPServer(("127.0.0.1", 0), _MevzuatStub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    monkeypatch.setattr("scripts.fetch_corpus.MEVZUAT_URL", base + "/mevzuatmetin/1.5.{number}.pdf")
    # The certificate workaround is for the real host; over plain HTTP to a
    # stub there is nothing to verify.
    monkeypatch.setattr("scripts.fetch_corpus._ssl_context", lambda *a, **k: None)
    yield
    server.shutdown()


def test_fetch_stops_at_count_skipping_404s(tmp_path, stub):
    from scripts.fetch_corpus import fetch_mevzuat_bulk

    got = fetch_mevzuat_bulk(tmp_path, count=3, pause_s=0)
    assert len(got) == 3
    assert all(p.exists() and p.read_bytes().startswith(b"%PDF") for p in got)


def test_fetch_is_idempotent(tmp_path, stub):
    from scripts.fetch_corpus import fetch_mevzuat_bulk

    first = fetch_mevzuat_bulk(tmp_path, count=2, pause_s=0)
    second = fetch_mevzuat_bulk(tmp_path, count=2, pause_s=0)
    assert first == second


def test_a_200_that_is_not_a_pdf_is_not_written(tmp_path, stub):
    """The failure mode that matters: mevzuat.gov.tr answers some missing
    numbers with an HTML page and a 200, not a 404. Writing those would
    poison the measurement corpus with files that parse to nothing."""
    from scripts.fetch_corpus import fetch_mevzuat_bulk

    got = fetch_mevzuat_bulk(tmp_path, count=6, pause_s=0)
    assert len(got) == 6
    assert all(p.read_bytes().startswith(b"%PDF") for p in got)
    written = {p.name for p in tmp_path.iterdir() if p.suffix == ".pdf"}
    assert not any(int(n.split("_")[1].removesuffix(".pdf")) % 10 == 0 for n in written)


def test_max_probes_caps_the_walk(tmp_path, stub):
    """A run can never quietly hammer the site: once max_probes URLs have been
    tried, the walk stops even if it has not reached `count`."""
    from scripts.fetch_corpus import fetch_mevzuat_bulk

    got = fetch_mevzuat_bulk(tmp_path, count=1000, pause_s=0, max_probes=5)
    # 5 probes over numbers 2000..2004: 2000 is HTML, 2001/2003 are 404,
    # 2002/2004 are PDFs -> at most 2 files, and never more than 5 requests.
    assert len(got) <= 2
