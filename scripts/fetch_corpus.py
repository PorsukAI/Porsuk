"""Bulk corpus fetch: walks mevzuat.gov.tr's predictable PDF path to build a
large sample for measuring first-index time and VRAM against something the
size of the real corpus (`scripts/fixtures/fetch.py` fetches a small hand-picked
set instead). CUAD/EDGAR is the English half and stays a manual download.

Nothing is redistributed: everything lands in `tests/fixtures/downloaded/`,
which is git-ignored. The certificate-chain and User-Agent workarounds for
mevzuat.gov.tr are imported from `scripts/fixtures/fetch.py` rather than
re-derived; that module's docstring explains both.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import httpx

from scripts.fixtures.fetch import _BROWSER_UA, CUAD_ZIP, _ssl_context

MEVZUAT_URL = "https://www.mevzuat.gov.tr/mevzuatmetin/1.5.{number}.pdf"

# Law numbers are roughly chronological and 1.5.<n> is far from dense, most
# numbers in this window 404. Walking rather than curating is the point: the
# measurement wants a few hundred ordinary documents, not the small set of
# pathological ones hand-picked elsewhere.
_LAW_NUMBERS = range(2000, 7500)

# This is a bulk scraper of a public government site. Be a good citizen:
# pause between requests, and cap how many we probe in one run. The default
# corpus size is deliberately small - the first-index measurement needs a few
# dozen ordinary documents, not hundreds.
_REQUEST_PAUSE_S = 1.0
_MAX_PROBES_DEFAULT = 300


def fetch_mevzuat_bulk(
    target: Path, count: int, *, pause_s: float = _REQUEST_PAUSE_S, max_probes: int | None = None
) -> list[Path]:
    """Download up to `count` mevzuat PDFs into `target`, skipping misses.

    Already-downloaded files are counted and reused, so re-running after an
    interrupted fetch resumes instead of starting over. `pause_s` throttles
    the walk; `max_probes` caps how many URLs are tried before giving up,
    so a run can never quietly hammer the site for thousands of requests.
    """
    target.mkdir(parents=True, exist_ok=True)
    url_template = MEVZUAT_URL
    # Only https needs the intermediate certificate the server omits; over
    # plain http (the tests' stub) there is nothing to verify.
    verify = _ssl_context(target.parent / "_issuer_chain.pem") if _is_https(url_template) else True
    probe_budget = max_probes if max_probes is not None else _MAX_PROBES_DEFAULT

    written: list[Path] = []
    probes = 0
    with httpx.Client(
        timeout=120.0,
        follow_redirects=True,
        verify=verify,
        headers={"User-Agent": _BROWSER_UA},
    ) as client:
        for number in _LAW_NUMBERS:
            if len(written) >= count or probes >= probe_budget:
                break
            destination = target / f"mevzuat_{number}.pdf"
            if destination.exists():
                written.append(destination)
                continue
            if probes:
                time.sleep(pause_s)
            probes += 1
            content = _get_pdf(client, url_template.format(number=number))
            if content is None:
                continue
            destination.write_bytes(content)
            written.append(destination)
    return written


def _is_https(url_template: str) -> bool:
    return url_template.startswith("https://")


def _get_pdf(client: httpx.Client, url: str) -> bytes | None:
    """The document, or None for every way this can legitimately miss.

    A missing law number is the normal case over a walk of 5500 of them, so
    none of these raise. The `%PDF` check is not belt-and-braces: the site
    answers some missing numbers with an HTML page under a 200, and writing
    those would fill the measurement corpus with files that parse to nothing
    and quietly drag the first-index numbers down.
    """
    try:
        response = client.get(url)
    except httpx.HTTPError:
        return None
    if response.status_code != 200:
        return None
    if not response.content.startswith(b"%PDF"):
        return None
    return response.content


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bulk corpus fetch. Scrapes a public site - run it "
        "sparingly; the default is small on purpose.",
    )
    parser.add_argument("--target", type=Path, default=Path("tests/fixtures/downloaded"))
    parser.add_argument("--mevzuat", type=int, default=40, help="how many mevzuat PDFs to keep")
    parser.add_argument(
        "--pause", type=float, default=_REQUEST_PAUSE_S, help="seconds between requests"
    )
    parser.add_argument(
        "--max-probes",
        type=int,
        default=_MAX_PROBES_DEFAULT,
        help="hard cap on URLs tried in one run",
    )
    parser.add_argument("--cuad", action="store_true", help="print the CUAD download steps")
    args = parser.parse_args()

    destination = args.target / "mevzuat"
    files = fetch_mevzuat_bulk(
        destination, args.mevzuat, pause_s=args.pause, max_probes=args.max_probes
    )
    print(f"mevzuat: {len(files)} PDF(s) in {destination}")
    if args.cuad:
        print(
            "CUAD is a 106 MB archive - download it manually:\n"
            f"  {CUAD_ZIP}\n"
            f"  unzip into {args.target / 'cuad'}"
        )


if __name__ == "__main__":
    main()
