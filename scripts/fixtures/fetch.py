"""Download real corpus samples. Not used by CI; run it locally when you want
the parser exercised against genuine documents rather than generated ones.

CUAD is CC BY 4.0. mevzuat.gov.tr publishes Turkish legislation publicly.
Neither is redistributed by this repository; this script fetches them to an
ignored directory.

mevzuat.gov.tr refuses the default httpx User-Agent (the request hangs until
timeout with zero bytes, not a clean failure) and serves an incomplete
certificate chain (leaf only, no intermediate), so this script sends a
browser User-Agent and fetches the missing intermediate certificate itself.
TLS verification stays ON throughout.
"""

from __future__ import annotations

import argparse
import ssl
from pathlib import Path

import httpx

MEVZUAT_URL = "https://www.mevzuat.gov.tr/mevzuatmetin/1.5.{number}.pdf"

# A few well-known Turkish laws with the structural patterns section 4.3
# tier 2 targets (MADDE / BOLUM / EK).
MEVZUAT_NUMBERS = ["6356", "5651", "2985", "4857", "6098"]

CUAD_ZIP = "https://zenodo.org/records/4595826/files/CUAD_v1.zip"

# Sent because the site drops the default httpx agent silently (see module
# docstring). Not an attempt to look like something we are not - the same
# public PDF is served either way.
_BROWSER_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

# The CA Issuers URI from the leaf certificate's Authority Information Access
# extension, read on 2026-09-05:
#   subject: C=TR, O=Cumhurbaskanligi, CN=*.tccb.gov.tr
#   issuer:  C=US, O=DigiCert Inc, CN=GeoTrust TLS RSA CA G1
#            (itself issued by DigiCert Global Root G2, already trusted)
# Hardcoded rather than parsed out of the live certificate because reading an
# X.509 extension would pull in `cryptography` for one string. If the site
# rotates its CA this stops working loudly, which is the right failure.
_INTERMEDIATE_URL = "http://cacerts.geotrust.com/GeoTrustTLSRSACAG1.crt"


def _ssl_context(cache: Path) -> ssl.SSLContext:
    """Default trust store plus the intermediate the server omits.

    `load_verify_locations` adds to the default roots rather than replacing
    them, so this widens the chain the server failed to send and weakens
    nothing.
    """
    context = httpx.create_ssl_context()
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        response = httpx.get(_INTERMEDIATE_URL, timeout=30.0)
        response.raise_for_status()
        cache.write_text(ssl.DER_cert_to_PEM_cert(response.content), encoding="ascii")
    context.load_verify_locations(cafile=str(cache))
    return context


def fetch_mevzuat(target: Path, limit: int) -> list[Path]:
    target.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    context = _ssl_context(target.parent / "_issuer_chain.pem")
    with httpx.Client(
        timeout=120.0,
        follow_redirects=True,
        verify=context,
        headers={"User-Agent": _BROWSER_UA},
    ) as client:
        for number in MEVZUAT_NUMBERS[:limit]:
            destination = target / f"mevzuat_{number}.pdf"
            if destination.exists():
                written.append(destination)
                continue
            response = client.get(MEVZUAT_URL.format(number=number))
            response.raise_for_status()
            destination.write_bytes(response.content)
            written.append(destination)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch real corpus samples.")
    parser.add_argument("--target", type=Path, default=Path("tests/fixtures/downloaded"))
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    files = fetch_mevzuat(args.target / "mevzuat", args.limit)
    print(f"mevzuat: {len(files)} file(s) in {args.target / 'mevzuat'}")
    print(
        "CUAD is a 106 MB archive; download it manually if you need it:\n"
        f"  {CUAD_ZIP}\n"
        f"  unzip into {args.target / 'cuad'}"
    )


if __name__ == "__main__":
    main()
