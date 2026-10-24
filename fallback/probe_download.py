"""Probe Data Store download access: print the HTTP status and message for two products."""

from __future__ import annotations

from pathlib import Path

import eumdac
import requests

COLLECTION = "EO:EUM:DAT:MSG:HRSEVIRI"
OLD_PRODUCT = "MSG3-SEVI-MSG15-0100-NA-20260915121243.372000000Z-NA"


def load_token() -> eumdac.AccessToken:
    """Build a token from the credentials file written by eumdac."""
    text = (Path.home() / ".eumdac" / "credentials").read_text().strip()
    key, secret = text.split(",")[:2]
    return eumdac.AccessToken((key.strip(), secret.strip()))


def probe(token: eumdac.AccessToken, product_id: str) -> None:
    """Request the first kilobyte of a product and report the response."""
    datastore = eumdac.DataStore(token)
    product = datastore.get_product(product_id=product_id, collection_id=COLLECTION)
    response = requests.get(
        product.url,
        headers={"Authorization": f"Bearer {token}", "Range": "bytes=0-1023"},
        timeout=60,
    )
    print(f"{product_id}\n  status {response.status_code}, {response.headers.get('content-type')}")
    if response.status_code >= 400:
        print(f"  body: {response.text[:800]}")


def main() -> None:
    token = load_token()
    datastore = eumdac.DataStore(token)
    latest = next(iter(datastore.get_collection(COLLECTION).search()))
    probe(token, OLD_PRODUCT)
    probe(token, str(latest))


if __name__ == "__main__":
    main()
