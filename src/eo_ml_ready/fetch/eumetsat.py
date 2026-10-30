"""EUMETSAT Data Store adapter for MSG SEVIRI (needs the eumetsat extra and licences)."""

from __future__ import annotations

import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

from eo_ml_ready.fetch.base import SceneRef

COLLECTION = "EO:EUM:DAT:MSG:HRSEVIRI"


class Eumetsat:
    """SEVIRI full-disk scenes; one product holds all channels, so channel is ignored."""

    name = "eumetsat"

    def __init__(self, collection: str = COLLECTION) -> None:
        self.collection = collection
        self._store = None

    def datastore(self):
        """Return a Data Store client using credentials saved by eumdac."""
        if self._store is None:
            import eumdac

            text = (Path.home() / ".eumdac" / "credentials").read_text().strip()
            key, secret = text.split(",")[:2]
            self._store = eumdac.DataStore(eumdac.AccessToken((key.strip(), secret.strip())))
        return self._store

    def list_scenes(self, start: datetime, end: datetime, channel: str = "all") -> list[SceneRef]:
        """Return products sensed in [start, end), sorted by time."""
        collection = self.datastore().get_collection(self.collection)
        refs = [
            SceneRef(
                provider=self.name,
                key=str(product),
                time=product.sensing_start.replace(tzinfo=UTC),
                channel="all",
            )
            for product in collection.search(dtstart=start, dtend=end)
        ]
        return sorted(refs, key=lambda ref: ref.time)

    def download(self, ref: SceneRef, dest: Path, attempts: int = 3) -> Path:
        """Download a product zip into dest unless it is already there.

        The file is written under a temporary name and renamed when complete, so an interrupted
        download never leaves a truncated zip behind. Failed attempts are retried after a pause.
        """
        dest.mkdir(parents=True, exist_ok=True)
        target = dest / f"{ref.key}.zip"
        if target.exists():
            return target
        partial = target.with_suffix(".part")
        for attempt in range(1, attempts + 1):
            try:
                product = self.datastore().get_product(
                    product_id=ref.key, collection_id=self.collection
                )
                with product.open() as source, partial.open("wb") as out:
                    shutil.copyfileobj(source, out)
                partial.replace(target)
                return target
            except Exception:
                partial.unlink(missing_ok=True)
                if attempt == attempts:
                    raise
                time.sleep(10 * attempt)
        return target
