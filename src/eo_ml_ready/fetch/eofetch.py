"""Scene provider on top of eo-fetch-api (the ``fetch`` extra): GOES on AWS and MSG SEVIRI."""

from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path

from eo_ml_ready.fetch.base import SceneRef

INSTALL_HINT = (
    'This step needs eo-fetch-api: run pip install -e ".[fetch]" in the repository '
    "(or pip install eo-fetch-api)."
)


def goes_pattern(channel: str) -> str:
    """File-name glob that selects one ABI channel of the CONUS cloud and moisture product."""
    return f"OR_ABI-L2-CMIPC-M?{channel}_*"


class EoFetchProvider:
    """Find and download scenes with eo-fetch-api."""

    def __init__(self, provider: str, product: str, pattern=None) -> None:
        try:
            import eo_fetch  # noqa: F401
        except ImportError as error:
            raise SystemExit(INSTALL_HINT) from error
        self.name = provider
        self.provider = provider
        self.product = product
        self.pattern = pattern
        self._items = {}
        self._backend = None
        self._gate = None

    def list_scenes(self, start: datetime, end: datetime, channel: str = "all") -> list[SceneRef]:
        """Return scenes sensed in [start, end), sorted by time."""
        import eo_fetch as eof

        text = self.pattern(channel) if self.pattern else ""
        found = eof.search(self.provider, self.product, start=start, end=end, text=text)
        self._backend = found.backend
        self._gate = self._gate or threading.BoundedSemaphore(found.backend.max_connections)
        refs = []
        for item in found:
            if item.time is None:
                continue
            self._items[item.key] = item
            refs.append(SceneRef(provider=self.name, key=item.key, time=item.time, channel=channel))
        return sorted(refs, key=lambda ref: ref.time)

    def download(self, ref: SceneRef, dest: Path) -> Path:
        """Download one scene into ``dest`` (skipped if a verified copy is there)."""
        from eo_fetch import downloader

        item = self._items[ref.key]
        with self._gate:
            result = downloader.download_items(self._backend, [item], dest, workers=1)[0]
        if result.path is None:
            raise RuntimeError(f"Download of {item.key} failed: {result.error}")
        return result.path


def goes(satellite: int = 19) -> EoFetchProvider:
    """GOES ABI CONUS cloud and moisture imagery from the open NOAA bucket on AWS."""
    return EoFetchProvider("noaa", f"goes{satellite}-cmip-conus", goes_pattern)


def seviri() -> EoFetchProvider:
    """Meteosat SEVIRI level 1.5 full disk (0 degrees) from the EUMETSAT Data Store."""
    return EoFetchProvider("eumetsat", "seviri")
