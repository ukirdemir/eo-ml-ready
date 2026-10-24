"""GOES ABI CONUS cloud and moisture imagery from the open NOAA bucket on AWS."""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

from eo_ml_ready.fetch.base import SceneRef

KEY_PATTERN = re.compile(
    r"OR_ABI-L2-CMIPC-M\dC(?P<band>\d{2})_G\d{2}_s(?P<year>\d{4})(?P<doy>\d{3})"
    r"(?P<hour>\d{2})(?P<minute>\d{2})(?P<second>\d{2})\d_e"
)


def hour_prefixes(product: str, start: datetime, end: datetime) -> list[str]:
    """Return the hourly key prefixes that cover [start, end)."""
    hour = start.replace(minute=0, second=0, microsecond=0)
    prefixes = []
    while hour < end:
        prefixes.append(f"{product}/{hour.year}/{hour.timetuple().tm_yday:03d}/{hour.hour:02d}/")
        hour += timedelta(hours=1)
    return prefixes


def parse_key(key: str) -> SceneRef | None:
    """Parse an ABI CMIP file key into a SceneRef, or None if it does not match."""
    match = KEY_PATTERN.search(key)
    if match is None:
        return None
    start = datetime(int(match["year"]), 1, 1, tzinfo=UTC) + timedelta(
        days=int(match["doy"]) - 1,
        hours=int(match["hour"]),
        minutes=int(match["minute"]),
        seconds=int(match["second"]),
    )
    return SceneRef(provider="goes_aws", key=key, time=start, channel=f"C{match['band']}")


class GoesAws:
    """Anonymous access to the NOAA GOES bucket (ABI-L2-CMIPC, 5-minute CONUS)."""

    name = "goes_aws"

    def __init__(self, satellite: int = 19, product: str = "ABI-L2-CMIPC") -> None:
        self.bucket = f"noaa-goes{satellite}"
        self.product = product
        self._fs = None

    def filesystem(self):
        """Return a cached anonymous S3 filesystem."""
        if self._fs is None:
            import s3fs

            self._fs = s3fs.S3FileSystem(anon=True)
        return self._fs

    def list_scenes(self, start: datetime, end: datetime, channel: str) -> list[SceneRef]:
        """Return scenes of one channel sensed in [start, end), sorted by time."""
        found = []
        for prefix in hour_prefixes(self.product, start, end):
            for path in self.filesystem().ls(f"{self.bucket}/{prefix}", detail=False):
                ref = parse_key(path)
                if ref and ref.channel == channel and start <= ref.time < end:
                    found.append(ref)
        return sorted(found, key=lambda ref: ref.time)

    def download(self, ref: SceneRef, dest: Path) -> Path:
        """Download a scene into dest unless it is already there."""
        dest.mkdir(parents=True, exist_ok=True)
        target = dest / Path(ref.key).name
        if not target.exists():
            self.filesystem().get(ref.key, str(target))
        return target
