"""Provider-neutral types for finding and downloading satellite scenes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class SceneRef:
    """One downloadable scene: provider name, provider key, sensing start and channel."""

    provider: str
    key: str
    time: datetime
    channel: str


class Provider(Protocol):
    """Interface every data source adapter implements."""

    name: str

    def list_scenes(self, start: datetime, end: datetime, channel: str) -> list[SceneRef]:
        """Return scenes sensed in [start, end), sorted by time."""
        ...

    def download(self, ref: SceneRef, dest: Path) -> Path:
        """Download a scene into dest and return the local file path."""
        ...
