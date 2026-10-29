"""End-to-end check of the Eumetsat adapter: list, download one scene, read it with satpy."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
from pathlib import Path

from eo_ml_ready.fetch.eumetsat import Eumetsat


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2026-09-15T12:00", help="UTC, ISO format")
    parser.add_argument("--minutes", type=int, default=20)
    parser.add_argument("--dest", type=Path, default=Path("data/seviri_test"))
    args = parser.parse_args()

    start = datetime.fromisoformat(args.start).replace(tzinfo=UTC)
    end = start + timedelta(minutes=args.minutes)

    provider = Eumetsat()
    refs = provider.list_scenes(start, end)
    print(f"Found {len(refs)} scenes between {start:%H:%M} and {end:%H:%M} UTC")
    for ref in refs:
        print(f"  {ref.time:%Y-%m-%d %H:%M}  {ref.key}")
    if not refs:
        raise SystemExit("No scenes found: check the window and the licence.")

    path = provider.download(refs[0], args.dest)
    print(f"Downloaded: {path} ({path.stat().st_size / 1e6:.0f} MB)")
    print(f"Next: python fallback\\inspect_scene.py {path}")


if __name__ == "__main__":
    main()
