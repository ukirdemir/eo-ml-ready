"""Inspect one downloaded SEVIRI scene: file size, channels, grid and brightness temperature."""

from __future__ import annotations

import argparse
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
from satpy import Scene


def find_native(path: Path, workdir: Path) -> Path:
    """Return a .nat file, extracting a zip first if needed."""
    if path.is_dir():
        nat = sorted(path.rglob("*.nat"))
        if nat:
            return nat[0]
        zips = sorted(path.rglob("*.zip"))
        if not zips:
            raise SystemExit(f"No .nat or .zip file found under {path}")
        path = zips[0]
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            archive.extractall(workdir)
        nat = sorted(workdir.rglob("*.nat"))
        if not nat:
            raise SystemExit(f"No .nat file inside {path}")
        return nat[0]
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="a .nat file, a .zip, or a folder holding one")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        nat = find_native(args.path, Path(tmp))
        print(f"File: {nat.name}")
        print(f"Size: {nat.stat().st_size / 1e6:.0f} MB")

        start = time.perf_counter()
        scene = Scene(filenames=[str(nat)], reader="seviri_l1b_native")
        print(f"Start time: {scene.start_time}")
        print(f"Channels: {sorted(scene.available_dataset_names())}")

        wanted = ["IR_108", "WV_062"]
        scene.load(wanted)
        for name in wanted:
            data = scene[name]
            values = np.asarray(data.values)
            valid = np.isfinite(values)
            print(
                f"{name}: shape {values.shape}, units {data.attrs.get('units')}, "
                f"valid {valid.mean():.1%}, min {np.nanmin(values):.1f}, "
                f"max {np.nanmax(values):.1f}"
            )
        print(f"Area: {scene['IR_108'].attrs['area']}")
        print(f"Read time: {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
