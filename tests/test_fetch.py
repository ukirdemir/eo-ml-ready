"""Tests for the eo-fetch adapter: channel glob, listing and download (no network)."""

import threading
from datetime import UTC, datetime
from fnmatch import fnmatch
from types import SimpleNamespace

import pytest

from eo_ml_ready.fetch import eofetch
from eo_ml_ready.fetch.base import SceneRef

KEY = (
    "noaa-goes19/ABI-L2-CMIPC/2026/258/18/"
    "OR_ABI-L2-CMIPC-M6C13_G19_s20262581801174_e20262581803547_c20262581804030.nc"
)
START = datetime(2026, 9, 15, 18, 0, tzinfo=UTC)
END = datetime(2026, 9, 15, 19, 0, tzinfo=UTC)


def test_goes_pattern_selects_one_channel_of_the_cmip_product():
    name = KEY.rsplit("/", 1)[-1]
    assert fnmatch(name, eofetch.goes_pattern("C13"))
    assert not fnmatch(name, eofetch.goes_pattern("C08"))
    assert not fnmatch(name.replace("CMIPC", "MCMIPC"), eofetch.goes_pattern("C13"))


def test_goes_pattern_accepts_other_scan_modes():
    assert fnmatch("OR_ABI-L2-CMIPC-M3C08_G19_s2026.nc", eofetch.goes_pattern("C08"))


def fake_search(items, backend):
    """Return a stand-in for eo_fetch.search that records its arguments."""
    calls = []

    class Found(list):
        pass

    def search(provider, product, **options):
        calls.append((provider, product, options))
        found = Found(items)
        found.backend = backend
        return found

    return search, calls


def test_list_scenes_asks_eo_fetch_and_returns_sorted_refs(monkeypatch):
    eof = pytest.importorskip("eo_fetch")
    later = SimpleNamespace(key="b", time=datetime(2026, 9, 15, 18, 15, tzinfo=UTC))
    earlier = SimpleNamespace(key="a", time=datetime(2026, 9, 15, 18, 0, tzinfo=UTC))
    undated = SimpleNamespace(key="c", time=None)
    search, calls = fake_search([later, undated, earlier], SimpleNamespace(max_connections=2))
    monkeypatch.setattr(eof, "search", search)

    provider = eofetch.goes(19)
    refs = provider.list_scenes(START, END, "C13")

    assert [ref.key for ref in refs] == ["a", "b"]
    assert all(isinstance(ref, SceneRef) and ref.channel == "C13" for ref in refs)
    name, product, options = calls[0]
    assert (name, product) == ("noaa", "goes19-cmip-conus")
    assert options["text"] == eofetch.goes_pattern("C13")
    assert options["start"] == START and options["end"] == END


def test_seviri_search_has_no_channel_filter(monkeypatch):
    eof = pytest.importorskip("eo_fetch")
    search, calls = fake_search([], SimpleNamespace(max_connections=2))
    monkeypatch.setattr(eof, "search", search)

    eofetch.seviri().list_scenes(START, END)

    assert calls[0][:2] == ("eumetsat", "seviri")
    assert calls[0][2]["text"] == ""


def test_download_returns_the_path_and_raises_on_failure(monkeypatch, tmp_path):
    pytest.importorskip("eo_fetch")
    from eo_fetch import downloader

    item = SimpleNamespace(key="a", time=START)
    provider = eofetch.EoFetchProvider("noaa", "goes19-cmip-conus", eofetch.goes_pattern)
    provider._items = {"a": item}
    provider._backend = object()
    provider._gate = threading.BoundedSemaphore(1)
    ref = SceneRef(provider="noaa", key="a", time=START, channel="C13")

    ok = SimpleNamespace(path=tmp_path / "a.nc", error="")
    monkeypatch.setattr(downloader, "download_items", lambda *args, **kwargs: [ok])
    assert provider.download(ref, tmp_path) == tmp_path / "a.nc"

    bad = SimpleNamespace(path=None, error="OSError: size does not match")
    monkeypatch.setattr(downloader, "download_items", lambda *args, **kwargs: [bad])
    with pytest.raises(RuntimeError, match="size does not match"):
        provider.download(ref, tmp_path)
