"""Tests for key parsing and prefix building in the GOES adapter (no network)."""

from datetime import UTC, datetime

from eo_ml_ready.fetch.goes_aws import hour_prefixes, parse_key

KEY = (
    "noaa-goes19/ABI-L2-CMIPC/2026/258/18/"
    "OR_ABI-L2-CMIPC-M6C13_G19_s20262581801174_e20262581803547_c20262581804030.nc"
)


def test_parse_key_reads_time_and_channel():
    ref = parse_key(KEY)
    assert ref is not None
    assert ref.channel == "C13"
    assert ref.time == datetime(2026, 9, 15, 18, 1, 17, tzinfo=UTC)


def test_parse_key_rejects_other_products():
    assert parse_key("noaa-goes19/ABI-L2-MCMIPC/2026/258/18/OR_ABI-L2-MCMIPC-M6_G19.nc") is None


def test_hour_prefixes_cover_range():
    start = datetime(2026, 9, 15, 17, 30, tzinfo=UTC)
    end = datetime(2026, 9, 15, 19, 0, tzinfo=UTC)
    assert hour_prefixes("ABI-L2-CMIPC", start, end) == [
        "ABI-L2-CMIPC/2026/258/17/",
        "ABI-L2-CMIPC/2026/258/18/",
    ]
