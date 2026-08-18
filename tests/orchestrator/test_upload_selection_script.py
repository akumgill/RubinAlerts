"""Unit tests for scripts/upload_selection_night.py's light-curve compaction
and attachment (no network: the fetcher is injected)."""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SCRIPT = os.path.join(_REPO, "scripts", "upload_selection_night.py")

spec = importlib.util.spec_from_file_location("upload_selection_night", _SCRIPT)
upl = importlib.util.module_from_spec(spec)
# register BEFORE exec: the script's @dataclass resolves its module via
# sys.modules at class-creation time
sys.modules[spec.name] = upl
spec.loader.exec_module(upl)


def _lc_df(n, mjd0=61200.0):
    return pd.DataFrame({
        "mjd": mjd0 + np.arange(n) * 0.5,
        "magnitude": 20.0 - 0.001 * np.arange(n),
        "mag_err": np.full(n, 0.0512),
        "band": ["g", "r"] * (n // 2) if n % 2 == 0 else ["g"] * n,
    })


def test_compact_lc_rounding_and_shape():
    df = _lc_df(4)
    df.loc[1, "mag_err"] = np.nan          # non-finite err -> null
    lc = upl.compact_lc(df)
    assert len(lc) == 4
    mjd, mag, err, band = lc[0]
    assert mjd == 61200.0 and mag == 20.0 and err == 0.051 and band == "g"
    assert lc[1][2] is None
    # mjd-sorted, 4dp / 3dp rounding
    assert lc == sorted(lc, key=lambda p: p[0])
    assert all(round(p[0], 4) == p[0] and round(p[1], 3) == p[1] for p in lc)


def test_compact_lc_caps_at_most_recent_150():
    lc = upl.compact_lc(_lc_df(200))
    assert len(lc) == upl.LC_MAX_POINTS == 150
    # the most RECENT points are kept (the first 50 epochs dropped)
    assert lc[0][0] == pytest.approx(61200.0 + 50 * 0.5)


def test_compact_lc_empty_and_none():
    assert upl.compact_lc(None) is None
    assert upl.compact_lc(pd.DataFrame()) is None
    # all-NaN magnitudes -> None, not an empty list
    bad = pd.DataFrame({"mjd": [61200.0], "magnitude": [np.nan],
                        "mag_err": [0.1], "band": ["g"]})
    assert upl.compact_lc(bad) is None


def test_attach_light_curves_skips_and_survives_failures():
    calls = []

    def fetcher(oid):
        calls.append(oid)
        if oid == "ZTFboom":
            raise RuntimeError("portal down")
        if oid == "ZTFempty":
            return pd.DataFrame()
        return _lc_df(2)

    cands = [
        {"ztf_oid": "ZTFok"},
        {"ztf_oid": None},          # no oid -> never fetched, lc omitted
        {"ztf_oid": "ZTFboom"},     # fetch failure -> lc omitted, no raise
        {"ztf_oid": "ZTFempty"},    # no detections -> lc omitted
    ]
    upl.attach_light_curves(cands, fetcher=fetcher)
    assert calls == ["ZTFok", "ZTFboom", "ZTFempty"]
    assert len(cands[0]["lc"]) == 2
    assert all("lc" not in c for c in cands[1:])
