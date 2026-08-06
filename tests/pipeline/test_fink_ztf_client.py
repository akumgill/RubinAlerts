"""Unit tests for FinkZTFClient schema translation + freshness cuts.

No network: query_sn_candidates is monkeypatched with canned Fink-ZTF frames.
The point is to lock the ZTF-specific translation the wide-mode funnel depends
on (jd→mjd, fid→band, magpsf-is-magnitude) and the wide-sky selection cuts.
"""
import pandas as pd

from broker_clients.fink_ztf_client import FinkZTFClient, _JD_TO_MJD


def test_normalize_columns_translates_ztf_schema():
    raw = pd.DataFrame([{
        "i:objectId": "ZTF26abcde",
        "i:jd": 2461262.5,          # -> mjd 61262.0
        "i:magpsf": 19.2,
        "i:sigmapsf": 0.08,
        "i:fid": 2,                 # -> r band
        "i:ra": 331.9, "i:dec": -7.5,
        "d:snn_snia_vs_nonia": 0.88,
    }])
    out = FinkZTFClient._normalize_columns(raw)
    assert out.loc[0, "objectId"] == "ZTF26abcde"
    assert out.loc[0, "mjd"] == 2461262.5 - _JD_TO_MJD == 61262.0
    assert out.loc[0, "band"] == "r"              # fid 2 -> r
    assert out.loc[0, "magnitude"] == 19.2        # magpsf is already a magnitude
    assert out.loc[0, "snn_snia_vs_nonia"] == 0.88


def test_fid_band_mapping_all_three():
    raw = pd.DataFrame({"i:fid": [1, 2, 3], "i:jd": [2461262.5] * 3})
    out = FinkZTFClient._normalize_columns(raw)
    assert list(out["band"]) == ["g", "r", "i"]


def test_fetch_fresh_sn_applies_wide_cuts(monkeypatch):
    mjd_now = 61262.0
    # last-detection jd for each object (jd = mjd + 2400000.5)
    def jd(mjd):
        return mjd + _JD_TO_MJD
    canned = pd.DataFrame([
        # fresh, observable, bright, high Ia, TNS-classified -> KEEP
        {"i:objectId": "keep1", "i:jd": jd(61260), "i:magpsf": 19.0,
         "i:fid": 2, "i:ra": 100.0, "i:dec": -10.0, "d:snn_snia_vs_nonia": 0.9,
         "d:tns": "SN Ia"},
        # too far north (dec > +22) -> DROP
        {"i:objectId": "north", "i:jd": jd(61261), "i:magpsf": 19.0,
         "i:fid": 2, "i:ra": 100.0, "i:dec": 40.0, "d:snn_snia_vs_nonia": 0.9},
        # too faint (mag > 21.5) -> DROP
        {"i:objectId": "faint", "i:jd": jd(61261), "i:magpsf": 22.5,
         "i:fid": 2, "i:ra": 100.0, "i:dec": -10.0, "d:snn_snia_vs_nonia": 0.9},
        # stale (last detection 40 d ago, days_back=15) -> DROP
        {"i:objectId": "stale", "i:jd": jd(61222), "i:magpsf": 19.0,
         "i:fid": 2, "i:ra": 100.0, "i:dec": -10.0, "d:snn_snia_vs_nonia": 0.9},
    ])

    client = FinkZTFClient()
    monkeypatch.setattr(client, "query_sn_candidates",
                        lambda finkclass, n=1000: canned.copy())

    out = client.fetch_fresh_sn_candidates(
        mjd_now=mjd_now, days_back=15, max_mag=21.5, dec_max=22.0,
        min_ia_score=0.5, classes=("SN candidate",))
    assert list(out["objectId"]) == ["keep1"]
    assert out.loc[0, "band"] == "r"
    assert out.loc[0, "ia_score"] == 0.9
    # prior-spectroscopy signal from Fink's d:tns
    assert out.loc[0, "tns_type"] == "SN Ia"
    assert bool(out.loc[0, "tns_classified"]) is True
