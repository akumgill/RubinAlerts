"""Unit tests for FinkZTFClient schema translation + freshness cuts.

No network: query_sn_candidates is monkeypatched with canned Fink-ZTF frames.
The point is to lock the ZTF-specific translation the wide-mode funnel depends
on (jd→mjd, fid→band, magpsf-is-magnitude) and the wide-sky selection cuts.
"""
import os

import pandas as pd
import pytest

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


# ---------------------------------------------------------------------------
# cone_search on the ZTF portal (the bug this file was extended to cover):
# the inherited Rubin cone_search sends `n` and no startdate, which returns
# ZERO rows on the ZTF /api/v1/conesearch even at an object's exact position.
# The override drops `n` from the payload and falls back to a date-windowed
# query when the plain one is empty.
# ---------------------------------------------------------------------------

def _row(oid, sep_deg):
    # Minimal ZTF conesearch row (the endpoint returns only these columns).
    return {"i:objectId": oid, "i:ra": 1.0, "i:dec": -0.2, "i:jd": 2461262.5,
            "d:classification": "SN candidate", "v:separation_degree": sep_deg}


def test_cone_search_payload_has_no_n_and_radius_in_arcsec():
    """`n` must NOT go in the POST body (it zeroes out the ZTF conesearch);
    radius is passed straight through as arcseconds."""
    client = FinkZTFClient()
    seen = {}

    def fake_post(endpoint, payload):
        seen["endpoint"] = endpoint
        seen["payload"] = payload
        return [_row("ZTF24abkllyo", 1.7e-5)]

    client._post = fake_post
    out = client.cone_search(1.0888, -0.2208, radius_arcsec=5.0, n=5)

    assert seen["endpoint"] == "/api/v1/conesearch"
    assert "n" not in seen["payload"]                 # the actual bug
    assert seen["payload"]["radius"] == 5.0           # arcseconds, verbatim
    assert seen["payload"]["ra"] == 1.0888
    assert out is not None and len(out) == 1
    assert out.loc[0, "i:objectId"] == "ZTF24abkllyo"


def test_cone_search_falls_back_to_startdate_when_plain_is_empty():
    """Plain (no-date) query returns 0 rows for fresh objects; the override
    retries with startdate+window, which finds them."""
    client = FinkZTFClient()
    calls = []

    def fake_post(endpoint, payload):
        calls.append(dict(payload))
        if "startdate" not in payload:
            return []                                  # rolling index misses it
        return [_row("ZTF26abjqico", 6.1e-5)]          # date path finds it

    client._post = fake_post
    out = client.cone_search(12.6749, 9.2214, radius_arcsec=5.0, n=5)

    assert len(calls) == 2                             # plain, then fallback
    assert "startdate" in calls[1] and "window" in calls[1]
    assert out is not None and len(out) == 1
    assert out.loc[0, "i:objectId"] == "ZTF26abjqico"


def test_cone_search_none_vs_empty_contract():
    client = FinkZTFClient()
    # transport error on the primary query -> None
    client._post = lambda e, p: None
    assert client.cone_search(1.0, -0.2, n=5) is None
    # both queries succeed with zero rows -> empty DataFrame (not None)
    client._post = lambda e, p: []
    res = client.cone_search(1.0, -0.2, n=5)
    assert res is not None and len(res) == 0


def test_cone_search_caps_to_nearest_n_clientside():
    client = FinkZTFClient()
    client._post = lambda e, p: [_row("far", 9e-4), _row("near", 1e-5),
                                 _row("mid", 5e-4)]
    out = client.cone_search(1.0, -0.2, radius_arcsec=60, n=2)
    assert list(out["i:objectId"]) == ["near", "mid"]  # sorted by sep, capped


def test_get_classifications_enriches_ia_score_from_objects():
    """conesearch omits d:snn_snia_vs_nonia; get_classifications must look it
    up per matched objectId via /api/v1/objects."""
    client = FinkZTFClient()

    def fake_post(endpoint, payload):
        if endpoint == "/api/v1/conesearch":
            return [_row("ZTF26abiavoi", 2.0e-5)]
        if endpoint == "/api/v1/objects":
            return [{"i:objectId": "ZTF26abiavoi", "d:snn_snia_vs_nonia": 0.61},
                    {"i:objectId": "ZTF26abiavoi", "d:snn_snia_vs_nonia": 0.91}]
        return []

    client._post = fake_post
    cand = pd.DataFrame([{"ra": 321.8157, "dec": 3.798}])
    out = client.get_classifications(cand, radius_arcsec=2.0)
    assert out.loc[0, "ztf_objectId"] == "ZTF26abiavoi"
    assert out.loc[0, "ztf_ia_score"] == 0.91          # best across alerts
    assert abs(out.loc[0, "ztf_sep_arcsec"] - 2.0e-5 * 3600) < 1e-9


# ---------------------------------------------------------------------------
# Live integration — guarded so it is skipped offline / in CI. Enable with
#   FINK_ZTF_LIVE=1 python -m pytest tests/pipeline/test_fink_ztf_client.py
# Verifies the real ZTF portal returns the known object at its own position.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(os.environ.get("FINK_ZTF_LIVE") != "1",
                    reason="live ZTF portal test; set FINK_ZTF_LIVE=1 to run")
@pytest.mark.parametrize("oid,ra,dec", [
    ("ZTF24abkllyo", 1.0888, -0.2208),
    ("ZTF26abiavoi", 321.8157, 3.798),
    ("ZTF26abjqico", 12.6749, 9.2214),
])
def test_cone_search_live_finds_known_object(oid, ra, dec):
    client = FinkZTFClient()
    res = client.cone_search(ra, dec, radius_arcsec=5.0, n=5)
    assert res is not None and len(res) >= 1, f"{oid} not found at its position"
    best = res.sort_values("v:separation_degree").iloc[0]
    assert best["i:objectId"] == oid
    assert best["v:separation_degree"] * 3600 < 3.0    # within a few arcsec
