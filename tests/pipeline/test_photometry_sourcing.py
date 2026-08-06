"""Photometry sourcing: live Fink-ZTF light curves + the blackout guard.

Covers the downtime path — when alerts come from ZTF, light curves must come
from ZTF (Fink-ZTF /objects) — and the proactive guard that turns "0 light
curves from any source" into a detectable failure instead of a silent INFO log
found only by manual inspection.
"""
import logging

import numpy as np
import pandas as pd

import run_tonight as rt


class _FakeZTFClient:
    """Stands in for FinkZTFClient.get_light_curve with canned magnitudes."""
    def __init__(self, lcs):
        self._lcs = lcs  # objectId -> DataFrame(mjd, band, magnitude, mag_err) or None

    def get_light_curve(self, oid, include_forced=False):
        return self._lcs.get(str(oid))


def _lc(mjd, band, mag, mag_err=0.05):
    return pd.DataFrame({"mjd": mjd, "band": band, "magnitude": mag,
                         "mag_err": [mag_err] * len(mjd)})


def test_finkztf_photometry_batch_converts_mag_to_njy():
    client = _FakeZTFClient({
        "ZTF26aaa": _lc([61255.0, 61256.0], ["r", "g"], [19.0, 19.5]),
        "ZTF26bbb": None,                       # queried, no data -> omitted
        "ZTF26ccc": _lc([], [], []),            # empty -> omitted
    })
    oid_lookup = {"obj1": "ZTF26aaa", "obj2": "ZTF26bbb", "obj3": "ZTF26ccc"}
    out = rt.fetch_finkztf_photometry_batch(client, oid_lookup)

    assert set(out.keys()) == {"obj1"}          # only the one with data
    df = out["obj1"]
    # nJy schema present and flux matches the pipeline's AB→nJy convention
    assert set(["mjd", "flux", "flux_err", "magnitude", "mag_err",
                "band", "survey", "source"]).issubset(df.columns)
    expect_flux0 = 10 ** ((rt.AB_ZP_NJY - 19.0) / 2.5)
    assert df.iloc[0]["flux"] == expect_flux0
    assert (df["survey"] == "ZTF").all()
    assert list(df["band"]) == ["r", "g"]


def test_finkztf_photometry_batch_empty_inputs():
    assert rt.fetch_finkztf_photometry_batch(None, {"a": "ZTF1"}) == {}
    assert rt.fetch_finkztf_photometry_batch(_FakeZTFClient({}), {}) == {}


def test_photometry_coverage_counts_any_source():
    dia = ["a", "b", "c"]
    # covered by different sources
    n, tot = rt.photometry_coverage(dia, {"a": 1}, {"b": 1}, {})
    assert (n, tot) == (2, 3)
    # string/nonstring keys unify
    n, tot = rt.photometry_coverage([1, 2], {"1": 1}, {2: 1})
    assert (n, tot) == (2, 2)


def test_photometry_blackout_is_zero_coverage():
    # every source empty with candidates present -> the blackout condition
    n, tot = rt.photometry_coverage(["a", "b"], {}, {}, {})
    assert tot == 2 and n == 0            # this is what the guard flags


class _RecordingFink:
    """Fake Fink-LSST client that records which diaObjectIds it was asked for."""
    def __init__(self):
        self.fetched = []

    def get_light_curve(self, did, include_forced=True):
        self.fetched.append(str(did))
        return pd.DataFrame()          # queried OK, no photometry


def test_fink_lsst_skipped_for_ztf_only_candidates():
    # Optimization: a ZTF-only object (coord-based id, no rubin_dia_object_id)
    # must NOT trigger a Fink-LSST fetch — only the Rubin-identified one does.
    fink = _RecordingFink()
    cands = pd.DataFrame([
        {"diaObjectId": "123456", "ra": 10.0, "dec": -5.0,
         "rubin_dia_object_id": "123456", "ztf_oid": None},        # Rubin
        {"diaObjectId": "329.79_1.36", "ra": 329.79, "dec": 1.36,
         "rubin_dia_object_id": None, "ztf_oid": "ZTF26aaa"},      # ZTF-only
    ])
    rt.fetch_and_fit(fink, cands, mjd_now=61265,
                     fetch_ztf=False, fetch_atlas=False)
    assert fink.fetched == ["123456"]      # ZTF-only object was skipped


def test_fetch_and_fit_logs_blackout(caplog):
    # Candidates exist but no photometry sources return anything -> the pipeline
    # must emit a loud, greppable error, not swallow it.
    cands = pd.DataFrame([
        {"diaObjectId": "obj1", "ra": 100.0, "dec": -10.0},
        {"diaObjectId": "obj2", "ra": 101.0, "dec": -11.0},
    ])
    with caplog.at_level(logging.ERROR, logger="run_tonight"):
        # fink=None and fetch_ztf/atlas off -> no source yields photometry
        rt.fetch_and_fit(None, cands, mjd_now=61265,
                         fetch_ztf=False, fetch_atlas=False)
    assert any("PHOTOMETRY BLACKOUT" in r.message for r in caplog.records)
