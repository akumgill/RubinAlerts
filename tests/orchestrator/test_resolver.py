"""Tests for api.resolver — name -> coordinates for the submission API.

Network is mocked (monkeypatched _post); the live Fink endpoints are
exercised manually, not in CI.
"""
import math

import pytest

from api import resolver
from api.resolver import resolve_name


ZTF_ROWS = [
    {"i:ra": 335.61, "i:dec": -2.886, "i:magpsf": 18.5, "i:jd": 2461269.5},
    {"i:ra": 335.6124736, "i:dec": -2.8867695, "i:magpsf": 18.382893,
     "i:jd": 2461270.8366667},  # latest
]
TNS_ROW = {"d:ra": "335.61247331429", "d:declination": "-2.8867830357143",
           "d:fullname": "AT 2026ydy", "d:redshift": "NaN", "d:type": "nan"}


def test_ztf_id_resolves_latest_alert(monkeypatch):
    calls = []

    def fake_post(endpoint, payload):
        calls.append((endpoint, payload))
        return ZTF_ROWS

    monkeypatch.setattr(resolver, "_post", fake_post)
    hit = resolve_name("ZTF26abmiytv")
    assert calls[0][0] == "/api/v1/objects"
    assert calls[0][1]["objectId"] == "ZTF26abmiytv"
    assert hit["scheme"] == "fink-ztf"
    assert hit["ra"] == pytest.approx(335.6124736)   # max-jd row, not row 0
    assert hit["mag"] == pytest.approx(18.382893)


def test_ztf_id_case_normalized(monkeypatch):
    seen = {}

    def fake_post(endpoint, payload):
        seen["oid"] = payload["objectId"]
        return ZTF_ROWS

    monkeypatch.setattr(resolver, "_post", fake_post)
    assert resolve_name("ztf26ABMIYTV") is not None
    assert seen["oid"] == "ZTF26abmiytv"


@pytest.mark.parametrize("name", ["2026ydy", "AT 2026ydy", "AT2026ydy",
                                  "sn 2026ydy", "SN2026YDY"])
def test_tns_variants_resolve(monkeypatch, name):
    tried = []

    def fake_post(endpoint, payload):
        tried.append(payload["name"])
        # Fink's TNS resolver matches only the exact full name
        return [TNS_ROW] if payload["name"] == "AT 2026ydy" else []

    monkeypatch.setattr(resolver, "_post", fake_post)
    hit = resolve_name(name)
    assert hit is not None, f"{name!r} should resolve via AT-prefix fallback"
    assert hit["scheme"] == "tns"
    assert hit["ra"] == pytest.approx(335.61247331429)
    assert "redshift" not in hit  # "NaN" must not leak through
    assert "AT 2026ydy" in tried


def test_stated_prefix_tried_first(monkeypatch):
    tried = []

    def fake_post(endpoint, payload):
        tried.append(payload["name"])
        return []

    monkeypatch.setattr(resolver, "_post", fake_post)
    assert resolve_name("SN 2026xyz") is None
    assert tried[0] == "SN 2026xyz"
    assert "AT 2026xyz" in tried


def test_unrecognized_and_failures_return_none(monkeypatch):
    monkeypatch.setattr(resolver, "_post", lambda e, p: None)  # network down
    assert resolve_name("ZTF26abmiytv") is None
    assert resolve_name("2026ydy") is None
    assert resolve_name("NGC 1234") is None   # unrecognized form, no HTTP call
    assert resolve_name("") is None
    assert resolve_name(None) is None


def test_app_wires_a_resolver():
    """The production bug: TargetQueueService built with resolver=None, so
    every name-only submission failed. Guard against regression."""
    import api.app as app_module
    assert app_module.svc._resolver is not None
