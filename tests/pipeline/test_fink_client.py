"""Offline tests for FinkLSSTClient retry/backoff and None-vs-empty contract.

All HTTP is mocked by monkeypatching ``requests.post`` — NO network is used.
``time.sleep`` is patched out so retries/backoff don't actually wait.
"""

import pandas as pd
import pytest
import requests

from broker_clients import fink_client
from broker_clients.fink_client import FinkLSSTClient


class FakeResponse:
    """Minimal stand-in for a requests.Response."""

    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json_data = json_data if json_data is not None else []

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(
                f"HTTP {self.status_code}")


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Never actually sleep during retry/backoff in tests."""
    monkeypatch.setattr(fink_client.time, "sleep", lambda *_a, **_k: None)


@pytest.fixture
def client():
    return FinkLSSTClient(max_retries=3, retry_backoff_base=0.0)


def _make_counting_post(behaviors):
    """Return a fake requests.post that follows ``behaviors`` per call.

    Each behavior is either a callable raising an exception, or a
    FakeResponse to return. Tracks the number of calls.
    """
    state = {"calls": 0}

    def fake_post(*_args, **_kwargs):
        i = state["calls"]
        state["calls"] += 1
        behavior = behaviors[min(i, len(behaviors) - 1)]
        if isinstance(behavior, Exception):
            raise behavior
        return behavior

    return fake_post, state


def test_post_retries_on_timeout_then_succeeds(client, monkeypatch):
    """Timeout twice, then a 200 with data on the third call -> data returned,
    and post was called exactly 3 times."""
    behaviors = [
        requests.exceptions.Timeout("t1"),
        requests.exceptions.Timeout("t2"),
        FakeResponse(200, [{"r:diaObjectId": "X", "r:band": "g"}]),
    ]
    fake_post, state = _make_counting_post(behaviors)
    monkeypatch.setattr(fink_client.requests, "post", fake_post)

    result = client._post("/api/v1/sources", {"diaObjectId": "X"})

    assert state["calls"] == 3
    assert isinstance(result, list)
    assert len(result) == 1


def test_post_does_not_retry_on_4xx(client, monkeypatch):
    """A 4xx client error is not retried: single call, returns None."""
    fake_post, state = _make_counting_post([FakeResponse(404, {"error": "nope"})])
    monkeypatch.setattr(fink_client.requests, "post", fake_post)

    result = client._post("/api/v1/sources", {"diaObjectId": "X"})

    assert state["calls"] == 1
    assert result is None


def test_post_empty_list_is_not_none(client, monkeypatch):
    """A 200 with an empty-list body returns [] (OK, zero rows), not None."""
    fake_post, _ = _make_counting_post([FakeResponse(200, [])])
    monkeypatch.setattr(fink_client.requests, "post", fake_post)

    result = client._post("/api/v1/sources", {"diaObjectId": "X"})

    assert result == []
    assert result is not None


def test_post_5xx_returns_none_after_exhausting_retries(client, monkeypatch):
    """Persistent 5xx -> None after exhausting all retries (3 calls)."""
    fake_post, state = _make_counting_post([FakeResponse(503, {"err": "down"})])
    monkeypatch.setattr(fink_client.requests, "post", fake_post)

    result = client._post("/api/v1/fp", {"diaObjectId": "X"})

    assert result is None
    assert state["calls"] == 3


def test_query_sources_empty_ok_returns_empty_df(client, monkeypatch):
    """An empty-OK _post -> empty DataFrame (not None)."""
    monkeypatch.setattr(client, "_post", lambda *_a, **_k: [])
    out = client.query_sources("X")
    assert isinstance(out, pd.DataFrame)
    assert len(out) == 0


def test_query_sources_error_returns_none(client, monkeypatch):
    """A None (transport error) _post -> None."""
    monkeypatch.setattr(client, "_post", lambda *_a, **_k: None)
    assert client.query_sources("X") is None


def test_get_light_curve_empty_ok_returns_empty_df(client, monkeypatch):
    """When both sub-queries return empty-OK, the light curve is an empty
    DataFrame (object queried OK, no photometry) — NOT None."""
    monkeypatch.setattr(client, "query_sources",
                        lambda *_a, **_k: pd.DataFrame())
    monkeypatch.setattr(client, "query_forced_photometry",
                        lambda *_a, **_k: pd.DataFrame())

    out = client.get_light_curve("X", include_forced=True)

    assert isinstance(out, pd.DataFrame)
    assert len(out) == 0


def test_get_light_curve_transport_error_returns_none(client, monkeypatch):
    """When a sub-query hits a transport error (None) and nothing else was
    retrieved, get_light_curve returns None."""
    monkeypatch.setattr(client, "query_sources", lambda *_a, **_k: None)
    monkeypatch.setattr(client, "query_forced_photometry",
                        lambda *_a, **_k: None)

    assert client.get_light_curve("X", include_forced=True) is None


def test_get_light_curve_with_data(client, monkeypatch):
    """When there IS data, get_light_curve returns a combined DataFrame."""
    df = pd.DataFrame({
        "mjd": [61000.0, 61001.0],
        "band": ["g", "r"],
        "flux": [1000.0, 2000.0],
        "flux_err": [10.0, 20.0],
    })
    monkeypatch.setattr(client, "query_sources", lambda *_a, **_k: df.copy())
    monkeypatch.setattr(client, "query_forced_photometry",
                        lambda *_a, **_k: pd.DataFrame())

    out = client.get_light_curve("X", include_forced=True)

    assert isinstance(out, pd.DataFrame)
    assert len(out) == 2
    assert "magnitude" in out.columns
