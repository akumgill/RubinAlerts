"""Tests for consolidated broker-liveness status (R5 / W3).

A silent broker failure (e.g. ALeRCE/ANTARES down) must be distinguishable
from a genuinely empty sky. These tests exercise SupernovaMonitor.query_all_brokers
with the per-broker query callables monkeypatched so that:
  - one broker RAISES        -> queried=True, responded=False, error set
  - one returns EMPTY frame  -> queried=True, responded=True, n_returned=0
  - one returns N>0 rows     -> queried=True, responded=True, n_returned=N

NO live broker/DB/API calls — every client.query_alerts is replaced in-memory.
"""

import pandas as pd
import pytest

from supernova_monitor import SupernovaMonitor


class _FakeClient:
    """Minimal stand-in for a broker client with a query_alerts() method."""

    def __init__(self, behavior):
        self._behavior = behavior

    def query_alerts(self, **kwargs):
        return self._behavior()


def _make_monitor_with_fake_brokers():
    """Build a SupernovaMonitor without running its real __init__ network setup."""
    monitor = SupernovaMonitor.__new__(SupernovaMonitor)
    monitor._last_broker_status = {}

    def _raises():
        raise RuntimeError("broker down")

    def _empty():
        return pd.DataFrame()

    def _n_rows():
        return pd.DataFrame({'ra': [150.0, 150.1], 'dec': [2.0, 2.1]})

    monitor.brokers = {
        'ANTARES': _FakeClient(_raises),
        'ALeRCE-LSST': _FakeClient(_empty),
        'Fink': _FakeClient(_n_rows),
    }
    return monitor


def test_broker_status_distinguishes_failure_empty_and_results():
    monitor = _make_monitor_with_fake_brokers()

    results = monitor.query_all_brokers(min_probability=0.3, days_back=30, limit=50)
    status = monitor._last_broker_status

    # Broker that raised: queried, did NOT respond, error recorded.
    assert status['ANTARES']['queried'] is True
    assert status['ANTARES']['responded'] is False
    assert status['ANTARES']['n_returned'] == 0
    assert 'broker down' in status['ANTARES']['error']

    # Broker that returned an empty frame: responded, but zero rows.
    assert status['ALeRCE-LSST']['queried'] is True
    assert status['ALeRCE-LSST']['responded'] is True
    assert status['ALeRCE-LSST']['n_returned'] == 0
    assert status['ALeRCE-LSST']['error'] is None

    # Broker that returned N>0 rows.
    assert status['Fink']['queried'] is True
    assert status['Fink']['responded'] is True
    assert status['Fink']['n_returned'] == 2
    assert status['Fink']['error'] is None

    # A raising broker must never crash the run, and only non-empty brokers
    # appear in the returned results dict.
    assert 'ANTARES' not in results
    assert 'ALeRCE-LSST' not in results
    assert 'Fink' in results


def test_failure_is_not_an_empty_sky():
    """An empty results dict from a down broker is flagged as unresponsive."""
    monitor = _make_monitor_with_fake_brokers()
    monitor.query_all_brokers(min_probability=0.3, days_back=30, limit=50)
    status = monitor._last_broker_status

    down = [b for b, s in status.items()
            if s['queried'] and not s['responded']]
    assert down == ['ANTARES']
