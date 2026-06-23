"""Tests for the ALeRCE-LSST bare-query + local-filter path.

The ALeRCE LSST object endpoint returns HTTP 500 for ANY server-side filter
(classifier=, class_name=, probability=). The client must therefore page
through bare query_objects(survey='lsst', ...) and filter LOCALLY for
class_name == 'SN', probability >= threshold, and DDF coordinates.

These tests are fully offline: the AlerceClient's `self.alerce.query_objects`
is monkeypatched to return synthetic DataFrames (or raise), and the on-disk
cache is neutralised so it cannot interfere. NO network/DB/API.
"""

import pandas as pd
import pytest

from broker_clients.alerce_client import AlerceClient, LSST_CLASS_NAME


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

# COSMOS DDF center is RA 150.11 / Dec +2.23 (radius 1.75 deg).
def _synthetic_objects():
    """Mixed bag of LSST objects covering every filter branch.

    Row meanings:
      0  SN, high prob, inside COSMOS         -> KEEP
      1  SN, high prob, far outside any DDF   -> drop (coords)
      2  SN, low prob, inside COSMOS          -> drop (probability)
      3  asteroid, high prob, inside COSMOS   -> drop (class)
    """
    return pd.DataFrame([
        {'oid': 'OID_SN_IN', 'meanra': 150.10, 'meandec': 2.20,
         'firstmjd': 61000.0, 'lastmjd': 61010.0, 'n_det': 12, 'n_forced': 3,
         'class_name': 'SN', 'classifier_name': 'lc_classifier_lsst',
         'probability': 0.95, 'stellar': False},
        {'oid': 'OID_SN_FAR', 'meanra': 10.0, 'meandec': -80.0,
         'firstmjd': 61000.0, 'lastmjd': 61010.0, 'n_det': 8, 'n_forced': 0,
         'class_name': 'SN', 'classifier_name': 'lc_classifier_lsst',
         'probability': 0.90, 'stellar': False},
        {'oid': 'OID_SN_LOWP', 'meanra': 150.12, 'meandec': 2.25,
         'firstmjd': 61000.0, 'lastmjd': 61010.0, 'n_det': 5, 'n_forced': 1,
         'class_name': 'SN', 'classifier_name': 'lc_classifier_lsst',
         'probability': 0.10, 'stellar': False},
        {'oid': 'OID_AST_IN', 'meanra': 150.11, 'meandec': 2.23,
         'firstmjd': 61000.0, 'lastmjd': 61010.0, 'n_det': 20, 'n_forced': 0,
         'class_name': 'asteroid', 'classifier_name': 'lc_classifier_lsst',
         'probability': 0.99, 'stellar': False},
    ])


def _make_lsst_client(tmp_path):
    """Construct an AlerceClient(survey='lsst') with no real ALeRCE library
    and a tmp cache dir, bypassing network init."""
    client = AlerceClient.__new__(AlerceClient)
    client.broker_name = 'ALeRCE-LSST'
    client.cache_dir = str(tmp_path)
    client.survey = 'lsst'
    client.use_db = False
    client.db_client = None

    class _FakeAlerce:
        pass

    client.alerce = _FakeAlerce()
    # Neutralise the on-disk cache so it never short-circuits the query.
    client._load_cache = lambda cache_path: None
    client._save_cache = lambda cache_path, data: None
    return client


class _APIError(Exception):
    """Stand-in for an alerce/requests API error (e.g. HTTP 500)."""


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------

def test_local_filter_keeps_only_in_ddf_sn_above_threshold(tmp_path):
    client = _make_lsst_client(tmp_path)

    calls = {}

    def fake_query_objects(**kwargs):
        calls.update(kwargs)
        # Page 1 returns the synthetic objects, page 2+ is empty.
        if kwargs.get('page', 1) == 1:
            return _synthetic_objects()
        return pd.DataFrame()

    client.alerce.query_objects = fake_query_objects

    df = client._query_alerts_lsst(min_probability=0.5, days_back=30, limit=200)

    # Exactly one survivor: the in-DDF SN above threshold.
    assert len(df) == 1
    row = df.iloc[0]
    assert row['object_id'] == 'OID_SN_IN'
    assert row['ddf_field'] == 'COSMOS'
    # sn_ia_prob populated from the row's `probability` column.
    assert row['sn_ia_prob'] == pytest.approx(0.95)
    # classifier read from classifier_name, not a removed constant.
    assert row['classifier'] == 'lc_classifier_lsst'
    assert row['class'] == LSST_CLASS_NAME

    # The query was BARE — no server-side filters were passed.
    assert calls.get('survey') == 'lsst'
    assert 'classifier' not in calls
    assert 'class_name' not in calls
    assert 'probability' not in calls


def test_api_error_propagates_instead_of_silent_empty(tmp_path):
    client = _make_lsst_client(tmp_path)

    def fake_query_objects(**kwargs):
        raise _APIError("HTTP 500 from ALeRCE LSST endpoint")

    client.alerce.query_objects = fake_query_objects

    # A genuine API error must surface, not be masked as an empty result.
    with pytest.raises(_APIError):
        client._query_alerts_lsst(min_probability=0.5, days_back=30, limit=200)


def test_empty_response_returns_empty_without_raising(tmp_path):
    client = _make_lsst_client(tmp_path)

    def fake_query_objects(**kwargs):
        return pd.DataFrame()

    client.alerce.query_objects = fake_query_objects

    df = client._query_alerts_lsst(min_probability=0.5, days_back=30, limit=200)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_lsst_error_surfaces_in_broker_status(tmp_path):
    """A hard ALeRCE-LSST failure shows up in broker_status as an error,
    not as a quiet n_returned=0 (the bug this fix addresses)."""
    from supernova_monitor import SupernovaMonitor

    client = _make_lsst_client(tmp_path)

    def fake_query_objects(**kwargs):
        raise _APIError("HTTP 500 from ALeRCE LSST endpoint")

    client.alerce.query_objects = fake_query_objects

    monitor = SupernovaMonitor.__new__(SupernovaMonitor)
    monitor._last_broker_status = {}
    monitor.brokers = {'ALeRCE-LSST': client}

    results = monitor.query_all_brokers(min_probability=0.5, days_back=30,
                                        limit=200)
    status = monitor._last_broker_status['ALeRCE-LSST']

    assert status['responded'] is False
    assert status['error'] is not None
    assert 'HTTP 500' in status['error']
    assert 'ALeRCE-LSST' not in results
