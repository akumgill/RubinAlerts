"""Tests for the multi-classifier fresh-SN union (BHRF + legacy)."""
import numpy as np
import pandas as pd
import pytest

from broker_clients.alerce_db_client import (AlerceDBClient,
                                             DEFAULT_FRESH_CLASSIFIERS)


def _pool(oids, clf_prob, class_name='SNIa'):
    """Fake query_fresh_sn_candidates result: one row per (oid, fid)."""
    rows = []
    for oid in oids:
        for fid in (1, 2):
            rows.append({'oid': oid, 'meanra': 10.0, 'meandec': -5.0,
                         'firstmjd': 61200.0, 'deltajd': 10.0,
                         'class_name': class_name, 'probability': clf_prob,
                         'fid': fid, 'magmin': 19.0, 'maglast': 19.5})
    return pd.DataFrame(rows)


@pytest.fixture
def client(monkeypatch):
    c = AlerceDBClient.__new__(AlerceDBClient)  # no DB connection
    pools = {
        'lc_classifier_BHRF_forced_phot': _pool(['A', 'B', 'C'], 0.4),
        'lc_classifier': _pool(['B', 'C', 'D'], 0.7),
    }
    calls = []

    def fake_query(mjd_now, classifier='lc_classifier', **kw):
        calls.append(classifier)
        return pools.get(classifier, pd.DataFrame()).copy()

    monkeypatch.setattr(c, 'query_fresh_sn_candidates', fake_query)
    c._calls = calls
    return c


def test_default_priority_is_bhrf_first():
    assert DEFAULT_FRESH_CLASSIFIERS[0] == 'lc_classifier_BHRF_forced_phot'
    assert 'lc_classifier' in DEFAULT_FRESH_CLASSIFIERS
    # ATAT deliberately excluded until calibration is checked
    assert not any('ATAT' in c for c in DEFAULT_FRESH_CLASSIFIERS)


def test_union_dedupes_by_oid_keeping_priority(client):
    df = client.query_fresh_sn_candidates_multi(61235.0)
    assert set(df['oid']) == {'A', 'B', 'C', 'D'}
    # B and C are in both pools: the BHRF (priority) rows must win
    by = df.drop_duplicates('oid').set_index('oid')['alerce_classifier']
    assert by['A'] == 'lc_classifier_BHRF_forced_phot'
    assert by['B'] == 'lc_classifier_BHRF_forced_phot'
    assert by['C'] == 'lc_classifier_BHRF_forced_phot'
    assert by['D'] == 'lc_classifier'
    # no cross-classifier probability mixing: B keeps BHRF's 0.4, not 0.7
    assert (df.loc[df['oid'] == 'B', 'probability'] == 0.4).all()


def test_union_keeps_all_band_rows(client):
    df = client.query_fresh_sn_candidates_multi(61235.0)
    # 4 objects x 2 fid rows each
    assert len(df) == 8
    assert df.groupby('oid').size().eq(2).all()


def test_union_queries_in_order(client):
    client.query_fresh_sn_candidates_multi(61235.0)
    assert client._calls == list(DEFAULT_FRESH_CLASSIFIERS)


def test_union_custom_classifiers(client):
    df = client.query_fresh_sn_candidates_multi(
        61235.0, classifiers=('lc_classifier',))
    assert set(df['oid']) == {'B', 'C', 'D'}
    assert (df['alerce_classifier'] == 'lc_classifier').all()


def test_union_all_empty(client, monkeypatch):
    monkeypatch.setattr(client, 'query_fresh_sn_candidates',
                        lambda *a, **k: pd.DataFrame())
    df = client.query_fresh_sn_candidates_multi(61235.0)
    assert len(df) == 0


def test_kwargs_passed_through(client, monkeypatch):
    seen = {}

    def spy(mjd_now, classifier='x', **kw):
        seen.update(kw)
        return pd.DataFrame()

    monkeypatch.setattr(client, 'query_fresh_sn_candidates', spy)
    client.query_fresh_sn_candidates_multi(61235.0, min_prob=0.4,
                                           dec_limit=15.0)
    assert seen == {'min_prob': 0.4, 'dec_limit': 15.0}
