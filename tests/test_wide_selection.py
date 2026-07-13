"""Tests for wide-sky payload-level candidate selection (run_tonight).

Covers select_wide_candidates() cuts (declination, freshness, variable/AGN
payload screen, brightness, redshift coalescing, fit cap) and the
aggregator-backed wide merge (phase 2) — all offline on synthetic frames.
"""

import numpy as np
import pandas as pd
import pytest

from core.alert_aggregator import AlertAggregator
from run_tonight import (
    select_wide_candidates, _coalesce_effective_prob,
    WIDE_DEC_LIMIT, WIDE_MAX_MAG, WIDE_MAX_Z, WIDE_HOSTLESS_MAX_MAG,
)

MJD_NOW = 61234.0


def make_payload(n=1, **overrides):
    """One good wide-mode candidate per row unless overridden."""
    base = {
        'diaObjectId': [f'obj{i}' for i in range(n)],
        'ra': [150.0] * n,
        'dec': [-20.0] * n,
        'sn_score': [0.9] * n,
        'early_ia_score': [0.5] * n,
        'brightest_mag': [19.5] * n,
        'last_mjd': [MJD_NOW - 3.0] * n,
        'z_tns': [np.nan] * n,
        'z_phot': [0.2] * n,
        'tns_name_xm': [''] * n,
        'tns_type_xm': [''] * n,
        'gcvs_type_xm': [''] * n,
        'vsx_type_xm': [''] * n,
        'simbad_otype_xm': [''] * n,
        'gaia_varflag_xm': [''] * n,
        'gaia_plx': [np.nan] * n,
        'gaia_plx_err': [np.nan] * n,
    }
    base.update(overrides)
    return pd.DataFrame(base)


class TestDeclinationCut:
    def test_magellan_visible_kept(self):
        df = make_payload(1, dec=[-30.0])
        assert len(select_wide_candidates(df, MJD_NOW)) == 1

    def test_too_far_north_dropped(self):
        df = make_payload(1, dec=[WIDE_DEC_LIMIT + 5.0])
        assert len(select_wide_candidates(df, MJD_NOW)) == 0

    def test_custom_dec_limit(self):
        df = make_payload(1, dec=[10.0])
        assert len(select_wide_candidates(df, MJD_NOW, dec_limit=0.0)) == 0


class TestFreshnessCut:
    def test_recent_kept(self):
        df = make_payload(1, last_mjd=[MJD_NOW - 10.0])
        assert len(select_wide_candidates(df, MJD_NOW, days_back=30)) == 1

    def test_stale_dropped(self):
        df = make_payload(1, last_mjd=[MJD_NOW - 100.0])
        assert len(select_wide_candidates(df, MJD_NOW, days_back=30)) == 0

    def test_unknown_mjd_kept(self):
        # Unknown recency must not be silently treated as stale
        df = make_payload(1, last_mjd=[np.nan])
        assert len(select_wide_candidates(df, MJD_NOW)) == 1


class TestVariableAGNScreen:
    def test_gcvs_variable_dropped(self):
        df = make_payload(1, gcvs_type_xm=['RRAB'])
        assert len(select_wide_candidates(df, MJD_NOW)) == 0

    def test_vsx_variable_dropped(self):
        df = make_payload(1, vsx_type_xm=['EW'])
        assert len(select_wide_candidates(df, MJD_NOW)) == 0

    def test_simbad_agn_dropped(self):
        for otype in ['QSO', 'Seyfert_1', 'BLLac']:
            df = make_payload(1, simbad_otype_xm=[otype])
            assert len(select_wide_candidates(df, MJD_NOW)) == 0, otype

    def test_simbad_galaxy_kept(self):
        # A plain galaxy cross-match is a HOST, not a variable source
        df = make_payload(1, simbad_otype_xm=['Galaxy'])
        assert len(select_wide_candidates(df, MJD_NOW)) == 1

    def test_gaia_variable_flag_dropped(self):
        df = make_payload(1, gaia_varflag_xm=['VARIABLE'])
        assert len(select_wide_candidates(df, MJD_NOW)) == 0

    def test_significant_parallax_dropped(self):
        df = make_payload(1, gaia_plx=[5.0], gaia_plx_err=[0.5])
        assert len(select_wide_candidates(df, MJD_NOW)) == 0

    def test_insignificant_parallax_kept(self):
        df = make_payload(1, gaia_plx=[0.5], gaia_plx_err=[0.5])
        assert len(select_wide_candidates(df, MJD_NOW)) == 1


class TestBrightnessCut:
    def test_bright_kept(self):
        df = make_payload(1, brightest_mag=[18.0])
        assert len(select_wide_candidates(df, MJD_NOW)) == 1

    def test_faint_dropped(self):
        df = make_payload(1, brightest_mag=[WIDE_MAX_MAG + 1.0])
        assert len(select_wide_candidates(df, MJD_NOW)) == 0

    def test_no_mag_dropped(self):
        # Cannot claim followable without a magnitude
        df = make_payload(1, brightest_mag=[np.nan])
        assert len(select_wide_candidates(df, MJD_NOW)) == 0


class TestRedshiftCoalescing:
    def test_tns_specz_preferred_over_photz(self):
        df = make_payload(1, z_tns=[0.1], z_phot=[0.9])
        out = select_wide_candidates(df, MJD_NOW)
        assert len(out) == 1
        assert out['z_best'].iloc[0] == pytest.approx(0.1)
        assert out['z_source'].iloc[0] == 'tns_specz'

    def test_photz_fallback(self):
        df = make_payload(1, z_tns=[np.nan], z_phot=[0.3])
        out = select_wide_candidates(df, MJD_NOW)
        assert out['z_source'].iloc[0] == 'legacy_photz'

    def test_high_z_dropped(self):
        df = make_payload(1, z_phot=[WIDE_MAX_Z + 0.3])
        assert len(select_wide_candidates(df, MJD_NOW)) == 0

    def test_no_z_kept_only_if_bright(self):
        kept = make_payload(1, z_phot=[np.nan],
                            brightest_mag=[WIDE_HOSTLESS_MAX_MAG - 0.5])
        dropped = make_payload(1, z_phot=[np.nan],
                               brightest_mag=[WIDE_HOSTLESS_MAX_MAG + 0.5])
        assert len(select_wide_candidates(kept, MJD_NOW)) == 1
        out = select_wide_candidates(kept, MJD_NOW)
        assert out['z_source'].iloc[0] == 'none'
        assert len(select_wide_candidates(dropped, MJD_NOW)) == 0

    def test_high_photz_overridden_by_low_specz(self):
        # A wrong photo-z must not kill a TNS-confirmed low-z object
        df = make_payload(1, z_tns=[0.05], z_phot=[1.5])
        assert len(select_wide_candidates(df, MJD_NOW)) == 1


class TestFitCap:
    def test_cap_keeps_best_scores(self):
        df = make_payload(10, sn_score=list(np.linspace(0.5, 0.95, 10)))
        out = select_wide_candidates(df, MJD_NOW, fit_cap=3)
        assert len(out) == 3
        assert out['sn_score'].min() >= 0.85

    def test_empty_frame_passthrough(self):
        df = make_payload(0)
        out = select_wide_candidates(df, MJD_NOW)
        assert len(out) == 0


class TestGalacticPlaneScreen:
    def test_plane_object_dropped(self):
        # RA 266, Dec -29 is the Galactic center (b ~ 0): nova/CV territory
        df = make_payload(1, ra=[266.4], dec=[-28.9])
        assert len(select_wide_candidates(df, MJD_NOW)) == 0

    def test_high_latitude_kept(self):
        df = make_payload(1, ra=[150.0], dec=[-20.0])  # b ~ +27
        assert len(select_wide_candidates(df, MJD_NOW)) == 1

    def test_configurable(self):
        df = make_payload(1, ra=[266.4], dec=[-28.9])
        assert len(select_wide_candidates(df, MJD_NOW, min_gal_b=0.0)) == 1


class TestWideAggregatorMerge:
    """The aggregator-backed wide merge (phase 2; replaces merge_wide_streams)."""

    @staticmethod
    def _agg(tmp_path):
        # Mirrors the wide branch of fetch_all_broker_candidates: 2" match
        # tolerance, no extinction lookup (network).
        return AlertAggregator(cache_dir=str(tmp_path / 'cache'),
                               match_tolerance_arcsec=2.0,
                               apply_extinction=False)

    @staticmethod
    def _fink(n=1, **ov):
        base = make_payload(n, **ov)
        base['object_id'] = base['diaObjectId']
        base['sn_ia_prob'] = base['sn_score']
        return base

    @staticmethod
    def _ztf(ra, dec, oid='ZTF26abc', cls='SNIa', prob=0.5):
        return pd.DataFrame({
            'diaObjectId': [oid], 'object_id': [oid], 'ra': [ra], 'dec': [dec],
            'sn_score': [prob],
            # wide schema: per-class probability is a Ia prob only for SNIa
            'sn_ia_prob': [prob if cls == 'SNIa' else np.nan],
            'alerce_class': [cls],
        })

    def test_coincident_pair_collapses_to_one_row(self, tmp_path):
        fink = self._fink(1, ra=[150.0], dec=[-20.0])
        ztf = self._ztf(150.0 + 0.0001, -20.0)  # ~0.34 arcsec away
        out = self._agg(tmp_path).merge_alerts({'Fink': fink,
                                                'ALeRCE-ZTF': ztf})
        assert len(out) == 1
        row = out.iloc[0]
        assert row['num_brokers'] == 2
        assert row['brokers_detected'] == 'ALeRCE-ZTF,Fink'
        assert row['ztf_oid'] == 'ZTF26abc'
        assert row['alerce_class'] == 'SNIa'
        # Fink's classifier score wins on the merged row
        assert row['sn_score'] == pytest.approx(0.9)
        # per-broker bookkeeping the old lightweight merge lacked
        assert row['object_id_Fink'] == 'obj0'
        assert row['object_id_ALeRCE-ZTF'] == 'ZTF26abc'
        # cross-survey agreement stats: both ML Ia probs pooled
        assert row['mean_ia_prob'] == pytest.approx((0.9 + 0.5) / 2)

    def test_distinct_objects_both_kept(self, tmp_path):
        fink = self._fink(1, ra=[150.0], dec=[-20.0])
        ztf = self._ztf(200.0, 10.0)
        out = self._agg(tmp_path).merge_alerts({'Fink': fink,
                                                'ALeRCE-ZTF': ztf})
        assert len(out) == 2
        assert set(out['brokers_detected']) == {'Fink', 'ALeRCE-ZTF'}

    def test_empty_streams(self, tmp_path):
        agg = self._agg(tmp_path)
        empty = pd.DataFrame()
        assert len(agg.merge_alerts({'Fink': self._fink(1),
                                     'ALeRCE-ZTF': empty})) == 1
        assert len(agg.merge_alerts({'Fink': empty,
                                     'ALeRCE-ZTF': self._ztf(1.0, 1.0)})) == 1
        assert len(agg.merge_alerts({'Fink': empty, 'ALeRCE-ZTF': empty})) == 0

    def test_ztf_only_non_ia_keeps_score_not_ia_prob(self, tmp_path):
        # A ZTF SNII: class probability survives as sn_score (ML confidence)
        # but must NOT be pooled into mean_ia_prob as if it were P(Ia).
        out = self._agg(tmp_path).merge_alerts({
            'ALeRCE-ZTF': self._ztf(150.0, -20.0, cls='SNII', prob=0.8)})
        assert len(out) == 1
        row = out.iloc[0]
        assert row['sn_score'] == pytest.approx(0.8)
        assert pd.isna(row['mean_ia_prob'])


class TestIaSpecificWeight:
    """w_iaspec: Ia-specific evidence folds into merit as [0.8, 1.2]."""

    @staticmethod
    def _merit(ia_evidence):
        from core.magellan_planning import compute_merit_breakdown
        return compute_merit_breakdown(
            delta_t=0.0, peak_mag=20.0, ia_evidence=ia_evidence)

    def test_spectroscopic_ia_boosted(self):
        b = self._merit(1.0)
        assert b['w_iaspec'] == pytest.approx(1.2)

    def test_known_non_ia_demoted(self):
        b = self._merit(0.0)
        assert b['w_iaspec'] == pytest.approx(0.8)

    def test_no_information_neutral(self):
        b = self._merit(np.nan)
        assert b['w_iaspec'] == pytest.approx(1.0)
        b2 = self._merit(None)
        assert b2['w_iaspec'] == pytest.approx(1.0)

    def test_partial_evidence_interpolates(self):
        b = self._merit(0.5)
        assert b['w_iaspec'] == pytest.approx(1.0)
        assert self._merit(0.75)['w_iaspec'] == pytest.approx(1.1)

    def test_folded_into_merit(self):
        m_ia = float(self._merit(1.0)['merit'])
        m_non = float(self._merit(0.0)['merit'])
        m_unk = float(self._merit(np.nan)['merit'])
        assert m_ia > m_unk > m_non
        assert m_ia / m_non == pytest.approx(1.2 / 0.8)


class TestCombined:
    def test_mixed_population(self):
        """One passing object among assorted rejects."""
        df = pd.concat([
            make_payload(1),                                   # good
            make_payload(1, dec=[40.0]),                       # too north
            make_payload(1, brightest_mag=[23.0]),             # too faint
            make_payload(1, z_phot=[0.9]),                     # too distant
            make_payload(1, simbad_otype_xm=['QSO']),          # AGN
            make_payload(1, last_mjd=[MJD_NOW - 200.0]),       # stale
        ], ignore_index=True)
        df['diaObjectId'] = [f'obj{i}' for i in range(len(df))]
        out = select_wide_candidates(df, MJD_NOW)
        assert len(out) == 1
        assert out['diaObjectId'].iloc[0] == 'obj0'


class TestXmFirstNonBlank:
    """TNS/xm cross-match must survive dedup even when only an OLDER alert
    carries it (the kept row is the most recent alert)."""

    def test_tns_from_older_alert_survives(self, monkeypatch):
        import run_tonight as rt

        # Two alerts of one object: older one has the TNS name, newest blank.
        payload = pd.DataFrame({
            'r:diaObjectId': ['obj1', 'obj1'],
            'r:ra': [150.0, 150.0],
            'r:dec': [-20.0, -20.0],
            'r:psfFlux': [10000.0, 8000.0],
            'r:midpointMjdTai': [61200.0, 61230.0],
            'f:clf_snnSnVsOthers_score': [0.9, 0.9],
            'f:clf_earlySNIa_score': [-1.0, -1.0],
            'f:xm_tns_fullname': ['AT 2026zz', ''],
            'f:xm_tns_redshift': [0.08, np.nan],
        })

        class FakeFink:
            def query_sn_candidates(self, tag, n):
                return payload.copy() if tag == 'sn_near_galaxy_candidate' else None

        out = rt.fetch_fink_candidates(FakeFink(), min_sn_score=0.3, n_fetch=10)
        assert len(out) == 1
        row = out.iloc[0]
        # Kept row is the most recent alert (mjd 61230)...
        assert row['alert_mjd'] == pytest.approx(61230.0)
        # ...but the TNS name and spec-z from the older alert survive.
        assert row['tns_name_xm'] == 'AT 2026zz'
        assert row['z_tns'] == pytest.approx(0.08)


class TestEffectiveProbCoalescing:
    """Chain: mean_ia_prob (ml) -> sn_score (ml) -> antares_proxy -> none."""

    def test_chain_and_labels(self):
        df = pd.DataFrame({
            'mean_ia_prob': [0.9, np.nan, np.nan],
            'sn_score': [0.8, 0.7, np.nan],
            'antares_proxy_prob': [np.nan, np.nan, 0.45],
        })
        out = _coalesce_effective_prob(df, min_prob=0.3)
        assert list(out['effective_prob']) == pytest.approx([0.9, 0.7, 0.45])
        assert list(out['prob_source']) == ['ml', 'ml', 'antares_proxy']
        assert list(out['needs_classification']) == [False, False, True]

    def test_no_probability_anywhere_dropped(self):
        df = pd.DataFrame({
            'mean_ia_prob': [np.nan],
            'sn_score': [np.nan],
        })
        out = _coalesce_effective_prob(df, min_prob=0.3)
        assert len(out) == 0

    def test_filter_applies_to_coalesced_value(self):
        df = pd.DataFrame({
            'mean_ia_prob': [np.nan, np.nan],
            'sn_score': [0.5, 0.1],
        })
        out = _coalesce_effective_prob(df, min_prob=0.3)
        assert len(out) == 1
        assert out['effective_prob'].iloc[0] == pytest.approx(0.5)


class TestZtfWideIaProbSemantics:
    """fetch_ztf_wide_candidates: sn_ia_prob only for SNIa-classified objects;
    sn_score keeps the raw class probability for ALL classes."""

    def test_sn_ia_prob_nan_for_non_ia(self, monkeypatch):
        import run_tonight as rt
        import broker_clients.alerce_db_client as adb

        rows = pd.DataFrame({
            'oid': ['ZTFia', 'ZTFii'],
            'meanra': [150.0, 152.0],
            'meandec': [-20.0, -22.0],
            'firstmjd': [MJD_NOW - 5.0] * 2,
            'deltajd': [3.0, 3.0],
            'class_name': ['SNIa', 'SNII'],
            'probability': [0.7, 0.8],
            'magmin': [18.5, 18.0],
        })

        class FakeDB:
            def connect(self):
                pass

            def query_fresh_sn_candidates(self, *a, **k):
                return rows

        monkeypatch.setattr(adb, 'AlerceDBClient', FakeDB)
        out, status = rt.fetch_ztf_wide_candidates(MJD_NOW, min_prob=0.3)
        assert status['responded'] and status['n_returned'] == 2

        ia = out[out['alerce_class'] == 'SNIa'].iloc[0]
        snii = out[out['alerce_class'] == 'SNII'].iloc[0]
        assert ia['sn_ia_prob'] == pytest.approx(0.7)
        assert ia['sn_score'] == pytest.approx(0.7)
        assert pd.isna(snii['sn_ia_prob'])
        assert snii['sn_score'] == pytest.approx(0.8)


class TestWideFetchIntegration:
    """fetch_all_broker_candidates in wide mode, end to end through the real
    aggregator: Fink stream + coincident ZTF SNIa + distinct ZTF SNII."""

    FINK_RA, FINK_DEC = 150.0, -20.0

    def _fake_fink(self):
        # ~mag 19.0: psfFlux = 10**((31.4 - 19.0)/2.5) nJy ~ 91201
        payload = pd.DataFrame({
            'r:diaObjectId': ['finkobj1'],
            'r:ra': [self.FINK_RA],
            'r:dec': [self.FINK_DEC],
            'r:psfFlux': [91201.0],
            'r:midpointMjdTai': [MJD_NOW - 3.0],
            'f:clf_snnSnVsOthers_score': [0.9],
            'f:clf_earlySNIa_score': [-1.0],
            'f:xm_legacydr8_zphot': [0.2],
        })

        class FakeFink:
            def query_sn_candidates(self, tag, n):
                if tag == 'extragalactic_lt20mag_candidate':
                    return payload.copy()
                return None

        return FakeFink()

    def _synthetic_ztf(self):
        # First row ~1" from the Fink object; second a distinct SNII.
        dra = (1.0 / 3600.0) / np.cos(np.radians(abs(self.FINK_DEC)))
        return pd.DataFrame({
            'diaObjectId': ['ZTF26aaa', 'ZTF26bbb'],
            'object_id': ['ZTF26aaa', 'ZTF26bbb'],
            'ra': [self.FINK_RA + dra, 152.0],
            'dec': [self.FINK_DEC, -22.0],
            'alerce_class': ['SNIa', 'SNII'],
            'sn_score': [0.7, 0.8],
            'sn_ia_prob': [0.7, np.nan],
            'brightest_mag': [18.9, 18.5],
            'last_mjd': [MJD_NOW - 2.0, MJD_NOW - 1.0],
            'z_phot': [np.nan, np.nan],
            'z_tns': [np.nan, np.nan],
            'broker': ['ALeRCE-ZTF'] * 2,
            'ddf_field': [None, None],
        })

    def test_wide_path_through_aggregator(self, monkeypatch):
        import run_tonight as rt

        ztf = self._synthetic_ztf()
        ztf_status = {'queried': True, 'responded': True,
                      'n_returned': len(ztf), 'error': None}
        monkeypatch.setattr(rt, 'fetch_ztf_wide_candidates',
                            lambda *a, **k: (ztf, ztf_status))

        out, status = rt.fetch_all_broker_candidates(
            self._fake_fink(), min_prob=0.3, days_back=30,
            sky_mode='wide', mjd_now=MJD_NOW)

        # ANTARES / ALeRCE-LSST still not queried in wide mode
        assert status['ANTARES']['queried'] is False
        assert status['ALeRCE-LSST']['queried'] is False
        assert status['ALeRCE-ZTF'] == ztf_status
        assert status['Fink']['queried'] is True

        # Coincident pair collapsed; distinct SNII kept -> 2 unique objects
        assert len(out) == 2

        pair = out[out['num_brokers'] == 2].iloc[0]
        assert pair['brokers_detected'] == 'ALeRCE-ZTF,Fink'
        assert pair['ztf_oid'] == 'ZTF26aaa'
        assert pair['alerce_class'] == 'SNIa'
        # both ML Ia probs pooled: Fink 0.9, ALeRCE-ZTF 0.7
        assert pair['mean_ia_prob'] == pytest.approx(0.8)
        assert pair['prob_source'] == 'ml'
        # payload selection columns survive the merge
        assert pair['z_source'] == 'legacy_photz'
        assert pair['z_best'] == pytest.approx(0.2)
        assert np.isfinite(pair['brightest_mag'])
        assert pair['diaObjectId'] == 'finkobj1'

        # The ZTF-only SNII survives via sn_score (ML, class-level), is NOT
        # treated as a Ia probability, and is not flagged for typing.
        snii = out[out['alerce_class'] == 'SNII'].iloc[0]
        assert pd.isna(snii['mean_ia_prob'])
        assert snii['effective_prob'] == pytest.approx(0.8)
        assert snii['prob_source'] == 'ml'
        assert not bool(snii['needs_classification'])
        assert snii['diaObjectId'] == 'ZTF26bbb'
        assert snii['ztf_oid'] == 'ZTF26bbb'

    def test_fink_unavailable_ztf_only_stream(self, monkeypatch):
        # August-2026 Rubin-downtime shape: merge_alerts with a single
        # non-empty frame must still work end to end.
        import run_tonight as rt

        ztf = self._synthetic_ztf()
        monkeypatch.setattr(
            rt, 'fetch_ztf_wide_candidates',
            lambda *a, **k: (ztf, {'queried': True, 'responded': True,
                                   'n_returned': len(ztf), 'error': None}))

        out, status = rt.fetch_all_broker_candidates(
            None, min_prob=0.3, days_back=30, sky_mode='wide',
            mjd_now=MJD_NOW)

        assert status['Fink']['responded'] is False
        assert len(out) == 2
        assert set(out['brokers_detected']) == {'ALeRCE-ZTF'}
        assert set(out['diaObjectId']) == {'ZTF26aaa', 'ZTF26bbb'}
        assert set(out['prob_source']) == {'ml'}

    def test_both_streams_empty(self, monkeypatch):
        import run_tonight as rt

        monkeypatch.setattr(
            rt, 'fetch_ztf_wide_candidates',
            lambda *a, **k: (pd.DataFrame(), {'queried': True,
                                              'responded': True,
                                              'n_returned': 0,
                                              'error': None}))

        out, status = rt.fetch_all_broker_candidates(
            None, min_prob=0.3, days_back=30, sky_mode='wide',
            mjd_now=MJD_NOW)

        assert len(out) == 0
        assert status['ALeRCE-ZTF']['responded'] is True


class TestZEnrichment:
    """Post-ranking TNS+NED redshift enrichment of finalists."""

    @staticmethod
    def _summary():
        return pd.DataFrame([
            {'diaObjectId': 'ZTFaaa', 'ra': 150.0, 'dec': -20.0,
             'redshift': np.nan, 'distmod': np.nan, 'ned_name': '',
             'tns_name': '', 'tns_type': np.nan, 'tns_redshift': np.nan,
             'tns_match': False, 'peak_mag': 19.0, 'absolute_mag': np.nan,
             'salt_chi2_dof': 0.5, 'salt_x1': 0.1, 'salt_c': 0.0,
             'salt_z': np.nan, 'salt_peak_mag_B': np.nan},
            {'diaObjectId': 'FK1', 'ra': 200.0, 'dec': 10.0,
             'redshift': 0.2, 'distmod': 40.0, 'ned_name': 'payload',
             'tns_name': '', 'tns_type': np.nan, 'tns_redshift': np.nan,
             'tns_match': False, 'peak_mag': 20.5, 'absolute_mag': -19.4,
             'salt_chi2_dof': 1.0, 'salt_x1': 0.0, 'salt_c': 0.0,
             'salt_z': 0.2, 'salt_peak_mag_B': np.nan},
        ])

    def test_tns_specz_gained(self, monkeypatch):
        import run_tonight as rt

        class FakeTNS:
            def verify_connection(self):
                return True, 'ok'
            def search_by_coordinates(self, ra, dec, radius_arcsec=5.0):
                if abs(ra - 150.0) < 0.01:
                    return [{'objname': '2026zz', 'prefix': 'SN',
                             'type': 'SN Ia', 'redshift': 0.07}]
                return []

        monkeypatch.setattr(rt, 'HAS_TNS', True)
        monkeypatch.setattr(rt, 'TNSClient', FakeTNS)
        monkeypatch.setattr(rt, 'HAS_NED', False)

        out = rt.enrich_finalist_redshifts(self._summary(), {}, use_salt=False)
        row = out[out['diaObjectId'] == 'ZTFaaa'].iloc[0]
        assert row['redshift'] == pytest.approx(0.07)
        assert row['tns_name'] == 'SN 2026zz'
        assert row['tns_type'] == 'SN Ia'
        assert np.isfinite(row['distmod'])
        # absolute mag computed from peak mag - distmod
        assert row['absolute_mag'] == pytest.approx(19.0 - row['distmod'])
        # the already-z'd row is untouched
        fk = out[out['diaObjectId'] == 'FK1'].iloc[0]
        assert fk['redshift'] == pytest.approx(0.2)

    def test_ned_fallback(self, monkeypatch):
        import run_tonight as rt

        class NoTNS:
            def verify_connection(self):
                return False, 'down'

        def fake_ned(df, cache=None, radius_arcsec=18.0):
            df = df.copy()
            df['ned_redshift'] = [0.05] * len(df)
            df['ned_distmod'] = [36.7] * len(df)
            df['ned_name'] = ['NGC 42'] * len(df)
            df['ned_sep_arcsec'] = [2.0] * len(df)
            return df

        monkeypatch.setattr(rt, 'HAS_TNS', True)
        monkeypatch.setattr(rt, 'TNSClient', NoTNS)
        monkeypatch.setattr(rt, 'HAS_NED', True)
        monkeypatch.setattr(rt, 'query_ned_batch', fake_ned)

        out = rt.enrich_finalist_redshifts(self._summary(), {}, use_salt=False)
        row = out[out['diaObjectId'] == 'ZTFaaa'].iloc[0]
        assert row['redshift'] == pytest.approx(0.05)
        assert 'ned:' in row['ned_name']

    def test_no_sources_no_change(self, monkeypatch):
        import run_tonight as rt
        monkeypatch.setattr(rt, 'HAS_TNS', False)
        monkeypatch.setattr(rt, 'HAS_NED', False)
        s = self._summary()
        out = rt.enrich_finalist_redshifts(s.copy(), {}, use_salt=False)
        assert not (pd.to_numeric(out[out['diaObjectId'] == 'ZTFaaa']['redshift'],
                                  errors='coerce') > 0).any()
