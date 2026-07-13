"""Tests for wide-sky payload-level candidate selection (run_tonight).

Covers select_wide_candidates() cuts (declination, freshness, variable/AGN
payload screen, brightness, redshift coalescing, fit cap) — all offline on
synthetic payload frames.
"""

import numpy as np
import pandas as pd
import pytest

from run_tonight import (
    select_wide_candidates, merge_wide_streams,
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


class TestMergeWideStreams:
    @staticmethod
    def _fink(n=1, **ov):
        base = make_payload(n, **ov)
        base['brokers_detected'] = 'Fink'
        base['num_brokers'] = 1
        return base

    @staticmethod
    def _ztf(ra, dec, oid='ZTF26abc', cls='SNIa'):
        return pd.DataFrame({
            'diaObjectId': [oid], 'object_id': [oid], 'ra': [ra], 'dec': [dec],
            'sn_score': [0.5], 'brokers_detected': ['ALeRCE-ZTF'],
            'num_brokers': [1], 'alerce_class': [cls],
        })

    def test_coincident_pair_collapses_to_fink_row(self):
        fink = self._fink(1, ra=[150.0], dec=[-20.0])
        ztf = self._ztf(150.0 + 0.0001, -20.0)  # ~0.36 arcsec away
        out = merge_wide_streams(fink, ztf)
        assert len(out) == 1
        row = out.iloc[0]
        assert row['num_brokers'] == 2
        assert row['brokers_detected'] == 'ALeRCE-ZTF,Fink'
        assert row['ztf_oid'] == 'ZTF26abc'
        assert row['alerce_class'] == 'SNIa'

    def test_distinct_objects_both_kept(self):
        fink = self._fink(1, ra=[150.0], dec=[-20.0])
        ztf = self._ztf(200.0, 10.0)
        out = merge_wide_streams(fink, ztf)
        assert len(out) == 2
        assert set(out['brokers_detected']) == {'Fink', 'ALeRCE-ZTF'}

    def test_empty_streams(self):
        fink = self._fink(1)
        empty = pd.DataFrame()
        assert len(merge_wide_streams(fink, empty)) == 1
        assert len(merge_wide_streams(empty, self._ztf(1.0, 1.0))) == 1
        assert len(merge_wide_streams(empty, empty)) == 0


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
