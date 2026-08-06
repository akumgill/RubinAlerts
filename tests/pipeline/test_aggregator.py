"""Tests for alert aggregation: ML-only w_prob and angular-separation dedup.

R7: the ANTARES heuristic proxy (_compute_antares_proxy_prob) must NOT be
    averaged into mean_ia_prob — that column pools ML-derived probabilities only.
R13: deduplication must use true angular separation (cos-dec aware), not a
     RA/Dec box, so southern-DDF dupes at moderate separation still merge.

All in-memory DataFrames — no live broker/DB/API calls.
"""

import numpy as np
import pandas as pd
import pytest

from core.alert_aggregator import AlertAggregator


@pytest.fixture
def aggregator(tmp_path):
    # apply_extinction=False to avoid any network/extinction-map lookups.
    return AlertAggregator(cache_dir=str(tmp_path / 'cache'),
                           match_tolerance_arcsec=1.5,
                           apply_extinction=False)


def test_antares_proxy_excluded_from_mean_ia_prob(aggregator):
    """R7: ANTARES proxy is preserved separately, never pooled into mean_ia_prob."""
    # ANTARES-only candidate (no sn_ia_prob -> proxy computed). Choose features
    # so the proxy lands well away from 0.25 to make exclusion unambiguous.
    antares = pd.DataFrame([{
        'ra': 150.10, 'dec': 2.20,
        'ztf_sgscore1': 0.1,      # galaxy-associated -> proxy bumps up
        'duration_days': 50.0,    # short -> proxy bumps up
        'object_id': 'ANT0001',
    }])

    # Multi-broker ML candidate: ALeRCE + Fink, both with real ML sn_ia_prob,
    # at a clearly different position so it does not merge with the ANTARES row.
    alerce = pd.DataFrame([{
        'ra': 200.0, 'dec': 10.0,
        'sn_ia_prob': 0.80,
        'object_id': 'AL_002',
    }])
    fink = pd.DataFrame([{
        'ra': 200.0, 'dec': 10.0,
        'sn_ia_prob': 0.60,
        'object_id': 'FK_002',
    }])

    merged = aggregator.merge_alerts({
        'ANTARES': antares,
        'ALeRCE': alerce,
        'Fink': fink,
    })

    assert len(merged) == 2

    # Locate the ANTARES-only row and the ML multi-broker row.
    ant_row = merged[merged['brokers_detected'] == 'ANTARES'].iloc[0]
    ml_row = merged[merged['num_brokers'] == 2].iloc[0]

    # The heuristic proxy value ANTARES would have produced.
    proxy_val = AlertAggregator._compute_antares_proxy_prob(antares.iloc[0])
    assert proxy_val > 0.25  # sanity: features pushed it above the prior

    # ANTARES-only: proxy preserved in its own column, NOT in mean_ia_prob.
    assert 'antares_proxy_prob' in merged.columns
    assert ant_row['antares_proxy_prob'] == pytest.approx(proxy_val)
    # No ML probabilities for this object -> mean_ia_prob must be NaN,
    # and in particular must NOT equal the proxy.
    assert pd.isna(ant_row['mean_ia_prob'])
    assert not (ant_row['mean_ia_prob'] == proxy_val)

    # Multi-broker ML row: mean_ia_prob averages ONLY the ML probs.
    assert ml_row['mean_ia_prob'] == pytest.approx((0.80 + 0.60) / 2)


def test_southern_dec_dedup_uses_angular_separation(aggregator):
    """R13: two detections ~1.4" apart at dec ~ -45 merge under separation match.

    The old box test |Δra|<tol used un-projected RA degrees, so at dec=-45
    (cos dec ~ 0.707) a true ~1.4" separation maps to a larger ΔRA in degrees
    and could fall outside a 1.5" box. True angular separation merges them.
    """
    dec = -45.0
    cosd = np.cos(np.radians(dec))
    # Offset purely in RA by ~1.4 arcsec of true angular separation.
    sep_arcsec = 1.4
    dra_deg = (sep_arcsec / 3600.0) / cosd  # RA degrees needed for that sep

    a = pd.DataFrame([{
        'ra': 58.9, 'dec': dec,
        'sn_ia_prob': 0.70,
        'object_id': 'S_A',
    }])
    b = pd.DataFrame([{
        'ra': 58.9 + dra_deg, 'dec': dec,
        'sn_ia_prob': 0.65,
        'object_id': 'S_B',
    }])

    # Confirm the un-projected ΔRA exceeds the 1.5" box half-width (the old bug
    # would have treated these as distinct objects).
    box_half_deg = aggregator.match_tolerance / 3600.0
    assert dra_deg > box_half_deg

    merged = aggregator.merge_alerts({'ALeRCE': a, 'Fink': b})

    # Separation-based matching collapses them into one object.
    assert len(merged) == 1
    assert merged.iloc[0]['num_brokers'] == 2


# ---------------------------------------------------------------------------
# Passthrough column carriage (wide-mode payload columns surviving the merge)
# ---------------------------------------------------------------------------

def _fink_row(**ov):
    row = {'ra': 150.0, 'dec': -20.0, 'sn_ia_prob': 0.8,
           'object_id': 'FK1', 'diaObjectId': 'FK1'}
    row.update(ov)
    return row


def _ztf_row(**ov):
    row = {'ra': 150.0, 'dec': -20.0, 'sn_ia_prob': 0.5,
           'object_id': 'ZTF26xx'}
    row.update(ov)
    return row


class TestPassthroughColumns:
    def test_brightest_mag_min_across_brokers(self, aggregator):
        merged = aggregator.merge_alerts({
            'Fink': pd.DataFrame([_fink_row(brightest_mag=19.5)]),
            'ALeRCE-ZTF': pd.DataFrame([_ztf_row(brightest_mag=18.8)]),
        })
        assert len(merged) == 1
        assert merged.iloc[0]['brightest_mag'] == pytest.approx(18.8)

    def test_last_mjd_max_across_brokers(self, aggregator):
        merged = aggregator.merge_alerts({
            'Fink': pd.DataFrame([_fink_row(last_mjd=61230.0)]),
            'ALeRCE-ZTF': pd.DataFrame([_ztf_row(last_mjd=61233.5)]),
        })
        assert merged.iloc[0]['last_mjd'] == pytest.approx(61233.5)

    def test_z_best_prefers_specz_over_photz(self, aggregator):
        merged = aggregator.merge_alerts({
            'Fink': pd.DataFrame([_fink_row(z_phot=0.35, z_tns=0.12)]),
        })
        row = merged.iloc[0]
        assert row['z_best'] == pytest.approx(0.12)
        assert row['z_source'] == 'tns_specz'

    def test_z_best_photz_fallback_and_none(self, aggregator):
        merged = aggregator.merge_alerts({
            'Fink': pd.DataFrame([_fink_row(z_phot=0.3, z_tns=np.nan)]),
        })
        assert merged.iloc[0]['z_source'] == 'legacy_photz'
        merged2 = aggregator.merge_alerts({
            'Fink': pd.DataFrame([_fink_row(z_phot=np.nan, z_tns=np.nan)]),
        })
        assert merged2.iloc[0]['z_source'] == 'none'

    def test_tns_xm_first_nonblank_across_brokers(self, aggregator):
        # TNS name blank on the Fink row, present on the ZTF row -> survives
        merged = aggregator.merge_alerts({
            'Fink': pd.DataFrame([_fink_row(tns_name_xm='')]),
            'ALeRCE-ZTF': pd.DataFrame([_ztf_row(tns_name_xm='AT 2026xy')]),
        })
        assert merged.iloc[0]['tns_name_xm'] == 'AT 2026xy'

    def test_alerce_class_join_not_overwrite(self, aggregator):
        merged = aggregator.merge_alerts({
            'Fink': pd.DataFrame([_fink_row(alerce_class='SNII')]),
            'ALeRCE-ZTF': pd.DataFrame([_ztf_row(alerce_class='SNIa')]),
        })
        joined = merged.iloc[0]['alerce_class']
        assert set(joined.split('|')) == {'SNIa', 'SNII'}

    def test_ztf_oid_alias_set(self, aggregator):
        merged = aggregator.merge_alerts({
            'Fink': pd.DataFrame([_fink_row()]),
            'ALeRCE-ZTF': pd.DataFrame([_ztf_row()]),
        })
        assert merged.iloc[0]['ztf_oid'] == 'ZTF26xx'

    def test_gaia_plx_and_err_paired_from_same_alert(self, aggregator):
        # Alert A: plx without err; alert B: both. Pairing must take both
        # from the row whose plx is first present (A), leaving err unset,
        # or from B if A's plx were missing.
        fink = pd.DataFrame([
            _fink_row(gaia_plx=np.nan, gaia_plx_err=0.1),
            _fink_row(gaia_plx=2.0, gaia_plx_err=0.4),
        ])
        merged = aggregator.merge_alerts({'Fink': fink})
        row = merged.iloc[0]
        assert row['gaia_plx'] == pytest.approx(2.0)
        assert row['gaia_plx_err'] == pytest.approx(0.4)

    def test_ddf_mode_noop_without_wide_columns(self, aggregator):
        # Frames lacking passthrough columns must not gain them (except the
        # always-present derived defaults), and existing columns unchanged.
        merged = aggregator.merge_alerts({
            'Fink': pd.DataFrame([_fink_row()]),
        })
        row = merged.iloc[0]
        for col in ('brightest_mag', 'last_mjd', 'z_best', 'z_source',
                    'tns_name_xm', 'alerce_class', 'ztf_oid'):
            assert col not in merged.columns or pd.isna(row.get(col)) or row.get(col) == ''
        assert row['sn_score'] == pytest.approx(0.8)

    def test_early_ia_score_max_and_not_pooled(self, aggregator):
        # max across alerts; and must NOT leak into mean_ia_prob (R7 guard)
        fink = pd.DataFrame([
            _fink_row(early_ia_score=0.2, sn_ia_prob=0.8),
            _fink_row(early_ia_score=0.6, sn_ia_prob=0.8),
        ])
        merged = aggregator.merge_alerts({'Fink': fink})
        row = merged.iloc[0]
        assert row['early_ia_score'] == pytest.approx(0.6)
        assert row['mean_ia_prob'] == pytest.approx(0.8)
