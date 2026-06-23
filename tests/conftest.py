"""Shared pytest fixtures for RubinAlerts tests.

All fixtures are fully in-memory or tmp_path-based. No fixture or test in this
suite may contact a live broker, database, or external API — network-using code
must be stubbed/monkeypatched at the call site.
"""

import os
import shutil

import numpy as np
import pandas as pd
import pytest

# Repository root (parent of tests/)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def sample_merged_alerts():
    """A small DataFrame of merged broker candidates.

    Columns mirror what core/magellan_planning.compute_merit_breakdown and
    run_tonight.build_summary_table actually read: ra, dec, num_brokers,
    sn_score / mean_ia_prob (the ia_prob columns), brokers_detected, ddf_field,
    diaObjectId, plus optional peak fit fields.

    Rows:
      0  ANTARES-only equatorial COSMOS candidate (single broker)
      1  ALeRCE+Fink multi-broker equatorial COSMOS candidate
      2  Southern EDFS_a candidate (dec ~ -47, single broker — no ZTF coverage)
      3  Equatorial M49 candidate (single broker)
    """
    rows = [
        {
            'diaObjectId': 'ANT0001',
            'object_id': 'ANT0001',
            'ra': 150.10,
            'dec': 2.20,
            'ddf_field': 'COSMOS',
            'brokers_detected': 'ANTARES',
            'num_brokers': 1,
            'sn_score': 0.42,
            'mean_ia_prob': 0.42,
            'E_BV': 0.02,
            'host_morphology': 'unknown',
            'peak_mag': 20.4,
            'peak_mjd': 61100.0,
            'delta_t': 1.0,
        },
        {
            'diaObjectId': 'AL_FK_002',
            'object_id': 'AL_FK_002',
            'ra': 150.20,
            'dec': 2.30,
            'ddf_field': 'COSMOS',
            'brokers_detected': 'ALeRCE-ZTF,Fink',
            'num_brokers': 2,
            'sn_score': 0.71,
            'mean_ia_prob': 0.68,
            'E_BV': 0.03,
            'host_morphology': 'elliptical',
            'peak_mag': 20.5,
            'peak_mjd': 61100.0,
            'delta_t': 0.0,
        },
        {
            'diaObjectId': 'EDFS_S_003',
            'object_id': 'EDFS_S_003',
            'ra': 58.90,
            'dec': -47.10,
            'ddf_field': 'EDFS_a',
            'brokers_detected': 'ALeRCE-LSST',
            'num_brokers': 1,
            'sn_score': 0.55,
            'mean_ia_prob': 0.55,
            'E_BV': 0.04,
            'host_morphology': 'spiral',
            'peak_mag': 20.6,
            'peak_mjd': 61100.0,
            'delta_t': 2.0,
        },
        {
            'diaObjectId': 'M49_004',
            'object_id': 'M49_004',
            'ra': 187.44,
            'dec': 8.00,
            'ddf_field': 'M49',
            'brokers_detected': 'Fink',
            'num_brokers': 1,
            'sn_score': 0.60,
            'mean_ia_prob': 0.60,
            'E_BV': 0.05,
            'host_morphology': 'unknown',
            'peak_mag': 20.5,
            'peak_mjd': 61100.0,
            'delta_t': -1.0,
        },
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def sample_allocations_path(tmp_path):
    """Copy ref/allocations_example.yaml into a tmp path and return the path."""
    src = os.path.join(REPO_ROOT, 'ref', 'allocations_example.yaml')
    dst = tmp_path / 'allocations.yaml'
    shutil.copy(src, dst)
    return str(dst)


@pytest.fixture
def sample_allocations(sample_allocations_path):
    """Parsed allocations YAML as a plain dict."""
    import yaml
    with open(sample_allocations_path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def tmp_output_dir(tmp_path):
    """A tmp_path-based output directory for orchestrator/pipeline writes."""
    out = tmp_path / 'output'
    out.mkdir()
    return str(out)
