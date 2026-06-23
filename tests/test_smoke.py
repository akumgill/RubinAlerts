"""Smoke test: prove core + orchestrator modules import and fixtures collect."""


def test_core_imports():
    import core.magellan_planning  # noqa: F401
    import core.ddf_fields  # noqa: F401


def test_orchestrator_imports():
    import orchestrator  # noqa: F401
    import orchestrator.normalize  # noqa: F401
    import orchestrator.models  # noqa: F401


def test_fixtures_collect(sample_merged_alerts, sample_allocations,
                           sample_allocations_path, tmp_output_dir):
    assert len(sample_merged_alerts) == 4
    assert {'ra', 'dec', 'num_brokers', 'sn_score'}.issubset(
        sample_merged_alerts.columns)
    # A southern row and an equatorial row are present.
    assert (sample_merged_alerts['dec'] < -32).any()
    assert (sample_merged_alerts['dec'] > 0).any()
    assert sample_allocations['default_program'] == 'MAGNETS-Stubbs'
    assert tmp_output_dir.endswith('output')
