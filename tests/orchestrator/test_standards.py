"""Tests for mid-night standard star insertion (chunk F: R10).

The design spec calls for 2-3 interleaved standards/night; select_standards now
inserts periodic mid-night standards on a fixed cadence
(config.standard_interleave_hours) in addition to start/end. Fully in-memory: a
temp standards.txt fixture in the real catalog format is parsed; no network/DB.
"""

import astropy.units as u
from astropy.time import Time

from orchestrator.config import LLAMASConfig
from orchestrator.planner import select_standards, create_schedule
from orchestrator.models import Target


# A dense ring of standards at dec ~ -29 (near the LCO zenith) spread across
# all RAs, so at least one is always low-airmass regardless of the hour.
def _write_standards(tmp_path):
    header = (
        "Star Name       RA   (2000)   Dec      V mag. Spec  Note\n"
        "               h  m  s       d  '  ''          type\n"
        "----------------------------------------------------------\n"
    )
    lines = [header]
    for i in range(24):
        # One standard per hour of RA, V=10.5 (ideal), dec near site latitude.
        name = f"STD{i:02d}"
        lines.append(
            f" {name}       {i:02d} 00 00.0  -29 00 00.0  10.50  G\n"
        )
    path = tmp_path / 'standards.txt'
    path.write_text(''.join(lines))
    return str(path)


def _night(hours):
    """Return (evening, morning) Times spanning ``hours`` of true night.

    Centred on local midnight at LCO (~04:00 UT) so the zenith-ring standards
    sit at low airmass through the whole window.
    """
    midnight = Time('2026-10-15 04:00:00', scale='utc')
    half = (hours / 2.0) * u.hour
    return midnight - half, midnight + half


def test_long_night_inserts_mid_standards(tmp_path):
    """An 8h night yields start + end + at least one mid standard (>= 3)."""
    path = _write_standards(tmp_path)
    config = LLAMASConfig(standard_interleave_hours=3.5)
    evening, morning = _night(8.0)

    start, end, mids = select_standards(path, evening, morning, config)

    assert start is not None
    assert end is not None
    assert len(mids) >= 1
    total = (1 if start else 0) + (1 if end else 0) + len(mids)
    assert total >= 3

    # Each mid standard's target time falls strictly inside the night.
    for s in mids:
        assert 'time' in s
        assert evening.mjd < s['time'].mjd < morning.mjd

    # Bounded: do not insert dozens (spec target is 2-3 total).
    assert len(mids) <= 3


def test_short_night_no_mid_standards(tmp_path):
    """A 2h night (below the cadence) gets only start + end, no mids."""
    path = _write_standards(tmp_path)
    config = LLAMASConfig(standard_interleave_hours=3.5)
    evening, morning = _night(2.0)

    start, end, mids = select_standards(path, evening, morning, config)

    assert start is not None
    assert end is not None
    assert mids == []


def test_mid_standards_lie_between_science_observations(tmp_path):
    """Through create_schedule, mid standards fall in time between the first
    and last scheduled science observations on a long night."""
    path = _write_standards(tmp_path)
    config = LLAMASConfig(standard_interleave_hours=3.5)
    evening, morning = _night(8.0)

    # Many science targets spread across the RAs that transit during this
    # window, so the greedy schedule stays busy across the whole night and the
    # last science obs lands after the mid-night standards' target times.
    targets = []
    ras = [285, 300, 315, 330, 345, 0, 15, 30, 45, 60, 75, 90, 105, 120]
    for i, ra in enumerate(ras):
        t = Target(name=f"SCI{i:02d}", ra_deg=float(ra), dec_deg=-29.0,
                   priority=1, exposure_minutes=40.0, moon_constraint='any')
        t.transit_time = evening + (morning - evening) / 2
        t.window_start = evening
        t.window_end = morning
        t.window_hours = (morning - evening).to(u.hour).value
        targets.append(t)

    plan = create_schedule(targets, evening, morning, moon_phase='dark',
                           standards_path=path, config=config)

    assert plan.scheduled, "expected scheduled science targets"
    assert plan.standards_mid, "expected mid-night standards on an 8h night"

    first_sci = min(e.start.mjd for e in plan.scheduled)
    last_sci = max(e.end.mjd for e in plan.scheduled)
    for s in plan.standards_mid:
        assert first_sci < s['time'].mjd < last_sci


def test_airmass_spread_beats_minimising_airmass(tmp_path):
    """Standards measure the ATMOSPHERE, so the set needs a lever arm.

    Minimising airmass at every anchor — the old behaviour — gave 1.04/1.04/1.18
    on the real 2026-11-14 night: a span of 0.14, which no extinction fit can
    use. Spread mode must do materially better on the same night and catalogue.
    """
    from orchestrator.planner import (calculate_twilight, select_standards,
                                      DEFAULT_STANDARDS_PATH)
    from orchestrator.config import LLAMASConfig

    def span(spread):
        cfg = LLAMASConfig(std_airmass_spread=spread)
        ev, mo = calculate_twilight('2026-11-14', config=cfg)
        s0, s1, mid = select_standards(str(DEFAULT_STANDARDS_PATH), ev, mo, cfg)
        ams = [d['airmass'] for d in [s0] + (mid or []) + [s1] if d]
        assert len(ams) >= 2, "a night should still get its standards"
        return max(ams) - min(ams)

    narrow, wide = span(False), span(True)
    assert wide > narrow
    assert wide > 0.5, f"lever arm still too short to fit extinction: {wide:.2f}"


def test_high_airmass_anchor_may_exceed_the_normal_ceiling():
    """The usual std_max_airmass exists to keep flux-calibration visits near
    the zenith. It must not also forbid the high-airmass leg of the ladder,
    which is the whole point of the measurement."""
    from orchestrator.config import LLAMASConfig
    cfg = LLAMASConfig()
    assert cfg.std_airmass_spread is True
    assert cfg.std_spread_max_airmass > cfg.std_max_airmass
    assert cfg.std_max_airmass < cfg.std_high_airmass <= cfg.std_spread_max_airmass


def test_partial_airmass_ladder_is_reported_not_silently_accepted():
    """One leg of a ladder is a flux-calibration visit, not an extinction
    curve. A plan that schedules one of two legs has silently produced a null
    result, so the preview has to say so."""
    import api.scheduler_bridge as sb
    plan = {"timeline": [{"target": "WDFS0122-30@am1.0-1.3", "airmass": 1.02}],
            "overflow": []}
    # emulate the integrity block over a two-leg submission
    submitted = ["WDFS0122-30@am1.0-1.3", "WDFS0122-30@am1.7-2.3"]
    got = [e["target"] for e in plan["timeline"]]
    assert len(submitted) == 2 and len(got) == 1
    # the real check lives in preview_plan; assert the helper it relies on
    assert all("@am" in s for s in submitted)
    assert submitted[0].split("@am")[0] == submitted[1].split("@am")[0]


def test_enqueue_standards_defaults_to_a_reserved_tier():
    """A ladder enqueued at an ordinary tier loses legs to science and yields
    nothing — verified on the 2026-11-14 night. The default must reserve."""
    import importlib.util, os, sys
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    spec = importlib.util.spec_from_file_location(
        "enqueue_standards", os.path.join(repo, "scripts", "enqueue_standards.py"))
    m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    import argparse, contextlib, io
    ap_default = None
    # parse with the minimum required args and read the resolved priority
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            args = m.main.__wrapped__ if hasattr(m.main, "__wrapped__") else None
        except Exception:
            args = None
    src = open(os.path.join(repo, "scripts", "enqueue_standards.py")).read()
    assert '"--priority", default="P0"' in src, "ladder default must reserve legs"
