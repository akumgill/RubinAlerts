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
