"""Tests for the night-primary policy gating must-see guarantees.

Each observing night has a designated PRIMARY program (looked up from a small
nights CSV). Only the primary's must-see (mandatory) targets are guaranteed
scheduling; everyone else's targets — and the primary's unmarked targets — go
through normal prioritization.

All in-memory: a temp nights CSV; Targets built directly with windows set the
way test_planner.py / test_mandatory_scheduling.py do; airmass monkeypatched
constant. No network.
"""

import logging

import astropy.units as u

from orchestrator.models import Target
from orchestrator.normalize import load_primary_program
from orchestrator.planner import calculate_twilight, create_schedule


DATE = '2026-10-15'


def _windowed_target(name, ra_deg, dec_deg, evening, morning,
                     exp_min=20.0, priority=1, moon='any', mandatory=False,
                     program='default'):
    """Build a Target with a full-night observable window populated."""
    t = Target(name=name, ra_deg=ra_deg, dec_deg=dec_deg, priority=priority,
               exposure_minutes=exp_min, moon_constraint=moon,
               mandatory=mandatory, program=program)
    t.transit_time = evening + (morning - evening) / 2
    t.window_start = evening
    t.window_end = morning
    t.window_hours = (morning - evening).to(u.hour).value
    return t


# ---------------------------------------------------------------------------
# load_primary_program
# ---------------------------------------------------------------------------

def test_load_primary_program_present_date(tmp_path):
    """Returns the listed primary for a date present in the CSV; tolerates
    comment lines and surrounding whitespace."""
    csv = tmp_path / 'nights.csv'
    csv.write_text(
        "# observing nights\n"
        "date,primary_program,primary_observer\n"
        "2026-10-15, MAGNETS-Stubbs , Chris Stubbs\n"
        "2026-10-16,MAGNETS-Villar,Yize Dong\n"
    )
    assert load_primary_program(str(csv), '2026-10-15') == 'MAGNETS-Stubbs'
    assert load_primary_program(str(csv), '2026-10-16') == 'MAGNETS-Villar'


def test_load_primary_program_missing_date(tmp_path):
    """Returns None for a date not listed in the CSV."""
    csv = tmp_path / 'nights.csv'
    csv.write_text(
        "date,primary_program\n"
        "2026-10-15,MAGNETS-Stubbs\n"
    )
    assert load_primary_program(str(csv), '2026-12-31') is None


def test_load_primary_program_missing_file(tmp_path):
    """Returns None when the file is absent."""
    missing = tmp_path / 'does_not_exist.csv'
    assert load_primary_program(str(missing), '2026-10-15') is None


# ---------------------------------------------------------------------------
# Gating the must-see reservation by primary
# ---------------------------------------------------------------------------

def test_primary_mustsee_is_reserved(monkeypatch):
    """With a designated primary, a mandatory target whose program matches the
    primary IS reserved/guaranteed."""
    import orchestrator.planner as planner_mod
    monkeypatch.setattr(planner_mod, '_get_airmass',
                        lambda coord, time, location: 1.1)

    evening, morning = calculate_twilight(DATE)

    must = _windowed_target('MUST', 20.0, -29.0, evening, morning,
                            exp_min=30.0, priority=4, mandatory=True,
                            program='MAGNETS-Stubbs')
    rich = _windowed_target('RICH', 24.0, -29.0, evening, morning,
                            exp_min=30.0, priority=1, program='MAGNETS-Villar')
    scores = {'MUST': 1.0, 'RICH': 10000.0}

    plan = create_schedule(
        [rich, must], evening, morning, moon_phase='dark',
        prioritizer_scores=scores, primary_program='MAGNETS-Stubbs',
    )

    names = [e.target.name for e in plan.scheduled]
    assert 'MUST' in names  # reserved (primary's must-see), despite low score
    assert plan.unschedulable_mandatory == []


def test_nonprimary_mustsee_demoted_and_warns(monkeypatch, caplog):
    """A mandatory target whose program is NOT tonight's primary is NOT
    reserved — it is demoted to a normal target (competes in the greedy fill)
    and a warning is logged."""
    import orchestrator.planner as planner_mod
    monkeypatch.setattr(planner_mod, '_get_airmass',
                        lambda coord, time, location: 1.1)

    evening, morning = calculate_twilight(DATE)

    # Mandatory but belongs to the NON-primary program, with a tiny score.
    other = _windowed_target('OTHER', 20.0, -29.0, evening, morning,
                             exp_min=30.0, priority=4, mandatory=True,
                             program='MAGNETS-Villar')
    # A high-scoring primary competitor.
    rich = _windowed_target('RICH', 24.0, -29.0, evening, morning,
                            exp_min=30.0, priority=1, program='MAGNETS-Stubbs')
    scores = {'OTHER': 1.0, 'RICH': 10000.0}

    with caplog.at_level(logging.WARNING):
        plan = create_schedule(
            [other, rich], evening, morning, moon_phase='dark',
            prioritizer_scores=scores, primary_program='MAGNETS-Stubbs',
        )

    # OTHER was demoted (not reserved): it did not land in the reservation-pass
    # "unschedulable_mandatory" bucket, and a demotion warning was logged.
    assert plan.unschedulable_mandatory == []  # demoted, not "unschedulable"
    assert any('OTHER' in r.message and 'primary' in r.message
               for r in caplog.records)
    # Demoted OTHER competes by score in the greedy fill. The high-scoring
    # primary RICH wins the first (best) slot — proving OTHER was NOT given a
    # guaranteed reservation ahead of the greedy loop.
    names = [e.target.name for e in plan.scheduled]
    assert names[0] == 'RICH'


def test_no_primary_honors_all_mustsee(monkeypatch):
    """With primary_program=None (no nights file), a mandatory target is
    honored regardless of its program (backward-compatible)."""
    import orchestrator.planner as planner_mod
    monkeypatch.setattr(planner_mod, '_get_airmass',
                        lambda coord, time, location: 1.1)

    evening, morning = calculate_twilight(DATE)

    must = _windowed_target('MUST', 20.0, -29.0, evening, morning,
                            exp_min=30.0, priority=4, mandatory=True,
                            program='MAGNETS-Villar')
    rich = _windowed_target('RICH', 24.0, -29.0, evening, morning,
                            exp_min=30.0, priority=1, program='MAGNETS-Stubbs')
    scores = {'MUST': 1.0, 'RICH': 10000.0}

    plan = create_schedule(
        [must, rich], evening, morning, moon_phase='dark',
        prioritizer_scores=scores, primary_program=None,
    )

    names = [e.target.name for e in plan.scheduled]
    assert 'MUST' in names  # honored even though program != any primary
    assert plan.unschedulable_mandatory == []


# ---------------------------------------------------------------------------
# Short-window guarantee
# ---------------------------------------------------------------------------

def test_primary_mustsee_short_window_still_reserved(monkeypatch):
    """A primary must-see target with a short-but-nonzero observable window is
    still reserved (not dropped). Exposure exceeds the window, but the override
    guarantee outranks the normal minimum-window cutoff."""
    import orchestrator.planner as planner_mod
    monkeypatch.setattr(planner_mod, '_get_airmass',
                        lambda coord, time, location: 1.1)

    evening, morning = calculate_twilight(DATE)

    must = _windowed_target('SHORT', 20.0, -29.0, evening, morning,
                            exp_min=120.0, priority=4, mandatory=True,
                            program='MAGNETS-Stubbs')
    # Short window: only 30 min wide, far less than the 120-min exposure.
    must.window_start = evening
    must.window_end = evening + 30 * u.minute
    must.transit_time = evening + 15 * u.minute
    must.window_hours = (must.window_end - must.window_start).to(u.hour).value

    plan = create_schedule(
        [must], evening, morning, moon_phase='dark',
        prioritizer_scores={'SHORT': 1.0}, primary_program='MAGNETS-Stubbs',
    )

    names = [e.target.name for e in plan.scheduled]
    assert 'SHORT' in names  # reserved despite the short window
    assert plan.unschedulable_mandatory == []


def test_primary_mustsee_no_window_unschedulable(monkeypatch, caplog):
    """A primary must-see target with NO observable window at all lands in
    unschedulable_mandatory and warns (physics still wins)."""
    import orchestrator.planner as planner_mod
    monkeypatch.setattr(planner_mod, '_get_airmass',
                        lambda coord, time, location: 1.1)

    evening, morning = calculate_twilight(DATE)

    # No window populated -> never reaches the airmass limit.
    never = Target(name='NEVER', ra_deg=10.0, dec_deg=80.0, priority=1,
                   exposure_minutes=30.0, mandatory=True,
                   program='MAGNETS-Stubbs')
    assert never.window_start is None

    with caplog.at_level(logging.WARNING):
        plan = create_schedule(
            [never], evening, morning, moon_phase='dark',
            primary_program='MAGNETS-Stubbs',
        )

    names = [e.target.name for e in plan.scheduled]
    assert 'NEVER' not in names
    assert any(t.name == 'NEVER' for t in plan.unschedulable_mandatory)
    assert any('NEVER' in r.message for r in caplog.records)
