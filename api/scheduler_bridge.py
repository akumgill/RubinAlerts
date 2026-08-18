"""Bridge from the API queue to the LLAMAS orchestrator.

The plan preview does NOT re-implement scheduling: it materializes the current
queue into the manual-CSV schema the orchestrator already accepts, runs the
real nightly scheduler as a dry run, and reshapes the resulting ObsPlan into
the preview response (per-program requested hours, a program-tagged timeline,
and the overflow of submitted-but-unscheduled targets).
"""
from __future__ import annotations

import csv
import logging
import math
import os
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_REF_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ref")
_NIGHTS_YAML = os.path.join(_REF_DIR, "observing_nights_2026B.yaml")
# Per-instrument allocation files (parallel budget universes).
_ALLOC_FILES = {
    "LLAMAS": os.path.join(_REF_DIR, "allocations_LLAMAS_2026B.yaml"),
    "LDSS3": os.path.join(_REF_DIR, "allocations_LDSS3_2026B.yaml"),
}


def load_allocations_overview() -> dict:
    """Season time allocations per program per instrument: initial / used /
    remaining hours. Reads BOTH instrument allocation files (LLAMAS + LDSS3 are
    parallel budgets). ``used`` is the reconciled observed time; it is 0 until a
    post-night reconciliation records observations, so ``remaining == initial``
    in the fresh demo. Returns {instruments, programs: {prog: {inst: {...}}}}."""
    import yaml
    instruments = list(_ALLOC_FILES)
    programs: dict = {}
    for inst, path in _ALLOC_FILES.items():
        try:
            with open(path) as f:
                data = yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning("could not load %s allocations: %s", inst, e)
            continue
        for row in data.get("programs", []) or []:
            prog = row.get("program")
            if not prog:
                continue
            hours = row.get("allocated_hours", {}) or {}
            initial = round(float(sum(hours.values())), 2)
            used = 0.0  # TODO: populate from post-night reconciliation ledger
            programs.setdefault(prog, {})[inst] = {
                "initial": initial, "used": round(used, 2),
                "remaining": round(initial - used, 2),
                "pi": row.get("pi", ""),
            }
    # ``tracked`` is False until a post-night reconciliation ledger populates
    # ``used``. The dashboard uses it to label the bars honestly ("not tracked
    # yet") instead of presenting a hardcoded 0 as a live figure.
    return {"instruments": instruments, "programs": programs, "tracked": False}


def load_nights() -> list:
    """The known MAGNETS observing calendar: one entry per night with its
    date, instrument, observer, program and length. Sorted by date. Returns an
    empty list if the calendar file is missing/unreadable (the dashboard then
    just shows the single requested night)."""
    try:
        import yaml
        with open(_NIGHTS_YAML) as f:
            data = yaml.safe_load(f) or {}
        nights = data.get("nights", []) or []
        return sorted(nights, key=lambda n: str(n.get("date", "")))
    except Exception as e:
        logger.warning("could not load observing calendar: %s", e)
        return []

# API priority tier -> (orchestrator integer priority, mandatory?). P0 is the
# "observe tonight" guarantee: top ordinary priority PLUS the mandatory
# reservation the orchestrator already implements. P1>P2>P3 are ordinary.
_TIER_MAP = {"P0": (1, True), "P1": (1, False), "P2": (2, False),
             "P3": (3, False), "P4": (4, False), "P5": (5, False)}


def _moon_phase_for(date: str) -> str:
    """Derive dark/grey/bright from the date's lunar illumination (astropy).
    The submitter never supplies this — it's a property of the night."""
    try:
        from astropy.time import Time
        from astropy.coordinates import get_body, get_sun
        import astropy.units as u
        t = Time(date) + 0.5 * u.day  # local midnight-ish
        elong = get_sun(t).separation(get_body("moon", t)).radian
        illum = (1 - math.cos(elong)) / 2
        return "dark" if illum < 0.25 else "grey" if illum < 0.65 else "bright"
    except Exception as e:
        logger.warning("moon derivation failed (%s); assuming grey", e)
        return "grey"


def _materialize_csv(targets, path: str) -> None:
    cols = ["name", "ra", "dec", "priority", "mag", "redshift",
            "exposure", "program", "keywords", "notes",
            "airmass_min", "airmass_max"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i, t in enumerate(targets):
            pri, mandatory = _TIER_MAP.get(t.priority, (3, False))
            w.writerow({
                "name": t.name or f"{t.program}-{t.id}",
                "ra": f"{t.canonical_ra:.6f}",
                "dec": f"{t.canonical_dec:.6f}",
                "priority": pri,
                "mag": "" if not math.isfinite(t.mag) else f"{t.mag:.3f}",
                "redshift": "" if not math.isfinite(t.redshift) else f"{t.redshift:.4f}",
                "exposure": "" if not math.isfinite(t.exposure_minutes)
                            else f"{t.exposure_minutes:.1f}",
                "program": t.program,
                "keywords": "mandatory" if mandatory else "",
                "notes": t.notes or "",
                # per-target airmass range (stamped #5) -> hard planner constraint
                "airmass_min": "" if not math.isfinite(t.airmass_min)
                               else f"{t.airmass_min:.2f}",
                "airmass_max": "" if not math.isfinite(t.airmass_max)
                               else f"{t.airmass_max:.2f}",
            })


def _tier_of(target, id_to_tier) -> str:
    return id_to_tier.get((target.program, target.name), "P?")


def _night_local(evening, morning):
    """Convert the night's twilight Times to Las Campanas local clock time
    (America/Santiago, so DST is handled: UTC-4 in winter, -3 in summer).
    Returns (start_local, end_local, tz_offset_label) — labels like 'UTC-4'."""
    def conv(t):
        if t is None:
            return None, None
        try:
            from datetime import timezone
            from zoneinfo import ZoneInfo
            dt = t.datetime.replace(tzinfo=timezone.utc).astimezone(
                ZoneInfo("America/Santiago"))
            off = int(dt.utcoffset().total_seconds() // 3600)
            return dt.strftime("%H:%M"), f"UTC{off:+d}"
        except Exception:
            return None, None
    start_local, tz = conv(evening)
    end_local, _ = conv(morning)
    return start_local, end_local, tz


def _empty_night(date: str, moon: str, instrument: str, cfg) -> dict:
    """Plan dict for a night with nothing queued for this instrument: real
    twilight/dark-hours from the site geometry, everything else zeroed."""
    twi_start = twi_end = None
    local_start = local_end = tz_off = None
    dark_h = 0.0
    try:
        from orchestrator.planner import calculate_twilight
        evening, morning = calculate_twilight(date, config=cfg)
        if evening is not None:
            twi_start = evening.datetime.strftime("%H:%M")
        if morning is not None:
            twi_end = morning.datetime.strftime("%H:%M")
        if evening is not None and morning is not None:
            dark_h = round((morning - evening).to_value("hr"), 2)
        local_start, local_end, tz_off = _night_local(evening, morning)
    except Exception as e:
        logger.warning("could not compute twilight for %s: %s", date, e)
    return {
        "date": date, "moon": moon, "instrument": instrument,
        "twilight_start": twi_start, "twilight_end": twi_end,
        "twilight_start_local": local_start, "twilight_end_local": local_end,
        "tz_offset": tz_off,
        "dark_hours": dark_h,
        "requested_hours": {}, "scheduled_hours": {},
        "scheduled_science_hours": 0.0, "n_scheduled": 0,
        "timeline": [], "overflow": [], "note": "nothing queued for this instrument yet",
    }


def preview_plan(service, date: str, moon: str = None,
                 instrument: str = "LLAMAS") -> dict:
    """Run the orchestrator over the live queue for one instrument; return the
    preview dict. LLAMAS and LDSS3 are parallel systems: only this instrument's
    targets (plus EITHER) are scheduled, with this instrument's overhead."""
    from orchestrator.run_nightly import run_nightly
    from orchestrator.config import LLAMASConfig

    instrument = (instrument or "LLAMAS").upper()
    active = service.active(instrument=instrument)
    moon = moon or _moon_phase_for(date)
    # Slit acquisition dominates LDSS3 per-target overhead; IFU is ~1 min.
    cfg = LLAMASConfig(overhead_minutes=10.0) if instrument == "LDSS3" else LLAMASConfig()

    if not active:
        # Nothing queued for this instrument yet, but the night still exists:
        # return its geometry (twilight, dark hours) with zeroed tallies so the
        # dashboard shows the night rather than "undefined".
        return _empty_night(date, moon, instrument, cfg)

    # requested hours per program (before scheduling), restricted to THIS
    # instrument so 'requested' reconciles with the per-instrument 'scheduled'.
    summary = service.queue_summary(instrument)
    requested = {p: v["requested_hours"] for p, v in summary.items()}

    # name/program -> tier, to tag timeline entries with the submitted tier
    id_to_tier = {}
    for t in active:
        key_name = t.name or f"{t.program}-{t.id}"
        id_to_tier[(t.program, key_name)] = t.priority

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = str(Path(tmp) / "queue.csv")
        _materialize_csv(active, csv_path)
        plan = run_nightly(
            date=date, candidates_path=csv_path,
            allocations_path=service.allocations_path,
            moon_phase=moon, output_dir=tmp, from_rubinalerts=False,
            config=cfg,
        )

    def _hhmm(t):
        return t.datetime.strftime("%H:%M") if t is not None else "--:--"

    timeline = []
    for e in plan.scheduled:
        tgt = e.target
        timeline.append({
            "utc": f"{_hhmm(e.start)}-{_hhmm(e.end)}",
            "target": tgt.name,
            "program": e.program or tgt.program,
            "tier": _tier_of(tgt, id_to_tier),
            "ra": round(tgt.ra_deg, 4), "dec": round(tgt.dec_deg, 4),
            "mag": None if not math.isfinite(tgt.mag) else round(tgt.mag, 1),
            "exp_min": None if not math.isfinite(e.charged_minutes)
                       else round(e.charged_minutes, 0),
            "airmass": None if not math.isfinite(e.airmass) else round(e.airmass, 2),
        })

    overflow = []
    for tgt in plan.backup:
        overflow.append({
            "target": tgt.name, "program": tgt.program,
            "tier": _tier_of(tgt, id_to_tier),
            "reason": "below the line for tonight (time/priority)",
        })
    for tgt in getattr(plan, "unschedulable_mandatory", []):
        overflow.append({
            "target": tgt.name, "program": tgt.program,
            "tier": _tier_of(tgt, id_to_tier),
            "reason": "P0 guaranteed but not observable tonight (never reaches airmass limit)",
        })

    sci_hours = sum(e.charged_minutes for e in plan.scheduled
                    if math.isfinite(e.charged_minutes)) / 60.0

    def _iso(t):
        return t.datetime.strftime("%H:%M") if t is not None else None
    twi_start, twi_end = _iso(plan.evening_twilight), _iso(plan.morning_twilight)
    local_start, local_end, tz_off = _night_local(
        plan.evening_twilight, plan.morning_twilight)
    dark_h = None
    if plan.evening_twilight is not None and plan.morning_twilight is not None:
        dark_h = round((plan.morning_twilight - plan.evening_twilight).to_value("hr"), 2)

    # per-program scheduled hours (what actually made the plan)
    sched_by_prog: dict = {}
    for e in plan.scheduled:
        if math.isfinite(e.charged_minutes):
            sched_by_prog[e.program] = sched_by_prog.get(e.program, 0.0) + e.charged_minutes / 60.0
    sched_by_prog = {k: round(v, 2) for k, v in sched_by_prog.items()}

    return {
        "date": date, "moon": moon, "instrument": instrument,
        "twilight_start": twi_start, "twilight_end": twi_end,
        "twilight_start_local": local_start, "twilight_end_local": local_end,
        "tz_offset": tz_off,
        "dark_hours": dark_h,
        "requested_hours": requested,
        "scheduled_hours": sched_by_prog,
        "scheduled_science_hours": round(sci_hours, 2),
        "n_scheduled": len(plan.scheduled),
        "timeline": timeline,
        "overflow": overflow,
    }


# Las Campanas Observatory (Magellan) — the site both LLAMAS and LDSS3 sit on.
_LCO = None


def _lco_location():
    """Cached EarthLocation for LCO (astropy import is deferred)."""
    global _LCO
    if _LCO is None:
        from astropy.coordinates import EarthLocation
        import astropy.units as u
        _LCO = EarthLocation(lat=-29.00833 * u.deg, lon=-70.68167 * u.deg,
                             height=2380 * u.m)
    return _LCO


def airmass_grid(service, plan: dict, date: str, instrument: str,
                 airmass_limit: float = 1.6, step_min: int = 10):
    """Per-target airmass tracks over the night, on a fixed time grid.

    Ported from the Aug-7 capture: builds a ``step_min``-spaced grid between
    the plan's evening and morning twilight, transforms every active target
    (for this instrument) into AltAz at each grid point, and returns
    (grid_labels, targets) where each target carries its sec(z) track (None
    above airmass 3 or below the horizon), scheduled/overflow/queued status,
    exposure estimate and scheduled UTC slot.

    IERS auto-download is disabled and ``auto_max_age`` set to None so the
    computation works fully offline for near-future (2026) dates.
    """
    import numpy as np
    from astropy.coordinates import AltAz, SkyCoord
    from astropy.time import Time
    from astropy.utils import iers
    import astropy.units as u

    from .service import estimate_exposure_minutes

    iers.conf.auto_download = False
    iers.conf.auto_max_age = None

    twi_start = plan.get("twilight_start")
    twi_end = plan.get("twilight_end")
    active = service.active(instrument=instrument)
    if not (twi_start and twi_end):
        # No usable night window; still return targets without tracks.
        sched = {e["target"] for e in plan.get("timeline", [])}
        over = {o["target"] for o in plan.get("overflow", [])}
        targets = [_target_row(t, [], plan, sched, over, estimate_exposure_minutes,
                               labels=[], airmass_limit=airmass_limit)
                   for t in active]
        return [], targets

    t0 = Time(f"{date}T{twi_start}:00")
    t1 = Time(f"{date}T{twi_end}:00")
    if t1 <= t0:
        t1 = t1 + 1 * u.day
    n = int(round((t1 - t0).to_value("min") / step_min)) + 1
    grid = t0 + np.arange(n) * step_min * u.min
    labels = [g.datetime.strftime("%H:%M") for g in grid]
    frame = AltAz(obstime=grid, location=_lco_location())

    sched = {e["target"] for e in plan.get("timeline", [])}
    over = {o["target"] for o in plan.get("overflow", [])}

    if not active:
        return labels, []

    # Vectorized airmass: transform ALL targets over the whole grid in ONE
    # AltAz call (coords shape (N,1) broadcast against the (M,) time grid ->
    # (N,M)) instead of a per-target transform. astropy's frame setup dominates,
    # so one call is far faster than N.
    ras = np.array([t.canonical_ra for t in active], dtype=float)
    decs = np.array([t.canonical_dec for t in active], dtype=float)
    aa = SkyCoord(ras[:, None] * u.deg,
                  decs[:, None] * u.deg).transform_to(frame)
    alt = np.atleast_2d(aa.alt.deg)
    secz = np.atleast_2d(aa.secz.value)

    targets = []
    for i, t in enumerate(active):
        am = [round(float(s), 2) if (a > 0 and 0 < s <= 3.0) else None
              for a, s in zip(alt[i], secz[i])]
        targets.append(_target_row(t, am, plan, sched, over,
                                   estimate_exposure_minutes,
                                   labels=labels, airmass_limit=airmass_limit))
    return labels, targets


def _obs_window(airmass, labels, limit) -> dict:
    """Turn an airmass track into an observability window, so the plan can show
    the observer their *freedom* (when a target is up, when it's best, when it
    leaves) instead of a single pinned clock slot. This is the flexibility the
    MAGNETS group asked for: bound + rank, don't dictate the minute.

    Returns start/end/best labels, the minimum airmass, and a human note +
    category flag: 'early' (sets during the night → do it first), 'late' (rises
    late), 'flexible' (up most of the night), or 'none' (never above the limit).
    """
    n = len(labels)
    obs = [i for i, a in enumerate(airmass) if a is not None and a <= limit]
    if not obs:
        finite = [a for a in airmass if a is not None]
        return {"obs_start": None, "obs_end": None, "obs_best": None,
                "min_airmass": (round(min(finite), 2) if finite else None),
                "window_note": f"never reaches airmass {limit} tonight",
                "window_flag": "none"}
    first, last = obs[0], obs[-1]
    best = min(obs, key=lambda i: airmass[i])
    sets_during = last < n - 1
    rises_during = first > 0
    if sets_during and last < n / 2:
        note, flag = f"observe early — sets ~{labels[last]}", "early"
    elif sets_during:
        note, flag = f"sets ~{labels[last]}", "early"
    elif rises_during and first > n / 2:
        note, flag = f"rises ~{labels[first]}", "late"
    else:
        note, flag = "up most of the night", "flexible"
    return {"obs_start": labels[first], "obs_end": labels[last],
            "obs_best": labels[best], "min_airmass": round(airmass[best], 2),
            "window_note": note, "window_flag": flag}


def _target_row(t, airmass, plan, sched, over, estimate_exposure_minutes,
                labels=None, airmass_limit=1.6) -> dict:
    """One dashboard target record (matches the frontend's expected shape)."""
    status = ("scheduled" if t.name in sched
              else "overflow" if t.name in over else "queued")
    sched_utc = next((e["utc"] for e in plan.get("timeline", [])
                      if e["target"] == t.name), None)
    window = _obs_window(airmass, labels or [], airmass_limit)
    return {
        "program": t.program,
        "name": t.name,
        "tier": t.priority,
        "ra": round(t.canonical_ra, 4),
        "dec": round(t.canonical_dec, 4),
        "mag": None if not math.isfinite(t.mag) else round(t.mag, 1),
        "redshift": None if not math.isfinite(t.redshift) else round(t.redshift, 4),
        "resolved_from": t.resolved_from,
        "exp_est": round(estimate_exposure_minutes(t.mag, t.redshift)),
        "exposure_minutes": None if not math.isfinite(t.exposure_minutes)
                            else round(t.exposure_minutes, 1),
        "n_exposures": t.n_exposures,
        "exposure_seconds": None if not math.isfinite(t.exposure_seconds)
                            else round(t.exposure_seconds),
        "airmass_min": None if not math.isfinite(t.airmass_min)
                       else round(t.airmass_min, 2),
        "airmass_max": None if not math.isfinite(t.airmass_max)
                       else round(t.airmass_max, 2),
        "sched_utc": sched_utc,
        "status": status,
        "instrument": t.instrument,
        "airmass": airmass,
        **window,
    }


def _compute_dashboard(service, date, instrument, airmass_limit) -> dict:
    """The expensive, caller-independent dashboard payload: the live plan
    preview + per-target airmass tracks + queue/programs/nights/allocations.
    Separated out so dashboard_data can cache it (this is the part that's slow —
    the full scheduler run plus an AltAz transform per target over the grid)."""
    plan = service.plan_preview(date, None, instrument)
    queue = service.queue_summary()
    grid, targets = airmass_grid(service, plan, date, instrument, airmass_limit)

    # program metadata: seed descriptions where available, else a stub.
    try:
        from .seed import load_seed_data
        seed_meta = load_seed_data().get("programs", {})
    except Exception:
        seed_meta = {}
    programs = {}
    for prog in sorted(set(service._programs.values())):
        programs[prog] = seed_meta.get(
            prog, {"kind": "manual", "science": prog})

    # Full shared queue across BOTH instruments (the queue section splits on
    # instrument since LLAMAS/LDSS3 are parallel systems). `targets` above is
    # only the selected night's instrument (it carries airmass tracks); this is
    # the lightweight all-instrument list. Status is tonight's plan outcome for
    # the selected instrument; other-instrument rows are "queued".
    sched = {e["target"] for e in plan.get("timeline", [])}
    over = {o["target"] for o in plan.get("overflow", [])}
    queue_targets = [{
        "id": t.id, "name": t.name, "program": t.program,
        "tier": t.priority, "instrument": t.instrument,
        "mag": None if not math.isfinite(t.mag) else round(t.mag, 1),
        "exposure_minutes": None if not math.isfinite(t.exposure_minutes)
                            else round(t.exposure_minutes, 1),
        "n_exposures": t.n_exposures,
        "exposure_seconds": None if not math.isfinite(t.exposure_seconds)
                            else round(t.exposure_seconds),
        "airmass_min": None if not math.isfinite(t.airmass_min)
                       else round(t.airmass_min, 2),
        "airmass_max": None if not math.isfinite(t.airmass_max)
                       else round(t.airmass_max, 2),
        "status": ("scheduled" if t.name in sched
                   else "overflow" if t.name in over else "queued"),
    } for t in service.active()]

    return {
        "plan": plan,
        "queue": queue,
        "targets": targets,
        "queue_targets": queue_targets,
        "grid": grid,
        "airmass_limit": airmass_limit,
        "programs": programs,
        "nights": load_nights(),
        "selected_night": {"date": date, "instrument": instrument},
        "allocations": load_allocations_overview(),
    }


def dashboard_data(service, date: str, instrument: str = "LLAMAS",
                   caller_program: Optional[str] = None,
                   airmass_limit: float = 1.6) -> dict:
    """The full aggregate the web dashboard renders in one request.

    CACHED per ``(date, instrument, airmass_limit)`` on the service instance and
    invalidated by the queue's revision counter, so the expensive plan+airmass
    computation runs only when the queue actually changes — not on every page
    load, refresh, or night switch. ``caller_program`` (which rows are "mine")
    is applied cheaply on top of the cached, caller-independent payload.
    """
    cache = getattr(service, "_dash_cache", None)
    if cache is None:
        cache = {}
        try:
            service._dash_cache = cache
        except Exception:
            cache = None                      # can't stash a cache; recompute
    rev = service.revision() if hasattr(service, "revision") else None
    key = (date, instrument, round(airmass_limit, 3))

    payload = None
    if cache is not None and rev is not None:
        hit = cache.get(key)
        if hit is not None and hit[0] == rev:
            payload = hit[1]
    if payload is None:
        payload = _compute_dashboard(service, date, instrument, airmass_limit)
        if cache is not None and rev is not None:
            cache[key] = (rev, payload)

    return {**payload, "caller_program": caller_program}
