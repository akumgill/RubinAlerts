"""Target loading, normalization, and exposure estimation."""

import logging
import re
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

from .models import Target
from .config import LLAMAS_CONFIG, LLAMASConfig
# Controlled keyword vocabulary lives in prioritizer (single source of truth for
# scoring). No import cycle: prioritizer never imports normalize.
from .prioritizer import KEYWORD_WEIGHTS, OVERRIDE_KEYWORDS

logger = logging.getLogger(__name__)


def parse_keywords(raw, target_name: str = '') -> tuple:
    """Parse a free-text ``keywords`` cell into structured, validated tags.

    Tokens are split on commas or semicolons, normalized (strip, lowercase,
    spaces -> underscores), then validated against the controlled vocabulary
    (KEYWORD_WEIGHTS keys ∪ OVERRIDE_KEYWORDS). Unknown tokens are dropped with
    a warning. Override tokens ('override'/'mandatory') set the mandatory flag
    rather than becoming scored tags.

    Returns
    -------
    (keywords, mandatory) : tuple of (list of str, bool)
        ``keywords`` is the recognized NON-override tags; ``mandatory`` is True
        if any override token was present.
    """
    if raw is None:
        return [], False
    text = str(raw).strip()
    if not text:
        return [], False

    valid = set(KEYWORD_WEIGHTS) | set(OVERRIDE_KEYWORDS)
    keywords = []
    mandatory = False
    for tok in re.split(r'[,;]', text):
        norm = tok.strip().lower().replace(' ', '_')
        if not norm:
            continue
        if norm in OVERRIDE_KEYWORDS:
            mandatory = True
        elif norm in KEYWORD_WEIGHTS:
            keywords.append(norm)
        else:
            logger.warning(
                "Target %s: dropping unknown keyword %r (not in controlled "
                "vocabulary)", target_name or '?', norm)
    return keywords, mandatory


def parse_coordinates(ra_str: str, dec_str: str) -> tuple:
    """Parse RA/Dec strings to degrees.

    Accepts sexagesimal (contains ':' or 'h') or decimal degree strings.

    Returns
    -------
    (ra_deg, dec_deg) : tuple of float
    """
    ra_str = str(ra_str).strip()
    dec_str = str(dec_str).strip()

    if ':' in ra_str or 'h' in ra_str:
        coord = SkyCoord(ra=ra_str, dec=dec_str, unit=(u.hourangle, u.deg))
        return coord.ra.deg, coord.dec.deg
    else:
        return float(ra_str), float(dec_str)


def parse_magnitude(mag_str: str) -> tuple:
    """Extract numeric magnitude and filter name from a string.

    Examples
    --------
    >>> parse_magnitude("19.5, r-band")
    (19.5, 'r')
    >>> parse_magnitude("20.1 ZTF-g")
    (20.1, 'g')
    >>> parse_magnitude("21.3")
    (21.3, '')

    Returns
    -------
    (mag, filter_name) : tuple of (float, str)
    """
    mag_str = str(mag_str).strip()
    mag_match = re.match(r'([0-9]+\.?[0-9]*)', mag_str)
    if not mag_match:
        return float('nan'), ''

    mag = float(mag_match.group(1))

    # Look for filter name after the number
    # Try explicit band patterns first (e.g., "r-band", "ZTF-g", "g-band")
    remainder = mag_str[mag_match.end():]
    # Match standalone filter letter: preceded by '-', space, or start; not inside a word
    filt_match = re.search(r'(?:^|[\s,\-])([ugrizy])(?:\b|-)', remainder, re.IGNORECASE)
    filt = filt_match.group(1).lower() if filt_match else ''

    return mag, filt


def predicted_mag_at_observation(peak_mag: float, delta_t: float,
                                 config: LLAMASConfig = None) -> float:
    """Apparent mag at the observation epoch, from the light-curve peak.

    A SN observed at phase ``delta_t`` (signed days from peak, at the observation
    night) is fainter than its peak: rising (delta_t<0) and declining (delta_t>0)
    both add magnitudes. Uses coarse optical fade rates from ``config`` (not a
    template) — adequate for exposure sizing. Returns ``peak_mag`` unchanged when
    either input is non-finite (e.g. the manual path, where mag is already the
    anticipated-at-observation value and delta_t is NaN)."""
    if config is None:
        config = LLAMAS_CONFIG
    if not (math.isfinite(peak_mag) and math.isfinite(delta_t)):
        return peak_mag
    rate = (getattr(config, 'mag_rise_per_day', 0.0) if delta_t < 0
            else getattr(config, 'mag_decline_per_day', 0.0))
    fade = min(abs(delta_t) * rate, getattr(config, 'mag_fade_cap', 3.0))
    return peak_mag + fade


def estimate_llamas_exposure(redshift: float, mag: float, moon: str = 'grey',
                             config: LLAMASConfig = None,
                             delta_t: float = float('nan')) -> tuple:
    """Estimate LLAMAS exposure time from target properties.

    ``mag`` is the light-curve peak; with a finite ``delta_t`` (phase at the
    observation night) it is faded to the mag AT OBSERVATION before sizing, so a
    target scheduled off peak isn't under-exposed. delta_t NaN -> mag used as-is.

    Strategy:
    1. If redshift is finite, interpolate proposal Table 1 (z -> minutes).
    2. Else if mag is finite, scale from reference (mag=20 -> 45 min).
    3. Else return fallback.

    The redshift branch LINEARLY INTERPOLATES between the discrete
    (z, minutes) rows of ``config.exposure_table`` instead of stepping
    between them. A bare lookup put a ~85-min cliff between adjacent rows
    (e.g. z=0.30 -> 95 min, z=0.301 -> 100 min); interpolation removes that.
    Below the lowest tabulated z the first row's value is used; above the
    highest, the last row's value is clamped (no runaway extrapolation).

    NOTE: this orchestrator exposure is intentionally PROPOSAL-TABLE driven
    and deliberately OMITS explicit moon/airmass scaling — those factors are
    baked into the proposal's per-z budgets (and airmass is handled by the
    observability/scheduling stage). This is the real divergence from the
    alert-pipeline estimate (core.magellan_planning), which models moon and
    airmass explicitly; it is documented here on purpose, not a bug to unify.

    Returns
    -------
    (exposure_minutes, moon_constraint) : tuple of (float, str)
    """
    if config is None:
        config = LLAMAS_CONFIG

    # Fade the peak mag to the observation epoch (no-op when delta_t is NaN).
    mag = predicted_mag_at_observation(mag, delta_t, config)

    # Strategy 0 (primary, if enabled): S/N-based ETC from Chris's LLAMAS SN Ia
    # curve — magnitude-driven, so it does not need a redshift and cleanly sizes
    # redshift-unknown candidates. Floored at snr_min_minutes; a moon multiplier
    # accounts for the curve being a dark-time calibration. Falls through to the
    # proposal-table / mag-scaling cascade if disabled or mag is unavailable.
    if getattr(config, 'use_snr_etc', False) and math.isfinite(mag):
        try:
            from core.snr_etc import snr_exposure_minutes, MAX_EXPOSURE_MIN
            t, _extrap = snr_exposure_minutes(
                mag, target_snr=config.snr_target_binned, n_bin=config.snr_binning)
            if math.isfinite(t):
                moon_mult = config.snr_moon_factor.get(moon, 1.0)
                t = max(config.snr_min_minutes, t * moon_mult)
                t = min(MAX_EXPOSURE_MIN, t)
                logger.debug("mag=%.1f -> %.1f min (S/N-ETC, binned S/N=%.0f, "
                             "moon=%s)", mag, t, config.snr_target_binned, moon)
                return float(t), moon
        except Exception as e:
            logger.debug("S/N-ETC unavailable (%s); falling back to cascade", e)

    # Strategy 1: redshift-based interpolation of proposal Table 1.
    if math.isfinite(redshift):
        table = sorted(config.exposure_table, key=lambda row: row[0])
        zs = [row[0] for row in table]
        mins = [row[1] for row in table]
        # numpy.interp clamps to the endpoints outside [zs[0], zs[-1]].
        exp_min = float(np.interp(redshift, zs, mins))
        # Moon constraint is a discrete label, so interpolation is meaningless:
        # use the constraint of the first row whose z-ceiling covers this
        # redshift (clamping to the last row beyond the table).
        constraint = table[-1][2]
        for max_z, _, c in table:
            if redshift <= max_z:
                constraint = c
                break
        logger.debug("z=%.3f -> %.1f min (%s, interpolated)",
                     redshift, exp_min, constraint)
        return exp_min, constraint

    # Strategy 2: magnitude-based scaling
    if math.isfinite(mag):
        # Reference: mag 20 -> 45 min, each magnitude -> 2.5x
        mag_diff = mag - 20.0
        exp_min = 45.0 * (2.5 ** mag_diff)
        exp_min = max(10.0, min(exp_min, 180.0))
        logger.debug("mag=%.1f -> %.0f min", mag, exp_min)
        return exp_min, moon if moon != 'any' else 'grey'

    # Strategy 3: fallback
    return config.fallback_exposure_minutes, 'any'


# Gaussian time-weight scale (days). Matches the alert-pipeline merit
# w_time = exp(-delta_t² / (2·tau²)) with tau=10 d (core.magellan_planning),
# so manual targets with a peak_mjd are weighted on the same footing.
PHASE_WEIGHT_TAU_DAYS = 10.0


def phase_weight_from_peak(peak_mjd: float, night_mjd: float,
                           tau_days: float = PHASE_WEIGHT_TAU_DAYS) -> float:
    """Gaussian phase weight from time-since-peak, matching the alert pipeline.

    Reuses the same form as core.magellan_planning's w_time
    (exp(-delta_t² / (2·tau²)), tau=10 d) so a manually-entered peak_mjd
    scores consistently with alert-sourced targets.

    Returns float('nan') if either input is non-finite.
    """
    if not (math.isfinite(peak_mjd) and math.isfinite(night_mjd)):
        return float('nan')
    delta_t = night_mjd - peak_mjd
    return math.exp(-delta_t ** 2 / (2.0 * tau_days ** 2))


def load_primary_program(nights_path: str, date: str) -> Optional[str]:
    """Look up the designated PRIMARY program for an observing night.

    Reads a small CSV with columns ``date, primary_program`` (and an optional
    human-readable ``primary_observer`` column, ignored by the logic). The row
    whose ``date`` equals ``date`` (string match on ISO ``YYYY-MM-DD``) gives
    tonight's primary program. Only that primary's must-see (override) targets
    are guaranteed scheduling; everyone else's targets — and the primary's
    *unmarked* targets — go through normal prioritization.

    Tolerates ``#`` comment lines and surrounding whitespace in cells.

    Parameters
    ----------
    nights_path : str
        Path to the observing-nights CSV.
    date : str
        Observing date YYYY-MM-DD to match.

    Returns
    -------
    Optional[str]
        The ``primary_program`` for ``date``, or None if the file is absent or
        has no row for that date (logged).
    """
    if not Path(nights_path).exists():
        logger.warning("Observing-nights file not found: %s "
                       "(no primary program — all must-see targets honored)",
                       nights_path)
        return None

    df = pd.read_csv(nights_path, comment='#', skipinitialspace=True)
    df.columns = [c.strip().lower() for c in df.columns]

    if 'date' not in df.columns or 'primary_program' not in df.columns:
        logger.warning("Observing-nights file %s missing date/primary_program "
                       "columns; ignoring", nights_path)
        return None

    target_date = str(date).strip()
    for _, row in df.iterrows():
        if str(row['date']).strip() == target_date:
            primary = str(row['primary_program']).strip()
            if primary:
                logger.info("Primary program for %s: %s", target_date, primary)
                return primary

    logger.warning("No primary program listed for %s in %s "
                   "(all must-see targets honored)", target_date, nights_path)
    return None


def load_targets_csv(path: str, night_mjd: float = float('nan'),
                     default_program: str = 'default') -> list:
    """Load targets from a user-provided CSV file.

    Expected columns: name, ra, dec (required); priority, mag, filter,
    redshift, exposure, moon, notes (optional). Optional manual-workflow
    columns also honored: ``program`` (per-PI accounting), ``phase_weight``
    (w_time, used directly), and ``peak_mjd`` (converted to a phase weight via
    :func:`phase_weight_from_peak` when ``night_mjd`` is supplied and
    phase_weight is absent).

    Coordinates can be sexagesimal (HH:MM:SS / +DD:MM:SS) or decimal degrees.

    Parameters
    ----------
    path : str
        Path to the target CSV.
    night_mjd : float
        MJD of the observing night, used only to convert a ``peak_mjd`` column
        into a phase weight. If non-finite (default), peak_mjd is ignored and
        phase stays neutral (a warning is logged per affected row).
    default_program : str
        Program assigned to rows lacking an explicit ``program`` column/value.
        A warning is logged for each target charged to the default so it is
        never silently mis-attributed.

    Returns
    -------
    list of Target
    """
    df = pd.read_csv(path, comment='#')
    df.columns = [c.strip().lower() for c in df.columns]

    required = {'name', 'ra', 'dec'}
    if not required.issubset(set(df.columns)):
        missing = required - set(df.columns)
        raise ValueError(f"CSV missing required columns: {missing}")

    targets = []
    for _, row in df.iterrows():
        try:
            ra_deg, dec_deg = parse_coordinates(str(row['ra']), str(row['dec']))
        except Exception as e:
            logger.warning("Skipping %s: coordinate parse error: %s", row['name'], e)
            continue

        mag_val = float('nan')
        mag_filt = ''
        if 'mag' in df.columns and pd.notna(row.get('mag')):
            raw = str(row['mag'])
            mag_val, mag_filt = parse_magnitude(raw)
        if 'filter' in df.columns and pd.notna(row.get('filter')):
            mag_filt = str(row['filter']).strip()

        name = str(row['name']).strip()

        # Optional program column for per-PI accounting. Absent/blank -> default,
        # but warn so manual targets are never silently mis-attributed.
        program = default_program
        if 'program' in df.columns and pd.notna(row.get('program')) \
                and str(row['program']).strip():
            program = str(row['program']).strip()
        else:
            logger.warning(
                "Target %s has no program; charging to default program %r",
                name, default_program)

        # Optional phase weight. An explicit phase_weight wins; otherwise derive
        # it from peak_mjd (needs night_mjd). Neither given -> neutral (NaN),
        # which _phase_factor treats as 1.0 (do NOT fabricate near-peak).
        phase_w = float('nan')
        if 'phase_weight' in df.columns and pd.notna(row.get('phase_weight')):
            phase_w = float(row['phase_weight'])
        elif 'peak_mjd' in df.columns and pd.notna(row.get('peak_mjd')):
            peak_mjd = float(row['peak_mjd'])
            if math.isfinite(night_mjd):
                phase_w = phase_weight_from_peak(peak_mjd, night_mjd)
            else:
                logger.warning(
                    "Target %s has peak_mjd but no night date was supplied; "
                    "leaving phase neutral", name)

        # Signed time-from-peak (days) for per-program phase preference. An
        # explicit delta_t column wins; otherwise derive it from peak_mjd and a
        # finite night_mjd (delta_t = night - peak). Absent -> NaN (neutral).
        delta_t = float('nan')
        if 'delta_t' in df.columns and pd.notna(row.get('delta_t')):
            delta_t = float(row['delta_t'])
        elif ('peak_mjd' in df.columns and pd.notna(row.get('peak_mjd'))
                and math.isfinite(night_mjd)):
            delta_t = night_mjd - float(row['peak_mjd'])

        # Optional structured keyword tags (controlled vocabulary). Comma- or
        # semicolon-separated; unknown tokens dropped with a warning; an
        # override token sets mandatory. Free-text ``notes`` is unaffected.
        keywords, mandatory = [], False
        if 'keywords' in df.columns and pd.notna(row.get('keywords')):
            keywords, mandatory = parse_keywords(row.get('keywords'), name)

        t = Target(
            name=name,
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            priority=int(row.get('priority', 3)) if pd.notna(row.get('priority')) else 3,
            mag=mag_val,
            mag_filter=mag_filt,
            redshift=float(row['redshift']) if 'redshift' in df.columns and pd.notna(row.get('redshift')) else float('nan'),
            exposure_minutes=float(row['exposure']) if 'exposure' in df.columns and pd.notna(row.get('exposure')) else float('nan'),
            moon_constraint=str(row.get('moon', 'any')).strip() if pd.notna(row.get('moon')) else 'any',
            notes=str(row.get('notes', '')).strip() if pd.notna(row.get('notes')) else '',
            keywords=keywords,
            mandatory=mandatory,
            source='csv',
            program=program,
            phase_weight=phase_w,
            delta_t=delta_t,
        )
        targets.append(t)

    logger.info("Loaded %d targets from %s", len(targets), path)
    return targets


def load_from_rubinalerts(path: str, max_targets: int = 30,
                          default_program: str = 'default',
                          program_profiles: dict = None) -> list:
    """Load targets from the RubinAlerts pipeline candidates.csv.

    Maps merit_score to priority via quartiles:
    top quartile = P1, second = P2, third = P3, rest = P4.

    Parameters
    ----------
    path : str
        Path to candidates.csv from the alert pipeline.
    max_targets : int
        Maximum number of targets to return.
    default_program : str
        Program to assign targets to (from allocations.yaml default_program).
    program_profiles : dict, optional
        {program_name: ranking_profile_name} from allocations.yaml. When a
        target's program uses a non-'ia' profile and the candidates CSV
        carries the matching merit_<profile> column, that merit ranks the
        target, and P1-P4 quartiles are computed WITHIN each program's own
        cohort — each program ranks by its own science, so a budget split
        actually binds.

    Returns
    -------
    list of Target
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    # Require finite coordinates and positive merit
    mask = (
        np.isfinite(df.get('ra', pd.Series(dtype=float))) &
        np.isfinite(df.get('dec', pd.Series(dtype=float)))
    )
    if 'merit' in df.columns:
        mask = mask & (df['merit'] > 0) & np.isfinite(df['merit'])
    elif 'merit_score' in df.columns:
        mask = mask & (df['merit_score'] > 0) & np.isfinite(df['merit_score'])

    df = df[mask].copy()

    if df.empty:
        logger.warning("No valid targets after filtering in %s", path)
        return []

    # Map merit to priority. NOTE: P1-P4 here are WITHIN-NIGHT RELATIVE labels
    # (tonight's quartiles), not absolute science classes — a P1 tonight may be
    # weaker than a P4 on a richer night. (R8; the summary header repeats this
    # caveat.) Absolute thresholds were deliberately deferred by the architect.
    merit_col = 'merit' if 'merit' in df.columns else 'merit_score'
    # Per-target ranking merit: the program's own profile column when
    # configured and present, else the shared (Ia) merit.
    if merit_col in df.columns:
        # Ranking merit: prefer the value-density column (merit_rate =
        # merit x (45min/exposure)^alpha) when the pipeline provides it —
        # ranking rewards science per hour, while `merit` itself remains the
        # pure science value for auditability.
        base_rate = 'merit_rate' if 'merit_rate' in df.columns else merit_col
        df['_rank_merit'] = df[base_rate]
        if program_profiles and 'program' in df.columns:
            for prog, prof in program_profiles.items():
                if not prof or prof == 'ia':
                    continue
                col = None
                for cand_col in (f'merit_{prof}_rate', f'merit_{prof}'):
                    if cand_col in df.columns:
                        col = cand_col
                        break
                if col:
                    sel = df['program'].fillna('').astype(str).str.strip() == prog
                    df.loc[sel, '_rank_merit'] = df.loc[sel, col]

    def _quartile_priority(sub):
        """P1-P4 within one cohort; <4 targets falls back to sorted rank
        (quartile bins degenerate) so it cannot collapse every target into
        one tier."""
        n = len(sub)
        if n < 4:
            order = sub.rank(method='first', ascending=False)
            return order.astype(int).clip(upper=4)
        q75, q50, q25 = np.percentile(sub.values, [75, 50, 25])
        return sub.apply(
            lambda m: 1 if m >= q75 else 2 if m >= q50 else 3 if m >= q25 else 4)

    if merit_col in df.columns:
        if program_profiles and 'program' in df.columns:
            # Quartiles WITHIN each program's cohort: a program's best
            # targets are its own P1s, whatever the other program's merit
            # scale looks like.
            grp_key = df['program'].fillna(default_program).astype(str).str.strip()
            df['_priority'] = (
                df.groupby(grp_key)['_rank_merit']
                  .transform(lambda sub: _quartile_priority(sub)))
        else:
            df['_priority'] = _quartile_priority(df['_rank_merit'])
        df['_priority'] = df['_priority'].astype(int)
    else:
        df['_priority'] = 3

    # Sort by ranking merit descending, limit to max_targets
    if merit_col in df.columns:
        df = df.sort_values('_rank_merit', ascending=False)
    df = df.head(max_targets)

    # Resolve name column
    name_col = None
    for col in ('object_id', 'name', 'object_id_alerce_lsst', 'object_id_antares'):
        if col in df.columns:
            name_col = col
            break
    if name_col is None:
        name_col = df.columns[0]

    targets = []
    for _, row in df.iterrows():
        mag_val = float('nan')
        mag_filt = ''
        if 'peak_mag' in df.columns and pd.notna(row.get('peak_mag')):
            mag_val = float(row['peak_mag'])
        elif 'mag' in df.columns and pd.notna(row.get('mag')):
            mag_val, mag_filt = parse_magnitude(str(row['mag']))

        # Phase weight from alert pipeline (w_time = exp(-dt²/2τ²))
        phase_w = float('nan')
        if 'w_time' in df.columns and pd.notna(row.get('w_time')):
            phase_w = float(row['w_time'])

        # Signed time-from-peak (days) from the alert pipeline, used for
        # per-program phase preference. Absent -> NaN (neutral).
        delta_t = float('nan')
        if 'delta_t' in df.columns and pd.notna(row.get('delta_t')):
            delta_t = float(row['delta_t'])

        name = str(row[name_col]).strip()

        # Optional structured keyword tags (same controlled vocabulary as the
        # manual CSV path). Unknown tokens dropped with a warning; override
        # tokens set mandatory.
        keywords, mandatory = [], False
        if 'keywords' in df.columns and pd.notna(row.get('keywords')):
            keywords, mandatory = parse_keywords(row.get('keywords'), name)

        # Optional per-target program (classification-based routing — the
        # alert-target ownership question, operating model (b)): honored when
        # a 'program' column is present and non-blank; otherwise the default
        # program owns alert-stream targets (operating model (a)).
        program = default_program
        if 'program' in df.columns:
            pv = row.get('program')
            if pd.notna(pv) and str(pv).strip():
                program = str(pv).strip()

        t = Target(
            name=name,
            ra_deg=float(row['ra']),
            dec_deg=float(row['dec']),
            priority=int(row['_priority']),
            mag=mag_val,
            mag_filter=mag_filt,
            redshift=float(row['redshift']) if 'redshift' in df.columns and pd.notna(row.get('redshift')) else float('nan'),
            merit_score=float(row['_rank_merit']) if '_rank_merit' in df.columns else float('nan'),
            keywords=keywords,
            mandatory=mandatory,
            source='rubinalerts',
            program=program,
            phase_weight=phase_w,
            delta_t=delta_t,
        )
        targets.append(t)

    logger.info("Loaded %d targets from RubinAlerts pipeline (%s)", len(targets), path)
    return targets
