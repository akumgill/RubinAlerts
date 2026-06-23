"""Target loading, normalization, and exposure estimation."""

import logging
import re
import math

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

from .models import Target
from .config import LLAMAS_CONFIG, LLAMASConfig

logger = logging.getLogger(__name__)


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


def estimate_llamas_exposure(redshift: float, mag: float, moon: str = 'grey',
                             config: LLAMASConfig = None) -> tuple:
    """Estimate LLAMAS exposure time from target properties.

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


def load_targets_csv(path: str) -> list:
    """Load targets from a user-provided CSV file.

    Expected columns: name, ra, dec (required); priority, mag, filter,
    redshift, exposure, moon, notes (optional).

    Coordinates can be sexagesimal (HH:MM:SS / +DD:MM:SS) or decimal degrees.

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

        t = Target(
            name=str(row['name']).strip(),
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            priority=int(row.get('priority', 3)) if pd.notna(row.get('priority')) else 3,
            mag=mag_val,
            mag_filter=mag_filt,
            redshift=float(row['redshift']) if 'redshift' in df.columns and pd.notna(row.get('redshift')) else float('nan'),
            exposure_minutes=float(row['exposure']) if 'exposure' in df.columns and pd.notna(row.get('exposure')) else float('nan'),
            moon_constraint=str(row.get('moon', 'any')).strip() if pd.notna(row.get('moon')) else 'any',
            notes=str(row.get('notes', '')).strip() if pd.notna(row.get('notes')) else '',
            source='csv',
        )
        targets.append(t)

    logger.info("Loaded %d targets from %s", len(targets), path)
    return targets


def load_from_rubinalerts(path: str, max_targets: int = 30,
                          default_program: str = 'default') -> list:
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
    if merit_col in df.columns:
        n = len(df)
        if n < 4:
            # Quartile bins degenerate (np.percentile collapses) with <4
            # targets. Assign P-labels by sorted rank instead so it cannot
            # crash or collapse every target into one tier: best→P1, etc.
            order = df[merit_col].rank(method='first', ascending=False)
            df['_priority'] = order.astype(int).clip(upper=4)
        else:
            q75, q50, q25 = np.percentile(df[merit_col].values, [75, 50, 25])
            def _priority(m):
                if m >= q75:
                    return 1
                elif m >= q50:
                    return 2
                elif m >= q25:
                    return 3
                return 4
            df['_priority'] = df[merit_col].apply(_priority)
    else:
        df['_priority'] = 3

    # Sort by merit descending, limit to max_targets
    if merit_col in df.columns:
        df = df.sort_values(merit_col, ascending=False)
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

        t = Target(
            name=str(row[name_col]).strip(),
            ra_deg=float(row['ra']),
            dec_deg=float(row['dec']),
            priority=int(row['_priority']),
            mag=mag_val,
            mag_filter=mag_filt,
            redshift=float(row['redshift']) if 'redshift' in df.columns and pd.notna(row.get('redshift')) else float('nan'),
            merit_score=float(row[merit_col]) if merit_col in df.columns else float('nan'),
            source='rubinalerts',
            program=default_program,
            phase_weight=phase_w,
        )
        targets.append(t)

    logger.info("Loaded %d targets from RubinAlerts pipeline (%s)", len(targets), path)
    return targets
