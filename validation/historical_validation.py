#!/usr/bin/env python
"""Historical validation of the RubinAlerts pipeline.

Runs the pipeline on a range of historical dates and evaluates performance
by tracking candidates across nights and comparing against available ground truth.

Ground truth sources:
1. TNS cross-match (tns_type contains 'Ia' = confirmed SN Ia)
2. High-confidence ML classifiers (Fink sn_score > 0.9)
3. Multi-night persistence (same object detected multiple nights)
4. Light curve quality (good Villar fit, reasonable peak mag)

Usage:
    python validation/historical_validation.py --start-mjd 61050 --end-mjd 61100 --step 5
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.time import Time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)
logger = logging.getLogger('validation')

# Classification thresholds
HIGH_CONFIDENCE_SN_SCORE = 0.8  # Fink classifier threshold for "likely SN Ia"
GOOD_FIT_QUALITY = 0.5  # Minimum merit score for "good" candidate
NUCLEAR_OFFSET_THRESHOLD = 1.0  # arcsec, below this = likely AGN


def load_candidates_from_night(night_dir: Path) -> pd.DataFrame:
    """Load candidates.csv from a night directory."""
    csv_path = night_dir / 'candidates.csv'
    if not csv_path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(csv_path)
        df['night_dir'] = str(night_dir)
        return df
    except Exception as e:
        logger.warning(f"Failed to load {csv_path}: {e}")
        return pd.DataFrame()


def classify_candidate(row: pd.Series) -> dict:
    """Classify a single candidate based on available signals.

    Returns dict with:
        - classification: 'confirmed_ia', 'likely_ia', 'possible_ia',
                         'likely_contaminant', 'unknown'
        - confidence: float 0-1
        - reasons: list of strings explaining classification
    """
    reasons = []
    score = 0.5  # Start neutral

    # TNS classification (strongest signal)
    tns_type = row.get('tns_type', '')
    if pd.notna(tns_type) and tns_type:
        tns_type_str = str(tns_type).lower()
        if 'ia' in tns_type_str and 'not' not in tns_type_str:
            return {
                'classification': 'confirmed_ia',
                'confidence': 1.0,
                'reasons': [f'TNS spectroscopic classification: {tns_type}']
            }
        elif 'sn' in tns_type_str:
            reasons.append(f'TNS type: {tns_type} (not Ia)')
            score -= 0.3
        else:
            reasons.append(f'TNS type: {tns_type}')

    # Fink SN classifier
    sn_score = row.get('sn_score', np.nan)
    if pd.notna(sn_score):
        if sn_score > HIGH_CONFIDENCE_SN_SCORE:
            reasons.append(f'High Fink sn_score: {sn_score:.2f}')
            score += 0.25
        elif sn_score > 0.5:
            reasons.append(f'Moderate Fink sn_score: {sn_score:.2f}')
            score += 0.1
        elif sn_score < 0.3:
            reasons.append(f'Low Fink sn_score: {sn_score:.2f}')
            score -= 0.2

    # Nuclear offset (AGN/TDE rejection)
    offset_class = row.get('offset_class', '')
    nuclear_offset = row.get('nuclear_offset_arcsec', np.nan)
    if offset_class == 'nuclear' or (pd.notna(nuclear_offset) and nuclear_offset < NUCLEAR_OFFSET_THRESHOLD):
        reasons.append(f'Nuclear offset: {nuclear_offset:.1f}" (likely AGN/TDE)')
        score -= 0.4
    elif offset_class == 'offset':
        reasons.append(f'Offset from nucleus: {nuclear_offset:.1f}" (SN-like)')
        score += 0.1

    # Host morphology
    host_morph = row.get('host_morphology', '')
    if host_morph == 'elliptical':
        reasons.append('Elliptical host (favorable for Ia)')
        score += 0.15
    elif host_morph == 'spiral':
        reasons.append('Spiral host (all SN types possible)')

    # Merit score (combines multiple factors)
    merit = row.get('merit', np.nan)
    if pd.notna(merit):
        if merit > 0.3:
            reasons.append(f'High merit: {merit:.3f}')
            score += 0.1
        elif merit < 0.05:
            reasons.append(f'Low merit: {merit:.3f}')
            score -= 0.1

    # Multi-broker agreement
    num_brokers = row.get('num_brokers', 1)
    if num_brokers >= 3:
        reasons.append(f'Detected by {num_brokers} brokers')
        score += 0.15
    elif num_brokers >= 2:
        reasons.append(f'Detected by {num_brokers} brokers')
        score += 0.05

    # Peak magnitude sanity check
    peak_mag = row.get('peak_mag', np.nan)
    if pd.notna(peak_mag):
        if 17 < peak_mag < 22:
            reasons.append(f'Reasonable peak mag: {peak_mag:.1f}')
        elif peak_mag < 16:
            reasons.append(f'Very bright peak: {peak_mag:.1f} (unusual)')
            score -= 0.1
        elif peak_mag > 23:
            reasons.append(f'Very faint peak: {peak_mag:.1f} (hard to confirm)')
            score -= 0.1

    # Clamp score
    score = np.clip(score, 0, 1)

    # Determine classification
    if score > 0.75:
        classification = 'likely_ia'
    elif score > 0.5:
        classification = 'possible_ia'
    elif score < 0.3:
        classification = 'likely_contaminant'
    else:
        classification = 'unknown'

    return {
        'classification': classification,
        'confidence': score,
        'reasons': reasons
    }


def run_pipeline_for_date(mjd: float, output_base: Path, args) -> Path:
    """Run the pipeline for a specific MJD and return the output directory."""
    from astropy.time import Time

    t = Time(mjd, format='mjd')
    ut_date = t.datetime.strftime('%Y%m%d')
    night_dir = output_base / f'ut{ut_date}'

    if night_dir.exists() and (night_dir / 'candidates.csv').exists():
        logger.info(f"MJD {mjd:.0f} ({ut_date}): Using existing results")
        return night_dir

    logger.info(f"MJD {mjd:.0f} ({ut_date}): Running pipeline...")

    # Build command
    cmd = [
        sys.executable, 'run_tonight.py',
        str(int(mjd)),
        '--min-prob', '0.3',
        '--days-back', '30',
    ]

    if args.no_atlas:
        cmd.append('--no-atlas')
    if args.no_tns:
        cmd.append('--no-tns')
    if args.fink_only:
        cmd.append('--fink-only')

    import subprocess
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.warning(f"Pipeline failed for MJD {mjd}: {result.stderr[-500:]}")
        return None

    return night_dir


def cross_match_candidates(all_candidates: pd.DataFrame,
                           tolerance_arcsec: float = 2.0) -> pd.DataFrame:
    """Cross-match candidates across nights to find persistent objects."""
    if len(all_candidates) == 0:
        return all_candidates

    from astropy.coordinates import SkyCoord
    from astropy import units as u

    # Group by approximate position
    all_candidates = all_candidates.copy()
    all_candidates['object_group'] = -1

    coords = SkyCoord(
        ra=all_candidates['ra'].values * u.deg,
        dec=all_candidates['dec'].values * u.deg
    )

    group_id = 0
    assigned = set()

    for i in range(len(all_candidates)):
        if i in assigned:
            continue

        # Find all matches within tolerance
        sep = coords[i].separation(coords).arcsec
        matches = np.where(sep < tolerance_arcsec)[0]

        for j in matches:
            all_candidates.iloc[j, all_candidates.columns.get_loc('object_group')] = group_id
            assigned.add(j)

        group_id += 1

    # Count nights per object
    nights_per_object = all_candidates.groupby('object_group')['night_dir'].nunique()
    all_candidates['n_nights_detected'] = all_candidates['object_group'].map(nights_per_object)

    return all_candidates


def generate_validation_report(all_candidates: pd.DataFrame,
                               output_path: Path,
                               mjd_start: float, mjd_end: float):
    """Generate a validation report summarizing results."""

    # Classify all candidates
    classifications = all_candidates.apply(classify_candidate, axis=1)
    all_candidates['val_classification'] = [c['classification'] for c in classifications]
    all_candidates['val_confidence'] = [c['confidence'] for c in classifications]
    all_candidates['val_reasons'] = ['; '.join(c['reasons']) for c in classifications]

    # Get unique objects (take highest-confidence detection per object group)
    unique_objects = all_candidates.sort_values('val_confidence', ascending=False).drop_duplicates('object_group')

    # Summary statistics
    t_start = Time(mjd_start, format='mjd')
    t_end = Time(mjd_end, format='mjd')

    report_lines = [
        "=" * 80,
        "RUBINALERTS HISTORICAL VALIDATION REPORT",
        "=" * 80,
        "",
        f"Date range: MJD {mjd_start:.0f} - {mjd_end:.0f}",
        f"           ({t_start.iso[:10]} to {t_end.iso[:10]})",
        f"Total candidate detections: {len(all_candidates)}",
        f"Unique objects (cross-matched): {len(unique_objects)}",
        "",
        "-" * 80,
        "CLASSIFICATION SUMMARY",
        "-" * 80,
    ]

    # Count by classification
    class_counts = unique_objects['val_classification'].value_counts()
    for cls in ['confirmed_ia', 'likely_ia', 'possible_ia', 'unknown', 'likely_contaminant']:
        count = class_counts.get(cls, 0)
        pct = 100 * count / len(unique_objects) if len(unique_objects) > 0 else 0
        marker = "***" if cls == 'confirmed_ia' else "   "
        report_lines.append(f"  {marker} {cls:20s}: {count:4d} ({pct:5.1f}%)")

    report_lines.extend([
        "",
        "-" * 80,
        "CONFIRMED + LIKELY SN Ia CANDIDATES",
        "-" * 80,
    ])

    # List confirmed and likely SNe Ia
    snia_candidates = unique_objects[
        unique_objects['val_classification'].isin(['confirmed_ia', 'likely_ia'])
    ].sort_values('val_confidence', ascending=False)

    if len(snia_candidates) > 0:
        for _, row in snia_candidates.iterrows():
            tns_name = row.get('tns_name', '')
            tns_str = f" [{tns_name}]" if pd.notna(tns_name) and tns_name else ""
            report_lines.append(
                f"\n  {row['diaObjectId']}{tns_str}"
            )
            report_lines.append(
                f"    RA, Dec: {row['ra']:.5f}, {row['dec']:.5f}"
            )
            report_lines.append(
                f"    Classification: {row['val_classification']} (confidence: {row['val_confidence']:.2f})"
            )
            report_lines.append(
                f"    Peak: {row.get('peak_mag', np.nan):.1f} mag, "
                f"P(Ia): {row.get('sn_score', np.nan):.2f}, "
                f"Merit: {row.get('merit', np.nan):.3f}"
            )
            report_lines.append(
                f"    Nights detected: {row.get('n_nights_detected', 1)}"
            )
            report_lines.append(
                f"    Reasons: {row['val_reasons']}"
            )
    else:
        report_lines.append("  (none)")

    report_lines.extend([
        "",
        "-" * 80,
        "LIKELY CONTAMINANTS (for review)",
        "-" * 80,
    ])

    # List likely contaminants
    contaminants = unique_objects[
        unique_objects['val_classification'] == 'likely_contaminant'
    ].head(10)

    if len(contaminants) > 0:
        for _, row in contaminants.iterrows():
            report_lines.append(
                f"  {row['diaObjectId']}: {row['val_reasons'][:80]}"
            )
    else:
        report_lines.append("  (none)")

    report_lines.extend([
        "",
        "=" * 80,
        f"Report generated: {datetime.now().isoformat()}",
        "=" * 80,
    ])

    report_text = '\n'.join(report_lines)

    # Write report
    with open(output_path, 'w') as f:
        f.write(report_text)

    # Also save full classified candidates
    csv_path = output_path.with_suffix('.csv')
    unique_objects.to_csv(csv_path, index=False)

    return report_text, unique_objects


def main():
    parser = argparse.ArgumentParser(description='Historical validation of RubinAlerts pipeline')
    parser.add_argument('--start-mjd', type=float, default=61050,
                        help='Start MJD for validation (default: 61050)')
    parser.add_argument('--end-mjd', type=float, default=61100,
                        help='End MJD for validation (default: 61100)')
    parser.add_argument('--step', type=int, default=5,
                        help='Days between pipeline runs (default: 5)')
    parser.add_argument('--output-dir', type=str, default='validation/results',
                        help='Output directory for validation results')
    parser.add_argument('--no-atlas', action='store_true',
                        help='Skip ATLAS photometry')
    parser.add_argument('--no-tns', action='store_true',
                        help='Skip TNS cross-matching')
    parser.add_argument('--fink-only', action='store_true',
                        help='Query Fink only (faster)')
    parser.add_argument('--use-existing', action='store_true',
                        help='Use existing night directories only (no new runs)')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nights_dir = Path('nights')

    # Collect all candidates
    all_candidates = []

    mjd_range = np.arange(args.start_mjd, args.end_mjd + 1, args.step)
    logger.info(f"Validation range: MJD {args.start_mjd:.0f} - {args.end_mjd:.0f} "
                f"({len(mjd_range)} dates, step={args.step}d)")

    for mjd in mjd_range:
        if args.use_existing:
            t = Time(mjd, format='mjd')
            ut_date = t.datetime.strftime('%Y%m%d')
            night_dir = nights_dir / f'ut{ut_date}'
        else:
            night_dir = run_pipeline_for_date(mjd, nights_dir, args)

        if night_dir and night_dir.exists():
            df = load_candidates_from_night(night_dir)
            if len(df) > 0:
                df['mjd'] = mjd
                all_candidates.append(df)
                logger.info(f"  Loaded {len(df)} candidates from {night_dir.name}")

    if not all_candidates:
        logger.error("No candidates found in validation range")
        sys.exit(1)

    all_candidates_df = pd.concat(all_candidates, ignore_index=True)
    logger.info(f"Total detections: {len(all_candidates_df)}")

    # Cross-match to find unique objects
    all_candidates_df = cross_match_candidates(all_candidates_df)
    n_unique = all_candidates_df['object_group'].nunique()
    logger.info(f"Unique objects after cross-match: {n_unique}")

    # Generate report
    report_path = output_dir / f'validation_mjd{args.start_mjd:.0f}-{args.end_mjd:.0f}.txt'
    report_text, unique_objects = generate_validation_report(
        all_candidates_df, report_path, args.start_mjd, args.end_mjd
    )

    print("\n" + report_text)

    logger.info(f"\nValidation report saved to: {report_path}")
    logger.info(f"Full candidates CSV: {report_path.with_suffix('.csv')}")


if __name__ == '__main__':
    main()
