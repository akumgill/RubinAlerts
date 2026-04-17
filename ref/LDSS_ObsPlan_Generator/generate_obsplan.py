#!/usr/bin/env python
"""
Automated Observing Plan Generator for Las Campanas Observatory (LDSS3)

Usage:
    python generate_obsplan.py <spreadsheet> <date> <output_dir> [standards_file]

Example:
    python generate_obsplan.py full_spreadsheet.txt 2026-01-23 ./output standards.txt

Input:
    - spreadsheet: Tab-separated file with columns:
        name, RA(HMS), Dec(DMS), RA(deg), Dec(deg), priority, date, N/A, mag, exposure, notes, url
    - date: Observation date in YYYY-MM-DD format
    - output_dir: Directory to write output files
    - standards_file: (optional) File with standard stars, defaults to standards.txt

Additional files (in same directory as script):
    - contact_info.txt: Contact information (name, email, phone - one per line)
      The first line should be the observer's name (used in filenames)
    - science_description.txt: Science program description

Output:
    - ObsPlan/<Observer>_LDSS_MMDD_catalog
    - ObsPlan/<Observer>_LDSS_MMDD_timeline
    - ObsPlan/Finders/*.png (finder charts)
    - LDSS_YYYYMMDD_<Observer>.tex and .pdf
"""

import sys
import os
import re
import subprocess
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun
from astropy.time import Time
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

# Las Campanas Observatory
LCO_LAT = -(29 + 0/60 + 51.12/3600)
LCO_LON = -(70 + 41/60 + 33.00/3600)
LCO_ELEVATION = 2380
LCO = EarthLocation(lat=LCO_LAT*u.deg, lon=LCO_LON*u.deg, height=LCO_ELEVATION*u.m)

# Observing constraints
MAX_AIRMASS = 1.6
OVERHEAD = 10  # minutes per observation sequence
MAX_EDGE_GAP = 5  # max minutes gap allowed at start/end of night

# Standard star selection criteria
STD_MIN_MAG = 9.0   # Minimum V magnitude (not too bright)
STD_MAX_MAG = 12.0  # Maximum V magnitude (not too faint)
STD_IDEAL_MAG = 10.5  # Ideal magnitude (prefer stars near this)
STD_MAX_AIRMASS = 1.5  # Max airmass for standard stars

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def calculate_twilight(date_str, location):
    """Calculate astronomical twilight times for a given date."""
    noon = Time(f'{date_str} 16:00:00', scale='utc')
    times = noon + np.linspace(0, 24, 1441) * u.hour
    sun_alt = get_sun(times).transform_to(AltAz(obstime=times, location=location)).alt.deg

    evening_idx = morning_idx = None
    for i in range(len(sun_alt) - 1):
        if sun_alt[i] > -18 and sun_alt[i+1] <= -18:
            evening_idx = i
        if sun_alt[i] <= -18 and sun_alt[i+1] > -18 and evening_idx:
            morning_idx = i
            break
    return times[evening_idx], times[morning_idx]

def get_airmass(coord, time, location):
    """Calculate airmass for a coordinate at a specific time."""
    alt = coord.transform_to(AltAz(obstime=time, location=location)).alt.deg
    return 1.0 / np.sin(np.radians(alt)) if alt > 0 else np.inf

def get_airmass_grid(coord, times, location):
    """Calculate airmass over a time grid."""
    alt = coord.transform_to(AltAz(obstime=times, location=location)).alt.deg
    with np.errstate(divide='ignore', invalid='ignore'):
        am = 1.0 / np.sin(np.radians(alt))
        am[alt <= 0] = np.inf
    return am

def find_window(coord, start, end, location):
    """Find observable window for a target."""
    times = start + np.linspace(0, (end - start).to(u.minute).value, 500) * u.minute
    am = get_airmass_grid(coord, times, location)
    obs = am <= MAX_AIRMASS
    if not np.any(obs):
        return None, None, None, None
    idx = np.where(obs)[0]
    return times[np.argmin(am)], am.min(), times[idx[0]], times[idx[-1]]

def parse_standards(filename):
    """Parse the standard stars file."""
    standards = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('Star') or line.startswith('-') or line.startswith('h '):
                continue

            # Parse the fixed-width format
            # Format: " Name        HH MM SS.ss  +DD MM SS.s  V.vv  Type"
            parts = line.split()
            if len(parts) < 7:
                continue

            try:
                name = parts[0]
                # RA: parts[1], parts[2], parts[3]
                ra_h, ra_m, ra_s = parts[1], parts[2], parts[3]
                ra_str = f"{ra_h}:{ra_m}:{ra_s}"

                # Dec: parts[4], parts[5], parts[6]
                dec_d, dec_m, dec_s = parts[4], parts[5], parts[6]
                dec_str = f"{dec_d}:{dec_m}:{dec_s}"

                # Magnitude: parts[7]
                vmag = float(parts[7])

                # Spectral type: parts[8] if exists
                spec_type = parts[8] if len(parts) > 8 else ""

                coord = SkyCoord(ra=ra_str, dec=dec_str, unit=(u.hourangle, u.deg))
                standards.append({
                    'name': name, 'ra': ra_str, 'dec': dec_str,
                    'coord': coord, 'vmag': vmag, 'spec_type': spec_type
                })
            except (ValueError, IndexError):
                continue

    return standards

def find_best_standards(standards, evening, morning, location):
    """Find the best standard stars for start and end of night."""

    # Filter to suitable magnitude range
    suitable = [s for s in standards if STD_MIN_MAG <= s['vmag'] <= STD_MAX_MAG]

    def score_standard(std, obs_time):
        """Score a standard star for observation at a given time."""
        am = get_airmass(std['coord'], obs_time, location)
        if am > STD_MAX_AIRMASS or am < 1.0:
            return None, am

        # Score: prefer lower airmass and magnitude closer to ideal
        mag_penalty = abs(std['vmag'] - STD_IDEAL_MAG)
        score = -am * 10 - mag_penalty * 2
        return score, am

    # Find best for start of night (observe ~15 min before science starts)
    start_time = evening - 15 * u.minute
    best_start = None
    best_start_score = -np.inf
    best_start_am = None

    for std in suitable:
        score, am = score_standard(std, start_time)
        if score is not None and score > best_start_score:
            best_start_score = score
            best_start = std
            best_start_am = am

    # Find best for end of night (observe ~15 min after science ends)
    end_time = morning + 15 * u.minute
    best_end = None
    best_end_score = -np.inf
    best_end_am = None

    for std in suitable:
        # Don't reuse the same star
        if best_start and std['name'] == best_start['name']:
            continue
        score, am = score_standard(std, end_time)
        if score is not None and score > best_end_score:
            best_end_score = score
            best_end = std
            best_end_am = am

    result = []
    if best_start:
        result.append({
            'name': best_start['name'],
            'ra': best_start['ra'],
            'dec': best_start['dec'],
            'vmag': best_start['vmag'],
            'airmass': best_start_am,
            'position': 'start'
        })
        print(f"  Start: {best_start['name']} (V={best_start['vmag']:.2f}, AM={best_start_am:.2f})")
    else:
        print(f"  WARNING: No suitable standard star found for start of night!")

    if best_end:
        result.append({
            'name': best_end['name'],
            'ra': best_end['ra'],
            'dec': best_end['dec'],
            'vmag': best_end['vmag'],
            'airmass': best_end_am,
            'position': 'end'
        })
        print(f"  End:   {best_end['name']} (V={best_end['vmag']:.2f}, AM={best_end_am:.2f})")
    else:
        print(f"  WARNING: No suitable standard star found for end of night!")

    return result

def parse_spreadsheet(filename):
    """Parse the target spreadsheet."""
    targets = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) < 10:
                continue

            name = parts[0]
            ra_str, dec_str = parts[1], parts[2]
            priority = int(parts[5])
            mag_info = parts[8]
            exp_str = parts[9]
            notes = parts[10] if len(parts) > 10 else ""

            # Parse magnitude (extract number from "19.7, ZTF-r" format)
            mag_match = re.match(r'([\d.]+)', mag_info)
            mag = float(mag_match.group(1)) if mag_match else 20.0

            # Parse exposure (e.g., "3x900s")
            exp_match = re.match(r'(\d+)x(\d+)s', exp_str)
            n_exp, exp_sec = (int(exp_match.group(1)), int(exp_match.group(2))) if exp_match else (3, 900)
            total_minutes = (n_exp * exp_sec / 60) + OVERHEAD

            try:
                coord = SkyCoord(ra=ra_str, dec=dec_str, unit=(u.hourangle, u.deg))
                targets.append({
                    'name': name, 'ra_str': ra_str, 'dec_str': dec_str,
                    'coord': coord, 'priority': priority, 'mag': mag,
                    'mag_info': mag_info, 'exp_str': exp_str,
                    'n_exp': n_exp, 'exp_sec': exp_sec,
                    'total_minutes': total_minutes, 'notes': notes
                })
            except Exception as e:
                print(f"Warning: Could not parse {name}: {e}")
    return targets

# ============================================================================
# SCHEDULING
# ============================================================================

def create_schedule(targets, evening, morning, location):
    """Create optimized observing schedule."""

    # Calculate windows for all targets
    for t in targets:
        transit, min_am, ws, we = find_window(t['coord'], evening, morning, location)
        t['transit'] = transit
        t['min_am'] = min_am
        t['ws'] = ws
        t['we'] = we
        if transit:
            t['window'] = (we - ws).to(u.minute).value
            t['can_obs'] = t['window'] >= t['total_minutes']
        else:
            t['window'] = 0
            t['can_obs'] = False

    observable = [t for t in targets if t['can_obs']]
    observable.sort(key=lambda x: x['transit'].mjd)

    # Greedy scheduling with gap-filling
    schedule = []
    scheduled = set()
    current = evening

    while current < morning:
        best = None
        best_score = -np.inf

        for t in observable:
            if t['name'] in scheduled:
                continue
            dur = t['total_minutes'] * u.minute

            if current < t['ws'] or current + dur > t['we'] or current + dur > morning:
                continue

            mid = current + dur / 2
            am = get_airmass(t['coord'], mid, location)
            if am > MAX_AIRMASS:
                continue

            score = (5 - t['priority']) * 100 - am * 10
            if score > best_score:
                best_score = score
                best = t

        if best:
            dur = best['total_minutes'] * u.minute

            # Check for small gap to next target and extend if possible
            next_available = None
            for t in observable:
                if t['name'] in scheduled or t['name'] == best['name']:
                    continue
                if t['ws'] > current + dur:
                    if next_available is None or t['ws'] < next_available:
                        next_available = t['ws']

            proposed_end = current + dur
            if next_available and next_available > proposed_end:
                gap = (next_available - proposed_end).to(u.minute).value
                if 0 < gap <= 10:
                    extended_end = next_available
                    if extended_end <= best['we']:
                        extended_mid = current + (extended_end - current) / 2
                        extended_am = get_airmass(best['coord'], extended_mid, location)
                        if extended_am <= MAX_AIRMASS:
                            dur = extended_end - current

            # Calculate final exposure time (rounded to nearest 10s)
            total_exp_min = dur.to(u.minute).value - OVERHEAD
            exp_per_frame = int(round(total_exp_min * 60 / best['n_exp'] / 10) * 10)

            mid = current + dur / 2
            am = get_airmass(best['coord'], mid, location)

            schedule.append({
                'name': best['name'], 'start': current, 'end': current + dur,
                'priority': best['priority'], 'mag': best['mag'],
                'n_exp': best['n_exp'], 'exp_sec': exp_per_frame,
                'exp_str': f"{best['n_exp']}x{exp_per_frame}s",
                'airmass': am, 'notes': best['notes']
            })
            scheduled.add(best['name'])
            current = current + dur
        else:
            future = [t for t in observable if t['name'] not in scheduled and t['ws'] > current]
            if future:
                next_t = min(future, key=lambda x: x['ws'].mjd)
                current = next_t['ws']
            else:
                break

    schedule.sort(key=lambda x: x['start'].mjd)

    # Post-process: fill end gap by extending observations
    if schedule:
        end_gap = (morning - schedule[-1]['end']).to(u.minute).value
        if end_gap > MAX_EDGE_GAP:
            extra_needed = end_gap - MAX_EDGE_GAP
            for i in range(len(schedule) - 1, -1, -1):
                if extra_needed <= 0:
                    break
                s = schedule[i]
                t = next((x for x in targets if x['name'] == s['name']), None)
                if t:
                    max_extend = 5
                    extended_end = s['end'] + min(extra_needed, max_extend) * u.minute
                    if extended_end > t['we']:
                        extended_end = t['we']
                    if extended_end > morning:
                        extended_end = morning
                    new_mid = s['start'] + (extended_end - s['start']) / 2
                    new_am = get_airmass(t['coord'], new_mid, location)
                    if new_am <= MAX_AIRMASS:
                        added = (extended_end - s['end']).to(u.minute).value
                        if added > 0:
                            s['end'] = extended_end
                            s['airmass'] = new_am
                            total_exp_min = (s['end'] - s['start']).to(u.minute).value - OVERHEAD
                            s['exp_sec'] = int(round(total_exp_min * 60 / s['n_exp'] / 10) * 10)
                            s['exp_str'] = f"{s['n_exp']}x{s['exp_sec']}s"
                            extra_needed -= added
                            for j in range(i + 1, len(schedule)):
                                schedule[j]['start'] = schedule[j]['start'] + added * u.minute
                                schedule[j]['end'] = schedule[j]['end'] + added * u.minute

    return schedule, targets

# ============================================================================
# OUTPUT GENERATION
# ============================================================================

def generate_catalog(targets, standards, output_path):
    """Generate the catalog file."""
    with open(output_path, 'w') as f:
        idx = 1
        for t in targets:
            f.write(f"{idx} {t['name']}\t{t['ra_str']}\t{t['dec_str']} 2000.0 0.0 0.0 -62.5 HRZ "
                   f"00:00:00.0   +00:00:00   2000.0   00:00:00.0   +00:00:00   2000.0\n")
            idx += 1
        for s in standards:
            f.write(f"{idx} {s['name']}\t{s['ra']}\t{s['dec']} 2000.0 0.0 0.0 -62.5 HRZ "
                   f"00:00:00.0   +00:00:00   2000.0   00:00:00.0   +00:00:00   2000.0\n")
            idx += 1
    print(f"  Written: {output_path}")

def generate_timeline(schedule, standards, targets, evening, output_path):
    """Generate the timeline file."""
    # Build catalog index mapping (targets first, then standards)
    catalog_idx = {}
    idx = 1
    for t in targets:
        catalog_idx[t['name']] = idx
        idx += 1
    for s in standards:
        catalog_idx[s['name']] = idx
        idx += 1

    with open(output_path, 'w') as f:
        f.write("#  Target     UT          Observation     Comments\n")

        # Start standard
        start_std = next((s for s in standards if s['position'] == 'start'), None)
        if start_std:
            std_idx = catalog_idx.get(start_std['name'], 'XX')
            f.write(f"{std_idx} {start_std['name']} before {evening.datetime.strftime('%H:%M')} "
                   f"spec: 2x30s Please adjust exposure time. {start_std['vmag']:.2f} mag star\n")

        # Science targets
        for s in schedule:
            tgt_idx = catalog_idx.get(s['name'], 'XX')
            start_str = s['start'].datetime.strftime('%H:%M')
            end_str = s['end'].datetime.strftime('%H:%M')
            f.write(f"{tgt_idx} {s['name']} {start_str} - {end_str} spec: {s['exp_str']}\n")

        # End standard
        end_std = next((s for s in standards if s['position'] == 'end'), None)
        if end_std and schedule:
            std_idx = catalog_idx.get(end_std['name'], 'XX')
            f.write(f"{std_idx} {end_std['name']} after {schedule[-1]['end'].datetime.strftime('%H:%M')} "
                   f"spec: 2x30s Please adjust exposure time. {end_std['vmag']:.2f} mag star\n")

    print(f"  Written: {output_path}")

def priority_to_str(p):
    """Convert priority number to ordinal string."""
    return {1: '1st', 2: '2nd', 3: '3rd', 4: '4th'}.get(p, f'{p}th')

def generate_latex(schedule, targets, standards, date_str, output_path,
                   observer, contact_info, science_description):
    """Generate the LaTeX file."""
    scheduled_names = {s['name'] for s in schedule}
    backup_targets = [t for t in targets if t['name'] not in scheduled_names]

    # Format date for display
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    date_display = date_obj.strftime('%B %d, %Y').replace(' 0', ' ')  # Remove leading zero
    date_mmdd = date_obj.strftime('%m%d')

    # Determine priority range in timeline
    timeline_priorities = set(s['priority'] for s in schedule)
    max_timeline_priority = max(timeline_priorities) if timeline_priorities else 1

    # Format contact info for LaTeX (replace newlines with \\)
    contact_latex = ' \\\\\n'.join(contact_info.split('\n'))

    latex = r"""\documentclass[a4paper]{article}

%% Language and font encodings
\usepackage[english]{babel}
\usepackage[utf8x]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage[colorlinks=true, allcolors=blue]{hyperref}

%% Sets page size and margins
\usepackage[a4paper,top=2cm,bottom=2cm,left=2cm,right=2cm,marginparwidth=1.75cm]{geometry}

\newcommand{\obsdate}{""" + date_display + r"""}

\title{\obsdate{} LDSS3 """ + observer + r""" Observing Request Package}
\date{}

\begin{document}
\maketitle

\section{Requested Instruments}
LDSS3-C will be used for all targets.

\section{Contact Info}

""" + contact_latex + r"""

\section{Description of Science Program}
""" + science_description + r"""

\section{Observing Strategy and Expected Products}
The observations will consist of direct imaging and long-slit spectroscopy.  Some targets need both imaging and spectroscopy and some need one or the other.  For targets that need both, there are two entries (named Spectrum and Photometry).  \textbf{Please align spectroscopic observations along the parallactic angle and imaging observations along N-S.}

\medskip
\noindent
\textbf{For all exposures use binning 1x1 and fast readout speed.}

\subsection{Imaging}

Please place targets in \textbf{chip 1}.

\subsection{Spectroscopy}

The setup for all spectroscopic targets is the following: \\
\\
\textbf{VPH-All grism} \\
\textbf{1" slit} \\
\textbf{Open filter} \\
\\
Please place all targets on \textbf{chip 1}. Please take all acquisition images with the $r$ filter. After each spectroscopic target please obtain an arc lamp (HeNeAr) and flat-field observation.

\section{Observing Catalog}
See file called {\tt """ + observer + r"""\_LDSS\_""" + date_mmdd + r"""\_catalog}.

\section{Finder Charts}
See folder called {\tt Finders}.

\section{Backup Program}
If the seeing is very poor, it may be necessary to increase the binning to 2x2, though this would require getting calibrations with the same binning and updating the to-slit offsets. Backup targets are listed below:

\begin{itemize}
"""

    # Add backup targets (all as 4th priority)
    for t in backup_targets:
        latex += f"\\item {t['name']} (4th Priority) mag$\\sim${t['mag']:.1f}\n"
        latex += "\\begin{enumerate}\n"
        latex += f"\t\\item Spectrum: {t['exp_str']}\n"
        latex += "\\end{enumerate}\n\n"

    latex += r"""\end{itemize}

\section{Calibration Plan}
\subsection{Daytime Calibrations}
\begin{itemize}
	\item 11 bias frames
	\item 10 Qh spectroscopic flats with peak counts of $\sim 20,000 - 30,000$
	\item Dome imaging flats in $g$, $r$, and $i$ (5 per filter) with peak counts of $\sim 20,000 - 30,000$ using variable voltage lamp
\end{itemize}

\subsection{Nighttime Calibrations}
\begin{itemize}
	\item One henear arc lamp and one Qh flat after each spectroscopic sequence
\end{itemize}
\section{Observing Timeline}
Listed in desired observing order. Please modify as necessary due to weather. As most targets are fading supernovae, some targets may be fainter than expected. Priorities ranked from 1st (highest) to """ + priority_to_str(max_timeline_priority) + r""" (lowest).

\subsection{\obsdate{}}

\begin{itemize}
"""

    # Add start standard star
    start_std = next((s for s in standards if s['position'] == 'start'), None)
    if start_std:
        latex += f"\\item Spectroscopic Standard Star {start_std['name']} (V mag = {start_std['vmag']:.2f})\n"
        latex += "\\begin{enumerate}\n"
        latex += "\t\\item Spectrum: 2x30s\n"
        latex += "\\end{enumerate}\n\n"

    # Add scheduled science targets
    for s in schedule:
        priority_str = priority_to_str(s['priority'])
        latex += f"\\item {s['name']} ({priority_str} Priority) mag$\\sim${s['mag']:.1f}\n"
        latex += "\\begin{enumerate}\n"
        latex += f"\t\\item Spectrum: {s['exp_str']}\n"
        latex += "\\end{enumerate}\n\n"

    # Add end standard star
    end_std = next((s for s in standards if s['position'] == 'end'), None)
    if end_std:
        latex += f"\\item Spectroscopic Standard Star: {end_std['name']} (V mag = {end_std['vmag']:.2f})\n"
        latex += "\\begin{enumerate}\n"
        latex += "        \\item Spectrum: 2x30s\n"
        latex += "\\end{enumerate}\n\n"

    latex += r"""\end{itemize}

\section{Data Transfer}
Please upload the resulting data to Google Drive and send us a link. We don't require live transfer of the data, it's ok to upload in the morning.

\end{document}
"""

    with open(output_path, 'w') as f:
        f.write(latex)
    print(f"  Written: {output_path}")

def compile_latex(tex_path):
    """Compile LaTeX to PDF."""
    directory = os.path.dirname(tex_path)
    filename = os.path.basename(tex_path)

    result = subprocess.run(
        ['pdflatex', '-interaction=nonstopmode', filename],
        cwd=directory,
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        pdf_path = tex_path.replace('.tex', '.pdf')
        print(f"  Written: {pdf_path}")
    else:
        print(f"  Warning: LaTeX compilation had issues (PDF may still be created)")

def generate_finder_charts(targets, standards, finders_dir):
    """Generate finder charts for all targets using mkFinderChart.py."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    finder_script = os.path.join(script_dir, 'mkFinderChart.py')

    if not os.path.exists(finder_script):
        print(f"  Warning: mkFinderChart.py not found, skipping finder charts")
        return

    os.makedirs(finders_dir, exist_ok=True)

    # Combine targets and standards
    all_targets = []
    for t in targets:
        all_targets.append({
            'name': t['name'],
            'ra_hms': t['ra_str'],
            'dec_dms': t['dec_str']
        })
    for s in standards:
        all_targets.append({
            'name': s['name'],
            'ra_hms': s['ra'],
            'dec_dms': s['dec']
        })

    # Generate finderList.py script
    finder_list_path = os.path.join(finders_dir, 'finderList.py')
    with open(finder_list_path, 'w') as f:
        for t in all_targets:
            output_file = os.path.join(finders_dir, f"{t['name']}.png")
            cmd = f"python {finder_script} -s {t['name']} -r {t['ra_hms']} -d {t['dec_dms']} -o {output_file}\n"
            f.write(cmd)
    print(f"  Written: {finder_list_path}")

    # Execute finder chart generation
    print(f"  Generating {len(all_targets)} finder charts...")
    success_count = 0
    for t in all_targets:
        output_file = os.path.join(finders_dir, f"{t['name']}.png")
        cmd = [
            'python', finder_script,
            '-s', t['name'],
            '-r', t['ra_hms'],
            '-d', t['dec_dms'],
            '-o', output_file
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0 and os.path.exists(output_file):
                success_count += 1
            else:
                print(f"    Warning: Failed to create finder for {t['name']}")
        except subprocess.TimeoutExpired:
            print(f"    Warning: Timeout creating finder for {t['name']}")
        except Exception as e:
            print(f"    Warning: Error creating finder for {t['name']}: {e}")

    print(f"  Created {success_count}/{len(all_targets)} finder charts")

# ============================================================================
# MAIN
# ============================================================================

def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)

    spreadsheet = sys.argv[1]
    date_str = sys.argv[2]
    output_dir = sys.argv[3]
    standards_file = sys.argv[4] if len(sys.argv) > 4 else 'standards.txt'

    # Get script directory for config files
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Read contact info and extract observer name
    contact_file = os.path.join(script_dir, 'contact_info.txt')
    if os.path.exists(contact_file):
        with open(contact_file, 'r') as f:
            contact_info = f.read().strip()
        # Extract observer name (last name from first line)
        first_line = contact_info.split('\n')[0].strip()
        observer = first_line.split()[-1] if first_line else "Observer"
    else:
        observer = "Observer"
        contact_info = f"{observer}\nemail@example.com\n(000) 000-0000"
        print(f"  Warning: contact_info.txt not found, using placeholder")

    # Read science description
    science_file = os.path.join(script_dir, 'science_description.txt')
    if os.path.exists(science_file):
        with open(science_file, 'r') as f:
            science_description = f.read().strip()
    else:
        science_description = "Science program description not provided."
        print(f"  Warning: science_description.txt not found, using placeholder")

    print(f"\n{'='*70}")
    print(f"LDSS3 Observing Plan Generator")
    print(f"{'='*70}")
    print(f"Input:     {spreadsheet}")
    print(f"Standards: {standards_file}")
    print(f"Observer:  {observer}")
    print(f"Date:      {date_str}")
    print(f"Output:    {output_dir}")

    # Parse targets
    print(f"\nParsing targets...")
    targets = parse_spreadsheet(spreadsheet)
    print(f"  Found {len(targets)} targets")

    # Parse standards
    print(f"\nParsing standard stars...")
    all_standards = parse_standards(standards_file)
    print(f"  Found {len(all_standards)} standard stars")

    # Calculate twilight
    print(f"\nCalculating twilight times...")
    evening, morning = calculate_twilight(date_str, LCO)
    night_mins = (morning - evening).to(u.minute).value
    print(f"  Evening twilight: {evening.datetime.strftime('%H:%M')} UTC")
    print(f"  Morning twilight: {morning.datetime.strftime('%H:%M')} UTC")
    print(f"  Night duration:   {night_mins:.0f} min ({night_mins/60:.1f} hrs)")

    # Find best standard stars
    print(f"\nSelecting standard stars...")
    selected_standards = find_best_standards(all_standards, evening, morning, LCO)

    # Create schedule
    print(f"\nCreating optimized schedule...")
    schedule, targets = create_schedule(targets, evening, morning, LCO)

    total_time = sum((s['end'] - s['start']).to(u.minute).value for s in schedule)
    p1_count = sum(1 for s in schedule if s['priority'] == 1)

    print(f"  Scheduled {len(schedule)} targets")
    print(f"  Priority 1: {p1_count}")
    print(f"  Total time: {total_time:.0f} min ({total_time/60:.1f} hrs)")
    print(f"  Efficiency: {total_time/night_mins*100:.1f}%")

    print(f"\nSchedule:")
    for s in schedule:
        print(f"  {s['start'].datetime.strftime('%H:%M')}-{s['end'].datetime.strftime('%H:%M')} "
              f"{s['name']:<16} P{s['priority']} {s['exp_str']:<10} AM={s['airmass']:.2f}")

    # Create output directories
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    date_mmdd = date_obj.strftime('%m%d')
    date_yyyymmdd = date_obj.strftime('%Y%m%d')

    obsplan_dir = os.path.join(output_dir, 'ObsPlan')
    os.makedirs(obsplan_dir, exist_ok=True)

    # Generate output files
    print(f"\nGenerating output files...")

    catalog_path = os.path.join(obsplan_dir, f'{observer}_LDSS_{date_mmdd}_catalog')
    generate_catalog(targets, selected_standards, catalog_path)

    timeline_path = os.path.join(obsplan_dir, f'{observer}_LDSS_{date_mmdd}_timeline')
    generate_timeline(schedule, selected_standards, targets, evening, timeline_path)

    tex_path = os.path.join(output_dir, f'LDSS_{date_yyyymmdd}_{observer}.tex')
    generate_latex(schedule, targets, selected_standards, date_str, tex_path,
                   observer, contact_info, science_description)

    print(f"\nCompiling LaTeX...")
    compile_latex(tex_path)

    # Generate finder charts
    print(f"\nGenerating finder charts...")
    finders_dir = os.path.join(obsplan_dir, 'Finders')
    generate_finder_charts(targets, selected_standards, finders_dir)

    print(f"\n{'='*70}")
    print(f"Done!")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()
