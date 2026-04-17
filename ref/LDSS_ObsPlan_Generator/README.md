# LDSS3 Observing Plan Generator

Automated observing plan generator for LDSS3 spectroscopy at Las Campanas Observatory (Magellan/Clay telescope).

## Features

- Calculates observable windows based on airmass constraints
- Creates optimized back-to-back observing schedule prioritizing P1 targets
- Automatically selects appropriate spectroscopic standard stars
- Generates three output files:
  - **Catalog file**: Target coordinates for telescope pointing
  - **Timeline file**: Observing sequence with UT times
  - **LaTeX document**: Complete observing package (compiled to PDF)

## Requirements

```bash
pip install -r requirements.txt
```

Requires:
- Python 3.7+
- astropy
- pdflatex (for PDF generation)

## Usage

```bash
python generate_obsplan.py <spreadsheet_file> <date> <output_dir> [standards_file]
```

### Arguments

- `spreadsheet_file`: Tab-separated file with target information (see format below)
- `date`: Observation date in YYYY-MM-DD format
- `output_dir`: Directory where output files will be created
- `standards_file`: (optional) Path to standard star catalog, defaults to `standards.txt`

The observer name is automatically extracted from `contact_info.txt` (last name from first line).

### Example

```bash
python generate_obsplan.py targets.txt 2026-01-23 /tmp/my_obsplan
```

## Configuration Files

The following files should be in the same directory as `generate_obsplan.py`:

### contact_info.txt
Your contact information, one item per line. The observer's **last name** is extracted from the first line and used in output filenames.
```
Jane Smith
jane.smith@university.edu
(555) 123-4567
```
This would use "Smith" as the observer name in output files.

### science_description.txt
A paragraph describing your science program. This will be included in the LaTeX document.

### standards.txt
Standard star catalog with columns: name, RA (HMS), Dec (DMS), V magnitude, spectral type.

## Input File Format

The input spreadsheet must be a tab-separated file with the following columns:

| Column | Description | Example |
|--------|-------------|---------|
| name | Target name | 2025aghn |
| RA | Right Ascension (HMS) | 03:01:05.33 |
| Dec | Declination (DMS) | -13:59:41.45 |
| RA | Right Ascension (degrees) | 45.272208 |
| Dec | Declination (degrees) | -13.994847 |
| Priority | 1 (highest) to 4 (lowest) | 1 |
| Date | Observation date (YYYY-MM-DD) | 2026-01-23 |
| N/A | Unused column | N/A |
| Mag | Apparent magnitude | 19.7 |
| Exposure | Exposure time format | 3x900s |
| Notes | Any notes | Classification needed |
| url | Reference URL | https://... |

See `example_targets.txt` for a template. Note: No header row.

### Exposure Time Format

Exposure times should be in the format `NxTTTTs` where:
- `N` = number of exposures (typically 3)
- `TTTT` = exposure time in seconds

Examples: `3x900s`, `3x1200s`, `2x30s`

## Script Configuration

Key parameters can be adjusted at the top of `generate_obsplan.py`:

```python
MAX_AIRMASS = 1.6        # Maximum allowed airmass during observation
OVERHEAD = 10            # Minutes overhead per observation sequence
MAX_EDGE_GAP = 5         # Maximum minutes gap at start/end of night
STD_MIN_MAG = 9.0        # Minimum standard star magnitude
STD_MAX_MAG = 12.0       # Maximum standard star magnitude
STD_IDEAL_MAG = 10.5     # Ideal standard star magnitude
STD_MAX_AIRMASS = 1.5    # Maximum airmass for standard stars
```

### Observatory Location

Default is Las Campanas Observatory:
- Latitude: -29.01420 degrees
- Longitude: -70.69250 degrees
- Elevation: 2380 meters

## Output Files

The script creates the following in the output directory:

```
output_dir/
├── ObsPlan/
│   ├── <Observer>_LDSS_MMDD_catalog    # Telescope catalog
│   ├── <Observer>_LDSS_MMDD_timeline   # Observing timeline
│   └── Finders/                        # (empty, for finder charts)
├── LDSS_YYYYMMDD_<Observer>.tex        # LaTeX document
└── LDSS_YYYYMMDD_<Observer>.pdf        # Compiled PDF
```

## Standard Stars

The script automatically selects standard stars from `standards.txt` for:
- Start of night: Observable at astronomical twilight
- End of night: Observable at end of observing window

Standard stars are selected based on:
- Magnitude close to 10.5 (range 9-12)
- Airmass < 1.5 at the relevant time

## Algorithm

1. **Parse targets**: Read spreadsheet and filter to observation date
2. **Calculate twilight**: Determine astronomical twilight times
3. **Compute observability**: Calculate when each target is above airmass limit
4. **Schedule P1 targets**: Greedy algorithm fitting highest-priority targets
5. **Fill gaps**: Add P2/P3 targets or extend exposure times (up to ~5 min per sequence)
6. **Add standards**: Select appropriate standard stars for start/end of night
7. **Generate outputs**: Create catalog, timeline, and LaTeX files
8. **Compile PDF**: Run pdflatex on the LaTeX document

## Notes

- Targets not scheduled become backup targets (Priority 4) in the LaTeX document
- Exposure times may be extended by up to ~5 minutes to fill scheduling gaps
- Final exposure times are rounded to the nearest 10 seconds
- The script allows gaps of up to 5 minutes at the start/end of night only
- All spectroscopic observations assume VPH-All grism, 1" slit, open filter
