"""PDF report generation for SN Ia monitoring pipeline.

Generates multi-page PDF reports with:
- Title page and summary statistics
- Merit breakdown tables
- Observing sequence sky maps
- Diagnostic scatter plots
- Light curve gallery
"""

import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate PDF reports for SN Ia candidate monitoring."""

    # Merit function documentation text
    # Note: Using plain text to avoid LaTeX escaping issues in monospace
    MERIT_REFERENCE_TEXT = """MERIT FUNCTION

    Merit = W_time x W_mag x W_prob x W_host x W_ext x W_broker

Each component ranges from 0 to 1 (W_broker can reach 1.2). The multiplicative
structure means a candidate needs to score well on ALL factors to rank highly.


COMPONENT DEFINITIONS

W_time  -- Time from Peak
    Gaussian decay with tau = 10 days
    Supernovae are most valuable for spectroscopy near peak brightness.

W_mag   -- Magnitude Suitability
    Gaussian centered at m_opt = 20.5 AB, sigma = 1.5 mag
    Penalizes targets too bright (use smaller telescope) or too faint (poor S/N).

W_prob  -- Type Ia Probability
    P(Ia) from ML classifier, clipped to [0.1, 1.0]
    From ALeRCE or Fink supernova classifiers.

W_host  -- Host Galaxy Morphology
    Elliptical = 1.0, Spiral = 0.6, Unknown = 0.7
    SNe Ia in elliptical hosts have lower Hubble diagram scatter.

W_ext   -- Galactic Extinction Penalty
    Exponential penalty based on E(B-V)
    Penalizes targets behind significant Milky Way dust.

W_broker -- Multi-broker Agreement
    1.0 + 0.1 x (N - 1)  where N = number of detecting brokers
    Independent detections increase confidence the transient is real.
"""

    def __init__(self, ddf_fields: List[Dict[str, Any]], style_path: Optional[str] = None):
        """Initialize report generator.

        Parameters
        ----------
        ddf_fields : list of dict
            DDF field definitions with 'name', 'ra', 'dec' keys.
        style_path : str, optional
            Path to matplotlib style file.
        """
        self.ddf_fields = ddf_fields
        self.style_path = style_path

        # Try to load style
        if style_path and Path(style_path).exists():
            try:
                plt.style.use(style_path)
                logger.info("Loaded matplotlib style: %s", style_path)
            except Exception as e:
                logger.warning("Could not load style file: %s", e)

    def generate_report(
        self,
        summary_df: pd.DataFrame,
        fit_results: Dict[str, Any],
        plot_paths: Dict[str, str],
        pdf_path: str,
        mjd_now: float,
        obs_date: str,
        observing_sequence: Optional[pd.DataFrame] = None,
        radec_formatter: Optional[callable] = None,
    ):
        """Generate complete PDF report.

        Parameters
        ----------
        summary_df : DataFrame
            Summary table with all candidates.
        fit_results : dict
            Mapping of diaObjectId to fit results.
        plot_paths : dict
            Mapping of diaObjectId to light curve plot paths.
        pdf_path : str
            Output PDF file path.
        mjd_now : float
            Current MJD.
        obs_date : str
            Observing date string.
        observing_sequence : DataFrame, optional
            Optimized observing sequence from optimize_observing_sequence().
        radec_formatter : callable, optional
            Function to format RA/Dec to sexagesimal strings.
        """
        with PdfPages(pdf_path) as pdf:
            # Title page
            self._add_title_page(pdf, summary_df, mjd_now, obs_date)

            # Summary table
            if len(summary_df) > 0:
                self._add_summary_table(pdf, summary_df, radec_formatter)

            # Merit breakdown
            if len(summary_df) > 0 and 'w_time' in summary_df.columns:
                self._add_merit_breakdown(pdf, summary_df)

            # Merit function reference
            self._add_merit_reference(pdf)

            # Observing sequence sky map
            if observing_sequence is not None and len(observing_sequence) > 0:
                self._add_observing_sequence_map(pdf, observing_sequence, obs_date)
                self._add_observing_sequence_table(pdf, observing_sequence)

            # Diagnostic scatter plots
            if len(summary_df) > 0:
                self._add_diagnostic_plots(pdf, summary_df)
                self._add_discovery_space_plot(pdf, summary_df)

            # Light curve pages
            self._add_light_curve_pages(pdf, summary_df, plot_paths)

        n_pages = self._count_pages(summary_df, plot_paths, observing_sequence)
        logger.info("PDF report: %s (%d pages)", pdf_path, n_pages)

    def _count_pages(self, summary_df, plot_paths, observing_sequence) -> int:
        """Estimate total number of pages."""
        n = 1  # title
        if len(summary_df) > 0:
            n += 1  # summary table
            if 'w_time' in summary_df.columns:
                n += 1  # merit breakdown
        n += 1  # merit reference
        if observing_sequence is not None and len(observing_sequence) > 0:
            n += 2  # sky map + table
        if len(summary_df) > 0:
            n += 2  # scatter plots + discovery space
        n += (len(plot_paths) + 3) // 4  # light curves
        return n

    def _add_title_page(self, pdf, summary_df, mjd_now, obs_date):
        """Add title page with summary statistics."""
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')

        # Convert MJD to UT date string
        from astropy.time import Time
        ut_stamp = Time(mjd_now, format='mjd').strftime('%Y%m%d')

        ax.text(0.5, 0.60, 'SN Ia Monitoring Report',
                ha='center', va='center', fontsize=28, fontweight='bold')
        ax.text(0.5, 0.48, f'ut{ut_stamp}',
                ha='center', va='center', fontsize=32, fontweight='bold',
                fontfamily='monospace')
        ax.text(0.5, 0.38, f'MJD {mjd_now:.1f}  |  {obs_date}',
                ha='center', va='center', fontsize=16, fontfamily='monospace')
        ax.text(0.5, 0.34, f'{len(summary_df)} candidates with peak fits',
                ha='center', va='center', fontsize=14, color='gray')

        # Summary stats
        if len(summary_df) > 0:
            n_atlas = (summary_df.get('n_atlas', pd.Series([0])) > 0).sum()
            n_ztf = (summary_df.get('n_ztf', pd.Series([0])) > 0).sum()
            fields = summary_df['ddf_field'].nunique() if 'ddf_field' in summary_df.columns else 0
            high_merit = (summary_df['merit'] > 0.1).sum() if 'merit' in summary_df.columns else 0

            stats = (f'{fields} DDFs  |  {n_atlas} with ATLAS  |  {n_ztf} with ZTF  |  '
                     f'{high_merit} high-merit (>0.1)')
            ax.text(0.5, 0.20, stats,
                    ha='center', va='center', fontsize=10, color='dimgray')

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def _add_summary_table(self, pdf, summary_df, radec_formatter):
        """Add summary table page."""
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        ax.set_title('Top 30 Candidates by Merit', fontsize=14, pad=20)

        table_df = summary_df.sort_values('merit', ascending=False).head(30).copy()

        if radec_formatter:
            table_df['RA_s'] = table_df.apply(
                lambda r: radec_formatter(r['ra'], r['dec'])[0], axis=1)
            table_df['Dec_s'] = table_df.apply(
                lambda r: radec_formatter(r['ra'], r['dec'])[1], axis=1)

        display_cols = ['diaObjectId', 'ddf_field', 'RA_s', 'Dec_s',
                        'peak_mag', 'peak_band', 'delta_t', 'merit',
                        'brokers_detected', 'num_brokers', 'fit_method', 'surveys']
        display_df = table_df[[c for c in display_cols if c in table_df.columns]].copy()

        # Format numbers
        for col in ['peak_mag', 'merit']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda x: f'{x:.2f}' if pd.notna(x) and np.isfinite(x) else '--')
        if 'delta_t' in display_df.columns:
            display_df['delta_t'] = display_df['delta_t'].apply(
                lambda x: f'{x:+.1f}d' if pd.notna(x) and np.isfinite(x) else '--')
        if 'diaObjectId' in display_df.columns:
            display_df['diaObjectId'] = display_df['diaObjectId'].astype(str).str[-10:]

        tbl = ax.table(
            cellText=display_df.values,
            colLabels=display_df.columns,
            loc='center',
            cellLoc='center',
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7)
        tbl.auto_set_column_width(range(len(display_df.columns)))
        tbl.scale(1.0, 1.3)

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def _add_merit_breakdown(self, pdf, summary_df):
        """Add merit breakdown table page."""
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        ax.set_title('Merit Breakdown — Top 30 Candidates', fontsize=14, pad=20)

        ax.text(0.5, 0.95,
                r'Merit = $W_{\rm time} \times W_{\rm mag} \times W_{\rm prob} \times W_{\rm host} \times W_{\rm ext} \times W_{\rm broker}$',
                ha='center', va='top', fontsize=10, transform=ax.transAxes)

        table_df = summary_df.sort_values('merit', ascending=False).head(30).copy()

        breakdown_cols = ['diaObjectId', 'merit', 'w_time', 'w_mag', 'w_prob',
                          'w_host', 'w_ext', 'w_broker']
        breakdown_df = table_df[[c for c in breakdown_cols if c in table_df.columns]].copy()

        for col in breakdown_df.columns:
            if col == 'diaObjectId':
                breakdown_df[col] = breakdown_df[col].astype(str).str[-10:]
            else:
                breakdown_df[col] = breakdown_df[col].apply(
                    lambda x: f'{x:.3f}' if pd.notna(x) and np.isfinite(x) else '--')

        col_names = {
            'diaObjectId': 'Object', 'merit': 'Merit',
            'w_time': r'$W_{\rm time}$', 'w_mag': r'$W_{\rm mag}$',
            'w_prob': r'$W_{\rm prob}$', 'w_host': r'$W_{\rm host}$',
            'w_ext': r'$W_{\rm ext}$', 'w_broker': r'$W_{\rm broker}$',
        }
        breakdown_df.columns = [col_names.get(c, c) for c in breakdown_df.columns]

        tbl = ax.table(
            cellText=breakdown_df.values,
            colLabels=breakdown_df.columns,
            loc='center',
            cellLoc='center',
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.auto_set_column_width(range(len(breakdown_df.columns)))
        tbl.scale(1.0, 1.4)

        # Note: Legend text handled in formula at top

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def _add_merit_reference(self, pdf):
        """Add merit function reference page."""
        import matplotlib as mpl

        # Temporarily disable LaTeX for this page (monospace text doesn't work well)
        original_usetex = mpl.rcParams.get('text.usetex', False)
        mpl.rcParams['text.usetex'] = False

        try:
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.axis('off')
            ax.set_title('Merit Function Reference', fontsize=16, fontweight='bold', pad=20)
            ax.text(0.05, 0.95, self.MERIT_REFERENCE_TEXT, ha='left', va='top',
                    fontsize=10, fontfamily='monospace', transform=ax.transAxes)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
        finally:
            # Restore LaTeX setting
            mpl.rcParams['text.usetex'] = original_usetex

    def _add_observing_sequence_map(self, pdf, sequence_df, obs_date):
        """Add optimized observing sequence sky map."""
        fig, ax = plt.subplots(figsize=(11, 7))

        df = sequence_df.sort_values('obs_order')
        n = len(df)

        # DDF field markers
        for f in self.ddf_fields:
            ax.scatter(f['ra'], f['dec'], s=400, marker='s', facecolors='none',
                       edgecolors='lightgray', linewidths=2, alpha=0.5, zorder=1)
            ax.annotate(f['name'], (f['ra'], f['dec']), fontsize=8,
                        ha='center', va='bottom', alpha=0.5, xytext=(0, 10),
                        textcoords='offset points')

        # Slew arrows
        ras, decs = df['ra'].values, df['dec'].values
        for i in range(n - 1):
            ax.annotate('', xy=(ras[i + 1], decs[i + 1]), xytext=(ras[i], decs[i]),
                        arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5, lw=1.2),
                        zorder=2)

        # Targets with color gradient
        sc = ax.scatter(ras, decs, c=range(n), cmap='plasma', s=120, zorder=3,
                        edgecolors='white', linewidths=1)

        # Order labels
        for i, (_, row) in enumerate(df.iterrows()):
            ax.annotate(f'{int(row["obs_order"])}', (row['ra'], row['dec']),
                        fontsize=8, ha='center', va='center', fontweight='bold',
                        color='white', zorder=4)

        # Colorbar
        cbar = plt.colorbar(sc, ax=ax, label='Observation Order', shrink=0.8)
        times = df['obs_time_ut'].values
        cbar.set_ticks([0, n // 2, n - 1])
        cbar.set_ticklabels([f'Start ({times[0]})', f'Mid ({times[n // 2]})',
                             f'End ({times[-1]})'])

        total_slew = df['slew_deg'].sum()
        ax.set_xlabel('RA (deg)', fontsize=12)
        ax.set_ylabel('Dec (deg)', fontsize=12)
        ax.set_title(f'Optimized Single-Night Observing Sequence — {obs_date}\n'
                     f'{n} targets, {total_slew:.1f}° total slew, ~{n * 0.5:.1f} hours',
                     fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def _add_observing_sequence_table(self, pdf, sequence_df):
        """Add observing sequence table page."""
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        ax.set_title('Optimized Observing Sequence (Slew-Minimized)', fontsize=14, pad=20)

        df = sequence_df.sort_values('obs_order')
        n = len(df)

        seq_display = df[['obs_order', 'obs_time_ut', 'diaObjectId', 'ra', 'dec',
                          'peak_mag', 'merit', 'slew_deg']].copy()
        seq_display['diaObjectId'] = seq_display['diaObjectId'].astype(str).str[-10:]
        seq_display['ra'] = seq_display['ra'].apply(lambda x: f'{x:.3f}')
        seq_display['dec'] = seq_display['dec'].apply(lambda x: f'{x:+.3f}')
        seq_display['peak_mag'] = seq_display['peak_mag'].apply(
            lambda x: f'{x:.1f}' if pd.notna(x) else '--')
        seq_display['merit'] = seq_display['merit'].apply(
            lambda x: f'{x:.3f}' if pd.notna(x) else '--')
        seq_display['slew_deg'] = seq_display['slew_deg'].apply(lambda x: f'{x:.1f}°')
        seq_display.columns = ['Ord', 'UT', 'Object', 'RA', 'Dec', 'Mag', 'Merit', 'Slew']

        tbl = ax.table(cellText=seq_display.values, colLabels=seq_display.columns,
                       loc='center', cellLoc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.auto_set_column_width(range(len(seq_display.columns)))
        tbl.scale(1.0, 1.5)

        ax.text(0.5, 0.02,
                f'Total slew: {df["slew_deg"].sum():.1f}° | '
                f'Estimated time: ~{n * 0.5:.1f} hours (30 min/target)',
                ha='center', va='bottom', fontsize=10, color='dimgray',
                transform=ax.transAxes)

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def _add_diagnostic_plots(self, pdf, summary_df):
        """Add diagnostic scatter plot page."""
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        valid = summary_df[summary_df['merit'].notna()]

        # Merit vs magnitude
        ax = axes[0]
        ax.scatter(valid['peak_mag'], valid['merit'], c='steelblue', s=20, alpha=0.7)
        ax.set_xlabel('Peak Magnitude (AB)')
        ax.set_ylabel('Merit Score')
        ax.set_title('Merit vs Peak Brightness')
        ax.grid(True, alpha=0.3)

        # Merit vs delta_t
        ax = axes[1]
        ax.scatter(valid['delta_t'], valid['merit'], c='firebrick', s=20, alpha=0.7)
        ax.set_xlabel('Days Since Peak')
        ax.set_ylabel('Merit Score')
        ax.set_title('Merit vs Time from Peak')
        ax.grid(True, alpha=0.3)

        # Sky distribution
        ax = axes[2]
        ax.scatter(summary_df['ra'], summary_df['dec'],
                   c=summary_df['merit'].fillna(0), cmap='YlOrRd',
                   s=30, alpha=0.7, edgecolors='gray', linewidths=0.3)
        for f in self.ddf_fields:
            ax.annotate(f['name'], (f['ra'], f['dec']),
                        fontsize=7, ha='center', alpha=0.5)
        ax.set_xlabel('RA (deg)')
        ax.set_ylabel('Dec (deg)')
        ax.set_title('Sky Distribution')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def _add_discovery_space_plot(self, pdf, summary_df):
        """Add discovery space (delta_t vs peak_mag) plot."""
        valid = summary_df.dropna(subset=['peak_mag', 'delta_t'])
        if len(valid) == 0:
            return

        fig, ax = plt.subplots(figsize=(8, 6))
        merit_vals = valid['merit'].fillna(0).values
        sc = ax.scatter(valid['delta_t'], valid['peak_mag'],
                        c=merit_vals, cmap='YlOrRd', s=40, alpha=0.8,
                        edgecolors='gray', linewidths=0.3,
                        vmin=0, vmax=max(merit_vals.max(), 0.01))
        plt.colorbar(sc, ax=ax, label='Merit Score')

        # Annotate high-merit targets
        high_merit = valid[valid['merit'] > 0.3]
        for _, row in high_merit.iterrows():
            label = str(row['diaObjectId'])[-6:]
            ax.annotate(label, (row['delta_t'], row['peak_mag']),
                        fontsize=6, alpha=0.7,
                        xytext=(4, 4), textcoords='offset points')

        ax.set_xlabel('Days Since Peak (negative = pre-peak)')
        ax.set_ylabel('Peak Magnitude (AB)')
        ax.invert_yaxis()
        ax.set_title(f'Discovery Space — {len(valid)} candidates with fits')
        ax.axvline(0, color='gray', linestyle='--', alpha=0.4, label='Peak')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def _add_light_curve_pages(self, pdf, summary_df, plot_paths):
        """Add light curve gallery pages (4 per page)."""
        # Sort by merit and get ranked list
        ordered = summary_df.sort_values('merit', ascending=False, na_position='last')
        ranked_dids = []

        for rank, (_, row) in enumerate(ordered.iterrows(), 1):
            did = row['diaObjectId']
            # Try to find matching plot
            did_str = str(did)
            short_did = did_str[-12:]
            if did in plot_paths:
                ranked_dids.append((rank, did, plot_paths[did]))
            elif did_str in plot_paths:
                ranked_dids.append((rank, did, plot_paths[did_str]))
            elif short_did in plot_paths:
                ranked_dids.append((rank, did, plot_paths[short_did]))
            else:
                # Try to find by suffix match
                for key, path in plot_paths.items():
                    if str(key).endswith(short_did) or short_did in str(key):
                        ranked_dids.append((rank, did, path))
                        break

        for page_start in range(0, len(ranked_dids), 4):
            page_items = ranked_dids[page_start:page_start + 4]
            n = len(page_items)
            if n == 0:
                continue

            fig, axes = plt.subplots(n, 1, figsize=(11, 4 * n))
            if n == 1:
                axes = [axes]

            for ax, (rank, did, path) in zip(axes, page_items):
                try:
                    img = plt.imread(path)
                    ax.imshow(img)
                    ax.axis('off')

                    # Add rank and merit annotation
                    row = summary_df[summary_df['diaObjectId'] == did]
                    if len(row) > 0:
                        r = row.iloc[0]
                        info = f"Rank {rank}"
                        if pd.notna(r.get('merit')):
                            info += f"  Merit={r['merit']:.3f}"
                        if pd.notna(r.get('ddf_field')):
                            info += f"  {r['ddf_field']}"
                        ax.set_title(info, fontsize=9, loc='right')
                except Exception as e:
                    logger.warning("Failed to load plot %s: %s", path, e)
                    ax.text(0.5, 0.5, f'Plot unavailable: {did}',
                            ha='center', va='center', transform=ax.transAxes)
                    ax.axis('off')

            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
