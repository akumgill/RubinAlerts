"""Plotting utilities for light curves and visualizations."""

import logging
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# Colors by band
BAND_COLORS = {
    'u': '#7b2cbf',   # violet
    'g': '#2ca02c',   # green
    'r': '#d62728',   # red
    'i': '#8c564b',   # brown
    'z': '#9467bd',   # purple
    'y': '#e377c2',   # pink
    'Y': '#e377c2',
    'R': '#d62728',
    'G': '#2ca02c',
    'o': '#ff7f0e',   # ATLAS orange
    'c': '#17becf',   # ATLAS cyan
    '1': '#2ca02c',   # ZTF g (fid=1)
    '2': '#d62728',   # ZTF r (fid=2)
    '3': '#e377c2',   # ZTF i (fid=3)
}

# Band colors for ATLAS named bands (from get_light_curve)
BAND_COLORS['ATLAS-cyan'] = '#17becf'
BAND_COLORS['ATLAS-orange'] = '#ff7f0e'

# Marker styles by survey
SURVEY_MARKERS = {
    'ZTF': 'o',            # circles
    'Rubin': 's',          # squares
    'ALeRCE': '^',         # triangles
    'ATLAS': 'P',          # plus (filled)
    'unknown': 'D',        # diamonds
}


class PlottingUtils:
    """Utilities for scientific plotting."""

    @staticmethod
    def prepare_light_curve(lc_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """Prepare light curve DataFrame for plotting.

        Ensures mjd, magnitude, mag_err, band, survey columns exist.
        Returns the cleaned DataFrame (not a dict).
        """
        if lc_df is None or len(lc_df) == 0:
            return None

        try:
            df = lc_df.copy()

            # Ensure required columns
            if 'mjd' not in df.columns:
                for col in df.columns:
                    if 'mjd' in col.lower() or 'jd' in col.lower():
                        df['mjd'] = df[col]
                        break

            if 'magnitude' not in df.columns:
                for col in df.columns:
                    cl = col.lower()
                    if 'mag' in cl and 'err' not in cl and 'sig' not in cl:
                        df['magnitude'] = df[col]
                        break

            if 'band' not in df.columns:
                for col in df.columns:
                    cl = col.lower()
                    if 'band' in cl or 'filter' in cl or 'passband' in cl or 'fid' in cl:
                        df['band'] = df[col]
                        break

            if 'mag_err' not in df.columns:
                df['mag_err'] = 0.0

            if 'survey' not in df.columns:
                df['survey'] = 'unknown'

            # Drop rows without essential data
            if 'mjd' in df.columns and 'magnitude' in df.columns:
                df = df.dropna(subset=['mjd', 'magnitude'])
                df = df.sort_values('mjd').reset_index(drop=True)
                return df

            return None

        except Exception as e:
            logger.warning(f"Error preparing light curve: {e}")
            return None

    @staticmethod
    def plot_light_curve_matplotlib(lc_df: pd.DataFrame,
                                    title: str = "Light Curve",
                                    figsize: tuple = (14, 7)):
        """Create a matplotlib light curve plot, colored by band, shaped by survey.

        Args:
            lc_df: DataFrame with mjd, magnitude, mag_err, band, survey columns.
            title: Plot title.
            figsize: Figure size.

        Returns:
            matplotlib Figure object.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize)

        if lc_df is None or len(lc_df) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=16)
            return fig

        # Ensure band column is string
        lc_df = lc_df.copy()
        lc_df['band'] = lc_df['band'].astype(str)
        lc_df['survey'] = lc_df['survey'].fillna('unknown').astype(str)

        # Plot each (survey, band) combination
        plotted_surveys = set()
        plotted_bands = set()

        for (survey, band), group in lc_df.groupby(['survey', 'band']):
            color = BAND_COLORS.get(str(band), '#333333')
            marker = SURVEY_MARKERS.get(str(survey), 'o')
            ms = 8 if survey == 'Rubin' else 6

            # Build label parts
            label_parts = []
            if band not in plotted_bands:
                label_parts.append(f'{band}-band')
                plotted_bands.add(band)
            if survey not in plotted_surveys:
                label_parts.append(f'[{survey}]')
                plotted_surveys.add(survey)
            label = f'{band} ({survey})'

            mag_err = group['mag_err'].values
            if np.all(mag_err == 0) or np.all(np.isnan(mag_err)):
                mag_err = None

            ax.errorbar(
                group['mjd'].values,
                group['magnitude'].values,
                yerr=mag_err,
                fmt=marker,
                label=label,
                color=color,
                markersize=ms,
                capsize=2,
                alpha=0.85,
                linewidth=0,
                elinewidth=1,
            )

        ax.set_xlabel('MJD', fontsize=12)
        ax.set_ylabel('Magnitude', fontsize=12)
        ax.invert_yaxis()
        ax.legend(fontsize=9, loc='best', ncol=2)
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_light_curve_plotly(lc_df: pd.DataFrame,
                                title: str = "Light Curve"):
        """Create interactive Plotly light curve, colored by band, shaped by survey."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            logger.warning("Plotly not available")
            return None

        fig = go.Figure()

        if lc_df is None or len(lc_df) == 0:
            return fig

        lc_df = lc_df.copy()
        lc_df['band'] = lc_df['band'].astype(str)
        lc_df['survey'] = lc_df['survey'].fillna('unknown').astype(str)

        plotly_symbols = {
            'ZTF': 'circle',
            'Rubin': 'square',
            'ALeRCE': 'triangle-up',
            'unknown': 'diamond',
        }

        for (survey, band), group in lc_df.groupby(['survey', 'band']):
            color = BAND_COLORS.get(str(band), '#333333')
            symbol = plotly_symbols.get(str(survey), 'circle')
            size = 10 if survey == 'Rubin' else 7

            mag_err = group['mag_err'].values
            show_err = not (np.all(mag_err == 0) or np.all(np.isnan(mag_err)))

            fig.add_trace(go.Scatter(
                x=group['mjd'].values,
                y=group['magnitude'].values,
                error_y=dict(array=mag_err, visible=show_err) if show_err else None,
                mode='markers',
                name=f'{band} ({survey})',
                marker=dict(color=color, size=size, symbol=symbol),
                hovertemplate=(
                    f'<b>{band}-band ({survey})</b><br>'
                    'MJD: %{x:.3f}<br>'
                    'Mag: %{y:.3f}<br>'
                    '<extra></extra>'
                ),
            ))

        fig.update_layout(
            title=title,
            xaxis_title='MJD',
            yaxis_title='Magnitude',
            yaxis_autorange='reversed',
            hovermode='closest',
            height=550,
            template='plotly_white',
            legend=dict(font=dict(size=10)),
        )

        return fig

    @staticmethod
    def create_classification_comparison_plot(classifications: Dict[str, Dict[str, float]]):
        """Create comparison bar chart of classifications from different brokers."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            return None

        class_types = set()
        for broker_class in classifications.values():
            class_types.update(broker_class.keys())
        class_types = sorted(list(class_types))

        fig = go.Figure()

        for broker, probs in classifications.items():
            values = [probs.get(cls, 0) for cls in class_types]
            fig.add_trace(go.Bar(
                name=broker,
                x=class_types,
                y=values,
                text=[f'{v:.2f}' for v in values],
                textposition='auto',
            ))

        fig.update_layout(
            title='Classification Comparison',
            xaxis_title='Classification',
            yaxis_title='Probability',
            barmode='group',
            height=400,
            template='plotly_white'
        )

        return fig
