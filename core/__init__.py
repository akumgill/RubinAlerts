"""Core alert aggregation and management."""

from .alert_aggregator import AlertAggregator
from .peak_fitting import (
    PeakFitter,
    villar_flux,
    villar_peak_from_params,
    extract_villar_peaks,
)
from .magellan_planning import compute_merit, filter_observable_targets, write_magellan_catalog

__all__ = [
    'AlertAggregator',
    'PeakFitter',
    'villar_flux',
    'villar_peak_from_params',
    'extract_villar_peaks',
    'compute_merit',
    'filter_observable_targets',
    'write_magellan_catalog',
]
