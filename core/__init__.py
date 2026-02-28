"""Core alert aggregation and management."""

from .alert_aggregator import AlertAggregator
from .peak_fitting import PeakFitter
from .magellan_planning import compute_merit, filter_observable_targets, write_magellan_catalog

__all__ = [
    'AlertAggregator',
    'PeakFitter',
    'compute_merit',
    'filter_observable_targets',
    'write_magellan_catalog',
]
