"""Utility functions for coordinates, plotting, catalog queries, and corrections."""

from .coordinates import CoordinateUtils
from .plotting import PlottingUtils
from .catalog_query import CatalogQuery
from .extinction import get_extinction, get_extinction_batch, correct_magnitude
from .ned_query import query_ned_redshift, query_ned_batch

__all__ = [
    'CoordinateUtils',
    'PlottingUtils',
    'CatalogQuery',
    'get_extinction',
    'get_extinction_batch',
    'correct_magnitude',
    'query_ned_redshift',
    'query_ned_batch',
]
