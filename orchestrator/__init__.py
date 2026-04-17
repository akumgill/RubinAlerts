"""LLAMAS observing plan generator for MAGNETS."""

from .models import Target, ObsPlan
from .planner import create_schedule

__all__ = ['Target', 'ObsPlan', 'create_schedule']
