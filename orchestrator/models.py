"""Data models for the LLAMAS observing plan generator."""

import logging
from dataclasses import dataclass, field
from functools import cached_property
from typing import Optional, List

from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u

logger = logging.getLogger(__name__)


@dataclass
class Target:
    """Canonical target representation for scheduling."""

    # Required
    name: str = ''
    ra_deg: float = float('nan')
    dec_deg: float = float('nan')

    # Optional with defaults
    priority: int = 3
    mag: float = float('nan')
    mag_filter: str = ''
    redshift: float = float('nan')
    exposure_minutes: float = float('nan')
    moon_constraint: str = 'any'
    notes: str = ''
    source: str = ''
    merit_score: float = float('nan')

    # Planner-populated fields
    transit_time: Optional[Time] = field(default=None, repr=False)
    min_airmass: float = float('nan')
    window_start: Optional[Time] = field(default=None, repr=False)
    window_end: Optional[Time] = field(default=None, repr=False)
    window_hours: float = 0.0

    @cached_property
    def coord(self) -> SkyCoord:
        """SkyCoord from ra_deg, dec_deg (cached)."""
        return SkyCoord(ra=self.ra_deg * u.deg, dec=self.dec_deg * u.deg)

    @property
    def ra_hms(self) -> str:
        """RA formatted as HH:MM:SS.ss."""
        hms = self.coord.ra.hms
        return '{:02d}:{:02d}:{:05.2f}'.format(
            int(hms.h), int(abs(hms.m)), abs(hms.s))

    @property
    def dec_dms(self) -> str:
        """Dec formatted as +DD:MM:SS.s."""
        dms = self.coord.dec.dms
        sign = '+' if dms.d >= 0 else '-'
        return '{}{:02d}:{:02d}:{:04.1f}'.format(
            sign, int(abs(dms.d)), int(abs(dms.m)), abs(dms.s))


@dataclass
class ScheduledEntry:
    """A target assigned to a specific time slot in the observing plan."""

    target: Target = field(default_factory=Target)
    start: Optional[Time] = None
    end: Optional[Time] = None
    airmass: float = float('nan')
    exp_str: str = ''
    n_exp: int = 1
    exp_sec: int = 0


@dataclass
class ObsPlan:
    """Complete observing plan for a single night."""

    date: str = ''
    evening_twilight: Optional[Time] = None
    morning_twilight: Optional[Time] = None
    moon_phase: str = 'grey'

    scheduled: List[ScheduledEntry] = field(default_factory=list)
    backup: List[Target] = field(default_factory=list)

    standards_start: Optional[dict] = None
    standards_end: Optional[dict] = None

    @property
    def night_duration_hours(self) -> float:
        """Total dark time in hours."""
        if self.evening_twilight is None or self.morning_twilight is None:
            return 0.0
        return (self.morning_twilight - self.evening_twilight).to(u.hour).value

    @property
    def scheduled_minutes(self) -> float:
        """Total scheduled observation time in minutes."""
        total = 0.0
        for entry in self.scheduled:
            if entry.start is not None and entry.end is not None:
                total += (entry.end - entry.start).to(u.minute).value
        return total

    @property
    def efficiency(self) -> float:
        """Fraction of night used for scheduled observations."""
        night_min = self.night_duration_hours * 60.0
        if night_min <= 0:
            return 0.0
        return self.scheduled_minutes / night_min
