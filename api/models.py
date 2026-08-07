"""Data model for the target-submission API (see the interface spec).

Plain dataclasses — no pydantic dependency. Validation is done in the service
layer so the model stays a transport container.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Optional

# Per-program priority tiers. P0 = "observe tonight, for sure" (target-of-
# opportunity / mandatory): reserves a slot and bypasses the budget throttle.
# P1..P5 are ordinary tiers on the collaboration's shared 1-5 scale (matches the
# Villar group's submissions): P1 highest, P5 lowest. Lower index = higher
# priority. P4/P5 are the low/opportunistic rungs (e.g. nearby fillers during a
# survey gap that should only run on otherwise-idle sky).
TIERS = ("P0", "P1", "P2", "P3", "P4", "P5")

# Statuses a submission moves through.
STATUSES = ("queued", "scheduled", "observed", "withdrawn")

# Instruments. LDSS3 and LLAMAS are budgeted/scheduled as two parallel systems;
# EITHER means the target may go on whichever instrument's night comes first.
INSTRUMENTS = ("LLAMAS", "LDSS3", "EITHER")


@dataclass
class Target:
    """One submitted target.

    Fields the submitter sets are up top; fields the system fills in are
    below the divider. Exactly one of (``name``) or (``ra`` & ``dec``) must be
    supplied; ``name`` is resolved to coordinates, which are the canonical key.
    """

    # --- submitted ---
    priority: str = "P2"
    instrument: str = "LLAMAS"          # LLAMAS | LDSS3 | EITHER — which
                                        # parallel budget universe this draws from
    name: Optional[str] = None
    ra: float = float("nan")            # deg, ICRS
    dec: float = float("nan")           # deg, ICRS
    mag: float = float("nan")           # anticipated, at observation
    band: str = "r"                     # band `mag` is in
    redshift: float = float("nan")
    exposure_minutes: float = float("nan")   # optional override
    valid_until: Optional[str] = None   # ISO date; drop after
    notes: str = ""

    # --- system-assigned ---
    id: Optional[int] = None
    program: str = ""                   # from the API key, never submitted
    status: str = "queued"
    canonical_ra: float = float("nan")  # resolved position (the dedup key)
    canonical_dec: float = float("nan")
    resolved_from: str = ""             # 'coords' | 'name:<scheme>' | ...

    def has_coords(self) -> bool:
        return math.isfinite(self.ra) and math.isfinite(self.dec)

    def to_dict(self) -> dict:
        d = asdict(self)
        # NaNs -> None so JSON is clean
        for k, v in d.items():
            if isinstance(v, float) and math.isnan(v):
                d[k] = None
        return d
