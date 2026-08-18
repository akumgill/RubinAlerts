"""Name -> coordinates resolver for the target-submission API.

The queue accepts name-only submissions ("Provide a name or an RA + Dec
pair"), but TargetQueueService only resolves names through an injected
resolver callable — and production historically wired none in, so EVERY
name-only submission failed with "no coordinates and no resolvable name"
(the "long ZTF name failed to resolve" bug from the 2026-08 dry-run prep;
in fact no name of any kind resolved).

This module is that resolver. It uses the public Fink API (no credentials,
already a pinned dependency path: requests) to handle the two name families
observers actually type:

  * ZTF object ids  (ZTF26abmiytv)      -> Fink /api/v1/objects
  * TNS designations (2026ydy, AT 2026ydy, SN2026ydy)
                                        -> Fink /api/v1/resolver (tns)

The Fink TNS resolver matches only the EXACT full name ("AT 2026ydy"), so
bare cores and unspaced forms are expanded into the AT/SN variants and tried
in order. Any network problem or unrecognized name returns None; the service
then tells the submitter to send ra/dec explicitly.
"""
from __future__ import annotations

import logging
import math
import re
from typing import Optional

import requests

logger = logging.getLogger(__name__)

FINK_ZTF_BASE = "https://api.ztf.fink-portal.org"
TIMEOUT_S = 10

_ZTF_RE = re.compile(r"^ztf\d{2}[a-z]{7}$", re.IGNORECASE)
# "2026ydy", "AT 2026ydy", "SN2026ydy", case-insensitive, optional space
_TNS_RE = re.compile(r"^(?:(at|sn)\s*)?(\d{4}[a-z]{1,4})$", re.IGNORECASE)


def _post(endpoint: str, payload: dict) -> Optional[list]:
    try:
        r = requests.post(f"{FINK_ZTF_BASE}{endpoint}", json=payload,
                          timeout=TIMEOUT_S)
        if r.status_code != 200:
            logger.warning("resolver %s -> HTTP %s", endpoint, r.status_code)
            return None
        body = r.json()
        return body if isinstance(body, list) else None
    except Exception as e:  # timeouts, DNS, bad JSON — resolve to "unknown"
        logger.warning("resolver %s failed: %s", endpoint, e)
        return None


def _f(v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x


def _resolve_ztf(oid: str) -> Optional[dict]:
    rows = _post("/api/v1/objects",
                 {"objectId": oid, "columns": "i:ra,i:dec,i:magpsf,i:jd",
                  "output-format": "json"})
    if not rows:
        return None
    latest = max(rows, key=lambda r: _f(r.get("i:jd")))
    ra, dec = _f(latest.get("i:ra")), _f(latest.get("i:dec"))
    if not (math.isfinite(ra) and math.isfinite(dec)):
        return None
    out = {"ra": ra, "dec": dec, "scheme": "fink-ztf"}
    mag = _f(latest.get("i:magpsf"))
    if math.isfinite(mag):
        out["mag"] = mag
    return out


def _resolve_tns(prefix: Optional[str], core: str) -> Optional[dict]:
    # Fink's TNS resolver wants the exact full name; try the stated prefix
    # first, then both designation families.
    prefixes = [prefix.upper()] if prefix else []
    prefixes += [p for p in ("AT", "SN") if p not in prefixes]
    for p in prefixes:
        rows = _post("/api/v1/resolver",
                     {"resolver": "tns", "name": f"{p} {core.lower()}"})
        if not rows:
            continue
        hit = rows[0]
        ra, dec = _f(hit.get("d:ra")), _f(hit.get("d:declination"))
        if not (math.isfinite(ra) and math.isfinite(dec)):
            continue
        out = {"ra": ra, "dec": dec, "scheme": "tns"}
        z = _f(hit.get("d:redshift"))
        if math.isfinite(z):
            out["redshift"] = z
        return out
    return None


def resolve_name(name: str) -> Optional[dict]:
    """Resolve a transient name to {ra, dec, mag?, redshift?, scheme}.

    Returns None when the name is unrecognized or the lookup fails — the
    caller (TargetQueueService.submit) turns that into a "send ra/dec"
    error for the submitter.
    """
    if not name:
        return None
    name = name.strip()
    if _ZTF_RE.match(name):
        oid = "ZTF" + name[3:].lower()
        return _resolve_ztf(oid)
    m = _TNS_RE.match(name)
    if m:
        return _resolve_tns(m.group(1), m.group(2))
    logger.info("resolver: unrecognized name form %r", name)
    return None
