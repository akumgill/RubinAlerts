"""LIVE end-to-end test against the real Fink LSST API.

Marked @pytest.mark.live and SKIPPED by default. Run with:

    pytest --run-live tests/test_fink_live.py

Purpose: prove we genuinely retrieve Fink photometry end-to-end — discover
real SN candidates, pick one, fetch its light curve (detections + forced
photometry), and assert we got real, finite-flux data. It does NOT hardcode a
transient ID (those age out); it discovers candidates at runtime and is
tolerant of which specific objects exist.
"""

import math

import pandas as pd
import pytest

from broker_clients.fink_client import FinkLSSTClient


def _find_object_id_column(df: pd.DataFrame):
    for col in ("r:diaObjectId", "diaObjectId", "i:objectId", "objectId"):
        if col in df.columns:
            return col
    return None


@pytest.mark.live
def test_fink_live_end_to_end():
    client = FinkLSSTClient()

    if not client.available:
        pytest.skip("Fink LSST API not reachable from this environment")

    # ---- Discover real SN candidates via the tag search the pipeline uses ---
    candidates = client.query_sn_candidates(
        tag="sn_near_galaxy_candidate", n=200)
    assert candidates is not None, "Fink candidate discovery returned None (error)"
    assert len(candidates) > 0, "Fink returned zero SN candidates"

    id_col = _find_object_id_column(candidates)
    assert id_col is not None, (
        f"No diaObjectId column among {list(candidates.columns)[:20]}")

    # ---- Walk candidates until one yields a real light curve ---------------
    obj_ids = (
        candidates[id_col].dropna().astype(str).unique().tolist()
    )
    assert obj_ids, "No usable diaObjectIds among discovered candidates"

    lc = None
    used_oid = None
    for oid in obj_ids[:25]:
        result = client.get_light_curve(oid, include_forced=True)
        # None = transport error; empty DataFrame = queried OK but no rows.
        if result is not None and len(result) > 0:
            lc = result
            used_oid = oid
            break

    assert lc is not None and len(lc) > 0, (
        "No discovered candidate yielded any Fink photometry — "
        "expected at least one with a real light curve")

    # ---- Assert the data is real ------------------------------------------
    assert "flux" in lc.columns, f"light curve missing flux: {list(lc.columns)}"
    finite_flux = pd.to_numeric(lc["flux"], errors="coerce").dropna()
    assert len(finite_flux) > 0, "no finite flux values in light curve"
    assert all(math.isfinite(v) for v in finite_flux), "non-finite flux present"

    # Ideally forced photometry contributed and we span multiple epochs/bands.
    if "source" in lc.columns:
        sources = set(lc["source"].unique())
        assert "forced_phot" in sources or "detection" in sources
    n_bands = lc["band"].nunique() if "band" in lc.columns else 0
    print(f"\nFink live: object {used_oid}: {len(lc)} points across "
          f"{n_bands} band(s); sources={set(lc.get('source', []))}")
