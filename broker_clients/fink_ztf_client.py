"""Fink broker client for the ZTF alert stream (api.ztf.fink-portal.org).

Why this exists: during the 2026-08 Chilean storm the Rubin/LSST alert stream
(Cerro Pachón) went dark on 2026-07-14 and the Chile-hosted ALeRCE broker froze
on 2026-07-08. ZTF is at Palomar (California) and Fink is hosted in France
(IN2P3), so this path is independent of the Chilean outage and stays live —
making it the working discovery ingress while Rubin is down.

It reuses ``FinkLSSTClient``'s HTTP transport (``_post`` retry/backoff,
``available``) but the ZTF schema differs substantially from Rubin, so nearly
every query/parse method is overridden:

  Rubin (LSST portal)                ZTF portal
  ------------------------------     -----------------------------------
  r:/f:/v: column prefixes           i: (instrument) / d: (derived) / v:
  r:diaObjectId (numeric)            i:objectId (string, "ZTF26...")
  r:midpointMjdTai (MJD)             i:jd (Julian Date → mjd = jd - 2400000.5)
  r:band (string g/r/i)              i:fid (int 1/2/3 = g/r/i)
  r:psfFlux (nJy → mag)              i:magpsf (already AB mag; no conversion)
  /sources + /fp for light curves    /objects (single call; /sources is 404)
  /tags for discovery                /latests (class = Fink classification)
  f:clf_snnSnVsOthers_score          d:snn_snia_vs_nonia / d:rf_snia_vs_nonia
"""

import logging
from datetime import timedelta as _timedelta
from typing import Optional

import numpy as np
import pandas as pd

from .fink_client import FinkLSSTClient

logger = logging.getLogger(__name__)

FINK_ZTF_URL = "https://api.ztf.fink-portal.org"

# ZTF filter id -> band
_ZTF_FID_BAND = {1: "g", 2: "r", 3: "i"}
_JD_TO_MJD = 2400000.5

# Fink classification labels (from /api/v1/classes) that flag SN-like transients
# worth spectroscopic follow-up. "Early SN Ia candidate" is the highest-value.
SN_FINK_CLASSES = ("Early SN Ia candidate", "SN candidate", "(TNS) SN Ia")


class FinkZTFClient(FinkLSSTClient):
    """Client for Fink's ZTF alert API — the live downtime discovery ingress.

    Inherits the ``_post`` transport and ``available`` probe from
    ``FinkLSSTClient``; overrides the schema-specific query/parse methods for
    ZTF (jd→mjd, fid→band, magpsf-is-magnitude, /objects light curves,
    /latests discovery).
    """

    def __init__(self, base_url: str = FINK_ZTF_URL, timeout: int = 60,
                 cache_dir: str = './cache/data',
                 max_retries: int = 3, retry_backoff_base: float = 1.0):
        super().__init__(base_url=base_url, timeout=timeout, cache_dir=cache_dir,
                         max_retries=max_retries, retry_backoff_base=retry_backoff_base)
        self.broker_name = 'Fink-ZTF'

    # ------------------------------------------------------------------
    # Schema translation
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Rename ZTF i:/d: columns to standard names and derive mjd + band.

        ZTF quotes ``i:magpsf`` directly in AB magnitudes, so unlike the Rubin
        client there is no flux→mag step; ``magnitude`` is populated here.
        """
        rename = {
            "i:objectId": "objectId",
            "i:jd": "jd",
            "i:magpsf": "magnitude",
            "i:sigmapsf": "mag_err",
            "i:fid": "fid",
            "i:ra": "ra",
            "i:dec": "dec",
            "i:diffmaglim": "diffmaglim",
            "i:isdiffpos": "isdiffpos",
            "i:magnr": "magnr",
            "d:classification": "classification",
            "d:snn_snia_vs_nonia": "snn_snia_vs_nonia",
            "d:snn_sn_vs_all": "snn_sn_vs_all",
            "d:rf_snia_vs_nonia": "rf_snia_vs_nonia",
            "d:tns": "tns",
            "d:cdsxmatch": "cdsxmatch",
            "v:separation_degree": "separation_degree",
        }
        out = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        if "jd" in out.columns:
            out["mjd"] = pd.to_numeric(out["jd"], errors="coerce") - _JD_TO_MJD
        if "fid" in out.columns:
            out["band"] = pd.to_numeric(out["fid"], errors="coerce").map(
                _ZTF_FID_BAND).fillna("?")
        return out

    # ------------------------------------------------------------------
    # Light curves — single /api/v1/objects call (ZTF has no /sources or /fp)
    # ------------------------------------------------------------------
    def get_light_curve(self, object_id: str,
                        include_forced: bool = False) -> Optional[pd.DataFrame]:
        """Full ZTF light curve for one objectId via ``/api/v1/objects``.

        Returns one row per alert epoch with mjd/band/magnitude/mag_err.
        ``include_forced`` is accepted for interface parity but ignored (the ZTF
        portal serves alert-epoch photometry from this single endpoint).

        None-vs-empty contract mirrors ``_post``: ``None`` on transport error,
        an empty DataFrame when the object simply has no usable detections.
        """
        data = self._post("/api/v1/objects", {"objectId": str(object_id)})
        if data is None:
            return None
        if len(data) == 0:
            return pd.DataFrame()
        df = self._normalize_columns(pd.DataFrame(data))
        # Keep real positive-difference detections with a measured magnitude.
        if "magnitude" in df.columns:
            df = df[pd.to_numeric(df["magnitude"], errors="coerce").notna()]
        if "isdiffpos" in df.columns:
            df = df[df["isdiffpos"].isin(["t", "1", 1, True])]
        if df.empty:
            return pd.DataFrame()
        df = df.copy()
        df["source"] = "detection"
        df["survey"] = "ZTF"
        df = df.sort_values("mjd").reset_index(drop=True)
        logger.info("Fink-ZTF light curve for %s: %d points, %s, MJD %.1f-%.1f",
                    object_id, len(df), sorted(df["band"].unique()),
                    df["mjd"].min(), df["mjd"].max())
        return df

    # ------------------------------------------------------------------
    # Discovery — /api/v1/latests by Fink classification
    # ------------------------------------------------------------------
    def query_sn_candidates(self, finkclass: str = "SN candidate",
                            n: int = 1000) -> Optional[pd.DataFrame]:
        """Most-recent ``n`` alerts of a given Fink class via ``/api/v1/latests``.

        ``finkclass`` is one of :data:`SN_FINK_CLASSES` (or any label from
        ``/api/v1/classes``). Returns the raw Fink frame (one row per object's
        latest alert) or None on transport error.
        """
        data = self._post("/api/v1/latests", {"class": finkclass, "n": n})
        if data is None:
            return None
        df = pd.DataFrame(data)
        logger.info("Fink-ZTF /latests class=%r: %d rows", finkclass, len(df))
        return df

    def fetch_fresh_sn_candidates(self, mjd_now: float, days_back: float = 15.0,
                                  max_mag: float = 21.5, dec_max: float = 22.0,
                                  min_ia_score: float = 0.0,
                                  classes=SN_FINK_CLASSES,
                                  n_per_class: int = 1000) -> pd.DataFrame:
        """Fresh SN candidates from the live ZTF stream, wide-mode selection.

        Pulls each SN class from ``/latests``, dedups by objectId (keeping the
        highest Ia score), converts jd→mjd and fid→band, and applies the
        wide-sky cuts: last detection within ``days_back`` of ``mjd_now``,
        dec ≤ ``dec_max`` (Magellan-observable), magpsf ≤ ``max_mag``, and Ia
        score ≥ ``min_ia_score``.

        Returns a DataFrame with objectId, ra, dec, mag, band, mjd (last
        detection), ia_score, sn_score, fink_class. Empty (not None) if the
        query worked but nothing survives the cuts.
        """
        frames = []
        for cls in classes:
            df = self.query_sn_candidates(finkclass=cls, n=n_per_class)
            if df is not None and len(df):
                df = self._normalize_columns(df)
                df["fink_class"] = cls
                frames.append(df)
        if not frames:
            return pd.DataFrame()

        cat = pd.concat(frames, ignore_index=True)
        # Effective Ia score: SNN Ia-vs-nonIa, fall back to RF.
        ia = pd.to_numeric(cat.get("snn_snia_vs_nonia"), errors="coerce")
        rf = pd.to_numeric(cat.get("rf_snia_vs_nonia"), errors="coerce")
        cat["ia_score"] = ia.fillna(rf)
        cat["sn_score"] = pd.to_numeric(cat.get("snn_sn_vs_all"), errors="coerce")
        for col in ("ra", "dec", "magnitude", "mjd"):
            cat[col] = pd.to_numeric(cat.get(col), errors="coerce")
        # Prior spectroscopy: Fink's d:tns carries the TNS spectroscopic type
        # (e.g. "SN Ia") when the object has been classified, else blank. This
        # is the "already followed up spectroscopically?" signal, free in the
        # /latests payload — non-empty means a classification spectrum exists.
        cat["tns_type"] = (cat.get("tns", "").fillna("").astype(str).str.strip()
                           if "tns" in cat.columns else "")
        cat["tns_classified"] = cat["tns_type"].astype(bool) & (cat["tns_type"] != "")

        # Dedup by object, keeping the row with the best Ia score.
        cat = (cat.sort_values("ia_score", ascending=False, na_position="last")
                  .drop_duplicates(subset="objectId", keep="first"))

        n0 = len(cat)
        mask = (
            cat["mjd"].ge(mjd_now - days_back)
            & cat["dec"].le(dec_max)
            & cat["magnitude"].le(max_mag)
            & cat["ia_score"].fillna(0.0).ge(min_ia_score)
        )
        out = cat[mask].copy()
        logger.info("Fink-ZTF fresh SN: %d raw -> %d after cuts "
                    "(days_back<=%.0f, dec<=%.0f, mag<=%.1f, ia>=%.2f)",
                    n0, len(out), days_back, dec_max, max_mag, min_ia_score)
        out = out.sort_values("ia_score", ascending=False, na_position="last")
        keep = ["objectId", "ra", "dec", "magnitude", "band", "mjd",
                "ia_score", "sn_score", "fink_class", "classification",
                "tns_type", "tns_classified"]
        return out[[c for c in keep if c in out.columns]].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Cone search — ZTF portal /api/v1/conesearch (NOT the inherited Rubin one)
    # ------------------------------------------------------------------
    def cone_search(self, ra: float, dec: float, radius_arcsec: float = 5.0,
                    n: int = 1000, startdate: Optional[str] = None,
                    window_days: float = 200.0,
                    lookback_days: float = 90.0) -> Optional[pd.DataFrame]:
        """Positional search on Fink's ZTF portal ``/api/v1/conesearch``.

        The inherited :meth:`FinkLSSTClient.cone_search` is wrong for the ZTF
        portal in two ways:

        * **It sends ``n``.** The ZTF ``/api/v1/conesearch`` does not accept a
          result-limit ``n``; when the key is present the query returns ZERO
          rows even at an object's exact position. (Empirically ``{ra,dec,
          radius:5}`` finds ZTF24abkllyo at 0.06", but adding ``n:5`` returns
          nothing.) ``n`` here is applied CLIENT-side as a nearest-``n`` cap.
        * **The plain query misses some very recent objects.** The default
          (no-date) conesearch hits a rolling spatial index that lags ingest,
          so freshly-added objects (e.g. ZTF26abjqico, first detection days
          ago) return 0 rows even at their exact position. Supplying
          ``startdate`` + ``window`` routes the request to the date-partitioned
          path, which does find them. Conversely, some older objects only
          answer on the plain (no-date) path. We therefore try the plain query
          first and fall back to a date-windowed query when it is empty — the
          union finds all of them.

        ``radius`` is in **arcseconds** (the ZTF portal's unit; a 5 here means
        5"). None-vs-empty contract: ``None`` on transport error, an empty
        DataFrame on a successful zero-row result.
        """
        payload = {"ra": float(ra), "dec": float(dec),
                   "radius": float(radius_arcsec)}
        data = self._post("/api/v1/conesearch", payload)
        if data is None:
            return None  # transport error on the primary query
        if len(data) == 0:
            # Fallback to the date-partitioned path for recent objects.
            if startdate is None:
                from astropy.time import Time
                startdate = (Time.now().datetime.date()
                             - _timedelta(days=lookback_days)).isoformat()
            fb = self._post("/api/v1/conesearch",
                            {"ra": float(ra), "dec": float(dec),
                             "radius": float(radius_arcsec),
                             "startdate": startdate, "window": window_days})
            # Primary already gave a valid zero-row answer; a transport error on
            # the fallback should not turn that into the error sentinel.
            data = fb if fb is not None else []
        if len(data) == 0:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        if "v:separation_degree" in df.columns:
            df = df.sort_values("v:separation_degree").reset_index(drop=True)
        if n and len(df) > n:
            df = df.head(int(n)).reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Cross-match / classification enrichment (ZTF field names)
    # ------------------------------------------------------------------
    def _object_ia_score(self, object_id: str) -> float:
        """Best SNN Ia-vs-nonIa score for one objectId (RF as fallback).

        The ZTF ``/api/v1/conesearch`` returns only positional/classification
        columns (``i:objectId``, ``i:ra``, ``i:dec``, ``i:jd``,
        ``d:classification``, ``v:separation_degree``) — never the
        ``d:snn_snia_vs_nonia`` float. That score lives per-alert in
        ``/api/v1/objects``; we pull it there and keep the best (max) value.
        Returns NaN on transport error or if no score exists.
        """
        data = self._post("/api/v1/objects",
                           {"objectId": str(object_id),
                            "columns": "i:objectId,i:jd,d:snn_snia_vs_nonia,"
                                       "d:rf_snia_vs_nonia"})
        if not data:
            return np.nan
        df = pd.DataFrame(data)
        for col in ("d:snn_snia_vs_nonia", "d:rf_snia_vs_nonia"):
            vals = pd.to_numeric(df.get(col), errors="coerce").dropna()
            if len(vals):
                return float(vals.max())
        return np.nan

    def get_classifications(self, candidates_df: pd.DataFrame,
                            radius_arcsec: float = 2.0) -> pd.DataFrame:
        """Attach Fink-ZTF Ia score + objectId to candidates by cone-match.

        Overrides the Rubin version to read ZTF fields (``i:objectId``,
        ``v:separation_degree``). The Ia score (``d:snn_snia_vs_nonia``) is not
        served by the ZTF conesearch, so it is fetched for the matched objectId
        via :meth:`_object_ia_score` (``/api/v1/objects``).
        """
        scores, ids, seps = [], [], []
        for _, row in candidates_df.iterrows():
            ra, dec = row.get("ra"), row.get("dec")
            if pd.isna(ra) or pd.isna(dec):
                scores.append(np.nan); ids.append(None); seps.append(np.nan)
                continue
            try:
                res = self.cone_search(float(ra), float(dec),
                                       radius_arcsec=radius_arcsec, n=5)
                if res is not None and len(res):
                    if "v:separation_degree" in res.columns:
                        res = res.sort_values("v:separation_degree")
                    best = res.iloc[0]
                    oid = str(best.get("i:objectId", ""))
                    ids.append(oid)
                    sep = best.get("v:separation_degree", np.nan)
                    seps.append(float(sep) * 3600 if pd.notna(sep) else np.nan)
                    # conesearch omits the Ia score; look it up per objectId.
                    sc = best.get("d:snn_snia_vs_nonia", np.nan)
                    if pd.isna(sc) and oid:
                        sc = self._object_ia_score(oid)
                    scores.append(float(sc) if pd.notna(sc) else np.nan)
                else:
                    scores.append(np.nan); ids.append(None); seps.append(np.nan)
            except Exception as e:
                logger.debug("Fink-ZTF classification lookup failed (%.4f,%.4f): %s",
                             ra, dec, e)
                scores.append(np.nan); ids.append(None); seps.append(np.nan)
        out = candidates_df.copy()
        out["ztf_ia_score"] = scores
        out["ztf_objectId"] = ids
        out["ztf_sep_arcsec"] = seps
        return out

    # ------------------------------------------------------------------
    # BaseBrokerClient interface
    # ------------------------------------------------------------------
    def query_alerts(self, class_name: str = 'SN Ia',
                     min_probability: float = 0.5,
                     days_back: int = 15, **kwargs) -> pd.DataFrame:
        """Fresh SN candidates as a plain DataFrame (BaseBrokerClient contract).

        Maps ``class_name`` onto the Fink class and filters by Ia score. Needs
        a reference MJD; pass ``mjd_now`` in kwargs, else uses today.
        """
        cls = ("Early SN Ia candidate" if "Ia" in class_name
               else "SN candidate")
        mjd_now = kwargs.get("mjd_now")
        if mjd_now is None:
            from astropy.time import Time
            mjd_now = Time.now().mjd
        return self.fetch_fresh_sn_candidates(
            mjd_now=mjd_now, days_back=days_back,
            min_ia_score=min_probability, classes=(cls,),
            max_mag=kwargs.get("max_mag", 21.5),
            dec_max=kwargs.get("dec_max", 22.0))
