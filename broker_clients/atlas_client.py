"""ATLAS forced photometry client.

Adapted from astrostubbs/ATLAS-snagger. Fetches forced photometry
from the ATLAS forced photometry server for transient candidates.

This is an enrichment source, not a primary search broker — ATLAS
doesn't classify objects, but provides independent light curves in
cyan (c) and orange (o) filters.

Requires ATLAS credentials: set ATLAS_USERNAME and ATLAS_PASSWORD
environment variables, or create ~/.atlas_credentials with:
    [atlas]
    username = your_username
    password = your_password

Register at: https://fallingstar-data.com/forcedphot/
"""

import io
import os
import re
import time
import logging
import configparser
from typing import Optional, Dict, Any, List, Tuple
import requests
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

ATLAS_BASE_URL = "https://fallingstar-data.com/forcedphot"
ATLAS_POLL_INTERVAL_SEC = 10
ATLAS_POLL_TIMEOUT_SEC = 300
ATLAS_BATCH_POLL_TIMEOUT_SEC = 1800  # 30 min for large batches
ATLAS_BATCH_SIZE = 100  # server limit per radeclist POST
ATLAS_MJD_MIN = 57000  # ~July 2015

ATLAS_NUMERIC_COLS = [
    'MJD', 'm', 'dm', 'uJy', 'duJy', 'RA', 'Dec', 'chi/N',
    'x', 'y', 'maj', 'min', 'phi', 'apfit', 'mag5sig', 'Sky',
]

CREDENTIALS_FILE = os.path.expanduser("~/.atlas_credentials")


def get_atlas_credentials():
    """Get ATLAS API credentials from environment or config file."""
    username = os.environ.get('ATLAS_USERNAME')
    password = os.environ.get('ATLAS_PASSWORD')

    if username and password:
        return username, password

    if os.path.exists(CREDENTIALS_FILE):
        config = configparser.ConfigParser()
        config.read(CREDENTIALS_FILE)
        if 'atlas' in config:
            username = config['atlas'].get('username')
            password = config['atlas'].get('password')
            if username and password:
                return username, password

    raise RuntimeError(
        "ATLAS credentials not found. Set ATLAS_USERNAME and ATLAS_PASSWORD "
        "environment variables, or create ~/.atlas_credentials with:\n"
        "  [atlas]\n  username = your_username\n  password = your_password\n"
        "Register at: https://fallingstar-data.com/forcedphot/"
    )


class AtlasClient:
    """ATLAS forced photometry client for enriching transient candidates."""

    def __init__(self):
        self.token = None
        self._available = None

    @property
    def available(self) -> bool:
        """Check if ATLAS credentials are configured."""
        if self._available is None:
            try:
                get_atlas_credentials()
                self._available = True
            except RuntimeError:
                self._available = False
        return self._available

    def _ensure_token(self):
        """Authenticate and cache the ATLAS API token."""
        if self.token is not None:
            return
        username, password = get_atlas_credentials()
        resp = requests.post(
            f"{ATLAS_BASE_URL}/api-token-auth/",
            data={'username': username, 'password': password},
        )
        if resp.status_code != 200:
            raise RuntimeError(f"ATLAS auth failed (HTTP {resp.status_code}): {resp.text}")
        self.token = resp.json().get('token')
        if not self.token:
            raise RuntimeError("ATLAS auth response missing 'token'")
        logger.info("ATLAS authenticated successfully")

    def _headers(self):
        return {'Authorization': f'Token {self.token}', 'Accept': 'application/json'}

    # ------------------------------------------------------------------
    # Single-target methods (kept for get_light_curve and direct use)
    # ------------------------------------------------------------------

    def fetch_photometry(self, ra: float, dec: float,
                         mjd_min: Optional[float] = None) -> Dict[str, pd.DataFrame]:
        """Fetch ATLAS forced photometry for a single coordinate.

        Returns dict with keys 'c' (cyan) and/or 'o' (orange),
        each containing a DataFrame of photometry. Either key may
        be absent if no data exists for that filter.
        """
        self._ensure_token()

        data = {'ra': ra, 'dec': dec}
        if mjd_min is not None:
            data['mjd_min'] = mjd_min

        task_url = self._submit_job(data)
        result_url = self._poll_job(task_url)
        text_data = self._download_results(result_url)
        self._cleanup(task_url)

        df = self._parse_data(text_data)
        return self._split_by_filter(df)

    def _submit_job(self, data: Dict) -> str:
        """Submit a single forced photometry job, handling rate limiting."""
        while True:
            resp = requests.post(
                f"{ATLAS_BASE_URL}/queue/",
                headers=self._headers(),
                data=data,
            )
            if resp.status_code == 201:
                task_url = resp.json().get('url')
                if not task_url:
                    raise RuntimeError("ATLAS queue response missing 'url'")
                return task_url
            elif resp.status_code == 429:
                detail = resp.json().get('detail', '')
                wait = self._parse_throttle_wait(detail)
                logger.info(f"ATLAS rate limited, waiting {wait}s")
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"ATLAS submit failed (HTTP {resp.status_code}): {resp.text}"
                )

    def _poll_job(self, task_url: str) -> str:
        """Poll until a single job completes. Returns result_url."""
        elapsed = 0
        while elapsed < ATLAS_POLL_TIMEOUT_SEC:
            resp = requests.get(task_url, headers=self._headers())
            resp.raise_for_status()
            data = resp.json()
            if data.get('finishtimestamp'):
                result_url = data.get('result_url')
                if not result_url:
                    raise RuntimeError("ATLAS task finished but no result_url")
                return result_url
            time.sleep(ATLAS_POLL_INTERVAL_SEC)
            elapsed += ATLAS_POLL_INTERVAL_SEC
        raise TimeoutError(f"ATLAS job timed out after {ATLAS_POLL_TIMEOUT_SEC}s")

    def _download_results(self, result_url: str) -> str:
        resp = requests.get(result_url, headers=self._headers())
        resp.raise_for_status()
        return resp.text

    def _cleanup(self, task_url: str):
        try:
            requests.delete(task_url, headers=self._headers())
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Batch submission methods
    # ------------------------------------------------------------------

    def _submit_batch(self, coords: List[Tuple[Any, float, float]],
                      mjd_min: Optional[float] = None) -> List[Tuple[Any, str]]:
        """Submit up to 100 coordinates in one POST via radeclist.

        Parameters
        ----------
        coords : list of (df_idx, ra, dec) tuples
        mjd_min : optional MJD lower bound (shared across all targets)

        Returns
        -------
        list of (df_idx, task_url) tuples
        """
        radeclist = "\n".join(f"{ra},{dec}" for _, ra, dec in coords)
        data = {'radeclist': radeclist}
        if mjd_min is not None:
            data['mjd_min'] = mjd_min

        while True:
            resp = requests.post(
                f"{ATLAS_BASE_URL}/queue/",
                headers=self._headers(),
                data=data,
            )
            if resp.status_code == 201:
                body = resp.json()
                tasks = body if isinstance(body, list) else [body]
                result = []
                for i, task in enumerate(tasks):
                    task_url = task.get('url')
                    if task_url and i < len(coords):
                        result.append((coords[i][0], task_url))
                return result
            elif resp.status_code == 429:
                detail = resp.json().get('detail', '')
                wait = self._parse_throttle_wait(detail)
                logger.info("ATLAS rate limited on batch submit, waiting %ds", wait)
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"ATLAS batch submit failed (HTTP {resp.status_code}): {resp.text}"
                )

    def _poll_all_tasks(self, task_map: Dict[Any, str],
                        timeout: int = ATLAS_BATCH_POLL_TIMEOUT_SEC
                        ) -> Dict[Any, str]:
        """Poll all tasks until complete or timeout.

        Parameters
        ----------
        task_map : dict of df_idx -> task_url
        timeout : max seconds to wait

        Returns
        -------
        dict of df_idx -> result_url (only for successfully completed tasks)
        """
        results = {}
        pending = dict(task_map)
        elapsed = 0
        total = len(pending)

        while pending and elapsed < timeout:
            newly_done = []
            for df_idx, task_url in pending.items():
                try:
                    resp = requests.get(task_url, headers=self._headers())
                    resp.raise_for_status()
                    data = resp.json()
                    if data.get('finishtimestamp'):
                        result_url = data.get('result_url')
                        if result_url:
                            results[df_idx] = result_url
                        else:
                            err = data.get('error_msg', 'no result_url')
                            logger.debug("ATLAS task finished without result: %s", err)
                        newly_done.append(df_idx)
                except Exception as e:
                    logger.debug("Error polling ATLAS task: %s", e)

            for key in newly_done:
                pending.pop(key)

            if pending:
                logger.info("ATLAS poll: %d/%d complete, %d pending (elapsed %ds)",
                            len(results), total, len(pending), elapsed)
                time.sleep(ATLAS_POLL_INTERVAL_SEC)
                elapsed += ATLAS_POLL_INTERVAL_SEC

        if pending:
            logger.warning("ATLAS: %d/%d tasks timed out after %ds",
                           len(pending), total, timeout)
        return results

    # ------------------------------------------------------------------
    # Enrichment (batch version)
    # ------------------------------------------------------------------

    def enrich_candidates(self, candidates_df: pd.DataFrame,
                          mjd_min: Optional[float] = None) -> pd.DataFrame:
        """Fetch ATLAS photometry for candidates using batch submission.

        Submits coordinates in batches of 100 via the radeclist API,
        polls all tasks in parallel, then downloads and summarizes results.

        Adds columns: atlas_n_cyan, atlas_n_orange, atlas_has_data.
        """
        df = candidates_df.copy()
        df['atlas_n_cyan'] = 0
        df['atlas_n_orange'] = 0
        df['atlas_has_data'] = False

        if not self.available:
            logger.warning("ATLAS credentials not configured; skipping enrichment")
            return df

        self._ensure_token()

        # Build coordinate list with DataFrame indices
        coords = []
        for idx, row in df.iterrows():
            ra, dec = row.get('ra'), row.get('dec')
            if pd.notna(ra) and pd.notna(dec):
                coords.append((idx, float(ra), float(dec)))

        total = len(coords)
        if total == 0:
            logger.info("ATLAS: no valid coordinates to enrich")
            return df

        n_batches = (total + ATLAS_BATCH_SIZE - 1) // ATLAS_BATCH_SIZE
        logger.info("ATLAS batch enrichment: %d targets in %d batches of <=%d",
                     total, n_batches, ATLAS_BATCH_SIZE)

        # Phase 1: Submit all batches
        all_tasks = {}  # df_idx -> task_url
        for batch_num, batch_start in enumerate(
                range(0, total, ATLAS_BATCH_SIZE), start=1):
            batch = coords[batch_start:batch_start + ATLAS_BATCH_SIZE]
            logger.info("ATLAS: submitting batch %d/%d (%d targets)",
                        batch_num, n_batches, len(batch))
            try:
                task_pairs = self._submit_batch(batch, mjd_min=mjd_min)
                for df_idx, task_url in task_pairs:
                    all_tasks[df_idx] = task_url
                logger.info("ATLAS: batch %d submitted (%d tasks queued so far)",
                            batch_num, len(all_tasks))
            except Exception as e:
                logger.warning("ATLAS batch %d/%d submit failed: %s",
                               batch_num, n_batches, e)

        if not all_tasks:
            logger.warning("ATLAS: no tasks were submitted successfully")
            return df

        # Phase 2: Poll all tasks
        logger.info("ATLAS: %d tasks queued, polling for results...", len(all_tasks))
        result_urls = self._poll_all_tasks(all_tasks)
        logger.info("ATLAS: %d/%d tasks returned results",
                     len(result_urls), len(all_tasks))

        # Phase 3: Download and parse results
        n_downloaded = 0
        for df_idx, result_url in result_urls.items():
            try:
                text_data = self._download_results(result_url)
                phot_df = self._parse_data(text_data)
                by_filter = self._split_by_filter(phot_df)
                n_c = len(by_filter.get('c', []))
                n_o = len(by_filter.get('o', []))
                df.at[df_idx, 'atlas_n_cyan'] = n_c
                df.at[df_idx, 'atlas_n_orange'] = n_o
                df.at[df_idx, 'atlas_has_data'] = (n_c + n_o) > 0
                n_downloaded += 1
            except Exception as e:
                logger.debug("ATLAS result download failed for idx %s: %s",
                             df_idx, e)

        # Phase 4: Cleanup all tasks from server queue
        logger.info("ATLAS: cleaning up %d tasks from server queue", len(all_tasks))
        for task_url in all_tasks.values():
            self._cleanup(task_url)

        n_with = int(df['atlas_has_data'].sum())
        logger.info("ATLAS batch enrichment complete: %d/%d candidates have data "
                     "(%d results downloaded)", n_with, total, n_downloaded)
        return df

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_throttle_wait(detail: str) -> int:
        m = re.search(r'available in (\d+) second', detail)
        if m:
            return int(m.group(1)) + 1
        m = re.search(r'available in (\d+) minute', detail)
        if m:
            return int(m.group(1)) * 60 + 1
        return 10

    @staticmethod
    def _parse_data(text_data: str) -> pd.DataFrame:
        # Extract column names from the ###-prefixed header line
        colnames = None
        for line in text_data.split('\n'):
            if line.startswith('###'):
                colnames = line.lstrip('#').split()
                break

        if colnames is None:
            logger.warning("ATLAS data has no header line; returning empty")
            return pd.DataFrame()

        df = pd.read_csv(
            io.StringIO(text_data),
            sep=r'\s+',
            header=None,
            names=colnames,
            comment='#',
            dtype=str,
            engine='python',
        )
        for c in ATLAS_NUMERIC_COLS:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna(subset=['MJD']).copy()
        return df

    @staticmethod
    def _split_by_filter(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        result = {}
        for filt in ['c', 'o']:
            mask = df['F'] == filt
            if mask.any():
                result[filt] = df[mask].sort_values('MJD').reset_index(drop=True)
        return result

    # ------------------------------------------------------------------
    # Single-target light curve (uses single-target fetch_photometry)
    # ------------------------------------------------------------------

    def get_light_curve(self, ra: float, dec: float,
                        mjd_min: Optional[float] = None) -> Optional[pd.DataFrame]:
        """Get ATLAS light curve in standardized format for plotting.

        Returns DataFrame with columns: mjd, magnitude, mag_err, band, survey
        """
        try:
            phot = self.fetch_photometry(ra, dec, mjd_min=mjd_min)
        except Exception as e:
            logger.warning(f"ATLAS light curve fetch failed: {e}")
            return None

        frames = []
        filter_names = {'c': 'ATLAS-cyan', 'o': 'ATLAS-orange'}
        for filt, fdf in phot.items():
            if fdf is None or len(fdf) == 0:
                continue
            # Convert microJy flux to magnitude where flux > 0
            valid = fdf['uJy'] > 0
            if not valid.any():
                continue
            sub = fdf[valid].copy()
            sub['magnitude'] = -2.5 * np.log10(sub['uJy']) + 23.9  # AB mag
            sub['mag_err'] = 1.0857 * sub['duJy'] / sub['uJy']  # error propagation
            frames.append(pd.DataFrame({
                'mjd': sub['MJD'].values,
                'magnitude': sub['magnitude'].values,
                'mag_err': sub['mag_err'].values,
                'band': filter_names.get(filt, filt),
                'survey': 'ATLAS',
            }))

        if frames:
            df = pd.concat(frames, ignore_index=True).sort_values('mjd').reset_index(drop=True)
            logger.info(f"ATLAS light curve: {len(df)} points")
            return df
        return None
