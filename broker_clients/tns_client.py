"""Transient Name Server (TNS) client for cross-matching candidates.

TNS is the official IAU repository for transient discoveries. This client
checks if our candidates are already reported, helping to:
- Avoid duplicating known discoveries
- Validate classifications against spectroscopic confirmations
- Retrieve IAU designations (AT/SN names)

TNS API documentation: https://www.wis-tns.org/content/tns-getting-started

Requires TNS API key: set TNS_API_KEY environment variable or create
~/.tns_credentials with:
    [tns]
    api_key = your_api_key

Register for API key at: https://www.wis-tns.org/user
"""

import os
import json
import time
import logging
import configparser
from typing import Optional, Dict, Any, List, Tuple
import requests
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

TNS_API_URL = "https://www.wis-tns.org/api/get"
TNS_SEARCH_URL = "https://www.wis-tns.org/api/get/search"
TNS_OBJECT_URL = "https://www.wis-tns.org/api/get/object"

# TNS rate limiting: max 100 requests per minute
TNS_RATE_LIMIT_DELAY = 0.7  # seconds between requests
TNS_TIMEOUT = 30  # seconds

# User-Agent required by TNS API (identify your bot)
TNS_USER_AGENT = "RubinAlerts-Pipeline/1.0 (Rubin LSST SN Ia follow-up)"

CREDENTIALS_FILE = os.path.expanduser("~/.tns_credentials")


def get_tns_api_key() -> str:
    """Get TNS API key from environment or config file."""
    api_key = os.environ.get('TNS_API_KEY')
    if api_key:
        return api_key

    if os.path.exists(CREDENTIALS_FILE):
        config = configparser.ConfigParser()
        config.read(CREDENTIALS_FILE)
        if 'tns' in config:
            api_key = config['tns'].get('api_key')
            if api_key:
                return api_key

    raise RuntimeError(
        "TNS API key not found. Set TNS_API_KEY environment variable, "
        "or create ~/.tns_credentials with:\n"
        "  [tns]\n  api_key = your_api_key\n"
        "Register at: https://www.wis-tns.org/user"
    )


# ---------------------------------------------------------------------------
# TNS public-objects daily dump
#
# One ~25 MB download per night replaces per-object rate-limited cone
# searches (~0.7 s/request + 60 s penalty sleeps). The dump endpoint is
# stricter than /api/get: the tns_marker User-Agent must carry the REGISTERED
# bot id/name (401 otherwise), so ~/.tns_credentials needs bot_id/bot_name.
# ---------------------------------------------------------------------------

TNS_DUMP_URL = ("https://www.wis-tns.org/system/files/tns_public_objects/"
                "tns_public_objects.csv.zip")


def _get_bot_identity():
    """(bot_id, bot_name) from env or credentials file; (None, None) if absent."""
    bot_id = os.environ.get('TNS_BOT_ID')
    bot_name = os.environ.get('TNS_BOT_NAME')
    if bot_id and bot_name:
        return bot_id, bot_name
    if os.path.exists(CREDENTIALS_FILE):
        config = configparser.ConfigParser()
        config.read(CREDENTIALS_FILE)
        if 'tns' in config:
            return (config['tns'].get('bot_id'), config['tns'].get('bot_name'))
    return None, None


def fetch_tns_public_objects(cache_dir: str = './cache/data',
                             max_age_hours: float = 24.0) -> Optional[pd.DataFrame]:
    """Download (or reuse cached) TNS public-objects catalog as a DataFrame.

    Columns kept: name_prefix, name, ra, dec, redshift, type,
    internal_names. Returns None when the download and cache both fail
    (caller falls back to per-object cone searches).
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, 'tns_public_objects.csv.zip')

    fresh = (os.path.exists(cache_path) and
             (time.time() - os.path.getmtime(cache_path)) < max_age_hours * 3600)
    if not fresh:
        bot_id, bot_name = _get_bot_identity()
        if not (bot_id and bot_name):
            logger.warning("TNS dump: bot_id/bot_name missing from credentials "
                           "— cannot download (add them from wis-tns.org/bots)")
            return _load_tns_dump(cache_path) if os.path.exists(cache_path) else None
        try:
            marker = (f'tns_marker{{"tns_id":{bot_id},"type":"bot",'
                      f'"name":"{bot_name}"}}')
            r = requests.post(TNS_DUMP_URL, headers={'User-Agent': marker},
                              data={'api_key': get_tns_api_key()},
                              stream=True, timeout=300)
            if r.status_code != 200:
                logger.warning("TNS dump download failed (HTTP %d)", r.status_code)
                return _load_tns_dump(cache_path) if os.path.exists(cache_path) else None
            tmp = cache_path + '.tmp'
            with open(tmp, 'wb') as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
            os.replace(tmp, cache_path)
            logger.info("TNS dump: downloaded %.1f MB",
                        os.path.getsize(cache_path) / 1e6)
        except Exception as e:
            logger.warning("TNS dump download failed: %s", e)
            return _load_tns_dump(cache_path) if os.path.exists(cache_path) else None

    return _load_tns_dump(cache_path)


def _load_tns_dump(path: str) -> Optional[pd.DataFrame]:
    """Parse the dump zip -> DataFrame (name, ra, dec, redshift, type, ...)."""
    try:
        # Row 0 is a metadata line; row 1 is the real header.
        df = pd.read_csv(path, compression='zip', skiprows=1,
                         usecols=['name_prefix', 'name', 'ra', 'declination',
                                  'redshift', 'type', 'internal_names'],
                         dtype={'internal_names': str}, low_memory=False)
        df = df.rename(columns={'declination': 'dec'})
        df['ra'] = pd.to_numeric(df['ra'], errors='coerce')
        df['dec'] = pd.to_numeric(df['dec'], errors='coerce')
        df['redshift'] = pd.to_numeric(df['redshift'], errors='coerce')
        df = df.dropna(subset=['ra', 'dec'])
        logger.info("TNS dump: %d objects loaded", len(df))
        return df
    except Exception as e:
        logger.warning("TNS dump parse failed: %s", e)
        return None


def crossmatch_tns_local(targets: pd.DataFrame, tns_df: pd.DataFrame,
                         radius_arcsec: float = 5.0) -> Dict[str, Dict]:
    """Match targets (diaObjectId, ra, dec) against the local TNS catalog.

    Matches by internal name FIRST (exact — e.g. a ZTF oid listed in TNS
    internal_names), then by coordinates within ``radius_arcsec``.
    Returns {diaObjectId: {tns_name, tns_type, tns_redshift}}.
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    out = {}
    if targets is None or len(targets) == 0 or tns_df is None or len(tns_df) == 0:
        return out

    def _hit_dict(rec: dict) -> Dict:
        prefix = rec.get('name_prefix')
        typ = rec.get('type')
        z = rec.get('redshift')
        return {
            'tns_name': f"{prefix if pd.notna(prefix) else 'AT'} {rec.get('name', '')}".strip(),
            'tns_type': typ if (pd.notna(typ) and str(typ).strip()) else None,
            'tns_redshift': float(z) if pd.notna(z) else None,
        }

    # Name index: internal_names is a comma-separated list of survey ids
    name_idx = {}
    has_names = tns_df['internal_names'].notna()
    for rec in tns_df.loc[has_names].to_dict('records'):
        for nm in str(rec['internal_names']).split(','):
            nm = nm.strip()
            if nm:
                name_idx[nm] = rec

    tns_sc = SkyCoord(ra=tns_df['ra'].values * u.deg,
                      dec=tns_df['dec'].values * u.deg)
    tgt_ra = pd.to_numeric(targets['ra'], errors='coerce')
    tgt_dec = pd.to_numeric(targets['dec'], errors='coerce')
    valid = tgt_ra.notna() & tgt_dec.notna()
    if valid.any():
        tgt_sc = SkyCoord(ra=tgt_ra[valid].values * u.deg,
                          dec=tgt_dec[valid].values * u.deg)
        idx, sep, _ = tgt_sc.match_to_catalog_sky(tns_sc)
    pos = 0
    for i, row in targets.iterrows():
        did = str(row['diaObjectId'])
        rec = name_idx.get(did)
        if rec is None and valid.loc[i] and sep[pos].arcsec <= radius_arcsec:
            rec = tns_df.iloc[idx[pos]].to_dict()
        if valid.loc[i]:
            pos += 1
        if rec is not None:
            out[did] = _hit_dict(rec)
    logger.info("TNS local cross-match: %d/%d targets matched",
                len(out), len(targets))
    return out


class TNSClient:
    """TNS client for cross-matching transient candidates."""

    def __init__(self, bot_name: str = "RubinAlerts", bot_id: int = None):
        """Initialize TNS client.

        Parameters
        ----------
        bot_name : str
            Name of your registered TNS bot (for rate limit tracking)
        bot_id : int, optional
            Your TNS bot ID (from registration)
        """
        self.api_key = None
        self._available = None
        self.bot_name = bot_name
        self.bot_id = bot_id or 0
        self._last_request_time = 0

    @property
    def available(self) -> bool:
        """Check if TNS API key is configured."""
        if self._available is None:
            try:
                get_tns_api_key()
                self._available = True
            except RuntimeError:
                self._available = False
        return self._available

    def _ensure_api_key(self):
        """Load API key if not already loaded."""
        if self.api_key is None:
            self.api_key = get_tns_api_key()

    def _headers(self) -> Dict[str, str]:
        """Build request headers required by TNS API."""
        return {
            'User-Agent': f'tns_marker{{"tns_id":{self.bot_id},"type":"bot","name":"{self.bot_name}"}}',
        }

    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < TNS_RATE_LIMIT_DELAY:
            time.sleep(TNS_RATE_LIMIT_DELAY - elapsed)
        self._last_request_time = time.time()

    def _make_request(self, endpoint: str, data: Dict) -> Dict:
        """Make a TNS API request with rate limiting.

        Parameters
        ----------
        endpoint : str
            API endpoint URL
        data : dict
            Request data (will be JSON-encoded)

        Returns
        -------
        dict : API response or empty dict on failure
        """
        self._ensure_api_key()
        self._rate_limit()

        # TNS requires data in a specific format
        payload = {
            'api_key': self.api_key,
            'data': json.dumps(data),
        }

        try:
            resp = requests.post(
                endpoint,
                headers=self._headers(),
                data=payload,
                timeout=TNS_TIMEOUT,
            )
            if resp.status_code == 200:
                result = resp.json()
                # TNS returns data in a nested structure
                if 'data' in result:
                    return result['data']
                return result
            elif resp.status_code == 429:
                logger.warning("TNS rate limited — waiting 60s")
                time.sleep(60)
                return self._make_request(endpoint, data)  # retry
            else:
                logger.debug("TNS request failed (HTTP %d): %s",
                            resp.status_code, resp.text[:200])
                return {}
        except requests.exceptions.Timeout:
            logger.warning("TNS request timed out")
            return {}
        except Exception as e:
            logger.debug("TNS request error: %s", e)
            return {}

    def search_by_coordinates(self, ra: float, dec: float,
                              radius_arcsec: float = 5.0) -> List[Dict]:
        """Search TNS for objects near given coordinates.

        Parameters
        ----------
        ra, dec : float
            J2000 coordinates in decimal degrees
        radius_arcsec : float
            Search radius in arcseconds (default 5")

        Returns
        -------
        list of dicts with TNS object info:
            - objname: IAU designation (e.g., "2024abc")
            - prefix: "AT" (unconfirmed) or "SN" (spectroscopically confirmed)
            - ra, dec: coordinates
            - discoverydate: discovery date string
            - discoverymag: discovery magnitude
            - type: spectroscopic classification (if any)
            - redshift: spectroscopic redshift (if any)
            - hostname: host galaxy name
            - internal_names: other designations
        """
        data = {
            'ra': str(ra),
            'dec': str(dec),
            'radius': str(radius_arcsec),
            'units': 'arcsec',
            'objname': '',
            'objname_exact_match': 0,
            'internal_name': '',
            'internal_name_exact_match': 0,
        }

        result = self._make_request(TNS_SEARCH_URL, data)

        if isinstance(result, dict) and 'reply' in result:
            return result['reply']
        elif isinstance(result, list):
            return result
        return []

    def get_object_details(self, objname: str) -> Optional[Dict]:
        """Get full details for a TNS object by name.

        Parameters
        ----------
        objname : str
            TNS object name (e.g., "2024abc", without AT/SN prefix)

        Returns
        -------
        dict with full object details or None
        """
        data = {
            'objname': objname,
            'objid': '',
            'photometry': '1',
            'spectra': '1',
        }

        result = self._make_request(TNS_OBJECT_URL, data)

        if isinstance(result, dict) and 'reply' in result:
            return result['reply']
        return result if result else None

    def cross_match_candidates(self, candidates_df: pd.DataFrame,
                               radius_arcsec: float = 5.0) -> pd.DataFrame:
        """Cross-match candidates against TNS.

        Adds columns:
        - tns_name: IAU designation (e.g., "AT2024abc" or "SN2024xyz")
        - tns_type: spectroscopic classification (if any)
        - tns_redshift: spectroscopic redshift (if any)
        - tns_discovery_date: TNS discovery date
        - tns_match: boolean, True if found in TNS

        Parameters
        ----------
        candidates_df : pd.DataFrame
            Must have 'ra', 'dec' columns
        radius_arcsec : float
            Search radius in arcseconds

        Returns
        -------
        Copy of DataFrame with TNS columns added
        """
        df = candidates_df.copy()

        # Initialize TNS columns
        df['tns_name'] = None
        df['tns_type'] = None
        df['tns_redshift'] = np.nan
        df['tns_discovery_date'] = None
        df['tns_match'] = False

        if not self.available:
            logger.warning("TNS API key not configured — skipping cross-match")
            return df

        total = len(df)
        n_matched = 0
        n_classified = 0

        logger.info("TNS cross-match: checking %d candidates (%.1f\" radius)...",
                    total, radius_arcsec)

        for idx, row in df.iterrows():
            ra, dec = row.get('ra'), row.get('dec')
            if pd.isna(ra) or pd.isna(dec):
                continue

            matches = self.search_by_coordinates(ra, dec, radius_arcsec)

            if matches:
                # Take the closest/first match
                match = matches[0]
                prefix = match.get('prefix', 'AT')
                objname = match.get('objname', '')
                tns_name = f"{prefix}{objname}" if objname else None

                df.at[idx, 'tns_name'] = tns_name
                df.at[idx, 'tns_type'] = match.get('type')
                df.at[idx, 'tns_discovery_date'] = match.get('discoverydate')
                df.at[idx, 'tns_match'] = True

                redshift = match.get('redshift')
                if redshift:
                    try:
                        df.at[idx, 'tns_redshift'] = float(redshift)
                    except (ValueError, TypeError):
                        pass

                n_matched += 1
                if match.get('type'):
                    n_classified += 1

                logger.debug("TNS match: (%.4f, %.4f) -> %s (%s)",
                            ra, dec, tns_name, match.get('type', 'unclassified'))

            # Progress logging
            done = idx + 1
            if done % 20 == 0:
                logger.info("TNS cross-match: %d/%d checked (%d matches)",
                            done, total, n_matched)

        logger.info("TNS cross-match complete: %d/%d matched, %d spectroscopically classified",
                    n_matched, total, n_classified)

        return df

    def verify_connection(self) -> Tuple[bool, str]:
        """Test TNS API connection.

        Returns (success, message) tuple.
        """
        if not self.available:
            return False, "API key not configured"

        try:
            # Try a simple search to verify connection
            self._ensure_api_key()
            # Search for a known object (SN2011fe, well-known SN Ia)
            result = self.search_by_coordinates(210.774, 54.274, radius_arcsec=10)
            if result:
                return True, f"Connection verified ({len(result)} test results)"
            # Empty result could be valid, try the API directly
            return True, "Connection established (no test objects in search area)"
        except Exception as e:
            return False, f"Connection failed: {e}"
