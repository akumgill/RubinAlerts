"""Abstract base class for alert broker clients."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import pandas as pd


@dataclass
class Alert:
    """Standard alert data model."""
    broker_name: str
    object_id: str
    ra: float
    dec: float
    discovery_date: str
    classification: Dict[str, float]  # e.g., {'SN Ia': 0.8, 'SN II': 0.1, ...}
    photometry: List[Dict[str, Any]]  # List of magnitude measurements
    metadata: Dict[str, Any]  # Additional broker-specific metadata


class BaseBrokerClient(ABC):
    """Base class defining interface for broker clients."""

    def __init__(self, broker_name: str, cache_dir: str = './cache/data'):
        """
        Initialize broker client.

        Args:
            broker_name: Name of the broker (e.g., 'ANTARES', 'ALeLRCE')
            cache_dir: Directory for caching broker data
        """
        self.broker_name = broker_name
        self.cache_dir = cache_dir

    @abstractmethod
    def query_alerts(self,
                    class_name: str = 'SN Ia',
                    min_probability: float = 0.7,
                    days_back: int = 30) -> pd.DataFrame:
        """
        Query broker for supernova alerts.

        Args:
            class_name: Target classification (e.g., 'SN Ia')
            min_probability: Minimum classification probability
            days_back: Query alerts from last N days

        Returns:
            DataFrame with columns: object_id, ra, dec, discovery_date,
                                   classification_prob, real_bogus, magnitude, band
        """
        pass

    @abstractmethod
    def get_light_curve(self, object_id: str) -> Optional[pd.DataFrame]:
        """
        Retrieve light curve for an object.

        Args:
            object_id: Broker-specific object identifier

        Returns:
            DataFrame with columns: jd, magnitude, mag_err, band, or None if unavailable
        """
        pass

    @abstractmethod
    def get_stamps(self, object_id: str, ra: float, dec: float) -> Dict[str, Any]:
        """
        Retrieve postage stamp images for an object.

        Args:
            object_id: Broker-specific object identifier
            ra: Right ascension
            dec: Declination

        Returns:
            Dictionary with stamp data (science, reference, difference images, etc.)
        """
        pass

    def to_dataframe(self, alerts: List[Alert]) -> pd.DataFrame:
        """
        Convert list of Alert objects to DataFrame.

        Args:
            alerts: List of Alert objects

        Returns:
            Unified DataFrame representation
        """
        records = []
        for alert in alerts:
            record = {
                'broker': alert.broker_name,
                'object_id': alert.object_id,
                'ra': alert.ra,
                'dec': alert.dec,
                'discovery_date': alert.discovery_date,
                'classification': str(alert.classification),
            }
            record.update(alert.metadata)
            records.append(record)

        return pd.DataFrame(records)
