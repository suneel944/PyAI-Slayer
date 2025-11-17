"""AI Testing Dashboard - Real-time and historical metrics reporting."""

from .api import create_app
from .collectors import DashboardCollector
from .data_store import DashboardDataStore
from .failure_analyzer import FailureAnalyzer

__all__ = ["create_app", "DashboardCollector", "DashboardDataStore", "FailureAnalyzer"]
