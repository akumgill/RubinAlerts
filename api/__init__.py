"""MAGNETS target-submission & prioritization API.

Framework-agnostic core (`TargetQueueService`) implementing the interface
spec: submit (upsert) / list / patch / withdraw / queue summary / plan
preview. The plan preview reuses the existing LLAMAS orchestrator as the
scheduler. A thin FastAPI wrapper lives in ``api.app`` (optional dependency);
the core has no web-framework dependency so it is importable and testable
without one.
"""
from .models import Target, TIERS
from .service import TargetQueueService

__all__ = ["Target", "TIERS", "TargetQueueService"]
