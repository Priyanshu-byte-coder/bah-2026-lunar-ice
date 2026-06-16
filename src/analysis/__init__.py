"""
Analysis module for quantitative lunar subsurface ice estimation.

Provides dielectric mixing models, ice volume estimation from radar
backscatter, and per-crater volume reporting for ISRO BAH 2026 PS-8.
"""

from .ice_volume import (
    DielectricModel,
    IceVolumeEstimator,
    cpr_to_ice_fraction,
    estimate_penetration_depth,
)

__all__ = [
    "DielectricModel",
    "IceVolumeEstimator",
    "cpr_to_ice_fraction",
    "estimate_penetration_depth",
]
