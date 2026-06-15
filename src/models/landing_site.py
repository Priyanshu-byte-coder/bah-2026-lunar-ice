"""
Landing Site Scoring Module.

Evaluates candidate landing sites for future Chandrayaan missions based on:
    1. Ice probability (from LunarIceNet predictions)
    2. Terrain slope (from LOLA DEM)
    3. Accessibility (Earth visibility for communication)
    4. Illumination (solar panel viability in nearby regions)
    5. Prediction confidence (model certainty)

Produces ranked list of optimal landing zones with composite scores.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class LandingSite:
    """Candidate landing site with multi-criteria scores."""
    name: str
    lat: float
    lon: float
    ice_probability: float
    terrain_slope: float        # degrees
    accessibility: float        # 0-1 (Earth visibility fraction)
    illumination: float         # 0-1 (solar illumination fraction nearby)
    confidence: float           # 0-1 (model prediction confidence)
    composite_score: float = 0.0
    rank: int = 0
    details: Dict = field(default_factory=dict)


class LandingSiteScorer:
    """
    Multi-criteria landing site evaluation system.

    Scoring weights from config (default):
        ice_probability: 0.35
        terrain_slope: 0.20
        accessibility: 0.15
        illumination: 0.15
        confidence: 0.15
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            'ice_probability': 0.35,
            'terrain_slope': 0.20,
            'accessibility': 0.15,
            'illumination': 0.15,
            'confidence': 0.15,
        }

        # Constraints — hard filters before scoring
        self.constraints = {
            'max_slope': 15.0,          # degrees — safe landing limit
            'min_ice_prob': 0.3,        # Minimum ice probability worth landing for
            'min_confidence': 0.4,       # Minimum model confidence
        }

    def score_site(
        self,
        ice_prob: float,
        slope: float,
        accessibility: float,
        illumination: float,
        confidence: float,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute composite score for a single site.

        Returns:
            (composite_score, component_scores_dict)
        """
        # Normalize each criterion to 0-1 (higher = better)
        scores = {}

        # Ice probability — direct (higher = better)
        scores['ice_probability'] = np.clip(ice_prob, 0, 1)

        # Terrain slope — inverse (flatter = better, max 15°)
        scores['terrain_slope'] = np.clip(1.0 - slope / 30.0, 0, 1)

        # Accessibility — direct
        scores['accessibility'] = np.clip(accessibility, 0, 1)

        # Illumination — direct (solar power availability nearby)
        scores['illumination'] = np.clip(illumination, 0, 1)

        # Confidence — direct
        scores['confidence'] = np.clip(confidence, 0, 1)

        # Weighted sum
        composite = sum(
            self.weights[k] * scores[k] for k in self.weights
        )

        return composite, scores

    def evaluate_region(
        self,
        ice_prob_map: np.ndarray,
        slope_map: np.ndarray,
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
        confidence_map: Optional[np.ndarray] = None,
        grid_spacing_km: float = 1.0,
        top_k: int = 10,
    ) -> List[LandingSite]:
        """
        Evaluate entire region and find best landing sites.

        Args:
            ice_prob_map: (H, W) ice probability from LunarIceNet
            slope_map: (H, W) terrain slope in degrees
            lat_grid: (H, W) latitude values
            lon_grid: (H, W) longitude values
            confidence_map: (H, W) prediction confidence
            grid_spacing_km: Spacing between grid cells in km
            top_k: Number of top sites to return

        Returns:
            List of top-K LandingSite objects, ranked by composite score
        """
        if confidence_map is None:
            confidence_map = np.ones_like(ice_prob_map) * 0.5

        h, w = ice_prob_map.shape
        candidates = []

        # Evaluate grid cells (with stride to avoid redundancy)
        stride = max(1, int(5.0 / grid_spacing_km))  # ~5km spacing between candidates

        for i in range(0, h - stride, stride):
            for j in range(0, w - stride, stride):
                # Average over local window
                patch = slice(i, i + stride), slice(j, j + stride)
                ice_prob = float(np.mean(ice_prob_map[patch]))
                slope = float(np.mean(slope_map[patch]))
                conf = float(np.mean(confidence_map[patch]))
                lat = float(np.mean(lat_grid[patch]))
                lon = float(np.mean(lon_grid[patch]))

                # Hard constraints
                if slope > self.constraints['max_slope']:
                    continue
                if ice_prob < self.constraints['min_ice_prob']:
                    continue
                if conf < self.constraints['min_confidence']:
                    continue

                # Compute accessibility (simplified: Earth visibility)
                accessibility = self._compute_accessibility(lat, lon)

                # Compute illumination (nearby ridge illumination)
                illumination = self._compute_illumination(lat, lon, slope_map, i, j, stride)

                score, components = self.score_site(
                    ice_prob, slope, accessibility, illumination, conf
                )

                candidates.append(LandingSite(
                    name=f"Site_{lat:.1f}_{lon:.1f}",
                    lat=lat,
                    lon=lon,
                    ice_probability=ice_prob,
                    terrain_slope=slope,
                    accessibility=accessibility,
                    illumination=illumination,
                    confidence=conf,
                    composite_score=score,
                    details=components,
                ))

        # Rank and return top-K
        candidates.sort(key=lambda s: s.composite_score, reverse=True)
        for rank, site in enumerate(candidates[:top_k], 1):
            site.rank = rank

        logger.info(f"Evaluated {len(candidates)} candidate sites, returning top {top_k}")
        return candidates[:top_k]

    @staticmethod
    def _compute_accessibility(lat: float, lon: float) -> float:
        """
        Simplified Earth visibility metric.

        Lunar south pole has limited but existing Earth visibility.
        Libration allows periodic visibility even near poles.
        """
        # Simplified: higher latitudes have less direct Earth view
        # But libration gives ~6.5° tilt
        lat_abs = abs(lat)
        if lat_abs < 83.5:
            return 0.8  # Good visibility
        elif lat_abs < 86.5:
            return 0.5  # Partial (libration-dependent)
        elif lat_abs < 89.0:
            return 0.3  # Limited
        else:
            return 0.1  # Very limited (near pole)

    @staticmethod
    def _compute_illumination(
        lat: float, lon: float,
        slope_map: np.ndarray,
        row: int, col: int,
        stride: int,
    ) -> float:
        """
        Estimate solar illumination availability in nearby region.

        PSRs have no direct sunlight, but nearby peaks of eternal light
        can provide solar power via relay stations.
        """
        # Check if nearby elevated terrain exists (potential illumination)
        h, w = slope_map.shape
        search_radius = stride * 3

        r_start = max(0, row - search_radius)
        r_end = min(h, row + search_radius)
        c_start = max(0, col - search_radius)
        c_end = min(w, col + search_radius)

        nearby_slopes = slope_map[r_start:r_end, c_start:c_end]

        # Elevated ridges (high slope) nearby suggest possible illuminated peaks
        ridge_fraction = np.mean(nearby_slopes > 20) if nearby_slopes.size > 0 else 0

        # Higher latitude → less illumination
        lat_factor = max(0, 1.0 - (abs(lat) - 80) / 10)

        return float(np.clip(0.3 + ridge_fraction * 0.4 + lat_factor * 0.3, 0, 1))

    def generate_report(self, sites: List[LandingSite]) -> str:
        """Generate human-readable report of landing site rankings."""
        lines = [
            "=" * 70,
            "LUNAR LANDING SITE ASSESSMENT REPORT",
            "LunarIceNet — Chandrayaan Mission Planning",
            "=" * 70,
            "",
        ]

        for site in sites:
            lines.extend([
                f"Rank #{site.rank}: {site.name}",
                f"  Location: {site.lat:.2f}°N, {site.lon:.2f}°E",
                f"  Composite Score: {site.composite_score:.3f}",
                f"  ├── Ice Probability:  {site.ice_probability:.3f}",
                f"  ├── Terrain Slope:    {site.terrain_slope:.1f}°",
                f"  ├── Accessibility:    {site.accessibility:.3f}",
                f"  ├── Illumination:     {site.illumination:.3f}",
                f"  └── Confidence:       {site.confidence:.3f}",
                "",
            ])

        lines.extend([
            "=" * 70,
            f"Total candidates evaluated: {len(sites)}",
            f"Constraint filters: slope < {self.constraints['max_slope']}°, "
            f"ice_prob > {self.constraints['min_ice_prob']}, "
            f"confidence > {self.constraints['min_confidence']}",
            "=" * 70,
        ])

        return "\n".join(lines)
