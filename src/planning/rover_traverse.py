"""
Optimized Rover Traverse Path Planner.

Plans optimal rover traverse paths from landing site to ice targets on the
lunar south pole, considering:
    1. Terrain slope constraints (safety)
    2. Surface roughness penalties
    3. Solar power availability (proximity to illuminated peaks)
    4. Ice probability reward (incentivize high-value sampling)
    5. Distance / energy budget

Uses A* search on a discretized grid (25 m/pixel) with a composite cost
function.  Outputs waypoint lists, distance/time estimates, and safety
analysis suitable for the ISRO BAH 2026 PS-8 presentation.

Rover reference specs (Pragyan-class):
    Max traversable slope : 25 deg
    Preferred slope        : < 15 deg
    Flat-terrain speed     : ~100 m/hr
    Max range from lander  : ~5 km
    Grid resolution        : 25 m/pixel
"""

import heapq
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CELL_SIZE_M = 25.0          # metres per pixel
MAX_SLOPE_DEG = 25.0        # impassable above this
PREFERRED_SLOPE_DEG = 15.0  # penalty ramps above this
FLAT_SPEED_M_PER_HR = 100.0 # rover speed on flat ground
MAX_RANGE_M = 5000.0        # max traverse budget

# 8-connected grid neighbours (row_offset, col_offset)
_NEIGHBOURS = [
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),          ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1),
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class TraverseResult:
    """Container for a planned traverse path and its metrics."""
    path_rc: List[Tuple[int, int]]          # grid (row, col) waypoints
    path_latlon: List[Tuple[float, float]]  # (lat, lon) waypoints
    total_distance_m: float
    estimated_time_hr: float
    cost_breakdown: Dict[str, float]
    success: bool
    message: str


@dataclass
class SafetyReport:
    """Safety and energy analysis of a traverse path."""
    max_slope_deg: float
    mean_slope_deg: float
    hazard_cell_count: int       # cells with slope > PREFERRED_SLOPE_DEG
    impassable_near_misses: int  # cells with slope in [20, 25) deg
    total_distance_m: float
    estimated_energy_wh: float
    sampling_waypoints: List[Tuple[float, float]]  # (lat, lon)
    report_text: str


# ---------------------------------------------------------------------------
# RoverTraversePlanner
# ---------------------------------------------------------------------------
class RoverTraversePlanner:
    """
    A* grid-based path planner for lunar rover traversal.

    Parameters
    ----------
    ice_prob_map : np.ndarray
        2-D array of ice probability values in [0, 1].
    slope_map : np.ndarray
        2-D array of terrain slope in degrees.
    lat_grid : np.ndarray
        2-D array of latitude values per pixel (same shape as ice_prob_map).
    lon_grid : np.ndarray
        2-D array of longitude values per pixel (same shape as ice_prob_map).
    landing_lat : float
        Latitude of the landing site.
    landing_lon : float
        Longitude of the landing site.
    illumination_map : np.ndarray, optional
        2-D array in [0, 1] representing solar illumination fraction.
        If *None*, a synthetic proxy is derived from the slope map.
    """

    def __init__(
        self,
        ice_prob_map: np.ndarray,
        slope_map: np.ndarray,
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
        landing_lat: float,
        landing_lon: float,
        illumination_map: Optional[np.ndarray] = None,
    ):
        self.ice_prob = np.asarray(ice_prob_map, dtype=np.float32)
        self.slope = np.asarray(slope_map, dtype=np.float32)
        self.lat_grid = np.asarray(lat_grid, dtype=np.float32)
        self.lon_grid = np.asarray(lon_grid, dtype=np.float32)
        self.rows, self.cols = self.ice_prob.shape
        self.landing_lat = landing_lat
        self.landing_lon = landing_lon

        if illumination_map is not None:
            self.illumination = np.asarray(illumination_map, dtype=np.float32)
        else:
            self.illumination = self._synthetic_illumination()

        # Pre-compute the landing-site grid cell
        self.landing_rc = self._latlon_to_rc(landing_lat, landing_lon)

        # Validate shapes
        for name, arr in [
            ("slope_map", self.slope),
            ("lat_grid", self.lat_grid),
            ("lon_grid", self.lon_grid),
            ("illumination", self.illumination),
        ]:
            if arr.shape != self.ice_prob.shape:
                raise ValueError(
                    f"{name} shape {arr.shape} does not match "
                    f"ice_prob_map shape {self.ice_prob.shape}"
                )

        logger.info(
            "RoverTraversePlanner initialised: grid %d x %d, "
            "landing (%+.4f, %+.4f) -> cell (%d, %d)",
            self.rows, self.cols,
            landing_lat, landing_lon,
            self.landing_rc[0], self.landing_rc[1],
        )

    # ----- coordinate helpers -----

    def _latlon_to_rc(self, lat: float, lon: float) -> Tuple[int, int]:
        """Return the (row, col) grid cell closest to (lat, lon)."""
        dist = (self.lat_grid - lat) ** 2 + (self.lon_grid - lon) ** 2
        idx = int(np.argmin(dist))
        return divmod(idx, self.cols)

    def _rc_to_latlon(self, r: int, c: int) -> Tuple[float, float]:
        """Return (lat, lon) for grid cell (r, c)."""
        return float(self.lat_grid[r, c]), float(self.lon_grid[r, c])

    # ----- illumination proxy -----

    def _synthetic_illumination(self) -> np.ndarray:
        """
        Heuristic illumination proxy: elevated ridges (low-slope peaks
        surrounded by higher slopes) tend to be permanently illuminated.
        We use inverse-slope smoothed as a rough stand-in.
        """
        flat_score = 1.0 - np.clip(self.slope / 45.0, 0, 1)
        kernel_size = max(3, int(200 / CELL_SIZE_M))  # ~200 m smoothing
        from scipy.ndimage import uniform_filter
        return uniform_filter(flat_score, size=kernel_size)

    # ----- cost function -----

    def _edge_cost(
        self,
        r1: int, c1: int,
        r2: int, c2: int,
    ) -> Optional[float]:
        """
        Compute traversal cost from cell (r1,c1) to neighbour (r2,c2).

        Returns None if the move is impassable.

        Cost components (all >= 0):
            distance   : Euclidean cell distance * CELL_SIZE_M
            slope_pen  : exponential penalty above PREFERRED_SLOPE_DEG
            solar_pen  : penalty for low illumination (energy risk)
            ice_reward : negative cost (reward) for high ice probability
        """
        slope_deg = self.slope[r2, c2]

        # Hard constraint: impassable
        if slope_deg > MAX_SLOPE_DEG:
            return None

        # Base distance (diagonal = sqrt(2))
        dr, dc = abs(r2 - r1), abs(c2 - c1)
        step_m = CELL_SIZE_M * math.sqrt(dr * dr + dc * dc)

        # Slope penalty — ramps exponentially between 15 and 25 deg
        if slope_deg <= PREFERRED_SLOPE_DEG:
            slope_pen = 0.0
        else:
            frac = (slope_deg - PREFERRED_SLOPE_DEG) / (
                MAX_SLOPE_DEG - PREFERRED_SLOPE_DEG
            )
            slope_pen = step_m * 2.0 * (math.exp(3.0 * frac) - 1.0)

        # Solar illumination penalty (prefer illuminated paths for power)
        illum = self.illumination[r2, c2]
        solar_pen = step_m * 0.5 * (1.0 - illum)

        # Ice probability reward (reduce cost in high-probability areas)
        ice_reward = step_m * 0.8 * self.ice_prob[r2, c2]

        cost = step_m + slope_pen + solar_pen - ice_reward
        return max(cost, 0.01)  # keep cost positive

    def _heuristic(self, r: int, c: int, gr: int, gc: int) -> float:
        """Admissible heuristic: Euclidean distance in metres."""
        dr = r - gr
        dc = c - gc
        return CELL_SIZE_M * math.sqrt(dr * dr + dc * dc)

    # ----- A* search -----

    def plan_path(
        self,
        target_lat: float,
        target_lon: float,
        max_range_m: float = MAX_RANGE_M,
    ) -> TraverseResult:
        """
        Run A* from the landing site to a target (lat, lon).

        Parameters
        ----------
        target_lat, target_lon : float
            Destination coordinates.
        max_range_m : float
            Hard cap on cumulative distance.

        Returns
        -------
        TraverseResult
        """
        start = self.landing_rc
        goal = self._latlon_to_rc(target_lat, target_lon)

        logger.info(
            "Planning path: start=%s  goal=%s", start, goal,
        )

        # A* structures
        # heap items: (f_score, counter, row, col)
        counter = 0
        open_heap: list = []
        heapq.heappush(open_heap, (0.0, counter, start[0], start[1]))

        g_score: Dict[Tuple[int, int], float] = {start: 0.0}
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        closed = set()

        # Cost accumulators for breakdown
        total_slope_pen = 0.0
        total_solar_pen = 0.0
        total_ice_reward = 0.0

        while open_heap:
            f, _, r, c = heapq.heappop(open_heap)
            node = (r, c)

            if node == goal:
                break

            if node in closed:
                continue
            closed.add(node)

            for dr, dc in _NEIGHBOURS:
                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= self.rows or nc < 0 or nc >= self.cols:
                    continue
                nb = (nr, nc)
                if nb in closed:
                    continue

                edge = self._edge_cost(r, c, nr, nc)
                if edge is None:
                    continue

                tentative_g = g_score[node] + edge

                # Enforce range budget (cumulative distance, not cost)
                step_m = CELL_SIZE_M * math.sqrt(dr * dr + dc * dc)
                dist_so_far = self._cumulative_distance(
                    came_from, node, start
                ) + step_m
                if dist_so_far > max_range_m:
                    continue

                if tentative_g < g_score.get(nb, math.inf):
                    came_from[nb] = node
                    g_score[nb] = tentative_g
                    f_new = tentative_g + self._heuristic(nr, nc, goal[0], goal[1])
                    counter += 1
                    heapq.heappush(open_heap, (f_new, counter, nr, nc))
        else:
            # Exhausted open set without reaching goal
            return TraverseResult(
                path_rc=[],
                path_latlon=[],
                total_distance_m=0.0,
                estimated_time_hr=0.0,
                cost_breakdown={},
                success=False,
                message="No feasible path found to target.",
            )

        # Reconstruct path
        path_rc = self._reconstruct(came_from, start, goal)
        path_latlon = [self._rc_to_latlon(r, c) for r, c in path_rc]

        # Compute detailed metrics along path
        total_dist, total_slope_pen, total_solar_pen, total_ice_reward = (
            self._path_cost_breakdown(path_rc)
        )
        est_time = self._estimate_time(path_rc)

        result = TraverseResult(
            path_rc=path_rc,
            path_latlon=path_latlon,
            total_distance_m=total_dist,
            estimated_time_hr=est_time,
            cost_breakdown={
                "distance_m": round(total_dist, 1),
                "slope_penalty": round(total_slope_pen, 1),
                "solar_penalty": round(total_solar_pen, 1),
                "ice_reward": round(total_ice_reward, 1),
                "total_cost": round(g_score.get(goal, 0.0), 1),
            },
            success=True,
            message=f"Path found: {len(path_rc)} waypoints, "
                    f"{total_dist:.0f} m, ~{est_time:.1f} hr.",
        )
        logger.info(result.message)
        return result

    # ----- internal helpers -----

    @staticmethod
    def _reconstruct(
        came_from: Dict[Tuple[int, int], Tuple[int, int]],
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        path = [goal]
        node = goal
        while node != start:
            node = came_from[node]
            path.append(node)
        path.reverse()
        return path

    @staticmethod
    def _cumulative_distance(
        came_from: Dict[Tuple[int, int], Tuple[int, int]],
        node: Tuple[int, int],
        start: Tuple[int, int],
    ) -> float:
        """Walk back through came_from to sum Euclidean distance."""
        dist = 0.0
        while node != start:
            prev = came_from.get(node)
            if prev is None:
                break
            dr = node[0] - prev[0]
            dc = node[1] - prev[1]
            dist += CELL_SIZE_M * math.sqrt(dr * dr + dc * dc)
            node = prev
        return dist

    def _path_cost_breakdown(
        self, path_rc: List[Tuple[int, int]]
    ) -> Tuple[float, float, float, float]:
        """Return (distance, slope_pen, solar_pen, ice_reward) along path."""
        total_dist = 0.0
        total_slope = 0.0
        total_solar = 0.0
        total_ice = 0.0

        for i in range(len(path_rc) - 1):
            r1, c1 = path_rc[i]
            r2, c2 = path_rc[i + 1]

            dr, dc = abs(r2 - r1), abs(c2 - c1)
            step_m = CELL_SIZE_M * math.sqrt(dr * dr + dc * dc)
            total_dist += step_m

            slope_deg = self.slope[r2, c2]
            if slope_deg > PREFERRED_SLOPE_DEG:
                frac = (slope_deg - PREFERRED_SLOPE_DEG) / (
                    MAX_SLOPE_DEG - PREFERRED_SLOPE_DEG
                )
                total_slope += step_m * 2.0 * (math.exp(3.0 * frac) - 1.0)

            illum = self.illumination[r2, c2]
            total_solar += step_m * 0.5 * (1.0 - illum)

            total_ice += step_m * 0.8 * self.ice_prob[r2, c2]

        return total_dist, total_slope, total_solar, total_ice

    def _estimate_time(self, path_rc: List[Tuple[int, int]]) -> float:
        """Estimate traverse time in hours, accounting for slope slowdown."""
        total_hr = 0.0
        for i in range(len(path_rc) - 1):
            r1, c1 = path_rc[i]
            r2, c2 = path_rc[i + 1]
            dr, dc = abs(r2 - r1), abs(c2 - c1)
            step_m = CELL_SIZE_M * math.sqrt(dr * dr + dc * dc)

            slope_deg = self.slope[r2, c2]
            # Speed decreases linearly with slope, reaching 20% at MAX_SLOPE
            speed_factor = max(
                0.2, 1.0 - 0.8 * (slope_deg / MAX_SLOPE_DEG)
            )
            speed = FLAT_SPEED_M_PER_HR * speed_factor
            total_hr += step_m / speed

        return total_hr


# ---------------------------------------------------------------------------
# TraverseAnalyzer
# ---------------------------------------------------------------------------
class TraverseAnalyzer:
    """
    Post-hoc analysis of a planned rover traverse.

    Parameters
    ----------
    planner : RoverTraversePlanner
        The planner instance (provides map data).
    result : TraverseResult
        A previously computed traverse result.
    """

    # Energy model constants (Wh)
    BASE_POWER_W = 50.0          # hotel load
    DRIVE_POWER_W = 80.0         # motor power on flat
    SLOPE_POWER_FACTOR = 3.0     # extra W per degree above 0

    def __init__(self, planner: RoverTraversePlanner, result: TraverseResult):
        self.planner = planner
        self.result = result

    def safety_metrics(self) -> Dict[str, float]:
        """Compute slope-related safety metrics along the path."""
        if not self.result.path_rc:
            return {"max_slope": 0, "mean_slope": 0, "hazard_count": 0}

        slopes = np.array([
            self.planner.slope[r, c] for r, c in self.result.path_rc
        ])
        return {
            "max_slope_deg": float(np.max(slopes)),
            "mean_slope_deg": float(np.mean(slopes)),
            "hazard_count": int(np.sum(slopes > PREFERRED_SLOPE_DEG)),
            "impassable_near_misses": int(np.sum(
                (slopes >= 20.0) & (slopes < MAX_SLOPE_DEG)
            )),
        }

    def estimate_energy_wh(self) -> float:
        """
        Estimate total energy consumption in Watt-hours.

        Model: E = sum over segments of
            (BASE_POWER + DRIVE_POWER + slope_extra) * segment_time
        """
        total_wh = 0.0
        for i in range(len(self.result.path_rc) - 1):
            r1, c1 = self.result.path_rc[i]
            r2, c2 = self.result.path_rc[i + 1]
            dr = abs(r2 - r1)
            dc = abs(c2 - c1)
            step_m = CELL_SIZE_M * math.sqrt(dr * dr + dc * dc)

            slope_deg = self.planner.slope[r2, c2]
            speed_factor = max(
                0.2, 1.0 - 0.8 * (slope_deg / MAX_SLOPE_DEG)
            )
            speed = FLAT_SPEED_M_PER_HR * speed_factor
            dt_hr = step_m / speed

            power_w = (
                self.BASE_POWER_W
                + self.DRIVE_POWER_W
                + self.SLOPE_POWER_FACTOR * slope_deg
            )
            total_wh += power_w * dt_hr

        return total_wh

    def ice_sampling_waypoints(
        self, top_n: int = 5, min_spacing_cells: int = 8
    ) -> List[Tuple[float, float]]:
        """
        Select the best ice-sampling stops along the traverse.

        Picks *top_n* cells with the highest ice probability, enforcing a
        minimum spacing of *min_spacing_cells* between selected stops.

        Returns list of (lat, lon).
        """
        if not self.result.path_rc:
            return []

        # Collect (ice_prob, index, row, col)
        candidates = []
        for idx, (r, c) in enumerate(self.result.path_rc):
            candidates.append((self.planner.ice_prob[r, c], idx, r, c))

        # Sort descending by ice probability
        candidates.sort(key=lambda x: x[0], reverse=True)

        selected: List[Tuple[float, float]] = []
        used_indices: List[int] = []

        for prob, idx, r, c in candidates:
            if len(selected) >= top_n:
                break
            # Enforce spacing
            if any(abs(idx - ui) < min_spacing_cells for ui in used_indices):
                continue
            selected.append(self.planner._rc_to_latlon(r, c))
            used_indices.append(idx)

        return selected

    def generate_report(self) -> SafetyReport:
        """Build a complete SafetyReport with human-readable text."""
        metrics = self.safety_metrics()
        energy = self.estimate_energy_wh()
        samples = self.ice_sampling_waypoints()

        lines = [
            "=" * 60,
            "  ROVER TRAVERSE ANALYSIS REPORT",
            "=" * 60,
            "",
            f"Path status       : {'SUCCESS' if self.result.success else 'FAILED'}",
            f"Waypoints         : {len(self.result.path_rc)}",
            f"Total distance    : {self.result.total_distance_m:.0f} m",
            f"Estimated time    : {self.result.estimated_time_hr:.1f} hr",
            "",
            "--- Safety Metrics ---",
            f"Max slope         : {metrics['max_slope_deg']:.1f} deg",
            f"Mean slope        : {metrics['mean_slope_deg']:.1f} deg",
            f"Hazardous cells   : {metrics['hazard_count']}  (>{PREFERRED_SLOPE_DEG} deg)",
            f"Near-impassable   : {metrics['impassable_near_misses']}  (20-{MAX_SLOPE_DEG} deg)",
            "",
            "--- Energy ---",
            f"Est. consumption  : {energy:.0f} Wh",
            "",
            "--- Cost Breakdown ---",
        ]
        for key, val in self.result.cost_breakdown.items():
            lines.append(f"  {key:20s}: {val}")
        lines.append("")
        lines.append("--- Ice Sampling Waypoints ---")
        for i, (lat, lon) in enumerate(samples, 1):
            lines.append(f"  Stop {i}: ({lat:+.4f}, {lon:+.4f})")
        lines.append("")
        lines.append("=" * 60)

        report_text = "\n".join(lines)

        return SafetyReport(
            max_slope_deg=metrics["max_slope_deg"],
            mean_slope_deg=metrics["mean_slope_deg"],
            hazard_cell_count=metrics["hazard_count"],
            impassable_near_misses=metrics["impassable_near_misses"],
            total_distance_m=self.result.total_distance_m,
            estimated_energy_wh=energy,
            sampling_waypoints=samples,
            report_text=report_text,
        )


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------
def plan_traverse(
    ice_prob_map: np.ndarray,
    slope_map: np.ndarray,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    landing_lat: float,
    landing_lon: float,
    target_lat: float,
    target_lon: float,
    illumination_map: Optional[np.ndarray] = None,
    max_range_m: float = MAX_RANGE_M,
) -> TraverseResult:
    """
    One-call convenience wrapper: plan a traverse and return the result.

    Parameters
    ----------
    ice_prob_map : np.ndarray   2-D ice probability [0,1].
    slope_map    : np.ndarray   2-D slope in degrees.
    lat_grid     : np.ndarray   2-D latitude per pixel.
    lon_grid     : np.ndarray   2-D longitude per pixel.
    landing_lat, landing_lon : float   Landing site coordinates.
    target_lat, target_lon   : float   Target coordinates.
    illumination_map : np.ndarray, optional   Solar illumination [0,1].
    max_range_m : float   Maximum traverse distance (default 5 km).

    Returns
    -------
    TraverseResult
    """
    planner = RoverTraversePlanner(
        ice_prob_map=ice_prob_map,
        slope_map=slope_map,
        lat_grid=lat_grid,
        lon_grid=lon_grid,
        landing_lat=landing_lat,
        landing_lon=landing_lon,
        illumination_map=illumination_map,
    )
    return planner.plan_path(target_lat, target_lon, max_range_m=max_range_m)


def visualize_traverse(
    path: TraverseResult,
    ice_prob_map: np.ndarray,
    slope_map: np.ndarray,
    lat_grid: Optional[np.ndarray] = None,
    lon_grid: Optional[np.ndarray] = None,
) -> "matplotlib.figure.Figure":
    """
    Render the traverse path overlaid on ice-probability and slope maps.

    Returns a matplotlib Figure with two side-by-side panels:
        Left  : ice probability map with path
        Right : slope map with path and hazard colouring

    Parameters
    ----------
    path : TraverseResult
        Output from RoverTraversePlanner.plan_path or plan_traverse.
    ice_prob_map : np.ndarray
        2-D ice probability array.
    slope_map : np.ndarray
        2-D slope array (degrees).
    lat_grid, lon_grid : np.ndarray, optional
        If provided, axes are labelled in lat/lon; otherwise pixel coords.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Determine axis extents
    if lat_grid is not None and lon_grid is not None:
        extent = [
            float(lon_grid.min()), float(lon_grid.max()),
            float(lat_grid.min()), float(lat_grid.max()),
        ]
        xlabel, ylabel = "Longitude (deg)", "Latitude (deg)"
        # Path in lat/lon
        if path.path_latlon:
            plats = [p[0] for p in path.path_latlon]
            plons = [p[1] for p in path.path_latlon]
        else:
            plats, plons = [], []
        use_latlon = True
    else:
        extent = None
        xlabel, ylabel = "Column (px)", "Row (px)"
        use_latlon = False

    # --- Left panel: ice probability ---
    im1 = ax1.imshow(
        ice_prob_map, cmap="YlOrRd", origin="lower",
        extent=extent, aspect="auto", vmin=0, vmax=1,
    )
    if path.path_rc:
        if use_latlon:
            ax1.plot(plons, plats, "c-", linewidth=2, label="Traverse")
            ax1.plot(plons[0], plats[0], "g^", markersize=10, label="Landing")
            ax1.plot(plons[-1], plats[-1], "r*", markersize=12, label="Target")
        else:
            rows = [p[0] for p in path.path_rc]
            cols = [p[1] for p in path.path_rc]
            ax1.plot(cols, rows, "c-", linewidth=2, label="Traverse")
            ax1.plot(cols[0], rows[0], "g^", markersize=10, label="Landing")
            ax1.plot(cols[-1], rows[-1], "r*", markersize=12, label="Target")
    ax1.set_title("Ice Probability + Traverse Path")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.legend(loc="upper right", fontsize=8)
    fig.colorbar(im1, ax=ax1, label="P(ice)")

    # --- Right panel: slope with hazard zones ---
    cmap_slope = plt.cm.terrain.copy()
    norm_slope = mcolors.TwoSlopeNorm(
        vmin=0, vcenter=PREFERRED_SLOPE_DEG, vmax=MAX_SLOPE_DEG + 10
    )
    im2 = ax2.imshow(
        slope_map, cmap=cmap_slope, norm=norm_slope, origin="lower",
        extent=extent, aspect="auto",
    )
    # Overlay impassable zones
    impassable_mask = slope_map > MAX_SLOPE_DEG
    if np.any(impassable_mask):
        ax2.contour(
            impassable_mask.astype(float),
            levels=[0.5], colors="red", linewidths=0.5,
            extent=extent if extent else None,
            origin="lower",
        )
    if path.path_rc:
        if use_latlon:
            ax2.plot(plons, plats, "w-", linewidth=2, label="Traverse")
            ax2.plot(plons[0], plats[0], "g^", markersize=10, label="Landing")
            ax2.plot(plons[-1], plats[-1], "r*", markersize=12, label="Target")
        else:
            rows = [p[0] for p in path.path_rc]
            cols = [p[1] for p in path.path_rc]
            ax2.plot(cols, rows, "w-", linewidth=2, label="Traverse")
            ax2.plot(cols[0], rows[0], "g^", markersize=10, label="Landing")
            ax2.plot(cols[-1], rows[-1], "r*", markersize=12, label="Target")
    ax2.set_title("Terrain Slope + Hazard Zones")
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel)
    ax2.legend(loc="upper right", fontsize=8)
    fig.colorbar(im2, ax=ax2, label="Slope (deg)")

    fig.suptitle(
        f"Rover Traverse  |  {path.total_distance_m:.0f} m  |  "
        f"~{path.estimated_time_hr:.1f} hr",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    return fig
