"""
Quantitative Subsurface Ice Volume Estimation for Lunar Polar Regions.

Implements dielectric mixing models and radar-based ice volume calculation
for the top 0-5 meters of lunar regolith, following methodology consistent
with Chandrayaan-2 DFSAR L-band and S-band observations.

Physical basis:
    - Dielectric mixing (Maxwell-Garnett / Lichtenecker) for ice-regolith
    - CPR-based empirical ice fraction estimation
    - Radar penetration depth from frequency and mixture permittivity
    - Monte Carlo uncertainty propagation

References:
    - PRL 2026: Chandrayaan-2 DFSAR polarimetric study of lunar PSRs
    - Thompson et al. (2011): Lunar ice constraints from Mini-SAR CPR
    - Campbell et al. (2006): Radar properties of lunar regolith
    - Hapke (1990): Coherent backscatter and lunar ice

Constants:
    - Water ice dielectric constant: 3.15 (at microwave, ~100 K)
    - Dry lunar regolith dielectric constant: ~3.0 (bulk, porosity ~50%)
    - Water ice density: 917 kg/m^3
    - Lunar regolith bulk density: 1500 kg/m^3
    - DFSAR L-band wavelength: 24 cm (1.25 GHz)
    - DFSAR S-band wavelength: 12 cm (2.5 GHz)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
EPSILON_REGOLITH = 3.0       # Dry lunar regolith real permittivity
EPSILON_ICE = 3.15           # Water ice real permittivity at ~100 K
DENSITY_ICE = 917.0          # kg/m^3
DENSITY_REGOLITH = 1500.0    # kg/m^3 (bulk, ~50% porosity)
SPEED_OF_LIGHT = 2.998e8     # m/s

# Loss tangent values for penetration depth calculation
LOSS_TAN_REGOLITH = 0.005    # Dry regolith, very low loss
LOSS_TAN_ICE = 0.0001        # Pure ice, extremely low loss at cryo temps

# Chandrayaan-2 DFSAR parameters
LBAND_FREQ_GHZ = 1.25       # L-band center frequency
SBAND_FREQ_GHZ = 2.5        # S-band center frequency
LBAND_WAVELENGTH_M = 0.24   # 24 cm
SBAND_WAVELENGTH_M = 0.12   # 12 cm

DEFAULT_PIXEL_AREA_M2 = 625.0  # 25 m x 25 m pixel


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def cpr_to_ice_fraction(cpr: np.ndarray) -> np.ndarray:
    """
    Empirical mapping from Circular Polarization Ratio to ice volume fraction.

    Based on radar scattering models for ice-regolith mixtures:
        - CPR <= 0.6: No anomalous scattering, ice fraction ~ 0
        - CPR = 1.0:  ~5% ice by volume (moderate coherent backscatter)
        - CPR = 1.5:  ~15% ice by volume (strong volumetric scattering)
        - CPR = 2.0:  ~25% ice by volume (very strong CB opposition effect)
        - CPR >= 2.5: Saturates at ~30% (physical upper bound from models)

    The relationship is derived from a piecewise-linear fit to scattering
    simulations of wavelength-scale ice inclusions embedded in regolith,
    consistent with Thompson et al. (2011) and Hapke (1990) CB models.

    Args:
        cpr: CPR values, array of any shape. Values below 0.6 yield 0.

    Returns:
        Ice volume fraction in range [0.0, 0.30].
    """
    cpr = np.asarray(cpr, dtype=np.float64)

    # Piecewise linear: knot points (cpr, fraction)
    # (0.6, 0.0), (1.0, 0.05), (1.5, 0.15), (2.0, 0.25), (2.5, 0.30)
    knot_cpr = np.array([0.6, 1.0, 1.5, 2.0, 2.5])
    knot_frac = np.array([0.0, 0.05, 0.15, 0.25, 0.30])

    fraction = np.interp(cpr, knot_cpr, knot_frac)
    return fraction.astype(np.float64)


def estimate_penetration_depth(
    frequency_ghz: float,
    ice_fraction: np.ndarray,
) -> np.ndarray:
    """
    Estimate one-way radar penetration depth in an ice-regolith mixture.

    Penetration depth (skin depth) for a low-loss dielectric:
        delta = lambda / (2 * pi * sqrt(epsilon') * tan_delta)

    where epsilon' is the real permittivity of the mixture and tan_delta
    is the effective loss tangent. Ice lowers the loss tangent and
    slightly increases permittivity, yielding deeper penetration.

    For DFSAR L-band (1.25 GHz, lambda=24 cm):
        - Pure regolith: delta ~ 2.0 m
        - 10% ice mixture: delta ~ 2.5 m
        - 30% ice mixture: delta ~ 3.5 m

    For DFSAR S-band (2.5 GHz, lambda=12 cm):
        - Pure regolith: delta ~ 1.0 m

    Args:
        frequency_ghz: Radar frequency in GHz.
        ice_fraction: Volumetric ice fraction, array of any shape, in [0, 1].

    Returns:
        Penetration depth in meters, same shape as ice_fraction.
    """
    ice_fraction = np.asarray(ice_fraction, dtype=np.float64)
    wavelength = SPEED_OF_LIGHT / (frequency_ghz * 1e9)

    # Effective permittivity via Lichtenecker logarithmic mixing
    epsilon_eff = _lichtenecker_permittivity(ice_fraction)

    # Effective loss tangent — linear mixing of loss tangents
    tan_delta_eff = (
        (1.0 - ice_fraction) * LOSS_TAN_REGOLITH
        + ice_fraction * LOSS_TAN_ICE
    )
    # Prevent zero loss tangent (would give infinite depth)
    tan_delta_eff = np.maximum(tan_delta_eff, 1e-6)

    # Skin depth formula for low-loss dielectric
    depth = wavelength / (2.0 * np.pi * np.sqrt(epsilon_eff) * tan_delta_eff)

    # Physical clamp: maximum meaningful depth is ~5 m for top regolith layer
    depth = np.clip(depth, 0.0, 5.0)

    return depth


def _lichtenecker_permittivity(ice_fraction: np.ndarray) -> np.ndarray:
    """
    Lichtenecker logarithmic mixing rule for effective permittivity.

    ln(epsilon_eff) = f_ice * ln(epsilon_ice) + (1 - f_ice) * ln(epsilon_reg)

    This model is appropriate for randomly distributed inclusions and is
    widely used in planetary radar studies of icy regolith.

    Args:
        ice_fraction: Volumetric ice fraction in [0, 1].

    Returns:
        Effective real permittivity of the mixture.
    """
    ice_fraction = np.asarray(ice_fraction, dtype=np.float64)
    log_eps = (
        ice_fraction * np.log(EPSILON_ICE)
        + (1.0 - ice_fraction) * np.log(EPSILON_REGOLITH)
    )
    return np.exp(log_eps)


# ---------------------------------------------------------------------------
# DielectricModel class
# ---------------------------------------------------------------------------

class DielectricModel:
    """
    Dielectric mixing model for ice-regolith mixtures on the lunar surface.

    Supports both Maxwell-Garnett and Lichtenecker formulations for
    computing effective permittivity of a two-phase mixture (regolith host
    with ice inclusions).

    The model can also invert polarimetric observables (CPR, SERD, T-ratio)
    to estimate the volumetric ice fraction.
    """

    def __init__(
        self,
        epsilon_host: float = EPSILON_REGOLITH,
        epsilon_inclusion: float = EPSILON_ICE,
        method: str = "lichtenecker",
    ):
        """
        Args:
            epsilon_host: Real permittivity of the host material (regolith).
            epsilon_inclusion: Real permittivity of the inclusion (ice).
            method: Mixing rule — 'lichtenecker' or 'maxwell_garnett'.
        """
        if method not in ("lichtenecker", "maxwell_garnett"):
            raise ValueError(
                f"Unknown mixing method '{method}'. "
                "Use 'lichtenecker' or 'maxwell_garnett'."
            )
        self.epsilon_host = epsilon_host
        self.epsilon_inclusion = epsilon_inclusion
        self.method = method

    def effective_permittivity(self, ice_fraction: np.ndarray) -> np.ndarray:
        """
        Compute effective permittivity for a given ice volume fraction.

        Args:
            ice_fraction: Volume fraction of ice in [0, 1].

        Returns:
            Effective real permittivity of the mixture.
        """
        f = np.asarray(ice_fraction, dtype=np.float64)
        eh = self.epsilon_host
        ei = self.epsilon_inclusion

        if self.method == "lichtenecker":
            log_eps = f * np.log(ei) + (1.0 - f) * np.log(eh)
            return np.exp(log_eps)

        # Maxwell-Garnett: inclusions (ice) in host (regolith)
        # epsilon_eff = eh * (ei + 2*eh + 2*f*(ei - eh)) /
        #               (ei + 2*eh - f*(ei - eh))
        numerator = ei + 2.0 * eh + 2.0 * f * (ei - eh)
        denominator = ei + 2.0 * eh - f * (ei - eh)
        return eh * numerator / denominator

    def estimate_ice_fraction(
        self,
        cpr: np.ndarray,
        serd: Optional[np.ndarray] = None,
        t_ratio: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Estimate ice volume fraction from polarimetric radar observables.

        Primary estimator: CPR-based empirical curve (always used).
        Secondary corrections when available:
            - SERD (Same-sense Enhanced Radar Darkness): reduces false positives
              from surface roughness. Pixels with high SERD (> 0.3) are likely
              rough surfaces, not ice — ice fraction is attenuated.
            - T-ratio (coherency eigenvalue ratio T22/T11): high values indicate
              double-bounce from dihedral ice lenses. Boosts ice fraction
              estimate when T-ratio > 0.5.

        Args:
            cpr: Circular Polarization Ratio, any shape.
            serd: Optional SERD map, same shape as cpr.
            t_ratio: Optional coherency T-matrix eigenvalue ratio, same shape.

        Returns:
            Estimated ice volume fraction in [0, 0.30].
        """
        cpr = np.asarray(cpr, dtype=np.float64)
        base_fraction = cpr_to_ice_fraction(cpr)

        # SERD correction: penalize pixels with high SERD (rough surface)
        if serd is not None:
            serd = np.asarray(serd, dtype=np.float64)
            # Smooth suppression above SERD = 0.3
            # At SERD=0.3, scale=1.0; at SERD=0.6, scale~0.1
            roughness_scale = np.exp(-5.0 * np.maximum(serd - 0.3, 0.0))
            base_fraction = base_fraction * roughness_scale

        # T-ratio correction: boost fraction for dihedral signatures
        if t_ratio is not None:
            t_ratio = np.asarray(t_ratio, dtype=np.float64)
            # Gentle boost when T22/T11 > 0.5 (indicates double-bounce)
            # Max boost factor of 1.3
            dihedral_boost = 1.0 + 0.3 * np.clip((t_ratio - 0.5) / 0.5, 0.0, 1.0)
            base_fraction = base_fraction * dihedral_boost

        return np.clip(base_fraction, 0.0, 0.30)


# ---------------------------------------------------------------------------
# IceVolumeEstimator class
# ---------------------------------------------------------------------------

class IceVolumeEstimator:
    """
    Quantitative ice volume estimation from radar-derived maps.

    Integrates ice probability, penetration depth, and ice fraction over
    a spatial grid to produce total and per-crater volume estimates with
    Monte Carlo uncertainty bounds.
    """

    def __init__(
        self,
        pixel_area_m2: float = DEFAULT_PIXEL_AREA_M2,
        frequency_ghz: float = LBAND_FREQ_GHZ,
        n_mc_samples: int = 1000,
        rng_seed: Optional[int] = 42,
    ):
        """
        Args:
            pixel_area_m2: Ground area per pixel in m^2 (default 625 for 25x25 m).
            frequency_ghz: Radar frequency in GHz for penetration depth.
            n_mc_samples: Number of Monte Carlo iterations for uncertainty.
            rng_seed: Random seed for reproducibility (None for random).
        """
        self.pixel_area_m2 = pixel_area_m2
        self.frequency_ghz = frequency_ghz
        self.n_mc_samples = n_mc_samples
        self.rng = np.random.default_rng(rng_seed)
        self.dielectric = DielectricModel()

        # Store last estimation results for report generation
        self._last_result: Optional[Dict] = None

    def estimate_volume(
        self,
        ice_prob: np.ndarray,
        depth_map: Optional[np.ndarray],
        cpr_map: np.ndarray,
        valid_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Estimate total subsurface ice volume over a region.

        For each valid pixel the ice volume contribution is:
            V_pixel = pixel_area * depth * ice_fraction * ice_probability

        Mass is computed as:
            M_pixel = V_pixel * rho_ice

        If depth_map is None, penetration depth is estimated from the radar
        frequency and the local ice fraction.

        Args:
            ice_prob: Ice probability map (H x W), values in [0, 1].
            depth_map: Optional estimated penetration depth map in meters.
                       If None, depth is computed from frequency and ice fraction.
            cpr_map: CPR map (H x W) for ice fraction estimation.
            valid_mask: Boolean mask (H x W). True = include pixel. If None,
                        all pixels with ice_prob > 0.1 are used.

        Returns:
            Dictionary with keys:
                'volume_m3': Total ice volume in cubic meters
                'mass_kg': Total ice mass in kilograms
                'area_m2': Total area of valid pixels
                'mean_ice_fraction': Mean ice fraction over valid pixels
                'mean_depth_m': Mean penetration depth over valid pixels
                'n_pixels': Number of valid pixels
                'volume_lower_m3': 5th percentile volume (MC)
                'volume_upper_m3': 95th percentile volume (MC)
                'mass_lower_kg': 5th percentile mass (MC)
                'mass_upper_kg': 95th percentile mass (MC)
        """
        ice_prob = np.asarray(ice_prob, dtype=np.float32)
        cpr_map = np.asarray(cpr_map, dtype=np.float32)

        if valid_mask is None:
            valid_mask = ice_prob > 0.1
        else:
            valid_mask = np.asarray(valid_mask, dtype=bool)

        # Ice fraction from CPR
        ice_fraction = self.dielectric.estimate_ice_fraction(cpr_map)

        # Depth estimation
        if depth_map is not None:
            depth = np.asarray(depth_map, dtype=np.float64)
        else:
            depth = estimate_penetration_depth(self.frequency_ghz, ice_fraction)

        # Per-pixel volume (NaN-safe: CPR NaN pixels → skip)
        pixel_volume = (
            self.pixel_area_m2
            * np.nan_to_num(depth, nan=0.0)
            * np.nan_to_num(ice_fraction, nan=0.0)
            * np.nan_to_num(ice_prob, nan=0.0)
        )
        pixel_volume = pixel_volume * valid_mask

        total_volume = float(np.nansum(pixel_volume))
        total_mass = total_volume * DENSITY_ICE

        n_pixels = int(np.sum(valid_mask))
        valid_area = n_pixels * self.pixel_area_m2

        frac_valid = ice_fraction[valid_mask]
        depth_valid = depth[valid_mask]
        mean_frac = float(np.nanmean(frac_valid)) if n_pixels > 0 else 0.0
        mean_depth = float(np.nanmean(depth_valid)) if n_pixels > 0 else 0.0

        # Monte Carlo uncertainty estimation
        mc_volumes = self._monte_carlo_volume(
            ice_prob, ice_fraction, depth, valid_mask
        )
        vol_lo = float(np.percentile(mc_volumes, 5))
        vol_hi = float(np.percentile(mc_volumes, 95))

        result = {
            "volume_m3": total_volume,
            "mass_kg": total_mass,
            "area_m2": valid_area,
            "mean_ice_fraction": mean_frac,
            "mean_depth_m": mean_depth,
            "n_pixels": n_pixels,
            "volume_lower_m3": vol_lo,
            "volume_upper_m3": vol_hi,
            "mass_lower_kg": vol_lo * DENSITY_ICE,
            "mass_upper_kg": vol_hi * DENSITY_ICE,
        }

        self._last_result = result
        return result

    def estimate_volume_per_crater(
        self,
        ice_prob: np.ndarray,
        depth_map: Optional[np.ndarray],
        cpr_map: np.ndarray,
        lat: np.ndarray,
        lon: np.ndarray,
        crater_catalog: List[Dict],
    ) -> List[Dict]:
        """
        Estimate ice volume for each crater in the catalog.

        Each crater entry must have:
            'name': str, 'lat': float (deg), 'lon': float (deg),
            'radius_km': float

        Pixels are assigned to a crater if their (lat, lon) falls within
        the crater radius. Pixels not in any crater are ignored.

        Args:
            ice_prob: Ice probability map (H x W), values in [0, 1].
            depth_map: Optional depth map in meters.
            cpr_map: CPR map (H x W).
            lat: Latitude array (H x W) in degrees.
            lon: Longitude array (H x W) in degrees.
            crater_catalog: List of crater dicts with name, lat, lon, radius_km.

        Returns:
            List of dicts, one per crater, each containing:
                'name', 'volume_m3', 'mass_kg', 'area_m2',
                'mean_ice_fraction', 'n_pixels',
                'volume_lower_m3', 'volume_upper_m3'
        """
        # float32 throughout — avoids OOM and 47-min runtime on 12237×12794 grids
        lat = np.asarray(lat, dtype=np.float32)
        lon = np.asarray(lon, dtype=np.float32)
        ice_prob = np.asarray(ice_prob, dtype=np.float32)
        cpr_map = np.asarray(cpr_map, dtype=np.float32)

        # Moon radius for angular distance
        MOON_RADIUS_KM = 1737.4

        results = []

        for crater in crater_catalog:
            c_lat = crater["lat"]
            c_lon = crater["lon"]
            # DPSR entries have d_eff_m (effective diameter, meters) instead of radius_km
            if "radius_km" in crater:
                c_radius_km = crater["radius_km"]
            elif "d_eff_m" in crater:
                c_radius_km = crater["d_eff_m"] / 2000.0  # m diameter → km radius
            elif "area_km2" in crater:
                c_radius_km = np.sqrt(crater["area_km2"] / np.pi)
            else:
                c_radius_km = 5.0  # default 5 km

            # Angular radius in degrees
            angular_radius_deg = np.degrees(c_radius_km / MOON_RADIUS_KM)

            # Great-circle distance approximation (small angle, high latitude)
            dlat = lat - c_lat
            dlon = (lon - c_lon) * np.cos(np.radians(c_lat))
            dist_deg = np.sqrt(dlat ** 2 + dlon ** 2)

            crater_mask = dist_deg <= angular_radius_deg

            if not np.any(crater_mask):
                results.append({
                    "name": crater["name"],
                    "volume_m3": 0.0,
                    "mass_kg": 0.0,
                    "area_m2": 0.0,
                    "mean_ice_fraction": 0.0,
                    "n_pixels": 0,
                    "volume_lower_m3": 0.0,
                    "volume_upper_m3": 0.0,
                })
                continue

            # Run volume estimation with crater mask
            crater_result = self.estimate_volume(
                ice_prob, depth_map, cpr_map, valid_mask=crater_mask
            )
            crater_result["name"] = crater["name"]
            results.append(crater_result)

        self._crater_results = results
        return results

    def generate_volume_report(self) -> str:
        """
        Generate a human-readable text report of ice volume estimates.

        Must be called after estimate_volume() and optionally after
        estimate_volume_per_crater().

        Returns:
            Multi-line string report with total volume, per-crater breakdown,
            and uncertainty bounds.
        """
        lines = []
        lines.append("=" * 70)
        lines.append("LUNAR SUBSURFACE ICE VOLUME ESTIMATION REPORT")
        lines.append("ISRO BAH 2026 — PS-8: Chandrayaan-2 DFSAR Analysis")
        lines.append("=" * 70)
        lines.append("")

        if self._last_result is None:
            lines.append("[No estimation has been run yet.]")
            return "\n".join(lines)

        r = self._last_result

        lines.append("METHODOLOGY")
        lines.append("-" * 40)
        lines.append(f"  Radar frequency:       {self.frequency_ghz:.2f} GHz")
        lines.append(f"  Pixel area:            {self.pixel_area_m2:.0f} m^2")
        lines.append(f"  Monte Carlo samples:   {self.n_mc_samples}")
        lines.append(f"  Dielectric model:      {self.dielectric.method}")
        lines.append(f"  Regolith permittivity: {self.dielectric.epsilon_host:.2f}")
        lines.append(f"  Ice permittivity:      {self.dielectric.epsilon_inclusion:.2f}")
        lines.append("")

        lines.append("TOTAL ICE VOLUME ESTIMATE")
        lines.append("-" * 40)
        lines.append(f"  Valid pixels:          {r['n_pixels']:,}")
        lines.append(f"  Survey area:           {r['area_m2']:,.0f} m^2 "
                      f"({r['area_m2'] / 1e6:.3f} km^2)")
        lines.append(f"  Mean ice fraction:     {r['mean_ice_fraction']:.3f} "
                      f"({r['mean_ice_fraction'] * 100:.1f}%)")
        lines.append(f"  Mean depth:            {r['mean_depth_m']:.2f} m")
        lines.append("")
        lines.append(f"  Ice volume (best):     {r['volume_m3']:,.1f} m^3")
        lines.append(f"  Ice mass (best):       {r['mass_kg']:,.1f} kg "
                      f"({r['mass_kg'] / 1e6:.2f} million tonnes)")
        lines.append("")
        lines.append(f"  90% confidence interval:")
        lines.append(f"    Volume:  [{r['volume_lower_m3']:,.1f}, "
                      f"{r['volume_upper_m3']:,.1f}] m^3")
        lines.append(f"    Mass:    [{r['mass_lower_kg']:,.1f}, "
                      f"{r['mass_upper_kg']:,.1f}] kg")

        # Per-crater breakdown
        crater_results = getattr(self, "_crater_results", None)
        if crater_results:
            lines.append("")
            lines.append("PER-CRATER BREAKDOWN")
            lines.append("-" * 40)
            lines.append(
                f"  {'Crater':<18s} {'Volume (m^3)':>14s} {'Mass (kg)':>14s} "
                f"{'Ice frac':>10s} {'Pixels':>8s}"
            )
            lines.append("  " + "-" * 66)

            total_vol = 0.0
            total_mass = 0.0
            for cr in crater_results:
                total_vol += cr["volume_m3"]
                total_mass += cr["mass_kg"]
                frac_str = (
                    f"{cr['mean_ice_fraction'] * 100:.1f}%"
                    if cr["n_pixels"] > 0 else "N/A"
                )
                lines.append(
                    f"  {cr['name']:<18s} {cr['volume_m3']:>14,.1f} "
                    f"{cr['mass_kg']:>14,.1f} {frac_str:>10s} "
                    f"{cr['n_pixels']:>8,d}"
                )

            lines.append("  " + "-" * 66)
            lines.append(
                f"  {'TOTAL':<18s} {total_vol:>14,.1f} {total_mass:>14,.1f}"
            )

        lines.append("")
        lines.append("NOTES")
        lines.append("-" * 40)
        lines.append("  - Ice fraction estimated from CPR via empirical curve")
        lines.append("  - Penetration depth derived from Lichtenecker mixing model")
        lines.append("  - Uncertainty via Monte Carlo perturbation of ice fraction,")
        lines.append("    depth, and probability inputs")
        lines.append("  - Consistent with 1-30% ice by volume (literature range)")
        lines.append("  - Ice density: 917 kg/m^3; Regolith density: 1500 kg/m^3")
        lines.append("=" * 70)

        return "\n".join(lines)

    # --- Private methods ---

    def _monte_carlo_volume(
        self,
        ice_prob: np.ndarray,
        ice_fraction: np.ndarray,
        depth: np.ndarray,
        valid_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Monte Carlo uncertainty estimation for total ice volume.

        Perturbs three input quantities with realistic uncertainties:
            - ice_prob:     +/- 15% (multiplicative, log-normal)
            - ice_fraction: +/- 30% (multiplicative, log-normal)
            - depth:        +/- 25% (multiplicative, log-normal)

        These fractional uncertainties reflect the dominant error sources:
        ice fraction depends on the CPR calibration curve, depth on the
        loss tangent assumption, and probability on the ML classifier.

        Args:
            ice_prob: Ice probability map (H x W).
            ice_fraction: Ice volume fraction map (H x W).
            depth: Penetration depth map in meters (H x W).
            valid_mask: Boolean mask (H x W).

        Returns:
            1-D array of length n_mc_samples with total volume per trial.
        """
        volumes = np.empty(self.n_mc_samples, dtype=np.float64)

        # Precompute the base per-pixel contribution (NaN-safe, only valid pixels)
        base = (self.pixel_area_m2
                * np.nan_to_num(depth, nan=0.0)
                * np.nan_to_num(ice_fraction, nan=0.0)
                * np.nan_to_num(ice_prob, nan=0.0)
                * valid_mask)

        # Relative uncertainty standard deviations (in log-space)
        sigma_prob = 0.15
        sigma_frac = 0.30
        sigma_depth = 0.25

        for i in range(self.n_mc_samples):
            # Multiplicative perturbation factors (scalar per trial for speed;
            # pixel-wise correlation is assumed high for systematic errors)
            perturb_prob = self.rng.lognormal(
                mean=-0.5 * sigma_prob ** 2, sigma=sigma_prob
            )
            perturb_frac = self.rng.lognormal(
                mean=-0.5 * sigma_frac ** 2, sigma=sigma_frac
            )
            perturb_depth = self.rng.lognormal(
                mean=-0.5 * sigma_depth ** 2, sigma=sigma_depth
            )

            total_perturb = perturb_prob * perturb_frac * perturb_depth
            volumes[i] = np.sum(base) * total_perturb

        return volumes
