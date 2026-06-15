"""
DFSAR Data Preprocessing Pipeline.

Handles:
    - Loading raw DFSAR SLC data
    - Radiometric calibration
    - Speckle filtering
    - Geocoding
    - Patch extraction
    - Integration with DEM data (LRO LOLA)
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class DFSARPreprocessor:
    """
    Preprocessor for Chandrayaan-2 DFSAR data products.

    DFSAR operates in two frequency bands:
        - L-band (24 cm wavelength): Better penetration depth (~1-2m)
        - S-band (12 cm wavelength): Higher surface sensitivity

    Modes: Full-pol (HH, HV, VH, VV) in stripmap mode at 2m resolution
    """

    def __init__(self, config: dict):
        self.config = config
        self.patch_size = config['data']['preprocessing']['patch_size']
        self.stride = config['data']['preprocessing']['stride']

    def load_slc(self, filepath: str) -> Dict[str, np.ndarray]:
        """
        Load DFSAR SLC (Single Look Complex) product.

        DFSAR SLC data format: Complex float32, separate files per polarization.
        Actual loading depends on data format from PRADAN portal.
        """
        path = Path(filepath)

        if path.suffix == '.tif' or path.suffix == '.tiff':
            return self._load_geotiff(path)
        elif path.suffix == '.npy':
            return self._load_numpy(path)
        else:
            logger.warning(f"Unknown format: {path.suffix}. Attempting generic load.")
            return self._load_generic(path)

    def _load_geotiff(self, path: Path) -> Dict[str, np.ndarray]:
        """Load multi-band GeoTIFF SAR data."""
        try:
            import rasterio
            with rasterio.open(path) as src:
                data = src.read()
                profile = src.profile
                transform = src.transform

            # Assume bands: [HH_real, HH_imag, HV_real, HV_imag, VH_real, VH_imag, VV_real, VV_imag]
            if data.shape[0] >= 8:
                return {
                    'hh': data[0] + 1j * data[1],
                    'hv': data[2] + 1j * data[3],
                    'vh': data[4] + 1j * data[5],
                    'vv': data[6] + 1j * data[7],
                    'profile': profile,
                    'transform': transform,
                }
            elif data.shape[0] >= 4:
                # Intensity only
                return {
                    'hh': np.sqrt(data[0]) + 0j,
                    'hv': np.sqrt(data[1]) + 0j,
                    'vh': np.sqrt(data[2]) + 0j,
                    'vv': np.sqrt(data[3]) + 0j,
                    'profile': profile,
                    'transform': transform,
                }
        except ImportError:
            logger.error("rasterio not installed. Install with: pip install rasterio")
            raise

    def _load_numpy(self, path: Path) -> Dict[str, np.ndarray]:
        """Load from numpy archive."""
        data = np.load(path, allow_pickle=True)
        return dict(data)

    def _load_generic(self, path: Path) -> Dict[str, np.ndarray]:
        """Attempt generic binary load."""
        data = np.fromfile(path, dtype=np.complex64)
        side = int(np.sqrt(len(data)))
        data = data[:side * side].reshape(side, side)
        return {'hh': data, 'hv': data * 0, 'vh': data * 0, 'vv': data}

    def apply_speckle_filter(
        self,
        data: np.ndarray,
        method: str = "lee",
        window_size: int = 5
    ) -> np.ndarray:
        """
        Apply speckle filter to SAR data.

        Args:
            data: Complex SAR image
            method: 'boxcar', 'lee', or 'refined_lee'
            window_size: Filter window size
        """
        if method == "boxcar":
            return self._boxcar_filter(data, window_size)
        elif method == "lee":
            return self._lee_filter(data, window_size)
        else:
            return self._boxcar_filter(data, window_size)

    @staticmethod
    def _boxcar_filter(data: np.ndarray, window_size: int) -> np.ndarray:
        """Simple boxcar (moving average) speckle filter."""
        from scipy.ndimage import uniform_filter
        real = uniform_filter(data.real.astype(np.float64), window_size)
        imag = uniform_filter(data.imag.astype(np.float64), window_size)
        return (real + 1j * imag).astype(np.complex64)

    @staticmethod
    def _lee_filter(data: np.ndarray, window_size: int) -> np.ndarray:
        """
        Lee speckle filter — adaptive filter preserving edges.

        Minimizes MSE between filtered and true signal under
        multiplicative noise model.
        """
        from scipy.ndimage import uniform_filter

        intensity = np.abs(data) ** 2
        mean_intensity = uniform_filter(intensity.astype(np.float64), window_size)
        sq_mean = uniform_filter(intensity.astype(np.float64) ** 2, window_size)

        variance = sq_mean - mean_intensity ** 2
        overall_var = np.var(intensity)

        # Lee filter weight
        weight = np.where(
            variance > 1e-10,
            np.clip(1.0 - overall_var / (variance + 1e-10), 0, 1),
            1.0
        )

        # Apply filter
        mean_data = uniform_filter(data.real.astype(np.float64), window_size) + \
                    1j * uniform_filter(data.imag.astype(np.float64), window_size)

        filtered = mean_data + weight * (data - mean_data)
        return filtered.astype(np.complex64)

    def radiometric_calibration(
        self,
        data: Dict[str, np.ndarray],
        calibration_factor: float = 1.0
    ) -> Dict[str, np.ndarray]:
        """
        Apply radiometric calibration to convert DN to sigma-nought.

        sigma0 = |DN|^2 * calibration_factor / sin(incidence_angle)
        """
        calibrated = {}
        for pol in ['hh', 'hv', 'vh', 'vv']:
            if pol in data:
                calibrated[pol] = data[pol] * np.sqrt(calibration_factor)

        # Preserve metadata
        for key in data:
            if key not in ['hh', 'hv', 'vh', 'vv']:
                calibrated[key] = data[key]

        return calibrated

    def extract_patches(
        self,
        features: Dict[str, np.ndarray],
        label: Optional[np.ndarray] = None,
        coordinates: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> List[Dict]:
        """
        Extract fixed-size patches from feature maps.

        Args:
            features: Dict of feature name -> 2D array
            label: Optional label map (H x W)
            coordinates: Optional (lat_grid, lon_grid) arrays

        Returns:
            List of patch dicts with features, labels, coordinates
        """
        # Get spatial dimensions from first feature
        first_key = list(features.keys())[0]
        h, w = features[first_key].shape
        ps = self.patch_size
        stride = self.stride

        patches = []

        for y in range(0, h - ps + 1, stride):
            for x in range(0, w - ps + 1, stride):
                patch_features = {}
                for name, arr in features.items():
                    patch_features[name] = arr[y:y+ps, x:x+ps]

                patch = {
                    'features': np.stack(list(patch_features.values()), axis=0),
                    'feature_names': list(patch_features.keys()),
                }

                if label is not None:
                    patch['label'] = label[y:y+ps, x:x+ps]

                if coordinates is not None:
                    lat_grid, lon_grid = coordinates
                    patch['lat'] = float(np.mean(lat_grid[y:y+ps, x:x+ps]))
                    patch['lon'] = float(np.mean(lon_grid[y:y+ps, x:x+ps]))

                patches.append(patch)

        return patches

    def save_patches(
        self,
        patches: List[Dict],
        output_dir: str,
        split: str = "train"
    ):
        """Save extracted patches to disk."""
        out_path = Path(output_dir) / "processed" / split
        out_path.mkdir(parents=True, exist_ok=True)

        for i, patch in enumerate(patches):
            filepath = out_path / f"patch_{i:06d}.npz"
            np.savez_compressed(
                filepath,
                features=patch['features'],
                label=patch.get('label', np.zeros((self.patch_size, self.patch_size))),
                lat=patch.get('lat', -85.0),
                lon=patch.get('lon', 0.0),
            )

        logger.info(f"Saved {len(patches)} patches to {out_path}")


def load_lola_dem(filepath: str) -> np.ndarray:
    """
    Load LRO LOLA Digital Elevation Model.

    LOLA provides lunar topography at ~118m/pixel resolution.
    Used for terrain slope and crater depth features.
    """
    try:
        import rasterio
        with rasterio.open(filepath) as src:
            dem = src.read(1)
            return dem.astype(np.float32)
    except ImportError:
        logger.error("rasterio required for DEM loading")
        raise


def compute_terrain_features(dem: np.ndarray, resolution_m: float = 118.0) -> Dict[str, np.ndarray]:
    """
    Compute terrain features from DEM.

    Returns:
        slope: Terrain slope in degrees
        roughness: Surface roughness (std of elevation in local window)
        aspect: Terrain aspect angle
        curvature: Surface curvature
    """
    # Slope (gradient magnitude)
    dy, dx = np.gradient(dem, resolution_m)
    slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))

    # Aspect
    aspect = np.degrees(np.arctan2(-dx, dy))
    aspect = np.where(aspect < 0, aspect + 360, aspect)

    # Roughness (local std)
    from scipy.ndimage import generic_filter
    roughness = generic_filter(dem, np.std, size=5)

    # Curvature (Laplacian)
    from scipy.ndimage import laplace
    curvature = laplace(dem)

    return {
        'slope': slope.astype(np.float32),
        'aspect': aspect.astype(np.float32),
        'roughness': roughness.astype(np.float32),
        'curvature': curvature.astype(np.float32),
    }
