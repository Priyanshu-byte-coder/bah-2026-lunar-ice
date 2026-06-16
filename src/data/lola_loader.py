"""
LOLA GDR Polar DEM Loader.

Loads NASA LRO LOLA Gridded Data Record (GDR) polar DEM products.

Supported products:
    ldem_875s_20m.img  — -90° to -87.5°, 20m/pixel (7584×7584)
    ldem_85s_20m.img   — -90° to -85°,   20m/pixel
    ldem_80s_40m.img   — -90° to -80°,   40m/pixel

Format: PDS3 binary, LSB_INTEGER 16-bit, scale_factor=0.5, offset=1737400m
Projection: South Pole Stereographic (center=-90°, 0°)

Usage:
    from src.data.lola_loader import LOLADEMLoader
    dem = LOLADEMLoader("data/raw/lola_dem/ldem_875s_20m.img")
    dem.load()
    slope = dem.compute_slope()  # degrees
    lat, lon = dem.get_coordinates()
"""

import numpy as np
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict

logger = logging.getLogger(__name__)

# Moon radius (matches DFSAR loader)
MOON_RADIUS_M = 1737400.0


def _lbl_value(lbl_path: Path, key: str) -> Optional[str]:
    """Extract value for a key from PDS .lbl file."""
    for ext in ['.lbl', '.lbl.txt', '.LBL']:
        p = lbl_path.with_suffix(ext) if ext.startswith('.lbl') else Path(str(lbl_path) + ext)
        if p.exists():
            for line in p.read_text(errors='ignore').splitlines():
                if key in line and '=' in line:
                    return line.split('=', 1)[1].strip().strip('"')
    return None


class LOLADEMLoader:
    """
    Loads LOLA GDR polar DEM binary products.

    The .img file is a raw binary array:
        - LINES × LINE_SAMPLES signed 16-bit integers (little-endian)
        - Height (m) = DN * SCALING_FACTOR  (0.5 by default)
        - Planetary radius (m) = DN * 0.5 + 1737400
    """

    # Known product metadata (fallback if .lbl unavailable)
    KNOWN_PRODUCTS = {
        'ldem_875s_20m': dict(lines=7584, samples=7584, scale=20.0,
                               lat_min=-90.0, lat_max=-87.5,
                               proj_offset=3791.5, scaling_factor=0.5),
        'ldem_875s_40m': dict(lines=3792, samples=3792, scale=40.0,
                               lat_min=-90.0, lat_max=-87.5,
                               proj_offset=1895.5, scaling_factor=0.5),
        'ldem_875s_10m': dict(lines=15168, samples=15168, scale=10.0,
                               lat_min=-90.0, lat_max=-87.5,
                               proj_offset=7583.5, scaling_factor=0.5),
        'ldem_85s_20m':  dict(lines=15168, samples=15168, scale=20.0,
                               lat_min=-90.0, lat_max=-85.0,
                               proj_offset=7583.5, scaling_factor=0.5),
        'ldem_85s_40m':  dict(lines=7584, samples=7584, scale=40.0,
                               lat_min=-90.0, lat_max=-85.0,
                               proj_offset=3791.5, scaling_factor=0.5),
        'ldem_80s_40m':  dict(lines=13501, samples=13501, scale=40.0,
                               lat_min=-90.0, lat_max=-80.0,
                               proj_offset=6750.0, scaling_factor=0.5),
        'ldem_80s_80m':  dict(lines=6751, samples=6751, scale=80.0,
                               lat_min=-90.0, lat_max=-80.0,
                               proj_offset=3375.0, scaling_factor=0.5),
    }

    def __init__(self, img_path: str):
        self.img_path = Path(img_path)
        self.elevation: Optional[np.ndarray] = None  # (H, W) float32 meters
        self.slope: Optional[np.ndarray] = None       # (H, W) float32 degrees
        self.lat: Optional[np.ndarray] = None
        self.lon: Optional[np.ndarray] = None
        self._meta: Optional[Dict] = None

    def _get_meta(self) -> Dict:
        """Read metadata from .lbl or use known product table."""
        stem = self.img_path.stem.lower()
        if stem in self.KNOWN_PRODUCTS:
            return self.KNOWN_PRODUCTS[stem].copy()

        # Try parsing .lbl
        lbl = self.img_path.with_suffix('.lbl')
        if not lbl.exists():
            lbl = Path(str(self.img_path) + '.txt')  # user renamed it .lbl.txt

        meta = {}
        for ext in ['.lbl', '.lbl.txt']:
            candidate = Path(str(self.img_path).replace('.img', ext))
            if candidate.exists():
                text = candidate.read_text(errors='ignore')
                for line in text.splitlines():
                    line = line.strip()
                    if '=' in line:
                        k, v = line.split('=', 1)
                        k, v = k.strip(), v.strip().strip('"').split()[0].rstrip(',')
                        if k == 'LINES':
                            meta['lines'] = int(v)
                        elif k == 'LINE_SAMPLES':
                            meta['samples'] = int(v)
                        elif k == 'MAP_SCALE':
                            meta['scale'] = float(v)
                        elif k == 'SCALING_FACTOR':
                            meta['scaling_factor'] = float(v)
                        elif k == 'MINIMUM_LATITUDE':
                            meta['lat_min'] = float(v)
                        elif k == 'MAXIMUM_LATITUDE':
                            meta['lat_max'] = float(v)
                        elif k in ('LINE_PROJECTION_OFFSET', 'SAMPLE_PROJECTION_OFFSET'):
                            meta['proj_offset'] = float(v)
                break

        if not meta:
            raise ValueError(
                f"Cannot determine metadata for {self.img_path}. "
                f"Known products: {list(self.KNOWN_PRODUCTS.keys())}"
            )
        meta.setdefault('scaling_factor', 0.5)
        return meta

    def load(self) -> np.ndarray:
        """
        Load LOLA DEM from binary .img file.

        Returns:
            (H, W) float32 elevation array in meters above reference sphere.
        """
        meta = self._get_meta()
        self._meta = meta
        h, w = meta['lines'], meta['samples']
        scaling = meta.get('scaling_factor', 0.5)

        logger.info(
            f"Loading LOLA DEM: {self.img_path.name} "
            f"({h}×{w} px, {meta.get('scale', '?')}m/px, "
            f"lat [{meta.get('lat_min')}°, {meta.get('lat_max')}°])"
        )

        raw = np.frombuffer(
            self.img_path.read_bytes(),
            dtype='<i2',           # LSB signed 16-bit
        ).reshape(h, w)

        # Convert DN to elevation (meters)
        self.elevation = (raw.astype(np.float32) * np.float32(scaling))
        logger.info(
            f"Elevation range: [{self.elevation.min():.1f}, {self.elevation.max():.1f}] m"
        )
        return self.elevation

    def get_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute selenographic lat/lon for each pixel.

        Uses south pole stereographic → lat/lon conversion.
        Matches the DFSAR UPS projection formulas.

        Returns:
            lat, lon: (H, W) float32 arrays in degrees
        """
        if self._meta is None:
            self.load()

        meta = self._meta
        h, w = meta['lines'], meta['samples']
        scale = np.float32(meta['scale'])         # metres/pixel
        proj_offset = np.float32(meta['proj_offset'])  # centre pixel offset

        # Pixel → projected metres (south-pole stereographic, y increasing north)
        cols = np.arange(w, dtype=np.float32)
        rows = np.arange(h, dtype=np.float32)
        C, R = np.meshgrid(cols, rows)

        x_m = (C - proj_offset) * scale     # east (+x) / west (-x)
        y_m = (proj_offset - R) * scale     # north (+y) / south (-y)

        # UPS → lat/lon  (same formula as DFSARMosaicLoader.get_coordinates)
        R_moon = np.float32(MOON_RADIUS_M)
        rho = np.sqrt(x_m**2 + y_m**2)
        lat = -(np.float32(90.0) - 2.0 * np.degrees(np.arctan(rho / (2.0 * R_moon))))
        lon = np.degrees(np.arctan2(x_m, -y_m))

        self.lat = lat.astype(np.float32)
        self.lon = lon.astype(np.float32)
        return self.lat, self.lon

    def compute_slope(self, smooth_sigma: float = 1.0) -> np.ndarray:
        """
        Compute terrain slope in degrees from elevation gradient.

        slope = atan(sqrt((dz/dx)² + (dz/dy)²))

        Args:
            smooth_sigma: Gaussian smoothing sigma (pixels) before gradient.
                          0 = no smoothing.

        Returns:
            (H, W) float32 slope in degrees.
        """
        if self.elevation is None:
            self.load()

        elev = self.elevation.copy()

        if smooth_sigma > 0:
            from scipy.ndimage import gaussian_filter
            elev = gaussian_filter(elev, sigma=smooth_sigma)

        scale = float(self._meta['scale'])  # metres/pixel

        # Gradient in metres/metre
        dy, dx = np.gradient(elev, scale, scale)
        slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
        self.slope = np.degrees(slope_rad).astype(np.float32)

        logger.info(
            f"Slope range: [{self.slope.min():.1f}, {self.slope.max():.1f}]°, "
            f"mean={self.slope.mean():.2f}°"
        )
        return self.slope

    def regrid_to_dfsar(
        self,
        dfsar_lat: np.ndarray,
        dfsar_lon: np.ndarray,
        field: str = 'slope',
    ) -> np.ndarray:
        """
        Interpolate a LOLA field onto the DFSAR pixel grid.

        Uses nearest-neighbour for speed (both grids at ~20-25m, close enough).

        Args:
            dfsar_lat: (H_dfsar, W_dfsar) lat grid from DFSARMosaicLoader
            dfsar_lon: (H_dfsar, W_dfsar) lon grid
            field: 'slope' or 'elevation'

        Returns:
            (H_dfsar, W_dfsar) float32 array — NaN where LOLA has no coverage
        """
        if self.lat is None:
            self.get_coordinates()

        src = getattr(self, field)
        if src is None:
            if field == 'slope':
                self.compute_slope()
                src = self.slope
            else:
                src = self.elevation

        lola_lat = self.lat
        lola_lon = self.lon
        lola_scale = float(self._meta['scale'])
        lola_offset = float(self._meta['proj_offset'])

        # Convert DFSAR lat/lon → LOLA pixel coords (float32 throughout — avoids OOM on large grids)
        # Invert the UPS formula:
        #   rho = 2R * tan((90 + lat)/2 * pi/180)   [for south-pole, lat negative]
        lat_f32 = dfsar_lat.astype(np.float32)
        lon_f32 = dfsar_lon.astype(np.float32)
        lon_rad = np.radians(lon_f32)
        R_m = np.float32(MOON_RADIUS_M)

        # South-pole stereographic rho (float32)
        rho = np.float32(2.0) * R_m * np.tan(np.radians((np.float32(90.0) + lat_f32) * np.float32(0.5)))
        x_m = rho * np.sin(lon_rad)
        y_m = -rho * np.cos(lon_rad)
        del rho, lon_rad, lat_f32, lon_f32  # free intermediates immediately

        # → pixel coords
        col_f = x_m / lola_scale + lola_offset
        row_f = lola_offset - y_m / lola_scale
        del x_m, y_m

        col_i = np.round(col_f).astype(np.int32)
        row_i = np.round(row_f).astype(np.int32)
        del col_f, row_f

        h_lola, w_lola = src.shape
        out = np.full(dfsar_lat.shape, np.nan, dtype=np.float32)

        valid = (row_i >= 0) & (row_i < h_lola) & (col_i >= 0) & (col_i < w_lola)
        out[valid] = src[row_i[valid], col_i[valid]]

        coverage = valid.mean() * 100
        logger.info(
            f"LOLA {field} regridded to DFSAR grid: "
            f"{valid.sum():,} pixels ({coverage:.1f}% coverage)"
        )
        return out

    def plot_summary(self, save_path: Optional[str] = None):
        """Quick visualization of elevation + slope."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        if self.elevation is None:
            self.load()
        if self.slope is None:
            self.compute_slope()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
        fig.patch.set_facecolor('#0a0a1a')
        for ax in (ax1, ax2):
            ax.set_facecolor('#0a0a1a')
            ax.tick_params(colors='gray')

        im1 = ax1.imshow(self.elevation, cmap='terrain', origin='upper')
        plt.colorbar(im1, ax=ax1, label='Elevation (m)', shrink=0.8)
        ax1.set_title('LOLA DEM — Elevation', color='white')

        im2 = ax2.imshow(self.slope, cmap='RdYlGn_r', vmin=0, vmax=30, origin='upper')
        plt.colorbar(im2, ax=ax2, label='Slope (°)', shrink=0.8)
        ax2.contour(self.slope, levels=[15, 25], colors=['yellow', 'red'], linewidths=0.8)
        ax2.set_title('Terrain Slope (yellow=15°, red=25°)', color='white')

        meta = self._meta
        plt.suptitle(
            f'LOLA GDR South Pole DEM | {meta.get("scale", "?")}m/px | '
            f'Lat [{meta.get("lat_min")}°, {meta.get("lat_max")}°]',
            color='white', fontsize=12
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0a0a1a')
            logger.info(f"Saved DEM visualization to {save_path}")
        else:
            plt.show()
        plt.close()


def load_lola_slope_for_dfsar(
    lola_img: str,
    dfsar_lat: np.ndarray,
    dfsar_lon: np.ndarray,
    smooth_sigma: float = 2.0,
) -> np.ndarray:
    """
    Convenience: load LOLA DEM, compute slope, regrid to DFSAR pixel coordinates.

    Returns (H_dfsar, W_dfsar) slope array in degrees. NaN outside LOLA coverage.
    """
    dem = LOLADEMLoader(lola_img)
    dem.load()
    dem.compute_slope(smooth_sigma=smooth_sigma)
    dem.get_coordinates()
    return dem.regrid_to_dfsar(dfsar_lat, dfsar_lon, field='slope')
