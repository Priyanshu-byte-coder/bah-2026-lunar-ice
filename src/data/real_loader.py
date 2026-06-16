"""
Real DFSAR Data Loader for Level 3C/L4 Mosaic Products.

Loads pre-computed polarimetric mosaics from Chandrayaan-2 DFSAR:
    - CPR (Circular Polarization Ratio) — primary ice indicator
    - SERD (Single-bounce Eigenvalue Relative Difference) — roughness/scattering
    - T-Ratio — polarimetric scattering type ratio

Data is in GeoTIFF format, Moon South Pole Stereographic projection,
25m/pixel resolution.
"""

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

# Mapping from file tag to feature name
FEATURE_TAGS = {
    'cpr': 'CPR',        # Circular Polarization Ratio
    'srd': 'SERD',       # Single-bounce Eigenvalue Relative Difference
    'trt': 'T-Ratio',    # T-Ratio
}


class DFSARMosaicLoader:
    """
    Loads and merges DFSAR Level 3C/L4 mosaic products.

    Handles east/west look direction mosaics, NaN masking,
    and coordinate extraction from GeoTIFF metadata.
    """

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.features = {}
        self.crs = None
        self.transform = None
        self.shape = None
        self.valid_mask = None

    def discover_datasets(self) -> Dict[str, List[Path]]:
        """Find all DFSAR GeoTIFF files organized by feature type."""
        datasets = {}
        for tif in sorted(self.data_dir.rglob("*.tif")):
            name = tif.stem
            for tag, feature_name in FEATURE_TAGS.items():
                if f'_d_{tag}_' in name:
                    if feature_name not in datasets:
                        datasets[feature_name] = []
                    datasets[feature_name].append(tif)
        return datasets

    def load_single_tif(self, path: Path) -> Tuple[np.ndarray, dict]:
        """Load single GeoTIFF and return data + metadata."""
        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float32)
            meta = {
                'crs': src.crs,
                'transform': src.transform,
                'shape': src.shape,
                'bounds': src.bounds,
                'nodata': src.nodata,
            }
        return data, meta

    def load_and_merge_feature(self, paths: List[Path]) -> Tuple[np.ndarray, dict]:
        """
        Load and merge multiple GeoTIFFs for same feature (east+west).

        Uses rasterio merge to combine overlapping mosaics.
        """
        if len(paths) == 1:
            return self.load_single_tif(paths[0])

        datasets = []
        for p in paths:
            datasets.append(rasterio.open(p))

        merged, merged_transform = merge(datasets, method='max')
        merged = merged[0].astype(np.float32)  # Single band

        meta = {
            'crs': datasets[0].crs,
            'transform': merged_transform,
            'shape': merged.shape,
            'bounds': datasets[0].bounds,
        }

        for ds in datasets:
            ds.close()

        return merged, meta

    def load_all_features(self, merge_directions: bool = False) -> Dict[str, np.ndarray]:
        """
        Load all available features.

        Args:
            merge_directions: If True, merge east+west into single mosaic.
                            If False, load each direction separately.
        Returns:
            Dict of feature_name -> numpy array
        """
        datasets = self.discover_datasets()
        logger.info(f"Found features: {list(datasets.keys())}")
        for name, paths in datasets.items():
            logger.info(f"  {name}: {len(paths)} files")

        if merge_directions:
            for name, paths in datasets.items():
                logger.info(f"Loading + merging {name}...")
                data, meta = self.load_and_merge_feature(paths)
                self.features[name] = data
                if self.crs is None:
                    self.crs = meta['crs']
                    self.transform = meta['transform']
                    self.shape = meta['shape']
        else:
            # Load separately with direction suffix
            for name, paths in datasets.items():
                for path in paths:
                    stem = path.stem
                    direction = 'east' if 'east' in stem else 'west'
                    key = f"{name}_{direction}"
                    logger.info(f"Loading {key}...")
                    data, meta = self.load_single_tif(path)
                    self.features[key] = data
                    if self.crs is None:
                        self.crs = meta['crs']
                        self.transform = meta['transform']

        # Build valid mask (where ALL features have data)
        masks = []
        for data in self.features.values():
            masks.append(~np.isnan(data) & (data != 0) if 'CPR' in list(self.features.keys())[0] else ~np.isnan(data))
        self.valid_mask = np.all(masks, axis=0) if masks else None

        return self.features

    def load_single_direction(
        self, direction: str = "west", subsample: int = 1
    ) -> Dict[str, np.ndarray]:
        """
        Load CPR/SERD/T-Ratio for one look direction.
        West recommended (larger coverage, 340M pixels).

        Args:
            subsample: Take every Nth pixel (1=full res, 2=half res, 4=quarter res).
                       Use 2 for west to avoid OOM (12397x12090 instead of 24794x24181).
        """
        prefix = f"mpcpsp{direction}"
        features = {}

        for tag, name in FEATURE_TAGS.items():
            pattern = f"*{prefix}*_d_{tag}_*.tif"
            matches = list(self.data_dir.rglob(pattern))
            if matches:
                data, meta = self.load_single_tif(matches[0])
                if subsample > 1:
                    data = data[::subsample, ::subsample]
                    # Update transform: pixel size × subsample
                    from rasterio.transform import Affine
                    t = meta['transform']
                    meta['transform'] = Affine(
                        t.a * subsample, t.b, t.c,
                        t.d, t.e * subsample, t.f,
                    )
                    meta['shape'] = data.shape
                features[name] = data
                if self.crs is None:
                    self.crs = meta['crs']
                    self.transform = meta['transform']
                    self.shape = meta['shape']
                logger.info(
                    f"  {name}: {data.shape} (subsample={subsample}x), "
                    f"valid={np.sum(~np.isnan(data)):,}"
                )

        # Valid mask
        masks = [~np.isnan(d) for d in features.values()]
        self.valid_mask = np.all(masks, axis=0)
        self.features = features

        valid_count = np.sum(self.valid_mask)
        logger.info(f"Total valid pixels: {valid_count:,} ({100*valid_count/self.valid_mask.size:.1f}%)")

        return features

    def get_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get lat/lon grids from CRS transform.

        Data is in Moon South Pole Stereographic (UPS) projection.
        Convert projected coords (meters) to selenographic lat/lon.
        """
        if self.transform is None or self.shape is None:
            raise ValueError("Load data first")

        h, w = self.shape
        R = np.float32(1737400.0)
        inv_2R = np.float32(1.0 / (2.0 * 1737400.0))

        # 1-D projected coordinate vectors (tiny: h+w floats)
        x_1d = (np.float32(self.transform.c)
                + np.arange(w, dtype=np.float32) * np.float32(self.transform.a))
        y_1d = (np.float32(self.transform.f)
                + np.arange(h, dtype=np.float32) * np.float32(self.transform.e))

        # Pre-allocate outputs (1.2 GB total for east mosaic, unavoidable)
        lat = np.empty((h, w), dtype=np.float32)
        lon = np.empty((h, w), dtype=np.float32)

        # Chunked row computation — avoids full 2-D x, y meshes in RAM
        # Peak per chunk ≈ CHUNK × w × 4 bytes = 50 MB @ CHUNK=1024
        CHUNK = 1024
        for r0 in range(0, h, CHUNK):
            r1 = min(r0 + CHUNK, h)
            y_chunk = y_1d[r0:r1, np.newaxis]   # (chunk, 1) — broadcast over cols
            x_row   = x_1d[np.newaxis, :]        # (1, w)   — view, no copy

            # rho = sqrt(x² + y²)  hypot avoids separate x², y² temps
            rho = np.hypot(x_row, y_chunk)        # (chunk, w)

            lat[r0:r1] = -(np.float32(90.0)
                           - np.float32(2.0) * np.degrees(np.arctan(rho * inv_2R)))
            lon[r0:r1] = np.degrees(np.arctan2(x_row, -y_chunk))

        return lat, lon

    def get_ice_candidates(self, cpr_threshold: float = 1.0) -> np.ndarray:
        """Get binary mask of ice candidate pixels (CPR > threshold)."""
        if 'CPR' not in self.features:
            raise ValueError("Load CPR data first")
        cpr = self.features['CPR']
        mask = (~np.isnan(cpr)) & (cpr > cpr_threshold)
        return mask


class RealDFSARDataset(Dataset):
    """
    PyTorch Dataset for real DFSAR mosaic data.

    Extracts patches from pre-computed feature mosaics
    with coordinate-based physical feature encoding.
    """

    def __init__(
        self,
        features: Dict[str, np.ndarray],
        valid_mask: np.ndarray,
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
        patch_size: int = 64,
        stride: int = 32,
        ice_enrichment: float = 0.5,
        raw_features: Optional[Dict[str, np.ndarray]] = None,
    ):
        """
        Args:
            features: Dict of feature_name -> (H,W) arrays (normalized for model input)
            valid_mask: (H,W) boolean mask of valid pixels
            lat_grid, lon_grid: Coordinate grids
            patch_size: Patch size in pixels
            stride: Stride between patches
            ice_enrichment: Fraction of patches that should contain CPR>1 pixels
            raw_features: Un-normalized features for label generation (CPR > 1.0 threshold)
        """
        self.features = features
        self.raw_features = raw_features if raw_features is not None else features
        self.valid_mask = valid_mask
        self.lat_grid = lat_grid
        self.lon_grid = lon_grid
        self.patch_size = patch_size
        self.feature_names = sorted(features.keys())
        self.n_features = len(self.feature_names)

        # Build patch index
        self.patches = self._build_patch_index(stride, ice_enrichment)
        logger.info(f"Built {len(self.patches)} patches ({patch_size}x{patch_size}, stride={stride})")

    def _build_patch_index(
        self, stride: int, ice_enrichment: float
    ) -> List[Tuple[int, int]]:
        """Build list of valid patch top-left coordinates."""
        h, w = self.valid_mask.shape
        ps = self.patch_size
        patches = []
        ice_patches = []

        cpr = self.raw_features.get('CPR')

        for y in range(0, h - ps, stride):
            for x in range(0, w - ps, stride):
                patch_mask = self.valid_mask[y:y+ps, x:x+ps]
                valid_frac = patch_mask.mean()

                # Skip patches with < 50% valid data
                if valid_frac < 0.5:
                    continue

                # Check for ice candidates
                has_ice = False
                if cpr is not None:
                    cpr_patch = cpr[y:y+ps, x:x+ps]
                    ice_pixels = np.nansum(cpr_patch > 1.0)
                    has_ice = ice_pixels > 5  # At least 5 pixels with CPR > 1

                if has_ice:
                    ice_patches.append((y, x))
                else:
                    patches.append((y, x))

        # Enrich with ice patches (oversample)
        repeats = 0
        if ice_patches and ice_enrichment > 0:
            n_target_ice = int(len(patches) * ice_enrichment / (1 - ice_enrichment))
            repeats = max(1, n_target_ice // max(len(ice_patches), 1))
            ice_repeated = ice_patches * repeats
            all_patches = patches + ice_repeated[:n_target_ice]
        else:
            all_patches = patches + ice_patches

        logger.info(f"  Normal patches: {len(patches)}, Ice patches: {len(ice_patches)} (repeated {repeats}x)")
        return all_patches

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        y, x = self.patches[idx]
        ps = self.patch_size

        # Stack feature patches
        feature_stack = np.zeros((self.n_features, ps, ps), dtype=np.float32)
        for i, name in enumerate(self.feature_names):
            patch = self.features[name][y:y+ps, x:x+ps].copy()
            patch = np.nan_to_num(patch, nan=0.0)
            feature_stack[i] = patch

        # Coordinates
        lat = float(np.nanmean(self.lat_grid[y:y+ps, x:x+ps]))
        lon = float(np.nanmean(self.lon_grid[y:y+ps, x:x+ps]))

        # Generate pseudo-label from RAW CPR threshold (before normalization)
        raw_cpr = self.raw_features.get('CPR')
        if raw_cpr is not None:
            raw_cpr_patch = raw_cpr[y:y+ps, x:x+ps].copy()
            raw_cpr_patch = np.nan_to_num(raw_cpr_patch, nan=0.0)
            label = (raw_cpr_patch > 1.0).astype(np.float32)
        else:
            cpr_idx = self.feature_names.index('CPR') if 'CPR' in self.feature_names else 0
            label = (feature_stack[cpr_idx] > 1.0).astype(np.float32)

        # Physical features
        psr_prob = self._estimate_psr_probability(lat, lon)
        physical = np.array([
            lat, lon, psr_prob,
            abs(lat),
            1.0 if psr_prob > 0.3 else 0.0,
        ], dtype=np.float32)

        return {
            'features': torch.from_numpy(feature_stack),
            'physical': torch.from_numpy(physical),
            'label': torch.from_numpy(label),
            'lat': torch.tensor(lat),
            'lon': torch.tensor(lon),
        }

    @staticmethod
    def _estimate_psr_probability(lat: float, lon: float) -> float:
        """Quick PSR probability based on known crater locations."""
        from src.data.dataset import LunarPSRCatalog
        catalog = LunarPSRCatalog()
        return catalog.get_ice_probability_at(lat, lon)


def load_real_data(
    data_dir: str = "data/raw",
    direction: str = "east",
    patch_size: int = 64,
    stride: int = 32,
    batch_size: int = 16,
) -> Tuple[DataLoader, DFSARMosaicLoader]:
    """
    Convenience function to load real DFSAR data and create dataloader.

    Args:
        data_dir: Path to raw data directory
        direction: 'east' or 'west' (west has more coverage)
        patch_size: Size of extracted patches
        stride: Stride between patches
        batch_size: Batch size for dataloader

    Returns:
        (DataLoader, DFSARMosaicLoader)
    """
    loader = DFSARMosaicLoader(data_dir)
    features = loader.load_single_direction(direction)
    lat_grid, lon_grid = loader.get_coordinates()

    dataset = RealDFSARDataset(
        features=features,
        valid_mask=loader.valid_mask,
        lat_grid=lat_grid,
        lon_grid=lon_grid,
        patch_size=patch_size,
        stride=stride,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    return dataloader, loader


def quick_analysis(data_dir: str = "data/raw", direction: str = "east"):
    """
    Run quick analysis on real DFSAR data.
    Prints statistics and generates initial visualizations.
    """
    loader = DFSARMosaicLoader(data_dir)
    features = loader.load_single_direction(direction)

    print("\n" + "=" * 60)
    print("DFSAR DATA ANALYSIS — Lunar South Pole")
    print("=" * 60)

    for name, data in features.items():
        valid = data[~np.isnan(data)]
        print(f"\n{name}:")
        print(f"  Shape: {data.shape}")
        print(f"  Valid pixels: {len(valid):,} ({100*len(valid)/data.size:.1f}%)")
        print(f"  Range: [{np.min(valid):.4f}, {np.max(valid):.4f}]")
        print(f"  Mean: {np.mean(valid):.4f}")
        print(f"  Std: {np.std(valid):.4f}")
        if name == 'CPR':
            print(f"  CPR > 1.0: {(valid > 1.0).sum():,} pixels ({100*(valid > 1.0).mean():.3f}%)")
            print(f"  CPR > 1.5: {(valid > 1.5).sum():,} pixels ({100*(valid > 1.5).mean():.4f}%)")

    # Ice candidate analysis
    ice_mask = loader.get_ice_candidates(cpr_threshold=1.0)
    lat_grid, lon_grid = loader.get_coordinates()

    ice_lats = lat_grid[ice_mask]
    ice_lons = lon_grid[ice_mask]

    print(f"\n{'='*60}")
    print(f"ICE CANDIDATES (CPR > 1.0)")
    print(f"  Total: {ice_mask.sum():,} pixels")
    print(f"  Latitude range: [{np.min(ice_lats):.2f}, {np.max(ice_lats):.2f}]")
    print(f"  Mean latitude: {np.mean(ice_lats):.2f}")

    # Cross-reference with known PSRs
    serd = features.get('SERD')
    if serd is not None:
        serd_at_ice = serd[ice_mask]
        serd_valid = serd_at_ice[~np.isnan(serd_at_ice)]
        print(f"\n  SERD at ice candidates:")
        print(f"    Mean: {np.mean(serd_valid):.4f}")
        print(f"    Low SERD (<0.5): {(serd_valid < 0.5).sum():,} ({100*(serd_valid < 0.5).mean():.1f}%)")

    print(f"\n{'='*60}")

    return features, loader
