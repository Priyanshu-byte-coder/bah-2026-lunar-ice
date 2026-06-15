"""
Dataset classes for Chandrayaan-2 DFSAR data loading and preprocessing.

Handles:
    - Loading raw DFSAR SAR data (SLC products)
    - Patch extraction with configurable size/stride
    - Physical feature integration (DEM, temperature, illumination)
    - Pseudo-label generation from PSR catalogs
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml
import logging

logger = logging.getLogger(__name__)


class LunarPSRCatalog:
    """
    Permanently Shadowed Region (PSR) catalog for pseudo-label generation.

    Uses known PSR locations as proxy labels for potential ice deposits.
    Regions with CPR > 1 AND DOP < 0.13 inside PSRs are labeled as ice-positive.
    """

    # Known PSR craters in south polar region with suspected ice
    # Source: Compiled from LRO Diviner + Chandrayaan-2 studies
    KNOWN_PSRS = [
        {"name": "Faustini", "lat": -87.3, "lon": 77.0, "radius_km": 19.5, "ice_confidence": "high"},
        {"name": "Shoemaker", "lat": -88.1, "lon": 44.9, "radius_km": 25.5, "ice_confidence": "high"},
        {"name": "Haworth", "lat": -87.4, "lon": -2.2, "radius_km": 28.5, "ice_confidence": "medium"},
        {"name": "Shackleton", "lat": -89.9, "lon": 0.0, "radius_km": 10.5, "ice_confidence": "high"},
        {"name": "Sverdrup", "lat": -88.5, "lon": -145.0, "radius_km": 16.5, "ice_confidence": "medium"},
        {"name": "de Gerlache", "lat": -88.5, "lon": -87.1, "radius_km": 16.0, "ice_confidence": "medium"},
        {"name": "Cabeus", "lat": -85.3, "lon": -35.7, "radius_km": 49.0, "ice_confidence": "confirmed"},
        {"name": "Amundsen", "lat": -84.5, "lon": -82.8, "radius_km": 51.0, "ice_confidence": "low"},
        {"name": "Nobile", "lat": -85.3, "lon": 53.5, "radius_km": 37.0, "ice_confidence": "medium"},
    ]

    def __init__(self):
        self.psrs = self.KNOWN_PSRS

    def get_ice_probability_at(self, lat: float, lon: float) -> float:
        """Get prior ice probability based on proximity to known PSRs."""
        confidence_map = {"confirmed": 0.9, "high": 0.7, "medium": 0.4, "low": 0.2}

        max_prob = 0.0
        for psr in self.psrs:
            dist_km = self._haversine_lunar(lat, lon, psr["lat"], psr["lon"])
            if dist_km < psr["radius_km"]:
                prob = confidence_map.get(psr["ice_confidence"], 0.1)
                max_prob = max(max_prob, prob)
            elif dist_km < psr["radius_km"] * 2:
                # Gradual falloff outside crater
                falloff = 1.0 - (dist_km - psr["radius_km"]) / psr["radius_km"]
                prob = confidence_map.get(psr["ice_confidence"], 0.1) * falloff * 0.3
                max_prob = max(max_prob, prob)

        return max_prob

    @staticmethod
    def _haversine_lunar(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Haversine distance on Moon (R = 1737.4 km)."""
        R = 1737.4
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = (np.sin(dlat / 2) ** 2 +
             np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
             np.sin(dlon / 2) ** 2)
        return 2 * R * np.arcsin(np.sqrt(a))


class DFSARDataset(Dataset):
    """
    PyTorch Dataset for Chandrayaan-2 DFSAR polarimetric data.

    Loads preprocessed feature patches and corresponding labels.
    """

    def __init__(
        self,
        data_dir: str,
        config_path: str = "configs/config.yaml",
        split: str = "train",
        transform: Optional[object] = None,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.patch_size = self.config['data']['preprocessing']['patch_size']
        self.psr_catalog = LunarPSRCatalog()

        self.patches = self._load_patches()
        logger.info(f"Loaded {len(self.patches)} patches for {split} split")

    def _load_patches(self) -> List[Dict]:
        """Load preprocessed feature patches from disk."""
        patch_dir = self.data_dir / "processed" / self.split
        patches = []

        if not patch_dir.exists():
            logger.warning(f"Patch directory not found: {patch_dir}. Using synthetic data.")
            return self._generate_synthetic_patches()

        for patch_file in sorted(patch_dir.glob("*.npz")):
            data = np.load(patch_file, allow_pickle=True)
            patches.append({
                'features': data['features'],
                'label': data['label'],
                'lat': float(data.get('lat', -85.0)),
                'lon': float(data.get('lon', 0.0)),
                'metadata': data.get('metadata', {}),
            })

        return patches

    def _generate_synthetic_patches(self, n_patches: int = 500) -> List[Dict]:
        """
        Generate synthetic training patches for development/testing.

        Simulates DFSAR-like polarimetric features with known ice signatures:
        - Ice regions: High CPR (>1), Low DOP (<0.13), high volume scattering
        - Non-ice regions: Low CPR (<1), Higher DOP, surface scattering dominant
        """
        patches = []
        ps = self.patch_size
        n_features = 16  # Number of polarimetric features

        for i in range(n_patches):
            has_ice = np.random.random() < 0.3  # 30% ice patches

            features = np.random.randn(n_features, ps, ps).astype(np.float32) * 0.1

            if has_ice:
                # Simulate ice-like signatures
                ice_mask = self._random_ice_region(ps)

                # CPR > 1 in ice regions (feature index 0)
                features[0] = np.where(ice_mask, 1.2 + np.random.randn(ps, ps) * 0.3,
                                       0.5 + np.random.randn(ps, ps) * 0.2)
                # DOP < 0.13 in ice regions (feature index 1)
                features[1] = np.where(ice_mask, 0.08 + np.random.randn(ps, ps) * 0.03,
                                       0.4 + np.random.randn(ps, ps) * 0.1)
                # High entropy in ice regions (feature index 2)
                features[2] = np.where(ice_mask, 0.8 + np.random.randn(ps, ps) * 0.1,
                                       0.3 + np.random.randn(ps, ps) * 0.1)
                # High volume scattering (feature index 3)
                features[3] = np.where(ice_mask, 0.7 + np.random.randn(ps, ps) * 0.1,
                                       0.2 + np.random.randn(ps, ps) * 0.1)

                label = ice_mask.astype(np.float32)
            else:
                features[0] = 0.5 + np.random.randn(ps, ps) * 0.2  # Low CPR
                features[1] = 0.4 + np.random.randn(ps, ps) * 0.1  # Higher DOP
                label = np.zeros((ps, ps), dtype=np.float32)

            # Clip to valid ranges
            features = np.clip(features, -2.0, 5.0)

            lat = np.random.uniform(-90, -80)
            lon = np.random.uniform(-180, 180)

            patches.append({
                'features': features,
                'label': label,
                'lat': lat,
                'lon': lon,
                'metadata': {'synthetic': True, 'has_ice': has_ice},
            })

        return patches

    @staticmethod
    def _random_ice_region(size: int) -> np.ndarray:
        """Generate random blob-shaped ice region mask."""
        from scipy.ndimage import gaussian_filter
        mask = np.zeros((size, size), dtype=np.float32)

        # Random blob center
        cx, cy = np.random.randint(size // 4, 3 * size // 4, size=2)
        r = np.random.randint(size // 8, size // 3)

        y, x = np.ogrid[:size, :size]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        mask[dist < r] = 1.0

        # Smooth edges
        mask = gaussian_filter(mask, sigma=r / 4)
        mask = (mask > 0.3).astype(np.float32)

        return mask

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        patch = self.patches[idx]

        features = torch.from_numpy(patch['features'])
        label = torch.from_numpy(patch['label'])

        # Physical features
        psr_prob = self.psr_catalog.get_ice_probability_at(patch['lat'], patch['lon'])
        physical = torch.tensor([
            patch['lat'],
            patch['lon'],
            psr_prob,
            np.abs(patch['lat']),          # Distance from pole
            1.0 if psr_prob > 0.3 else 0.0  # Inside PSR flag
        ], dtype=torch.float32)

        if self.transform:
            features = self.transform(features)

        return {
            'features': features,
            'physical': physical,
            'label': label,
            'lat': torch.tensor(patch['lat']),
            'lon': torch.tensor(patch['lon']),
        }


class DFSARDataModule:
    """Data module managing train/val/test splits and dataloaders."""

    def __init__(
        self,
        data_dir: str = "data",
        config_path: str = "configs/config.yaml",
        batch_size: int = 16,
        num_workers: int = 4,
    ):
        self.data_dir = data_dir
        self.config_path = config_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self):
        self.train_dataset = DFSARDataset(self.data_dir, self.config_path, split="train")
        self.val_dataset = DFSARDataset(self.data_dir, self.config_path, split="val")
        self.test_dataset = DFSARDataset(self.data_dir, self.config_path, split="test")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
