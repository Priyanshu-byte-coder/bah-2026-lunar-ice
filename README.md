# LunarIceNet — Subsurface Ice Detection for Chandrayaan-2

> **Bharatiya Antariksh Hackathon 2026 | Problem Statement 8**  
> Detection and Characterization of Subsurface Ice in Lunar South Polar Regions  
> Using Chandrayaan-2 DFSAR Data for Landing Site and Rover Traverse Planning

---

## Overview

LunarIceNet is a physics-informed deep learning system that processes **Chandrayaan-2 DFSAR (Dual-frequency Synthetic Aperture Radar)** Level 3C mosaics to:

1. Detect subsurface water ice in the lunar south polar region
2. Score and rank optimal landing sites for future missions (Chandrayaan-4+)
3. Plan safe rover traverse paths to ice-rich targets
4. Estimate total ice volume and mass with uncertainty quantification

The system achieved **F1 = 0.843** on held-out DFSAR patches, processing 55 million valid radar pixels from the east-look Chandrayaan-2 mosaic.

---

## Why This Matters

The lunar south pole permanently shadowed regions (PSRs) are among the most strategically important locations in the solar system. Water ice confirmed there by Chandrayaan-1 (MINI-RF), LRO, and Chandrayaan-2 DFSAR can provide:

- Drinking water and oxygen for astronauts
- Hydrogen fuel for rockets (in-situ resource utilization)
- A scientific record of ancient solar system volatiles

**PRL 2026** (Sinha et al.) recently confirmed ice in four west-look DPSRs near Faustini crater using the exact same DFSAR dataset this system processes.

---

## Architecture: LunarIceNet

```
Input: DFSAR Level 3C Mosaic (CPR, SERD, T-Ratio channels)
          │
          ▼
┌─────────────────────────────────────────────────────────┐
│                   CNN Feature Extractor                  │
│   Conv2d → BatchNorm → ReLU × 4 levels (stride 2)       │
│   Skip connections (ResNet-style)                        │
└────────────────────┬────────────────────────────────────┘
                     │                    ▲
                     ▼                    │
┌─────────────────────────┐   ┌───────────────────────────┐
│  Physics Feature Branch │   │   Cross-Attention Fusion   │
│                         │◄──┤                           │
│  • Temperature prior    │   │  Radar ↔ Physics context  │
│  • Solar illumination   │   │  (4 heads, 2 layers)      │
│  • Crater depth proxy   │   │                           │
│  • Latitude encoding    │   └───────────────────────────┘
│  • Slope estimate       │
└─────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                    Decoder Head                          │
│   ConvTranspose2d × 3 → 1×1 Conv                        │
└────────────────┬──────────────────┬─────────────────────┘
                 │                  │
                 ▼                  ▼
          Ice Probability      Depth Estimate
          (per pixel, 0-1)     (metres below surface)
```

**Physics-Informed Loss:**
```
L = BCE(ice_prob, label)
  + 0.5 × L1(depth, depth_target)
  + 0.3 × physics_consistency_penalty
  + 0.2 × temperature_prior_penalty
```

---

## Five-Stage Pipeline

```
Stage 1 ── Ice Detection
           LunarIceNet inference on 12237×12794 DFSAR mosaic
           Outputs: ice_probability, depth_estimate, confidence maps

Stage 2 ── Landing Site Selection
           Composite scoring: ice_prob × slope_safety × illumination × confidence
           Real terrain slopes from NASA LRO LOLA GDR DEM (93.3% coverage)
           Top-10 candidates ranked and reported

Stage 3 ── Rover Traverse Planning
           A* pathfinding on 25m/pixel grid
           Cost: distance + slope_penalty + solar_penalty − ice_reward
           Hard constraints: slope < 25°, range < 8 km

Stage 4 ── Ice Volume Estimation
           Lichtenecker dielectric mixing model (CPR → ice fraction)
           Monte Carlo uncertainty (1000 samples)
           Per-DPSR crater breakdown

Stage 5 ── Visualization & Mission Report
           Multi-panel ice analysis maps
           Polar stereographic coverage plots
           Mission planning report (text + PNG)
```

---

## Key Results (East Look Direction)

| Metric | Value |
|--------|-------|
| Model F1 | 0.8428 |
| Precision | 0.8906 |
| Recall | 0.7999 |
| Valid DFSAR pixels | 54,903,877 (35.1%) |
| LOLA terrain coverage | 93.3% of DFSAR grid |
| Training patches | 34,840 |
| Model parameters | 12.4 million |

**Top Landing Site:** lat = −86.035°, lon = −28.650°
- Ice probability: 0.946 | Terrain slope: 3.5° | Score: 0.728

**Ice Volume Estimate:**
- Total: 8,423,807 m³
- Mass: 7,725 million tonnes
- Mean depth: 1.05 m
- 90% CI: [3.7M – 14.6M] m³

> Note: East-look DPSRs show minimal ice signal — this is scientifically expected.
> West-look direction (where PRL 2026 confirmed ice in Faustini) is the primary science target.

---

## Data Sources

| Dataset | Source | Description |
|---------|--------|-------------|
| Chandrayaan-2 DFSAR L3C | ISRO PRADAN portal | East + West look CPR, SERD, T-Ratio mosaics |
| LRO LOLA GDR ldem_85s_40m | NASA PDS | South pole DEM, −90° to −85°, 40 m/px |
| LRO LOLA GDR ldem_875s_20m | NASA PDS | Inner pole DEM, −90° to −87.5°, 20 m/px |
| Lunar PSR catalog | Hayne et al. 2015 | Known permanently shadowed regions |

```
data/raw/
├── mpcpsp_east/          # DFSAR east-look Level 3C tiles
│   ├── cpr_mosaic.npy
│   ├── serd_mosaic.npy
│   └── tratio_mosaic.npy
├── mpcpsp_west/          # DFSAR west-look Level 3C tiles
│   └── ...
├── my4rsp_east/          # Alternative east mosaic (Level 4)
├── my4rsp_west/          # Alternative west mosaic (Level 4)
└── lola_dem/
    ├── ldem_85s_40m.img  # 7584×7584 binary DEM (~110 MB)
    └── ldem_875s_20m.img # 7584×7584 high-res inner pole (~110 MB)
```

---

## Installation

```bash
git clone https://github.com/Priyanshu-byte-coder/bah-2026-lunar-ice.git
cd bah-2026-lunar-ice

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

**GPU strongly recommended** for training. Inference runs on CPU but takes ~2 hr for full mosaic.

---

## Usage

### Train on real DFSAR data
```bash
# East direction (smaller mosaic, recommended for initial runs)
python train_real.py --direction east --epochs 30 --batch-size 16

# West direction (larger, needs GPU with >=8 GB VRAM + AMP)
python train_real.py --direction west --epochs 30 --batch-size 8 --subsample 2

# Analyze data statistics only (no training)
python train_real.py --analyze-only --direction east
```

### Run full pipeline
```bash
# East direction — full run including inference
python full_pipeline.py \
    --direction east \
    --checkpoint checkpoints/east_best_model.pth \
    --lola-dem data/raw/lola_dem/ldem_85s_40m.img

# Re-run stages 2-5 only (skip slow CNN inference using cached .npy)
python full_pipeline.py \
    --use-cached \
    --direction east \
    --checkpoint checkpoints/east_best_model.pth \
    --lola-dem data/raw/lola_dem/ldem_85s_40m.img

# West direction (PRL 2026 confirmed ice region)
python full_pipeline.py \
    --direction west \
    --checkpoint checkpoints/east_best_model.pth \
    --lola-dem data/raw/lola_dem/ldem_85s_40m.img \
    --output-dir outputs/west
```

---

## Project Structure

```
bah-2026-lunar-ice/
├── full_pipeline.py          # End-to-end 5-stage pipeline
├── train_real.py             # Training on real DFSAR mosaics
├── predict_real.py           # Standalone inference script
├── main.py                   # Entry point / demo
│
├── src/
│   ├── data/
│   │   ├── real_loader.py    # DFSARMosaicLoader — reads Level 3C .npy mosaics
│   │   ├── lola_loader.py    # LOLADEMLoader — reads PDS3 binary DEM
│   │   ├── dataset.py        # LunarPSRCatalog, patch datasets, augmentation
│   │   └── features.py       # Polarimetric feature extraction
│   ├── models/
│   │   ├── lunaricenet.py    # LunarIceNet architecture + PhysicsInformedLoss
│   │   └── trainer.py        # Training utilities, Metrics
│   ├── analysis/
│   │   └── ice_volume.py     # IceVolumeEstimator (Lichtenecker + Monte Carlo)
│   ├── planning/
│   │   └── rover_traverse.py # RoverTraversePlanner (A* on slope/ice grid)
│   ├── visualization/
│   │   └── maps.py           # Polar maps, ice overlays
│   └── utils/
│       └── geo.py            # UPS projection utilities
│
├── configs/
│   └── config.yaml           # Hyperparameters and data paths
├── checkpoints/
│   ├── east_best_model.pth   # Best east-direction model (F1=0.843)
│   └── training_history.json # Loss/F1 training curves
├── outputs/
│   ├── mission_report_east.txt   # Full mission planning report
│   ├── landing_sites_east.txt    # Ranked landing site candidates
│   ├── ice_volume_east.txt       # Ice volume estimation report
│   ├── traverse_report_east.txt  # Rover traverse safety analysis
│   ├── ice_analysis_east.png     # Multi-panel ice detection map
│   ├── polar_ice_map_east.png    # Polar projection ice coverage
│   ├── rover_traverse_east.png   # Traverse path visualization
│   └── lola_dem_summary.png      # LOLA DEM elevation + slope
├── docs/
│   └── proposal/                 # BAH 2026 idea proposal document
├── notebooks/                    # Exploration and analysis notebooks
├── tests/                        # Unit tests
└── requirements.txt
```

---

## Technical Details

### Polarimetric Features

| Feature | Formula | Ice Indicator |
|---------|---------|---------------|
| CPR (Circular Polarization Ratio) | SC / OC backscatter | > 1.0 = volumetric scattering (ice) |
| SERD (Single Expected Return Deviation) | Std of multi-look | Low = homogeneous subsurface |
| T-Ratio | T11 / (T22 + T33) | < 0.3 = multiple volume reflections |

### Ice Detection Physics
- CPR > 1.0 — volumetric scattering diagnostic for ice
- CPR > 1.5 — strong ice candidate
- Combined with SERD and T-Ratio via LunarIceNet cross-attention for probabilistic output

### Depth Estimation (Lichtenecker Model)
```
ε_eff^0.5 = f_ice × ε_ice^0.5 + (1 − f_ice) × ε_regolith^0.5
penetration_depth = λ / (4π × Im(sqrt(ε_eff)))
```
Parameters: λ = 0.24 m (L-band), ε_regolith = 3.0, ε_ice = 3.15

### LOLA DEM Integration
- Primary: `ldem_85s_40m` — covers −90° to −85°, gives 93.3% DFSAR coverage
- Secondary: `ldem_875s_20m` — covers −90° to −87.5°, fills inner pole at 20 m/px (higher res)
- Reprojected via UPS (Universal Polar Stereographic) nearest-neighbour to DFSAR 25 m grid
- Slope: finite-difference gradient with Gaussian smoothing (σ = 1.5 px)

### Memory Optimisation
The full DFSAR east mosaic is 12237×12794 pixels. Each float32 array = 600 MB.

- `mmap_mode='r'` for cached `.npy` predictions (avoids loading 1.8 GB into RAM)
- Chunked coordinate computation (1024-row chunks, ~50 MB peak vs 2.4 GB for full meshgrid)
- All arrays kept in float32 (never float64)
- Explicit `del` of large intermediates before next allocation
- Mixed precision training (AMP/fp16) halves VRAM usage during training

---

## Limitations and Future Work

- **East direction**: Ice signal absent in east-look DPSRs (expected from polarimetry theory). West direction is the key science run.
- **No ground truth**: Labels derived from CPR > 1.0 threshold — same data used for training and labeling creates optimistic metrics.
- **Depth accuracy**: Lichtenecker model assumes homogeneous ice distribution; real subsurface is layered.
- **West training**: West mosaic (24794×24181 px, 340M pixels) causes CUDA OOM during training on consumer GPUs; east model used for west inference.

**Next steps for finale:**
1. Run west inference → expect CPR > 1 in Faustini DPSR (PRL 2026 confirmation site)
2. Multi-temporal analysis (if future DFSAR passes available)
3. Integration with Chandrayaan-2 IIRS thermal data for temperature-constrained ice fraction
4. Interactive web dashboard (Streamlit) for judge demonstration

---

## References

1. Sinha, R.K. et al. (2026). "Subsurface ice detection in DPSRs of lunar south pole using Chandrayaan-2 DFSAR." *PRL Technical Report.*
2. Hayne, P.O. et al. (2015). "Evidence for exposed water ice in the Moon's south polar regions from Lunar Reconnaissance Orbiter ultraviolet albedo and temperature measurements." *Icarus*, 255, 58–69.
3. Spudis, P.D. et al. (2013). "Evidence for water ice on the Moon: Results for anomalous polar craters from the LRO Mini-RF imaging radar." *JGR Planets*, 118(10), 2016–2029.
4. Nozette, S. et al. (2010). "The Lunar Reconnaissance Orbiter Miniature Radio Frequency (Mini-RF) Technology Demonstration." *Space Sci Rev*, 150, 285–302.
5. Smith, D.E. et al. (2010). "The Lunar Orbiter Laser Altimeter investigation on the Lunar Reconnaissance Orbiter Mission." *Space Sci Rev*, 150, 209–241.
6. ISRO. (2019). "Chandrayaan-2 DFSAR: Instrument Description and Data Products." *ISSDC Technical Note.*

---

## License

MIT License — open source for the Indian space science community.

---

*Built for Bharatiya Antariksh Hackathon 2026 by Team LunarIceNet*
*ISRO PS-8 | Chandrayaan-2 DFSAR | Physics-Informed Deep Learning*
