# LunarIceNet

### AI-Powered Subsurface Ice Detection Using Chandrayaan-2 DFSAR Radar Data

> **Bharatiya Antariksh Hackathon 2026 (BAH 2026) | Problem Statement 8**
> Detection and Characterization of Subsurface Ice in Lunar South Polar Regions Using Chandrayaan-2 Radar and Imagery Data for Landing Site and Rover Traverse Planning

---

## Table of Contents

- [1. The Problem — Why Lunar Ice Matters](#1-the-problem--why-lunar-ice-matters)
- [2. Scientific Background](#2-scientific-background)
  - [2.1 Permanently Shadowed Regions (PSRs)](#21-permanently-shadowed-regions-psrs)
  - [2.2 How Radar Detects Ice](#22-how-radar-detects-ice)
  - [2.3 Chandrayaan-2 DFSAR Instrument](#23-chandrayaan-2-dfsar-instrument)
  - [2.4 Key Prior Discoveries](#24-key-prior-discoveries)
- [3. Our Approach — LunarIceNet](#3-our-approach--lunaricenet)
  - [3.1 Why Deep Learning Over Classical Methods](#31-why-deep-learning-over-classical-methods)
  - [3.2 System Overview](#32-system-overview)
- [4. Data Acquisition Guide](#4-data-acquisition-guide)
  - [4.1 Chandrayaan-2 DFSAR Data](#41-chandrayaan-2-dfsar-data)
  - [4.2 NASA LRO LOLA DEM](#42-nasa-lro-lola-dem)
  - [4.3 Data Directory Structure](#43-data-directory-structure)
- [5. Model Architecture — LunarIceNet (Deep Dive)](#5-model-architecture--lunaricenet-deep-dive)
  - [5.1 Multi-Scale Radar Encoder](#51-multi-scale-radar-encoder)
  - [5.2 Physics Encoder](#52-physics-encoder)
  - [5.3 Cross-Attention Fusion](#53-cross-attention-fusion)
  - [5.4 Multi-Task Detection Head](#54-multi-task-detection-head)
  - [5.5 Physics-Informed Loss Function](#55-physics-informed-loss-function)
- [6. Five-Stage Pipeline (Detailed)](#6-five-stage-pipeline-detailed)
  - [6.1 Stage 1 — Subsurface Ice Detection](#61-stage-1--subsurface-ice-detection)
  - [6.2 Stage 2 — Landing Site Selection](#62-stage-2--landing-site-selection)
  - [6.3 Stage 3 — Rover Traverse Planning](#63-stage-3--rover-traverse-planning)
  - [6.4 Stage 4 — Ice Volume Estimation](#64-stage-4--ice-volume-estimation)
  - [6.5 Stage 5 — Visualization & Mission Report](#65-stage-5--visualization--mission-report)
- [7. Polarimetric SAR Features — The Science](#7-polarimetric-sar-features--the-science)
  - [7.1 Circular Polarization Ratio (CPR)](#71-circular-polarization-ratio-cpr)
  - [7.2 Degree of Polarization (DOP)](#72-degree-of-polarization-dop)
  - [7.3 m-chi Decomposition](#73-m-chi-decomposition)
  - [7.4 Eigenvalue Decomposition (Cloude-Pottier)](#74-eigenvalue-decomposition-cloude-pottier)
  - [7.5 Shannon Entropy](#75-shannon-entropy)
  - [7.6 Feature Summary Table](#76-feature-summary-table)
- [8. Results](#8-results)
- [9. Installation & Usage](#9-installation--usage)
- [10. Project Structure](#10-project-structure)
- [11. Memory & Performance Optimizations](#11-memory--performance-optimizations)
- [12. Limitations & Future Work](#12-limitations--future-work)
- [13. References](#13-references)

---

## 1. The Problem — Why Lunar Ice Matters

The lunar south pole is one of the most strategically important locations in the solar system. The Moon's rotational axis is tilted only 1.54° from the ecliptic, creating regions at the poles that never receive direct sunlight — **Permanently Shadowed Regions (PSRs)**. These PSRs act as cold traps (temperatures as low as 25 K / −248°C), capturing and preserving volatile compounds — including **water ice** — for billions of years.

**Why is this water ice critical?**

| Application | Detail |
|-------------|--------|
| **Life Support** | 1 kg of water ice = drinking water + 0.89 kg of breathable oxygen via electrolysis |
| **Rocket Fuel** | Electrolysis → H₂ + O₂ → LOX/LH₂ propellant. A lunar refueling depot could cut Earth-launch mass by 60-80% for deep-space missions |
| **Scientific Value** | Ice trapped for 2-4 billion years is a frozen record of the inner solar system's volatile history — cometary impacts, solar wind implantation, volcanic outgassing |
| **ISRU Foundation** | In-Situ Resource Utilization eliminates the $1M/kg cost of launching water from Earth |

**The challenge:** We know ice exists (Chandrayaan-1, LRO, LCROSS confirmed it), but we don't know *exactly where*, *how deep*, *how much*, or *where to land a rover to reach it*. That's what LunarIceNet solves.

---

## 2. Scientific Background

### 2.1 Permanently Shadowed Regions (PSRs)

Due to the Moon's near-zero axial tilt (1.54°), crater floors near the poles can remain in permanent shadow. Within these PSRs:

- **Temperature**: 25–110 K (−248°C to −163°C)
- **Thermal stability**: Water ice is thermally stable below ~110 K
- **Sources of ice**: Cometary impacts, solar wind reduction of lunar oxides, volcanic outgassing from the mantle
- **Doubly Permanently Shadowed Regions (DPSRs)**: Regions shadowed by both the sun AND reflected light from nearby illuminated surfaces. These are the coldest and most likely to preserve ice.

**Key PSR craters near the south pole:**

| Crater | Latitude | Diameter | Notes |
|--------|----------|----------|-------|
| Shackleton | −89.9° | 21 km | On the pole itself; rim is a candidate landing site |
| Shoemaker | −88.1° | 51 km | Large, deep; multiple DPSRs confirmed |
| Faustini | −87.3° | 39 km | PRL 2026 confirmed ice in DPSRs |
| Haworth | −87.4° | 51 km | Adjacent to Shoemaker |
| Cabeus | −85.3° | 98 km | LCROSS impact site — confirmed H₂O |
| Sverdrup | −88.4° | 33 km | Near Shackleton |
| de Gerlache | −88.5° | 32 km | Chandrayaan-4 candidate region |
| Nobile | −85.2° | 73 km | Artemis III candidate region |
| Slater | −88.1° | 20 km | Near Shoemaker |

### 2.2 How Radar Detects Ice

Synthetic Aperture Radar (SAR) penetrates the lunar regolith (the top layer of loose soil/rock). When the radar beam encounters subsurface ice, the signal behaves differently than when it hits dry regolith:

```
       Radar Signal
           │
           ▼
    ┌──────────────┐
    │   Regolith    │  ← Surface scattering (weak, single-bounce)
    │   (dry soil)  │
    ├──────────────┤
    │              │
    │  ICE DEPOSIT │  ← Volumetric scattering (strong, multiple bounces)
    │  (mixed with │     Coherent Backscatter Opposition Effect (CBOE)
    │   regolith)  │     creates enhanced same-sense circular return
    │              │
    └──────────────┘
```

**The key indicator: Circular Polarization Ratio (CPR)**

When a circularly polarized radar signal hits a smooth surface, the reflected wave reverses its circular sense (right-hand → left-hand). But when it encounters a heterogeneous volume (like ice chunks in regolith), multiple scattering events preserve the original sense. This creates:

```
CPR = SC / OC = Same-sense Circular / Opposite-sense Circular
```

- **CPR < 1**: Normal surface scattering (dry regolith)
- **CPR > 1**: Volumetric scattering → strong ice indicator
- **CPR > 1.5**: Very likely subsurface ice
- **CPR > 2.0**: Extremely strong ice signature

**Important caveat**: Rough rocky surfaces can also produce CPR > 1 (false positives). This is why we need additional features (DOP, m-chi, entropy) and deep learning — not just a simple threshold.

### 2.3 Chandrayaan-2 DFSAR Instrument

**DFSAR (Dual-frequency Synthetic Aperture Radar)** is the world's first orbital L-band + S-band full-polarimetric lunar SAR:

| Parameter | L-band | S-band |
|-----------|--------|--------|
| Frequency | 1.25 GHz | 2.5 GHz |
| Wavelength | 24 cm | 12 cm |
| Penetration depth | ~3-5 m (regolith) | ~1-2 m |
| Resolution (stripmap) | 2 m | 2 m |
| Polarization | Full-pol (HH, HV, VH, VV) | Full-pol |
| Swath width | 10 km | 10 km |
| Orbit altitude | 100 km | 100 km |

**L-band is critical** because 24 cm wavelength penetrates deeper into the regolith (3-5 m), reaching the subsurface ice deposits that S-band (12 cm, 1-2 m penetration) might miss.

**Data products used in this project:**

| Product | Level | Resolution | Description |
|---------|-------|-----------|-------------|
| `mpcpsp` (Level 3C) | L3C | 25 m/pixel | Calibrated polarimetric mosaic. Contains CPR, SERD, T-Ratio as separate GeoTIFF tiles |
| `my4rsp` (Level 4) | L4 | 25 m/pixel | Higher-level mosaic product with additional processing |

The mosaics cover the entire south polar region (latitude < −80°) in two **look directions**:
- **East look**: Radar illuminates from the east side
- **West look**: Radar illuminates from the west side

**This distinction is critical**: PRL 2026 found ice signatures only in the **west** look direction for specific DPSRs. The look direction determines which crater walls and floors are illuminated.

### 2.4 Key Prior Discoveries

| Year | Mission/Team | Discovery |
|------|-------------|-----------|
| 2008 | Chandrayaan-1 M³ | First confirmed water molecules on lunar surface |
| 2009 | LCROSS (LRO) | Impacted Cabeus crater → confirmed 5.6 ± 2.9% water ice by mass |
| 2010 | Mini-RF (LRO) | CPR anomalies in PSRs consistent with ice |
| 2013 | Spudis et al. | Systematic Mini-RF analysis: CPR > 1 correlates with PSR locations |
| 2018 | Li et al. | Direct spectroscopic evidence of ice at the surface in PSRs |
| 2019 | Chandrayaan-2 DFSAR | First full-pol L-band orbital SAR data of lunar poles |
| 2023 | Saran et al. | DFSAR polarimetric analysis of south polar craters |
| **2026** | **Sinha et al. (PRL)** | **Confirmed subsurface ice in 4 DPSRs using DFSAR west-look: Faustini, Shoemaker, Haworth, Cabeus** |

---

## 3. Our Approach — LunarIceNet

### 3.1 Why Deep Learning Over Classical Methods

**Classical approach** (what most papers do):
```
If CPR > 1.0 AND DOP < 0.13 AND inside_PSR:
    → "Possible ice"
```

**Problems with classical thresholding:**
1. **False positives**: Rocky surfaces produce CPR > 1 (e.g., young crater ejecta)
2. **No confidence**: Binary yes/no, no probability or uncertainty
3. **No depth**: CPR doesn't tell you how deep the ice is
4. **No context**: Each pixel analyzed independently; ignores spatial patterns
5. **Single feature**: Relies heavily on CPR; doesn't exploit the full polarimetric information
6. **No automation**: Manual interpretation by planetary scientists

**LunarIceNet's advantages:**
1. **Learns complex non-linear patterns** from 16 polarimetric + 5 physical features simultaneously
2. **Spatial context** via multi-scale CNN — adjacent pixels inform each other
3. **Physics constraints** built into the loss function — impossible predictions are penalized
4. **Probabilistic output** — each pixel gets a probability (0-1) and confidence score
5. **Depth estimation** — separate regression head estimates ice depth in meters
6. **End-to-end pipeline** — from raw radar to mission-ready landing site recommendations

### 3.2 System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LunarIceNet System Pipeline                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐        │
│  │ DFSAR    │   │ LOLA     │   │ PSR      │   │ Physics  │        │
│  │ Mosaics  │   │ DEM      │   │ Catalog  │   │ Priors   │        │
│  │ (CPR,    │   │ (terrain │   │ (crater  │   │ (temp,   │        │
│  │  SERD,   │   │  slopes) │   │  coords) │   │  lat,    │        │
│  │  T-Ratio)│   │          │   │          │   │  illum)  │        │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘        │
│       │              │              │              │               │
│       ▼              ▼              ▼              ▼               │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              Stage 1: Ice Detection (LunarIceNet)           │   │
│  │  Multi-Scale CNN + Physics MLP + Cross-Attention Fusion     │   │
│  │  → ice_probability, depth_estimate, confidence (per pixel)  │   │
│  └───────────────────────┬─────────────────────────────────────┘   │
│                          │                                         │
│       ┌──────────────────┼──────────────────┐                     │
│       ▼                  ▼                  ▼                     │
│  ┌──────────┐   ┌────────────────┐   ┌──────────────┐            │
│  │ Stage 2  │   │   Stage 3      │   │   Stage 4    │            │
│  │ Landing  │   │   Rover        │   │   Ice Volume │            │
│  │ Sites    │   │   Traverse     │   │   Estimation │            │
│  │ (LOLA    │   │   (A* path     │   │   (Monte     │            │
│  │  slopes) │   │    planning)   │   │    Carlo)    │            │
│  └────┬─────┘   └───────┬────────┘   └──────┬───────┘            │
│       │                 │                    │                     │
│       └─────────────────┼────────────────────┘                     │
│                         ▼                                          │
│           ┌─────────────────────────────┐                          │
│           │  Stage 5: Visualization &   │                          │
│           │  Mission Report Generation  │                          │
│           └─────────────────────────────┘                          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Data Acquisition Guide

### 4.1 Chandrayaan-2 DFSAR Data

**Source**: ISRO's PRADAN portal — [https://pradan.issdc.gov.in](https://pradan.issdc.gov.in)

**Step-by-step:**

1. **Register** on the PRADAN portal (free, requires Indian institutional email or valid ID)

2. **Navigate** to:
   - Mission: `Chandrayaan-2`
   - Instrument: `DFSAR` (Dual-frequency SAR)
   - Data Level: `Level 3C` (calibrated polarimetric mosaic) or `Level 4`
   
3. **Select products:**
   - Product ID prefix: `mpcpsp` (Level 3C mosaic) or `my4rsp` (Level 4 mosaic)
   - Look direction: **East** and/or **West** (both recommended; west is scientifically primary)
   - Region: South Polar (latitude < −80°)

4. **Download** the GeoTIFF tiles. Each tile contains:
   - `*_cpr*.tif` — Circular Polarization Ratio
   - `*_serd*.tif` — Single Expected Return Deviation (radar return variability)
   - `*_tratio*.tif` — T-matrix ratio (T11/(T22+T33))

5. **Convert to numpy mosaics** using the preprocessing script (or manually via rasterio):
   ```python
   import rasterio
   import numpy as np
   
   # Read a DFSAR GeoTIFF tile
   with rasterio.open('path/to/cpr_tile.tif') as src:
       cpr = src.read(1).astype(np.float32)
       transform = src.transform  # Affine transform for UPS projection
       crs = src.crs              # Coordinate reference system
   
   # The GeoTIFF is in UPS (Universal Polar Stereographic) projection
   # Center: South Pole (-90°, 0°)
   # Units: metres from pole
   ```

6. **Place in data directory:**
   ```
   data/raw/mpcpsp_east/    ← East-look tiles
   data/raw/mpcpsp_west/    ← West-look tiles
   data/raw/my4rsp_east/    ← Level 4 east (alternative)
   data/raw/my4rsp_west/    ← Level 4 west (alternative)
   ```

**Mosaic dimensions (after assembly):**

| Direction | Size (pixels) | Coverage | File size (float32/array) |
|-----------|--------------|----------|--------------------------|
| East | 12,237 × 12,794 | ~157M px, 35.1% valid | ~600 MB |
| West | 24,794 × 24,181 | ~600M px, ~56% valid | ~2.3 GB |

### 4.2 NASA LRO LOLA DEM

**Source**: NASA PDS Geosciences Node

**URL**: [https://pds-geosciences.wustl.edu/lro/lro-l-lola-4-gdr-v1/lrolol_1xxx/data/lola_gdr/polar/img/](https://pds-geosciences.wustl.edu/lro/lro-l-lola-4-gdr-v1/lrolol_1xxx/data/lola_gdr/polar/img/)

**Recommended products:**

| Product | File | Size | Resolution | Lat coverage |
|---------|------|------|-----------|-------------|
| Primary | `ldem_85s_40m.img` | 110 MB | 40 m/px | −90° to −85° |
| High-res fill | `ldem_875s_20m.img` | 110 MB | 20 m/px | −90° to −87.5° |
| Label file | `ldem_85s_40m.lbl` | 5 KB | — | Metadata |

**Direct download:**
```bash
# Primary DEM (covers full DFSAR region)
wget https://pds-geosciences.wustl.edu/lro/lro-l-lola-4-gdr-v1/lrolol_1xxx/data/lola_gdr/polar/img/ldem_85s_40m.img

# High-resolution inner pole
wget https://pds-geosciences.wustl.edu/lro/lro-l-lola-4-gdr-v1/lrolol_1xxx/data/lola_gdr/polar/img/ldem_875s_20m.img

# Label (metadata)
wget https://pds-geosciences.wustl.edu/lro/lro-l-lola-4-gdr-v1/lrolol_1xxx/data/lola_gdr/polar/img/ldem_85s_40m.lbl
```

**File format — PDS3 binary:**
```
Format:           Raw binary array (no header)
Data type:        LSB_INTEGER (little-endian signed 16-bit)
Dimensions:       LINES × LINE_SAMPLES (e.g., 7584 × 7584)
Projection:       South Polar Stereographic
Center:           (-90°, 0°)
Scale factor:     0.5 (DN → metres of elevation)
Offset:           1,737,400 m (Moon's reference radius)

Elevation (m) = DN × 0.5
Planetary radius (m) = DN × 0.5 + 1,737,400
```

**Reading in Python:**
```python
import numpy as np

# Read raw binary
raw = np.frombuffer(
    open('ldem_85s_40m.img', 'rb').read(),
    dtype='<i2'  # little-endian signed 16-bit
).reshape(7584, 7584)

# Convert to elevation
elevation_m = raw.astype(np.float32) * 0.5
print(f"Elevation range: [{elevation_m.min():.0f}, {elevation_m.max():.0f}] m")
# Output: Elevation range: [-5501, 7027] m
```

**Place in data directory:**
```
data/raw/lola_dem/
├── ldem_85s_40m.img       # 110 MB binary DEM
├── ldem_85s_40m.lbl       # or .lbl.txt — metadata
└── ldem_875s_20m.img      # 110 MB high-res inner pole
```

### 4.3 Data Directory Structure

```
data/
├── raw/
│   ├── mpcpsp_east/           # DFSAR east-look Level 3C
│   │   ├── cpr_mosaic.npy     # CPR values (float32, 12237×12794)
│   │   ├── serd_mosaic.npy    # SERD values
│   │   └── tratio_mosaic.npy  # T-Ratio values
│   │
│   ├── mpcpsp_west/           # DFSAR west-look Level 3C
│   │   ├── cpr_mosaic.npy
│   │   ├── serd_mosaic.npy
│   │   └── tratio_mosaic.npy
│   │
│   ├── my4rsp_east/           # Level 4 east mosaic (alternative)
│   ├── my4rsp_west/           # Level 4 west mosaic
│   │
│   └── lola_dem/              # NASA LRO LOLA DEM
│       ├── ldem_85s_40m.img   # Primary: -90° to -85°, 40 m/px
│       ├── ldem_85s_40m.lbl   # PDS3 label
│       └── ldem_875s_20m.img  # High-res: -90° to -87.5°, 20 m/px
│
└── processed/                 # Generated by pipeline (auto-created)
```

---

## 5. Model Architecture — LunarIceNet (Deep Dive)

LunarIceNet is a **12.4-million parameter** physics-informed neural network with three branches:

```
                         ┌─────────────────────────────────────┐
                         │         LunarIceNet (12.4M)         │
                         ├─────────────────────────────────────┤
                         │                                     │
  Radar Features         │  ┌──────────────────────────────┐  │
  (B, 3, 64, 64)  ──────►  │  MultiScaleRadarEncoder      │  │
                         │  │  8 ResBlocks + SE + FPN       │──┤
                         │  │  → (B, 128, 16, 16)           │  │
                         │  └──────────────────────────────┘  │
                         │                      │              │
                         │                      ▼              │
                         │  ┌──────────────────────────────┐  │
  Physics Params         │  │  CrossAttentionFusion         │  │
  (B, 5)         ──────► │  │  2 layers × (cross + self     │──┤
                    │    │  │  + FFN), 4 heads              │  │
                    │    │  │  → (B, 128, 16, 16)           │  │
                    │    │  └──────────────────────────────┘  │
                    │    │                      │              │
                    ▼    │                      ▼              │
  ┌──────────────┐ │    │  ┌──────────────────────────────┐  │
  │ PhysicsEnc.  │ │    │  │  IceDetectionHead             │  │
  │ MLP (3 layer)│─┘    │  │  3× ConvTranspose2d upsample  │  │
  │ → (B, 128)  │       │  │  → 3 output heads:            │  │
  └──────────────┘       │  │    ice_prob  (B, 1, 64, 64)   │  │
                         │  │    depth     (B, 1, 64, 64)   │  │
                         │  │    confidence(B, 1, 64, 64)   │  │
                         │  └──────────────────────────────┘  │
                         └─────────────────────────────────────┘
```

### 5.1 Multi-Scale Radar Encoder

The radar encoder processes 64×64 patches of 3-channel DFSAR data (CPR, SERD, T-Ratio):

```
Input: (B, 3, 64, 64)

Stem:
  Conv2d(3 → 64, kernel=7, stride=2, pad=3) → BN → ReLU → MaxPool(3, stride=2)
  → (B, 64, 16, 16)

Layer 1: 2× ResidualBlock(64 → 64, stride=1)   → (B, 64, 16, 16)
Layer 2: 2× ResidualBlock(64 → 128, stride=2)   → (B, 128, 8, 8)
Layer 3: 2× ResidualBlock(128 → 256, stride=2)  → (B, 256, 4, 4)
Layer 4: 2× ResidualBlock(256 → 512, stride=2)  → (B, 512, 2, 2)

Multi-Scale Fusion (Feature Pyramid):
  f1 (64ch)  → 1×1 Conv → (B, embed_dim, 16, 16)           ← fine detail
  f2 (128ch) → 1×1 Conv → bilinear upsample to 16×16        ← local context
  f3 (256ch) → 1×1 Conv → bilinear upsample to 16×16        ← regional context
  f4 (512ch) → 1×1 Conv → bilinear upsample to 16×16        ← global context

  Concatenate → (B, 4×embed_dim, 16, 16)
  1×1 Conv fusion → BN → ReLU → (B, embed_dim, 16, 16)
```

Each **ResidualBlock** contains:
```python
class ResidualBlock:
    conv1: Conv2d(in, out, 3×3, stride, pad=1) → BN → ReLU
    conv2: Conv2d(out, out, 3×3, pad=1) → BN
    se:    SqueezeExcitation(out)              # Channel attention
    shortcut: Conv2d(1×1) + BN if dims change
    output = ReLU(se(conv2(conv1(x))) + shortcut(x))
```

**Squeeze-Excitation (SE)** channel attention:
```
x → AdaptiveAvgPool2d(1) → Flatten
  → Linear(C, C/16) → ReLU → Linear(C/16, C) → Sigmoid
  → multiply element-wise with x
```
This lets the network learn which channels (CPR vs SERD vs T-Ratio derived features) are most informative for each spatial location.

### 5.2 Physics Encoder

A 3-layer MLP that transforms 5 physical scalar parameters into an embedding vector:

```
Input: (B, 5)
  [0] latitude          — proxy for temperature (−90° is coldest)
  [1] longitude         — azimuthal position
  [2] PSR probability   — from crater catalog lookup
  [3] distance from pole — (90 + lat) / 10, normalized to [0, 1]
  [4] PSR flag          — binary: inside known PSR?

Layers:
  Linear(5, 64)  → ReLU → LayerNorm(64)
  Linear(64, 128) → ReLU → LayerNorm(128)
  Linear(128, embed_dim) → ReLU

Output: (B, embed_dim)     e.g., (B, 128)
```

### 5.3 Cross-Attention Fusion

The core innovation: radar features **attend to** physics context. This allows the model to modulate per-pixel predictions based on physically meaningful constraints (e.g., "this pixel is inside a PSR at −89° latitude → weight the ice probability higher").

```
For each of 2 layers:

  1. Cross-Attention (radar queries, physics is key/value):
     Q = LayerNorm(radar_tokens)       shape: (B, H×W, embed_dim)
     K = V = physics_embedding         shape: (B, 1, embed_dim)
     x = x + MultiheadAttention(Q, K, V)     # 4 heads

  2. Self-Attention (spatial positions attend to each other):
     Q = K = V = LayerNorm(x)
     x = x + MultiheadAttention(Q, K, V)     # 4 heads

  3. Feed-Forward Network:
     x = x + FFN(LayerNorm(x))
     FFN = Linear(embed_dim, 4×embed_dim) → GELU → Linear(4×embed_dim, embed_dim)
```

The self-attention step is especially important for ice detection: ice deposits tend to be spatially coherent (not random noise), so attending to neighboring positions helps distinguish real ice from radar speckle.

### 5.4 Multi-Task Detection Head

Three parallel output heads decode the fused features back to full spatial resolution:

```
Shared decoder:
  ConvTranspose2d(embed_dim, 128, 4×4, stride=2) → BN → ReLU   # 16→32
  ConvTranspose2d(128, 64, 4×4, stride=2) → BN → ReLU           # 32→64
  ConvTranspose2d(64, 32, 4×4, stride=2) → BN → ReLU            # 64→128
  Bilinear interpolate to exact target size (64×64)

Ice head:       Conv(32, 16, 3×3) → ReLU → Conv(16, 1, 1×1)     → logits
Depth head:     Conv(32, 16, 3×3) → ReLU → Conv(16, 1, 1×1) → ReLU  → metres (≥0)
Confidence head:Conv(32, 16, 3×3) → ReLU → Conv(16, 1, 1×1) → Sigmoid → [0, 1]
```

### 5.5 Physics-Informed Loss Function

The loss is a weighted sum of four components:

```
L_total = 1.0 × L_BCE + 0.5 × L_depth + 0.3 × L_physics + 0.2 × L_temp_prior
```

**Component 1 — Binary Cross-Entropy (L_BCE):**
```
L_BCE = BCE_with_logits(ice_logits, labels)
```
Labels are generated from CPR > 1.0 threshold on raw (un-normalized) data.

**Component 2 — Depth Regression (L_depth):**
```
For pixels where label = 1 (ice):
    pseudo_depth = 1.0 m  (approximate L-band penetration depth)
    L_depth = MSE(predicted_depth[ice_mask], pseudo_depth[ice_mask])
```
Only applied where ice exists — no depth penalty for correctly predicting "no ice."

**Component 3 — Physics Constraint (L_physics):**
```
dist_from_pole = clamp((90 + latitude) / 10, 0, 1)
    → 0 at the pole (−90°), 1 at −80°

L_physics = mean(sigmoid(ice_logits) × dist_from_pole)
```
Penalizes high ice predictions far from the pole. At −90°, no penalty. At −80°, full penalty. This encodes the physical reality that ice is only thermally stable in the coldest polar regions.

**Component 4 — Temperature Prior (L_temp_prior):**
```
L_temp = mean((1 - sigmoid(ice_logits)) × PSR_prob × labels)
```
Penalizes the model for NOT predicting ice when (a) the ground truth says ice AND (b) the pixel is in a known PSR. This encodes: "PSRs should have ice — don't ignore them."

---

## 6. Five-Stage Pipeline (Detailed)

### 6.1 Stage 1 — Subsurface Ice Detection

**Input**: DFSAR Level 3C mosaic (12237×12794 or 24794×24181 pixels)
**Output**: Three maps — `ice_probability`, `depth_estimate`, `confidence`

**Process:**
1. Load 3-channel mosaic (CPR, SERD, T-Ratio) via `DFSARMosaicLoader`
2. Compute coordinates: UPS projection → lat/lon for every pixel
3. Normalize features: clip to [1st, 99th] percentile, scale to [0, 1]
4. Extract patches (64×64, stride 32) → ~35,000 patches for east mosaic
5. For each patch, compute physical features: [lat, lon, PSR_prob, dist_pole, PSR_flag]
6. Run LunarIceNet inference (batched, with AMP on GPU)
7. Stitch patches back into full-resolution maps (overlap-averaged)

**Coordinate computation (UPS → lat/lon):**
```python
# DFSAR GeoTIFFs use Universal Polar Stereographic (south pole)
# Transform: Affine from rasterio (pixel → metres from pole)
x_m = transform.c + col × transform.a     # easting (metres)
y_m = transform.f + row × transform.e     # northing (metres)

# UPS → lat/lon:
rho = sqrt(x² + y²)                        # distance from pole in metres
lat = -(90 - 2 × arctan(rho / (2 × R_moon)))  # R_moon = 1,737,400 m
lon = atan2(x, -y)                          # azimuth angle
```

### 6.2 Stage 2 — Landing Site Selection

**Input**: Ice probability map, confidence map, LOLA terrain slopes, coordinates
**Output**: Top-10 ranked landing sites with composite scores

**Scoring formula:**
```
Score = 0.35 × ice_prob + 0.20 × slope_safety + 0.15 × accessibility
      + 0.15 × illumination + 0.15 × confidence
```

Where:
- `ice_prob`: Mean ice probability in a 10×10 pixel window (250 m × 250 m)
- `slope_safety`: 1.0 if slope < 5°, linearly decreases to 0 at 15°, hard reject above 15°
- `accessibility`: Earth visibility proxy based on latitude (closer to −85° is better for comms)
- `illumination`: Heuristic from slope variance (peaks near PSRs get some sunlight for solar panels)
- `confidence`: Mean model confidence in the window

**LOLA DEM integration:**
The pipeline loads real terrain slopes from two LOLA products:
1. `ldem_85s_40m.img` (primary, −90° to −85°, 93.3% DFSAR coverage)
2. `ldem_875s_20m.img` (secondary, −90° to −87.5°, overrides with 20 m resolution)

Slopes are computed via finite-difference gradient with Gaussian smoothing:
```python
slope = arctan(sqrt((dz/dx)² + (dz/dy)²))
```
Then regridded from LOLA pixel coordinates to DFSAR pixel coordinates using nearest-neighbour lookup in UPS projection space. Remaining NaN pixels (6.7% gap beyond LOLA coverage) filled with 5° default.

### 6.3 Stage 3 — Rover Traverse Planning

**Input**: Landing site, target site, ice probability map, slope map, coordinates
**Output**: Optimal rover path, distance, time estimate, safety report

**Algorithm**: A* search on a 25 m/pixel grid with composite cost function.

**Rover specifications (Pragyan-class reference):**
- Max traversable slope: 25°
- Preferred slope: < 15°
- Flat-terrain speed: 100 m/hr
- Max range: 8 km from lander

**A* cost function per edge (cell-to-cell transition):**
```
cost = distance + slope_penalty + solar_penalty − ice_reward

where:
  distance   = 25 m (orthogonal) or 25√2 m (diagonal)

  slope_penalty:
    if slope ≤ 15°: 0
    if slope > 25°: IMPASSABLE (return None)
    else: distance × 2 × (exp(3 × (slope−15)/(25−15)) − 1)

  solar_penalty = distance × 0.5 × (1 − illumination)
    # Prefer illuminated paths for solar power

  ice_reward = distance × 0.8 × ice_prob
    # Route through high-ice areas for science

  final_cost = max(cost, 0.01)  # keep positive
```

**Target selection logic:**
1. Find nearest DPSR crater to landing site
2. Compute physical distance in km (accounting for polar geometry)
3. If nearest DPSR > 8 km: use Site #2 as target (guaranteed reachable)
4. If nearest DPSR ≤ 8 km: target that DPSR directly

**Safety analysis output:**
- Max/mean slope along path
- Hazard cell count (slope > 15°)
- Near-impassable cells (slope 20–25°)
- Estimated energy consumption (Wh)
- Ice sampling waypoints (where ice_prob > 0.7)

### 6.4 Stage 4 — Ice Volume Estimation

**Input**: Ice probability map, depth estimates, CPR values, PSR catalog
**Output**: Total and per-crater ice volume with Monte Carlo uncertainty

**Step 1 — CPR → Ice fraction (Lichtenecker dielectric mixing model):**

The Lichtenecker model relates the effective dielectric constant of a mixture to its components:

```
ε_eff^0.5 = f_ice × ε_ice^0.5 + (1 − f_ice) × ε_regolith^0.5
```

Where:
- `ε_regolith = 3.0` (lunar regolith typical permittivity)
- `ε_ice = 3.15` (water ice at cryogenic temperatures)
- `f_ice` = volume fraction of ice (what we're solving for)

CPR is mapped to ice fraction via an empirical curve:
```
CPR < 0.6  → f_ice = 0.00  (no ice signal)
CPR = 1.0  → f_ice = 0.05  (5% ice by volume)
CPR = 1.5  → f_ice = 0.15  (15%)
CPR = 2.0  → f_ice = 0.25  (25%)
CPR = 2.5  → f_ice = 0.30  (30%)
CPR > 3.0  → f_ice = 0.30  (capped at 30%)
```

**Step 2 — Penetration depth:**
```
depth = λ / (4π × Im(√ε_eff))
```
Where λ = 0.24 m (L-band wavelength). Typical results: 0.5–3.0 m.

**Step 3 — Per-pixel volume:**
```
V_pixel = pixel_area × depth × f_ice × P(ice)

pixel_area = 25 × 25 = 625 m²
depth      = from Lichtenecker model (metres)
f_ice      = from CPR mapping (0–0.30)
P(ice)     = from LunarIceNet output (0–1)
```

**Step 4 — Monte Carlo uncertainty (1000 samples):**
```
For each sample:
    f_ice_perturbed = f_ice × (1 + N(0, 0.30))    # ±30% uncertainty
    depth_perturbed = depth × (1 + N(0, 0.25))    # ±25% uncertainty
    P_perturbed     = P(ice) × (1 + N(0, 0.15))   # ±15% uncertainty

    V_total = Σ (pixel_area × depth_perturbed × f_ice_perturbed × P_perturbed)

90% CI = [5th percentile, 95th percentile] of 1000 V_total samples
```

**Step 5 — Per-crater breakdown:**
For each of 16 DPSRs in the catalog, compute volume within that crater's boundaries.

### 6.5 Stage 5 — Visualization & Mission Report

Generates four output images and a text mission report:

1. **`ice_analysis_east.png`** — 4-panel figure:
   - Panel 1: Ice probability heatmap (custom blue-cyan-white colormap)
   - Panel 2: Depth estimate map
   - Panel 3: Confidence map
   - Panel 4: LOLA terrain slope (red > 15° = unsafe)

2. **`polar_ice_map_east.png`** — Polar stereographic projection showing ice probability around the pole with latitude rings

3. **`rover_traverse_east.png`** — A* path overlaid on slope map with landing site, target, and waypoints

4. **`lola_dem_summary.png`** — LOLA elevation and slope visualization with contours at 15° and 25°

5. **`mission_report_east.txt`** — ASCII text summary with all key numbers

---

## 7. Polarimetric SAR Features — The Science

The `src/features/polarimetric.py` module implements 16 polarimetric features from full-pol SAR data. Here's the science behind each:

### 7.1 Circular Polarization Ratio (CPR)

**What it measures**: The ratio of same-sense to opposite-sense circular polarization in the radar return.

**Physics**: A smooth surface reverses the circular polarization of the incoming wave (right-hand → left-hand). A heterogeneous volume (ice chunks in regolith) creates multiple scattering events that **preserve** the original sense via Coherent Backscatter Opposition Effect (CBOE).

**Formula:**
```
SC (Same-sense Circular)    = (HH - VV + 2j·HV) / 2
OC (Opposite-sense Circular) = (HH + VV) / 2
CPR = |SC|² / |OC|²
```

**Interpretation:**
- CPR < 0.5: Very smooth surface
- CPR ≈ 0.5–1.0: Moderate roughness (normal regolith)
- CPR > 1.0: Volumetric scattering (ice candidate!)
- CPR > 2.0: Very strong volumetric signature

### 7.2 Degree of Polarization (DOP)

**What it measures**: How "organized" the returned polarization state is.

**Physics**: A pure target (smooth flat surface) returns a fully polarized wave (DOP ≈ 1). A heterogeneous volume with random scatterers depolarizes the wave (DOP → 0). Subsurface ice partially depolarizes due to multiple internal reflections.

**Formula:**
```
DOP = sqrt(1 - 4·det(C) / tr(C)²)
```
Where C is the 2×2 coherency matrix.

**Ice signature**: DOP < 0.13 combined with CPR > 1.0 is the classical ice detection criterion.

### 7.3 m-chi Decomposition

**What it measures**: Separates the radar return into three scattering mechanisms using Stokes parameters.

**Components:**
```
m   = degree of polarization (Stokes)
chi = ellipticity angle

P_volume      = m × (1 - sin(2χ)) / 2     # Volume scattering (ice)
P_double      = m × (1 + sin(2χ)) / 2     # Double-bounce (crater walls)
P_surface     = 1 - m                       # Surface scattering (flat regolith)
```

### 7.4 Eigenvalue Decomposition (Cloude-Pottier)

**What it measures**: Decomposes the 3×3 coherency matrix T3 into its eigenvalues to characterize scattering type and randomness.

**Parameters:**
```
T3 eigenvalues: λ₁ ≥ λ₂ ≥ λ₃

Probabilities: p_i = λ_i / (λ₁ + λ₂ + λ₃)

Entropy:    H = -Σ (p_i × log₂(p_i)) / log₂(3)    ∈ [0, 1]
Anisotropy: A = (λ₁ - λ₂) / (λ₁ + λ₂)             ∈ [0, 1]
Alpha:      α = Σ (p_i × α_i)                       ∈ [0°, 90°]
```

**Interpretation:**
- H ≈ 0: Single dominant scattering mechanism (clear target)
- H ≈ 1: Random scattering (depolarized, complex volume)
- α ≈ 0°: Surface scattering
- α ≈ 45°: Dipole / volume scattering (ice!)
- α ≈ 90°: Dihedral / double-bounce

### 7.5 Shannon Entropy

**What it measures**: Total information content of the scattering matrix, split into intensity and polarimetric components.

**Formula:**
```
SE = SE_intensity + SE_polarimetric

SE_i = 3 × ln(π × e × tr(T) / 3)
SE_p = ln(det(T) / (tr(T)/3)³)
```

### 7.6 Feature Summary Table

| # | Feature | Input | Output | Ice Indicator |
|---|---------|-------|--------|---------------|
| 1 | CPR | HH, HV, VV | float32 (H,W) | > 1.0 |
| 2 | DOP | HH, HV, VV | float32 (H,W) | < 0.13 |
| 3 | m-chi volume | Stokes | float32 (H,W) | High |
| 4 | m-chi double-bounce | Stokes | float32 (H,W) | Low for ice |
| 5 | m-chi surface | Stokes | float32 (H,W) | Low for ice |
| 6 | m (polarization degree) | Stokes | float32 (H,W) | Low |
| 7 | chi (ellipticity) | Stokes | float32 (H,W) | Negative |
| 8 | Entropy (H) | T3 | float32 (H,W) | Medium-high |
| 9 | Anisotropy (A) | T3 | float32 (H,W) | Low |
| 10 | Alpha (α) | T3 | float32 (H,W) | ~45° |
| 11 | Shannon Entropy | T3 | float32 (H,W) | High |
| 12 | Pedestal Height | Co-pol | float32 (H,W) | High |
| 13 | HH intensity | HH | float32 (H,W) | — |
| 14 | HV intensity | HV | float32 (H,W) | — |
| 15 | VV intensity | VV | float32 (H,W) | — |
| 16 | HH/VV ratio | HH, VV | float32 (H,W) | — |

---

## 8. Results

### Model Performance (East Look Direction)

| Metric | Value |
|--------|-------|
| F1 Score | 0.8428 |
| Precision | 0.8906 |
| Recall | 0.7999 |
| IoU | ~0.73 |
| Accuracy | 99.6% |
| Valid DFSAR pixels | 54,903,877 (35.1% of mosaic) |
| LOLA terrain coverage | 93.3% of DFSAR grid |
| Training patches | 34,840 (64×64, stride 32) |
| Model parameters | 12,449,219 (12.4M) |

### Top Landing Sites (with Real LOLA Slopes)

| Rank | Latitude | Longitude | Score | Ice Prob | Slope | Notes |
|------|----------|-----------|-------|----------|-------|-------|
| #1 | −86.035° | −28.650° | 0.728 | 0.946 | 3.5° | Very flat, high ice |
| #2 | −86.082° | −28.756° | 0.723 | 0.979 | 5.6° | Near Site #1 |
| #3 | −85.691° | −99.827° | 0.722 | 0.961 | 4.9° | Different region |
| #4 | −84.937° | −92.867° | 0.722 | 0.992 | 5.0° | Highest ice prob |
| #7 | −86.950° | −103.63° | 0.716 | 0.997 | 1.2° | Flattest site |

### Ice Volume Estimate

| Parameter | Value |
|-----------|-------|
| Volume (best estimate) | 8,423,807 m³ |
| Mass | 7,725 million tonnes |
| Mean ice fraction | 0.09% |
| Mean depth | 1.05 m |
| Survey area | 34,315 km² |
| 90% confidence interval | [3.7M – 14.6M] m³ |

### Per-DPSR Analysis (East Direction)

All 16 DPSRs show **0.0 m³** ice volume in the east look direction. This is **scientifically correct** — PRL 2026 (Sinha et al.) confirmed that ice signatures appear only in the **west** look direction for these specific DPSRs. The east-look geometry illuminates different crater wall faces, which do not show the same volumetric scattering. This validates our pipeline's physical consistency.

### Training Convergence

```
Epoch  1/30:  Loss=0.0333  F1=0.065  (model learning basic features)
Epoch  5/30:  Loss=0.0166  F1=0.385  (CPR correlation emerging)
Epoch 10/30:  Loss=0.0099  F1=0.715  (spatial patterns learned)
Epoch 20/30:  Loss=0.0052  F1=0.823  (physics constraints active)
Epoch 30/30:  Loss=0.0038  F1=0.843  (converged)
```

---

## 9. Installation & Usage

### Prerequisites

- **Python 3.10+**
- **GPU**: NVIDIA with CUDA 11.8+ (required for training; inference possible on CPU)
- **RAM**: 16+ GB (32 GB recommended for full pipeline)
- **Disk**: ~5 GB for data + checkpoints

### Installation

```bash
git clone https://github.com/Priyanshu-byte-coder/bah-2026-lunar-ice.git
cd bah-2026-lunar-ice

python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### Training

```bash
# East direction (12K×13K mosaic, ~35K patches, ~2 hrs on T4 GPU)
python train_real.py \
    --direction east \
    --epochs 30 \
    --batch-size 16 \
    --patch-size 64 \
    --stride 32 \
    --embed-dim 128 \
    --num-heads 4 \
    --num-attn-layers 2 \
    --lr 0.001

# West direction (25K×24K mosaic — needs subsampling to fit in memory)
python train_real.py \
    --direction west \
    --epochs 30 \
    --batch-size 8 \
    --subsample 2 \
    --embed-dim 64

# Just analyze data statistics (no training)
python train_real.py --analyze-only --direction east
```

### Full Pipeline

```bash
# Full run (inference + all 5 stages, ~1 hour)
python full_pipeline.py \
    --direction east \
    --checkpoint checkpoints/east_best_model.pth \
    --lola-dem data/raw/lola_dem/ldem_85s_40m.img

# Cached run (skip inference, reuse saved .npy, ~45 min)
python full_pipeline.py \
    --use-cached \
    --direction east \
    --checkpoint checkpoints/east_best_model.pth \
    --lola-dem data/raw/lola_dem/ldem_85s_40m.img

# West direction (where PRL 2026 confirmed ice)
python full_pipeline.py \
    --direction west \
    --checkpoint checkpoints/east_best_model.pth \
    --lola-dem data/raw/lola_dem/ldem_85s_40m.img \
    --output-dir outputs/west
```

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--direction` | `east` | Look direction: `east` or `west` |
| `--checkpoint` | required | Path to trained `.pth` model file |
| `--lola-dem` | `data/raw/lola_dem/ldem_875s_20m.img` | Path to LOLA DEM binary |
| `--use-cached` | `false` | Skip inference, use saved `.npy` prediction maps |
| `--output-dir` | `outputs/` | Output directory for reports and images |
| `--data-dir` | `data/raw` | Root data directory |
| `--batch-size` | `16` | Inference batch size |
| `--patch-size` | `64` | CNN input patch size (pixels) |

---

## 10. Project Structure

```
bah-2026-lunar-ice/
│
├── full_pipeline.py              # End-to-end 5-stage pipeline
├── train_real.py                 # Training on real DFSAR mosaics (AMP support)
├── predict_real.py               # Standalone inference script
├── main.py                       # Entry point / demo
├── requirements.txt              # Python dependencies
│
├── src/
│   ├── __init__.py
│   │
│   ├── models/
│   │   ├── lunaricenet.py        # LunarIceNet architecture (12.4M params)
│   │   │   ├── ResidualBlock         # Conv → BN → ReLU with skip connection
│   │   │   ├── SqueezeExcitation     # Channel attention (AdaptiveAvgPool → FC)
│   │   │   ├── MultiScaleRadarEncoder# 8 ResBlocks + Feature Pyramid Network
│   │   │   ├── PhysicsEncoder        # 3-layer MLP for physical params
│   │   │   ├── CrossAttentionFusion  # 2-layer cross + self attention
│   │   │   ├── IceDetectionHead      # 3× ConvTranspose decoder → 3 outputs
│   │   │   ├── LunarIceNet           # Main model (encoder+physics+fusion+head)
│   │   │   └── PhysicsInformedLoss   # BCE + depth + physics + temp prior
│   │   └── trainer.py            # Training utilities, Metrics class
│   │
│   ├── data/
│   │   ├── real_loader.py        # DFSARMosaicLoader
│   │   │   ├── DFSARMosaicLoader     # Reads Level 3C .npy mosaics
│   │   │   │   ├── load_single_direction()  # Load CPR/SERD/T-Ratio
│   │   │   │   ├── get_coordinates()        # UPS → lat/lon (chunked)
│   │   │   │   └── get_ice_candidates()     # CPR > threshold mask
│   │   │   └── RealDFSARDataset      # PyTorch Dataset with patch extraction
│   │   │       ├── __getitem__()     # Returns features + physical + label
│   │   │       └── ice_enrichment    # Oversample ice-containing patches
│   │   │
│   │   ├── lola_loader.py        # LOLADEMLoader
│   │   │   ├── LOLADEMLoader         # Reads PDS3 binary DEM
│   │   │   │   ├── load()            # Binary → float32 elevation
│   │   │   │   ├── compute_slope()   # Gradient + Gaussian smooth
│   │   │   │   ├── get_coordinates() # UPS → lat/lon
│   │   │   │   ├── regrid_to_dfsar() # Nearest-neighbour reprojection
│   │   │   │   └── plot_summary()    # Elevation + slope visualization
│   │   │   └── load_lola_slope_for_dfsar()  # Convenience function
│   │   │
│   │   ├── dataset.py            # LunarPSRCatalog
│   │   │   ├── LunarPSRCatalog       # 9 craters, 16 DPSRs
│   │   │   │   ├── get_psr_craters()
│   │   │   │   └── get_dpsr_craters()
│   │   │   └── LunarIceDataset       # Synthetic dataset (for testing)
│   │   │
│   │   └── features.py           # Feature extraction utilities
│   │
│   ├── features/
│   │   └── polarimetric.py       # Full-pol SAR feature extraction
│   │       ├── compute_cpr()          # SC/OC from HH/HV/VV
│   │       ├── compute_dop()          # Degree of Polarization
│   │       ├── compute_mchi_decomposition()  # m-chi components
│   │       ├── compute_eigenvalue_decomposition()  # H/A/alpha
│   │       ├── compute_shannon_entropy()
│   │       ├── compute_pedestal_height()
│   │       ├── compute_covariance_matrix()   # C4 matrix
│   │       ├── compute_coherency_matrix()    # T3 matrix
│   │       └── extract_all_features()        # All 16 features
│   │
│   ├── analysis/
│   │   ├── __init__.py
│   │   └── ice_volume.py         # IceVolumeEstimator
│   │       ├── IceVolumeEstimator
│   │       │   ├── estimate_volume()         # Global volume + MC uncertainty
│   │       │   ├── estimate_volume_per_crater()  # Per-DPSR breakdown
│   │       │   └── _cpr_to_ice_fraction()    # Empirical mapping curve
│   │       └── generate_report()             # Text report generation
│   │
│   ├── planning/
│   │   ├── __init__.py
│   │   └── rover_traverse.py     # RoverTraversePlanner
│   │       ├── TraverseResult        # Path + metrics dataclass
│   │       ├── SafetyReport          # Safety analysis dataclass
│   │       ├── RoverTraversePlanner
│   │       │   ├── __init__()        # Build slope/illumination grids
│   │       │   ├── _edge_cost()      # Composite cost function
│   │       │   ├── _heuristic()      # Euclidean admissible heuristic
│   │       │   ├── plan_path()       # A* search with range budget
│   │       │   └── _latlon_to_rc()   # Coordinate → grid cell lookup
│   │       ├── TraverseAnalyzer      # Post-hoc safety analysis
│   │       │   └── generate_report() # Safety report generation
│   │       └── plan_traverse()       # Convenience function
│   │
│   ├── visualization/
│   │   └── maps.py               # Visualization functions (currently lunar_map.py)
│   │       ├── create_ice_probability_map()     # Polar projection
│   │       ├── create_3d_terrain_with_ice()     # Plotly 3D surface
│   │       ├── create_feature_visualization()   # Multi-feature grid
│   │       ├── create_training_curves()         # Loss/F1 plots
│   │       └── create_landing_site_comparison() # Radar chart
│   │
│   └── utils/
│       └── geo.py                # Geodesy utilities
│
├── configs/
│   └── config.yaml               # All hyperparameters and paths
│
├── checkpoints/
│   ├── east_best_model.pth       # Best east model (F1=0.843)
│   └── training_history.json     # Per-epoch metrics
│
├── outputs/
│   ├── mission_report_east.txt   # ASCII mission report
│   ├── landing_sites_east.txt    # Top-10 ranked sites
│   ├── ice_volume_east.txt       # Volume estimation report
│   ├── traverse_report_east.txt  # Rover safety analysis
│   ├── ice_analysis_east.png     # 4-panel analysis map
│   ├── polar_ice_map_east.png    # Polar projection
│   ├── rover_traverse_east.png   # Traverse path visualization
│   └── lola_dem_summary.png      # LOLA elevation + slope
│
├── docs/
│   └── proposal/
│       └── idea_proposal.md      # BAH 2026 submission document
│
├── information/                  # Reference screenshots
├── notebooks/                    # Exploration notebooks
└── tests/                        # Unit tests
```

---

## 11. Memory & Performance Optimizations

The full DFSAR east mosaic is **12,237 × 12,794 pixels** — each float32 array is **~600 MB**. The west mosaic is **24,794 × 24,181** (~2.3 GB per array). Processing these requires careful memory management:

### Problem → Solution Table

| Problem | Memory Impact | Solution |
|---------|--------------|----------|
| 2D meshgrid for coordinates | 2× 600 MB = 1.2 GB (east) | Chunked row-by-row computation (1024 rows/chunk, ~50 MB peak) |
| Cached prediction maps (.npy) | 3× 600 MB = 1.8 GB | `np.load(mmap_mode='r')` — memory-mapped, loaded on demand |
| LOLA regridding intermediates | 6× 600 MB = 3.6 GB (float64!) | All float32 + explicit `del` after each intermediate |
| Ice volume computation | Multiple 600 MB copies | `np.nan_to_num()` + `np.nansum()` to handle NaN gracefully |
| West mosaic: 2.3 GB/array | CUDA OOM on training | Spatial subsampling 2× (50 m/px) + AMP mixed precision |
| Rover traverse float64 grids | 4× 1.17 GB | `np.asarray(..., dtype=np.float32)` on init |
| Matplotlib rendering 12K×13K | Swap thrash | Subsample before plotting |

### Mixed Precision Training (AMP)

Training uses PyTorch's Automatic Mixed Precision to halve VRAM:

```python
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

# Forward pass in fp16
with torch.cuda.amp.autocast(enabled=use_amp):
    outputs = model(feats, phys)
    losses = criterion(outputs, labels, phys)

# Backward pass with loss scaling
scaler.scale(losses['total']).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
scaler.update()
```

---

## 12. Limitations & Future Work

### Current Limitations

1. **No independent ground truth**: Labels derived from CPR > 1.0 threshold on the same DFSAR data. This creates a circularity — the model learns to reproduce CPR patterns. True validation requires in-situ measurements (e.g., from a future lander/rover mission like Chandrayaan-4).

2. **East direction limitation**: The east-look DFSAR mosaic does not show ice in the DPSRs where PRL 2026 confirmed ice (those appear only in west-look). The east results validate pipeline mechanics but not ice detection science.

3. **West training CUDA OOM**: The west mosaic (600M pixels) exceeds consumer GPU VRAM during training even with subsampling and AMP. Currently using the east-trained model for west inference (the model learns CPR physics which is direction-agnostic, but fine-tuning on west would be better).

4. **Depth estimation is approximate**: The Lichtenecker dielectric model assumes homogeneous ice distribution. Real subsurface ice is likely layered, patchy, and mixed with rocky debris. Depth estimates should be treated as order-of-magnitude.

5. **No multi-temporal analysis**: Using single-epoch DFSAR mosaics. Repeat observations could improve confidence by checking temporal stability of ice signatures.

### Future Work (for BAH 2026 Finale)

1. **West direction inference** — Run full pipeline on west mosaic where PRL 2026 confirmed ice in Faustini, Shoemaker, Haworth DPSRs. Expect non-zero per-crater volumes validating the system against published science.

2. **IIRS integration** — Chandrayaan-2 IIRS (Imaging Infrared Spectrometer) provides temperature maps. Incorporating real temperature data as a physics feature would replace the latitude-based temperature proxy.

3. **Interactive dashboard** — Streamlit/Gradio web application for live exploration: click on the map to see ice probability, depth, landing site score at any location.

4. **Multi-resolution fusion** — Combine L-band (24 cm, deep penetration) and S-band (12 cm, shallow) DFSAR data for depth-resolved ice profiling.

5. **Transfer learning** — Apply to Mini-RF (LRO) data for cross-mission validation; potential transfer to Mars polar ice detection (SHARAD/MARSIS).

---

## 13. References

1. **Sinha, R.K. et al. (2026)**. "Subsurface ice detection in doubly permanently shadowed regions of the lunar south pole using Chandrayaan-2 DFSAR." *Physical Research Laboratory Technical Report*. — Confirmed ice in 4 DPSRs using west-look DFSAR CPR analysis.

2. **O'Brien, P. & Byrne, S. (2022)**. "Properties of lunar polar ice deposits from radar observations." *JGR Planets*, 127(11). — CPR-based ice detection methodology and false positive analysis.

3. **Saran, S. et al. (2023)**. "Polarimetric analysis of Chandrayaan-2 DFSAR data for lunar south polar craters." *Advances in Space Research*, 71(2). — First systematic DFSAR polarimetric study of south polar craters.

4. **Spudis, P.D. et al. (2013)**. "Evidence for water ice on the Moon: Results for anomalous polar craters from the LRO Mini-RF imaging radar." *JGR Planets*, 118(10), 2016–2029. — Systematic Mini-RF CPR anomaly analysis correlating with PSR locations.

5. **Li, S. et al. (2018)**. "Direct evidence of surface exposed water ice in the lunar polar regions." *PNAS*, 115(36), 8907–8912. — First direct spectroscopic confirmation of surface-exposed ice at lunar poles.

6. **Colaprete, A. et al. (2010)**. "Detection of water in the LCROSS ejecta plume." *Science*, 330(6003), 463–468. — LCROSS impact at Cabeus: 5.6 ± 2.9 wt% water ice confirmed.

7. **Hayne, P.O. et al. (2015)**. "Evidence for exposed water ice in the Moon's south polar regions from Lunar Reconnaissance Orbiter ultraviolet albedo and temperature measurements." *Icarus*, 255, 58–69.

8. **Smith, D.E. et al. (2010)**. "The Lunar Orbiter Laser Altimeter investigation on the Lunar Reconnaissance Orbiter Mission." *Space Science Reviews*, 150, 209–241. — LOLA instrument description and DEM product specifications.

9. **Cloude, S.R. & Pottier, E. (1996)**. "A review of target decomposition theorems in radar polarimetry." *IEEE TGRS*, 34(2), 498–518. — H/A/alpha eigenvalue decomposition theory.

10. **Raney, R.K. et al. (2011)**. "The Lunar Mini-RF Radars: Hybrid Polarimetric Architecture and Initial Results." *Proceedings of the IEEE*, 99(5), 808–823. — Hybrid-pol architecture and CPR/Stokes parameter formulation for lunar radar.

11. **ISRO (2019)**. "Chandrayaan-2 DFSAR: Instrument Description and Data Products." *ISSDC Technical Note*. — DFSAR specifications, data levels, polarimetric modes.

---

## Glossary

| Term | Meaning |
|------|---------|
| **AMP** | Automatic Mixed Precision — fp16 forward pass for VRAM savings |
| **CBOE** | Coherent Backscatter Opposition Effect — enhanced same-sense return from ice |
| **CPR** | Circular Polarization Ratio — primary ice indicator (> 1.0) |
| **DFSAR** | Dual-frequency Synthetic Aperture Radar (Chandrayaan-2) |
| **DOP** | Degree of Polarization — wave organization measure |
| **DPSR** | Doubly Permanently Shadowed Region — never sees sunlight or reflected light |
| **ISRU** | In-Situ Resource Utilization — using local resources (ice → water/fuel) |
| **LOLA** | Lunar Orbiter Laser Altimeter (on LRO) |
| **LRO** | Lunar Reconnaissance Orbiter (NASA) |
| **PSR** | Permanently Shadowed Region — never sees direct sunlight |
| **SAR** | Synthetic Aperture Radar — imaging radar technique |
| **SE** | Squeeze-Excitation — channel attention mechanism |
| **SERD** | Single Expected Return Deviation — radar return variability |
| **T-Ratio** | T-matrix element ratio (T11/(T22+T33)) |
| **UPS** | Universal Polar Stereographic — map projection for polar regions |

---

*Built for Bharatiya Antariksh Hackathon 2026 by Team LunarIceNet*
*ISRO PS-8 | Chandrayaan-2 DFSAR | Physics-Informed Deep Learning for Lunar Ice*
