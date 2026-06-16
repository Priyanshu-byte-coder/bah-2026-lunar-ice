# IDEA PROPOSAL — Bharatiya Antariksh Hackathon 2026

## Problem Statement 8: Detection and Characterization of Subsurface Ice in Lunar South Polar Regions Using Chandrayaan-2 Radar and Imagery Data for Landing Site and Rover Traverse Planning

---

## Team Name: [TO BE FILLED]
## Team Members: [TO BE FILLED]

---

## 1. Problem Understanding

### 1.1 The Challenge

The Lunar South Polar Region harbors **Doubly Permanently Shadowed Regions (DPSRs)** — small craters nested within larger permanently shadowed craters, where rims additionally block scattered light and thermal emission from nearby illuminated surfaces. These are the coldest places in the inner solar system (~25K / -248°C), providing ideal environments for preserving subsurface water ice accumulated over billions of years. Detecting and characterizing this subsurface ice is critical for:

- **Future Chandrayaan mission landing site selection** — identifying safe, resource-rich locations
- **In-Situ Resource Utilization (ISRU)** — water ice can be converted to drinking water, oxygen, and hydrogen fuel for sustainable lunar exploration
- **Scientific understanding** — constraining models of volatile delivery and retention on airless bodies

### 1.2 Current State & Gap

India's **Chandrayaan-2 orbiter** carries the **Dual Frequency Synthetic Aperture Radar (DFSAR)** — the first full-polarimetric orbital SAR operating at both L-band (24 cm) and S-band (12 cm) wavelengths on the Moon. DFSAR provides unprecedented capability to probe the lunar subsurface up to ~2 meters depth.

Recent work by the **Physical Research Laboratory (PRL), Ahmedabad** (2026) detected possible subsurface ice in four doubly-shadowed craters including **Faustini** using classical polarimetric indicators:
- **CPR (Circular Polarization Ratio) > 1** — indicates volumetric scattering from subsurface structures
- **DOP (Degree of Polarization) < 0.13** — indicates depolarized returns consistent with ice-regolith mixtures

**The Gap**: Current detection methods rely on **manual thresholding** of individual polarimetric parameters. This approach:
- Cannot capture complex non-linear relationships between multiple radar signatures
- Provides binary yes/no detection without confidence quantification
- Does not integrate physical constraints (temperature, geometry, illumination)
- Cannot generalize across different crater morphologies
- Does not provide actionable landing site recommendations

### 1.3 Our Opportunity

We propose **LunarIceNet** — a **physics-informed deep learning system** that goes beyond classical thresholding to deliver a complete mission planning pipeline:
1. Continuous ice probability maps with uncertainty estimation in doubly shadowed craters
2. Subsurface ice depth and **volume estimation (0–5m)** using dielectric mixing models
3. Automated landing site scoring and ranking with terrain safety analysis
4. **Optimized rover traverse planning** from landing site to ice targets with hazard avoidance
5. Explainable predictions grounded in physical constraints

---

## 2. Proposed Solution

### 2.1 System Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                       LunarIceNet — Complete Mission Pipeline               │
│                                                                              │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐ │
│  │  STAGE 1     │   │  STAGE 2     │   │  STAGE 3     │   │  STAGE 4     │ │
│  │ Ice Detection│ → │Landing Site  │ → │Rover Traverse│ → │Ice Volume    │ │
│  │              │   │Selection     │   │Planning      │   │Estimation    │ │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘ │
│         │                  │                   │                  │          │
│  DFSAR L+S band     Multi-criteria      A* pathfinding     Dielectric      │
│  CPR/SERD/T-Ratio   scoring (ice,       terrain hazards    mixing models   │
│  OHRC imagery       slope, access,      solar power        Volume 0-5m    │
│  Physics priors     illumination)       rover constraints  Per-crater      │
│         │                  │                   │                  │          │
│         ▼                  ▼                   ▼                  ▼          │
│  Ice Probability     Ranked Sites      Optimal Path       Ice Volume      │
│  Depth Estimate      Safety Report     Waypoints           (m³ / kg)      │
│  Confidence Map      Coordinates       Energy Budget       Uncertainty     │
│                                                                              │
│                    Interactive 3D Dashboard + Mission Report                  │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Sources

| Source | Data | Purpose |
|--------|------|---------|
| **Chandrayaan-2 DFSAR** | L-band & S-band full-pol SAR (HH, HV, VH, VV) | Primary radar data for ice detection |
| **Chandrayaan-2 OHRC** | High-resolution camera (0.25m/pixel) | Crater morphology, boulder detection, surface roughness |
| **PRADAN Portal** (ISRO) | Processed DFSAR Level 3C/L4 mosaics (CPR, SERD, T-Ratio) | Pre-computed polarimetric products at 25m/pixel |
| **LRO LOLA** (NASA) | Digital Elevation Model (118m/pixel) | Terrain slope, rover traverse planning |
| **LRO Diviner** (NASA) | Surface temperature maps | Temperature constraints for ice stability |
| **DPSR Catalogs** | Doubly shadowed regions (O'Brien & Byrne 2022) | Focus targets & pseudo-labels |
| **PRL 2026 Paper** | 9 investigated DPSRs, 4 with CPR > 1 | Validation & ground truth |

### 2.3 Feature Engineering

From full-polarimetric DFSAR data (HH, HV, VH, VV), we extract **16 polarimetric features**:

**Primary Ice Indicators:**
- CPR (Circular Polarization Ratio) — volumetric scattering indicator
- DOP (Degree of Polarization) — depolarization indicator

**Decomposition Features:**
- m-chi decomposition: surface, double-bounce, volume scattering powers, m, χ
- Cloude-Pottier eigenvalue decomposition: Entropy (H), Anisotropy (A), Alpha (α)
- Shannon Entropy — information content of polarimetric response

**Scattering Features:**
- Pedestal Height — multiple scattering indicator
- HH/HV/VV backscatter intensities
- HH/VV co-pol ratio

**Physical Features (from external data):**
- Terrain slope and roughness (LOLA DEM)
- Estimated surface temperature (Diviner)
- Solar illumination geometry
- PSR membership probability

### 2.4 Model Architecture — LunarIceNet

LunarIceNet is a **multi-branch deep learning architecture** with three components:

#### Branch 1: Multi-Scale Radar Encoder
- Modified ResNet with Squeeze-Excitation attention
- Processes 16-channel polarimetric feature maps
- Extracts features at 4 spatial scales (capturing both fine texture and broad geological context)
- Multi-scale feature fusion via 1×1 convolutions + bilinear upsampling

#### Branch 2: Physics Encoder
- MLP network encoding physical parameters
- Inputs: latitude, PSR probability, distance from pole, temperature estimate, illumination
- Learns physics-aware representations that constrain the model

#### Branch 3: Cross-Attention Fusion
- Multi-head cross-attention between radar spatial features and physics embeddings
- Radar features attend to physical context → model learns to modulate ice predictions based on physical plausibility
- Self-attention among spatial positions → captures long-range geological dependencies
- 2-layer transformer with 8 attention heads

#### Output: Multi-Task Detection Head
- **Ice Probability Map** (per-pixel sigmoid) — where is ice?
- **Depth Estimate** (regression, 0-3m) — how deep is it?
- **Confidence Map** (0-1) — how certain is the model?

### 2.5 Physics-Informed Loss Function

Our custom loss combines four components:

```
L_total = λ₁·L_BCE + λ₂·L_depth + λ₃·L_physics + λ₄·L_temperature
```

| Component | Purpose | Weight |
|-----------|---------|--------|
| L_BCE | Binary cross-entropy for ice/no-ice classification | 1.0 |
| L_depth | MSE for depth estimation (where ice exists) | 0.5 |
| L_physics | Penalizes ice predictions far from south pole (temperature proxy) | 0.3 |
| L_temperature | Rewards high ice probability inside known PSRs | 0.2 |

**Key Insight**: The physics constraints prevent the model from making physically impossible predictions (e.g., ice in sunlit regions at T > 110K where ice sublimates).

### 2.6 Pseudo-Label Generation Strategy

Ground truth for lunar subsurface ice is extremely scarce. We address this with a multi-source pseudo-labeling strategy:

1. **PSR Catalog Labels**: Known permanently shadowed craters with suspected ice (Faustini, Shackleton, Cabeus, etc.) provide spatial priors
2. **Classical Indicator Labels**: Regions satisfying CPR > 1 AND DOP < 0.13 are labeled as positive candidates
3. **Thermal Model Labels**: Regions with modeled temperature < 110K in PSRs receive higher pseudo-label weight
4. **LCROSS Confirmation**: Cabeus crater (LCROSS 2009 impact site) provides one confirmed positive sample

### 2.7 Landing Site Scoring Module

Post-detection, we evaluate candidate landing sites using weighted multi-criteria scoring:

| Criterion | Weight | Source |
|-----------|--------|--------|
| Ice Probability | 35% | LunarIceNet output |
| Terrain Slope | 20% | LOLA DEM (< 15° safe landing) |
| Earth Accessibility | 15% | Orbital geometry + libration |
| Solar Illumination | 15% | Nearby peak illumination for power |
| Prediction Confidence | 15% | LunarIceNet uncertainty |

Output: Ranked list of optimal landing zones with composite scores and detailed breakdown.

### 2.8 Rover Traverse Planning Module

Once a landing site is selected, we plan an optimal rover path to ice targets:

**Algorithm**: A* pathfinding on the 25m/pixel terrain grid with custom cost function:

| Cost Component | Weight | Description |
|---------------|--------|-------------|
| Slope Penalty | 40% | Exponential cost for slopes >15°; impassable >25° |
| Distance Cost | 20% | Euclidean distance penalty |
| Ice Reward | -25% | Negative cost (attraction) toward high ice probability |
| Solar Access | -15% | Prefer paths near illuminated ridges for power |

**Rover Constraints**:
- Maximum traversable slope: 25°
- Maximum range: ~5 km from landing site
- Speed: ~100 m/hour (flat), decreasing with slope
- Solar power requirement: periodic access to illuminated terrain

**Outputs**:
- Optimal waypoint sequence from landing site to ice target
- Total traverse distance and estimated time
- Safety metrics (max slope, hazard count, energy budget)
- Ice sampling waypoints ranked by probability

### 2.9 Ice Volume Estimation Module

Quantitative ice volume estimation using radar backscatter and dielectric models:

**Dielectric Mixing Model** (Maxwell-Garnett approximation):
- Lunar regolith permittivity: ε_r ≈ 3.0
- Water ice permittivity: ε_ice ≈ 3.15
- Mixed permittivity depends on ice volume fraction (0–30%)

**Volume Calculation**:
1. Estimate ice fraction from CPR: CPR=1.0→~5%, CPR=1.5→~15%, CPR=2.0→~25%
2. Estimate penetration depth from radar wavelength: L-band (24cm) → ~2m, S-band (12cm) → ~1m
3. Per-pixel volume: pixel_area (625 m²) × depth × ice_fraction × ice_probability
4. Aggregate per crater and total with Monte Carlo uncertainty bounds

**Output**: Total ice volume (m³), mass (kg), per-crater breakdown, confidence intervals

---

## 3. Technical Feasibility

### 3.1 Technology Stack

| Component | Technology |
|-----------|-----------|
| ML Framework | PyTorch + PyTorch Lightning |
| SAR Processing | ISRO MIDAS, ESA SNAP |
| Geospatial | rasterio, geopandas, pyproj, GDAL |
| Path Planning | A* algorithm, NumPy, SciPy optimization |
| 3D Visualization | Plotly, CesiumJS, matplotlib |
| Dashboard | Streamlit |
| Compute | Google Colab Pro (NVIDIA T4/A100 GPU) |

### 3.2 Data Accessibility

- **DFSAR data**: Available via ISRO's PRADAN portal (pradan.issdc.gov.in) — free for Indian researchers
- **LOLA DEM**: Freely available from NASA PDS Geosciences Node
- **Diviner temperature**: Freely available from NASA PDS
- **Fallback**: If DFSAR access is delayed, NASA Mini-RF (LRO) provides similar S-band SAR data from PDS

### 3.3 Hackathon Timeline (30-Hour Finale)

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Data Loading & Feature Extraction | 4 hours | Preprocessed DFSAR + OHRC feature maps |
| Model Training & Ice Detection | 6 hours | Trained LunarIceNet, ice probability maps |
| Landing Site + Rover Traverse | 6 hours | Ranked sites, optimal rover paths |
| Ice Volume Estimation | 4 hours | Per-crater volume estimates with uncertainty |
| Dashboard & Visualization | 6 hours | Interactive 3D mission planning demo |
| Presentation Preparation | 4 hours | Final pitch + live demo |

### 3.4 Team Competencies

| Role | Skills |
|------|--------|
| ML Engineer | PyTorch, CNN/Transformer architectures, training pipelines |
| GIS/Remote Sensing | SAR processing, polarimetry, geospatial analysis |
| Full-Stack Developer | Streamlit, Plotly, API development, deployment |
| Research Lead | Paper reading, domain knowledge, validation strategy |

---

## 4. Innovation & Impact

### 4.1 What Makes This Novel

1. **First deep learning approach on DFSAR data for ice detection** — existing studies use only classical polarimetric thresholds
2. **Physics-informed architecture** — domain knowledge embedded in model design and loss function, not just applied as post-processing
3. **Multi-task learning** — simultaneously predicts ice presence, depth, and confidence rather than binary detection
4. **Complete mission planning pipeline** — from radar data to ice detection → landing site → rover traverse → ice volume — a single integrated system
5. **Doubly Shadowed Crater focus** — specifically targets DPSRs (coldest lunar locations, ~25K) rather than generic PSRs, aligned with PRL 2026 findings
6. **Quantitative ice volume estimation** — dielectric mixing models providing actionable resource estimates for ISRU planning

### 4.2 Impact on ISRO's Lunar Program

- **Chandrayaan-4/5 mission planning**: Directly supports landing site selection for India's next lunar missions
- **ISRU roadmap**: Quantified ice probability maps inform resource utilization planning
- **Scientific contribution**: New methodology for lunar volatiles research using Indian satellite data
- **Technology transfer**: Architecture applicable to other planetary SAR missions (Mars, Titan)

### 4.3 Comparison with Existing Approaches

| Approach | Method | Limitation | Our Improvement |
|----------|--------|-----------|----------------|
| PRL 2026 | CPR + DOP thresholds | Binary, no confidence | Continuous probability + uncertainty |
| Mini-RF studies | Single-frequency CPR | No depth estimation | Dual-frequency depth inference |
| General SAR ML | Transfer learning | No physics constraints | Physics-informed loss |
| Our: LunarIceNet | Multi-scale CNN + Physics + Cross-Attention | — | Integrated end-to-end system |

### 4.4 Future Extensions

- Integration with Chandrayaan-3 rover data for ground-truth validation
- Extension to Mars polar ice caps using SHARAD/MARSIS data
- Real-time inference for mission operations
- Open-source release for Indian space science community

---

## 5. References

1. Sinha, R.K., Bharti, R.R., Acharyya, K., Mishra, S.K., Srivastava, N. & Bhardwaj, A. (2026). "Subsurface ice in doubly shadowed craters as revealed by Chandrayaan-2 dual frequency synthetic aperture radar." *npj Space Exploration*, 2, Article 22.
2. O'Brien, R. & Byrne, S. (2022). "Doubly Shadowed Craters on the Moon." *Geophysical Research Letters*.
3. Saran, S. et al. (2023). "Chandrayaan-2 Dual Frequency SAR (DFSAR): Performance characterization and initial results." *Planetary and Space Science*.
4. Spudis, P.D. et al. (2013). "Evidence for water ice on the Moon: Results for anomalous polar craters from the LRO Mini-RF imaging radar." *JGR Planets*.
5. Li, S. et al. (2018). "Direct evidence of surface exposed water ice in the lunar polar regions." *PNAS*.
6. Colaprete, A. et al. (2010). "Detection of water in the LCROSS ejecta plume." *Science*.
7. Hayne, P.O. et al. (2015). "Evidence for exposed water ice in the Moon's south polar regions from LRO." *Icarus*.

---

*Document Version: 1.0 | Date: June 2026 | BAH 2026 — Problem Statement 8*
