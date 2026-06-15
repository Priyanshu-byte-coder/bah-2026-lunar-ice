# LunarIceNet — AI-Powered Subsurface Ice Detection for Chandrayaan Missions

**BAH 2026 | Problem Statement 8 | Bharatiya Antariksh Hackathon by ISRO**

## Problem

Detecting and characterizing subsurface ice in the Lunar South Polar Region using Chandrayaan-2 DFSAR radar and imagery data for future landing site planning.

## Our Approach

Physics-informed deep learning on Chandrayaan-2 Dual-Frequency SAR (DFSAR) data to:
1. Extract polarimetric features (CPR, DOP, m-chi decomposition)
2. Detect subsurface ice deposits with spatial probability maps
3. Score candidate landing sites for future Chandrayaan missions

## Architecture

```
DFSAR L+S Band Data → Polarimetric Feature Extraction → LunarIceNet
                                                            ├── Radar Encoder (Multi-scale CNN)
Physical Parameters ──────────────────────────────────────→ ├── Physics Branch (MLP)
                                                            ├── Cross-Attention Fusion
                                                            └── Output: Ice Probability | Depth | Confidence
                                                                    ↓
                                                        Landing Site Scoring Module
                                                                    ↓
                                                        Interactive 3D Lunar Map
```

## Project Structure

```
├── src/
│   ├── data/           # Data loading & preprocessing
│   ├── features/       # Polarimetric feature extraction
│   ├── models/         # LunarIceNet architecture
│   ├── visualization/  # 3D maps & plotting
│   └── utils/          # Helpers
├── notebooks/          # Exploration & analysis
├── dashboard/          # Streamlit app
├── configs/            # YAML configurations
├── data/               # Raw & processed data
├── docs/               # Proposal & references
└── tests/              # Unit tests
```

## Tech Stack

- **ML**: PyTorch, PyTorch Lightning
- **SAR Processing**: ESA SNAP, rasterio
- **Geospatial**: geopandas, pyproj
- **Visualization**: Plotly 3D, CesiumJS, Streamlit
- **Compute**: Google Colab Pro (T4/A100)

## Key Innovation

Unlike classical polarimetric analysis (CPR thresholds), LunarIceNet uses physics-informed deep learning that:
- Learns complex non-linear relationships between radar signatures and ice presence
- Incorporates physical constraints (temperature, illumination, crater geometry)
- Provides uncertainty-aware predictions with confidence maps
- Generates actionable landing site recommendations

## Team

BAH 2026 Participant Team

## References

- Chandrayaan-2 DFSAR: ISRO PRADAN Portal
- PRL 2026: Subsurface ice detection in Faustini crater
- LRO LOLA: Lunar terrain data
