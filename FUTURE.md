# LunarIceNet — Roadmap & Hackathon Timeline

## Hackathon Dates (BAH 2026) — OFFICIAL

| Phase | Date | Status |
|-------|------|--------|
| **Idea Submission** | Jun 09 → **Jul 01, 11:59 PM IST** | LIVE — ~12 days left |
| **Evaluation** | Jul 02 → Jul 19 | Pending |
| **Shortlist Announced** | Jul 20 | Pending |
| **Induction Session** | Jul 21, 4-5 PM IST | Pending |
| **Finale (30-hr hack)** | **Aug 06, 8 AM** → **Aug 07, 7 PM IST** | Pending |
| **Final Submission** | Aug 06, 5 PM → **Aug 07, 9 AM IST** | Pending |

### What Phase 1 Requires (Idea Submission by Jul 1)
- **PPTX presentation** using official template (10 slides)
- Concept/idea proposal — NOT a working prototype
- Upload via Hack2skill portal
- We already have a working system = massive advantage

### What Phase 2 Requires (Finale Aug 6-7)
- 30-hour live coding hackathon
- Working prototype demonstrated to ISRO scientists
- Code + results + presentation

---

## Current Status (Updated 2026-06-19)

### Models

| Model | F1 | Precision | Recall | Checkpoint |
|-------|----|-----------|--------|------------|
| East (embed=128, attn=2) | **0.8428** | 0.8906 | 0.7999 | `checkpoints/best_model.pth` |
| West (embed=64, attn=1) | **0.9383** | 0.9412 | 0.9353 | `checkpoints/west_best_model.pth` |

### Pipeline Results

| Stage | East | West |
|-------|------|------|
| 1. Ice detection | **8.4M m³** (54.9M pixels) | **5.6M m³** (85M pixels) |
| 2. Landing sites | Top: -86.0°, -28.7° (score 0.728) | Top: -82.9°, 126.8° (score 0.771) |
| 3. Rover traverse | **60 wp, 1.76 km**, max slope 18.5° | **6 wp, 0.12 km**, max slope 5.0° |
| 4. Ice volume | 8.4M m³, CI [3.7M–14.6M] | 5.6M m³, CI [2.5M–9.7M] |
| 5. Reports + maps | All PNGs + TXT | All PNGs + TXT |
| LOLA DEM coverage | **93.3%** (real slopes) | 24.6% (75% default 5°) |

### Presentation
- `presentation/LunarIceNet_BAH2026_PS8.pptx` — filled with content + team
- Team: Priyanshu Doshi (leader), Shub Patel, Meer Patel

---

## Before Jul 1 — Idea Submission Checklist

- [x] East pipeline complete with real LOLA slopes
- [x] West pipeline complete (F1=0.938)
- [x] Both mission reports clean
- [x] Per-crater ice volume breakdown
- [x] README comprehensive
- [x] .gitignore clean
- [x] Presentation filled with content
- [ ] Presentation upgraded with visuals + result images
- [ ] East vs West comparison plot (replicates PRL 2026 finding)
- [ ] Upload to Hack2skill portal
- [ ] Verify GitHub repo is public and polished

---

## Finale Preparation (Aug 6-7) — What to Build

### P0 — Must Have

1. **Streamlit Dashboard**
   - Interactive map: click → see ice prob, depth, slope
   - Side-by-side east/west comparison
   - Real-time inference on cropped DFSAR tiles
   - Demo-ready for judges

2. **GeoTIFF Export**
   - Ice probability as GeoTIFF with UPS projection
   - Importable into QGIS/ArcGIS — science-grade output
   - Shows system produces real spatial data, not just PNGs

3. **PRL 2026 Validation Table**
   - Quantitative comparison: our CPR>1 detections vs Sinha et al. DPSRs
   - Table: crater, our ice_prob, published CPR, agreement Y/N
   - This is the scientific credibility killer

### P1 — Should Have

4. **Multi-Resolution Depth Profile**
   - Combine L-band (deep, λ=24cm) + S-band (shallow, λ=12cm)
   - Depth-resolved ice stratigraphy — novel contribution
   - Goes beyond what PRL 2026 published

5. **Improved West LOLA Coverage**
   - Download additional LOLA DEMs covering west DFSAR extent
   - Target: 80%+ real slope coverage (currently 24.6%)

6. **3D Terrain Visualization**
   - plotly 3D mesh of LOLA DEM + ice probability overlay
   - Interactive rotation — judges can explore landing sites
   - Wow factor for live demo

### P2 — Nice to Have

7. **Ensemble Model** — Average east+west predictions for robust ice map
8. **Uncertainty Calibration** — Temperature scaling on validation set
9. **Transfer to New Data** — Show model works on unseen DFSAR tiles
10. **Paper Draft** — arXiv-style writeup for credibility

---

## Commands Cheatsheet

```bash
# East (cached, ~15 min)
python full_pipeline.py --use-cached --direction east \
  --checkpoint checkpoints/best_model.pth \
  --lola-dem data/raw/lola_dem/ldem_85s_40m.img

# West (cached, ~20 min)
python full_pipeline.py --use-cached --direction west \
  --checkpoint checkpoints/west_best_model.pth \
  --lola-dem data/raw/lola_dem/ldem_85s_40m.img \
  --output-dir outputs/west

# Generate presentation
cd presentation && python fill_presentation.py
```

---

*Last updated: 2026-06-19*
