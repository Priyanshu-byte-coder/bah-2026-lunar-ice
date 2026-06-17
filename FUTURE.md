# LunarIceNet — Remaining Work & Hackathon Timeline

## Hackathon Dates (BAH 2026)

> Official dates not published in repo. Based on document metadata and "June 2026" stamp in proposal.
> **ACTION REQUIRED**: Confirm exact dates at https://bah.isro.gov.in

Typical BAH structure (verify official portal):

| Round | Event | Est. Date |
|-------|-------|-----------|
| Round 1 | Idea submission | **DONE** (Jun 2026) |
| Round 2 | Prototype + code submission | ~Jul 2026 |
| Finale | 30-hour hackathon + presentation | ~Aug 2026 |

---

## Current Status

### Models Trained

| Model | F1 | Precision | Recall | Status |
|-------|----|-----------|--------|--------|
| East (embed=128, patch=64) | **0.8428** | 0.8906 | 0.7999 | `checkpoints/east_best_model.pth` ✓ |
| West (embed=128, patch=32) | **0.9383** | 0.9412 | 0.9353 | `checkpoints/west_best_model.pth` ✓ |

### Pipeline Runs

| Stage | East | West |
|-------|------|------|
| 1. Ice detection (inference) | ✓ (cached .npy) | ✗ NOT RUN |
| 2. Landing sites (real LOLA slopes) | ✓ 93.3% coverage | ✗ |
| 3. Rover traverse | ✗ FAILED (traverse fix applied, needs re-run) | ✗ |
| 4. Ice volume | ✓ global=8.4M m³, per-crater=0 (expected east) | ✗ |
| 5. Visualization + report | ✓ | ✗ |

---

## What's Left — Priority Order

### P0 — Critical (must do before submission)

#### 1. Run east pipeline with traverse fix
```bash
python full_pipeline.py \
  --use-cached \
  --direction east \
  --checkpoint checkpoints/east_best_model.pth \
  --lola-dem data/raw/lola_dem/ldem_85s_40m.img
```
- Fix applied: now targets Site #2 (5.3 km away) not Cabeus (100 km)
- Expected output: working rover traverse map + report
- Time: ~45 min

#### 2. Run west pipeline (MOST IMPORTANT — this is where PRL 2026 confirmed ice)
```bash
python full_pipeline.py \
  --direction west \
  --checkpoint checkpoints/west_best_model.pth \
  --lola-dem data/raw/lola_dem/ldem_85s_40m.img \
  --output-dir outputs/west
```
- West model F1=0.938, trained on 67K patches from 37M valid pixels
- West has 85,525 pixels with CPR>1 — real ice signal present
- Expect: non-zero per-crater volumes in Faustini, Shoemaker, Haworth DPSRs
- This validates the entire system against PRL 2026 published science
- Time: ~2-3 hours (inference on 8265×8061 grid)

#### 3. West cached pipeline (after inference completes)
```bash
python full_pipeline.py \
  --use-cached \
  --direction west \
  --checkpoint checkpoints/west_best_model.pth \
  --lola-dem data/raw/lola_dem/ldem_85s_40m.img \
  --output-dir outputs/west
```

### P1 — High Priority

#### 4. Fix per-crater ice volume (east)
East per-crater = 0. Two reasons:
- DPSRs in east-look don't show ice (scientifically correct per PRL 2026)
- But the crater-boundary lookup may also have coord mismatch issues

After west run, compare per-crater west volumes. If still 0, debug `estimate_volume_per_crater()` coordinate matching.

#### 5. Add LOLA slope to west outputs
West LOLA coverage will be same 93.3% (same DEM, same DFSAR mosaic size).
Landing sites for west should show REAL slopes too.

#### 6. Commit checkpoints metadata
```bash
git add checkpoints/training_history.json
git commit -m "feat: west model training complete F1=0.938"
```

### P2 — Should Do

#### 7. Compare east vs west ice maps side-by-side
Generate overlay plot: same craters, east CPR vs west CPR, show ice appears only in west.
This is the killer visual for judges — directly replicating PRL 2026 finding.

```python
# outputs/comparison_east_west.png
# Left panel: east ice_prob on Faustini
# Right panel: west ice_prob on Faustini
# Annotation: "Ice confirmed in West look (PRL 2026)"
```

#### 8. Per-crater volume table
Build clean output table for west pipeline:

| Crater | DPSR | Vol (m³) | Mass (Mt) | Ice frac | Depth (m) |
|--------|------|----------|-----------|----------|-----------|
| Faustini | DPSR-1 | ? | ? | ? | ? |
| Shoemaker | DPSR-1 | ? | ? | ? | ? |
| Haworth | DPSR-1 | ? | ? | ? | ? |

This is direct validation against PRL 2026 Table 1.

#### 9. Improve traverse visualization
Current: A* path on slope map.
Better: Overlay ice probability + slope + annotate sampling stops.

#### 10. Update training_history.json with west epochs
Current file has east metrics only. Append west training curve.

### P3 — Nice to Have

#### 11. Streamlit dashboard
```bash
streamlit run dashboard/app.py
```
Interactive: click map → see ice prob, depth, landing score at that location.
Demo-ready for judges.

#### 12. Comparison with PRL 2026 Table 1
Quantitative comparison of our CPR>1 detections vs Sinha et al. 4 DPSRs.

#### 13. Export GeoTIFF
Ice probability map as GeoTIFF with UPS projection — importable into QGIS/ArcGIS.
Shows system produces science-grade spatial data.

#### 14. Multi-resolution depth profile
Combine L-band (deep) + S-band (shallow) for depth-resolved ice stratigraphy.

---

## Commands Cheatsheet

```bash
# East full re-run (use cached inference)
python full_pipeline.py --use-cached --direction east \
  --checkpoint checkpoints/east_best_model.pth \
  --lola-dem data/raw/lola_dem/ldem_85s_40m.img

# West full run (PRIORITY — needs ~3 hrs)
python full_pipeline.py --direction west \
  --checkpoint checkpoints/west_best_model.pth \
  --lola-dem data/raw/lola_dem/ldem_85s_40m.img \
  --output-dir outputs/west

# West cached run (after inference .npy saved)
python full_pipeline.py --use-cached --direction west \
  --checkpoint checkpoints/west_best_model.pth \
  --lola-dem data/raw/lola_dem/ldem_85s_40m.img \
  --output-dir outputs/west

# Retrain east (if needed, ~2 hrs)
python train_real.py --direction east --epochs 30 --batch-size 16 --embed-dim 128

# Retrain west (~9.5 hrs, do overnight)
python train_real.py --direction west --epochs 30 --batch-size 4 \
  --patch-size 32 --stride 32 --subsample 3
```

---

## Key Files

| File | Purpose | Status |
|------|---------|--------|
| `full_pipeline.py` | 5-stage pipeline | ✓ (traverse fix applied) |
| `train_real.py` | Training on DFSAR | ✓ |
| `checkpoints/east_best_model.pth` | East model F1=0.843 | ✓ |
| `checkpoints/west_best_model.pth` | West model F1=0.938 | ✓ NEW |
| `outputs/ice_probability_east.npy` | East predictions cached | ✓ (600 MB) |
| `outputs/west/` | West outputs | ✗ EMPTY |
| `outputs/mission_report_east.txt` | East mission report | ✓ |
| `outputs/mission_report_west.txt` | West mission report | ✗ |

---

## What Winning Looks Like

1. **West per-crater ice volumes > 0** in Faustini DPSR — matches PRL 2026 published finding
2. **Working rover traverse** from landing site to nearby ice-rich target
3. **Side-by-side east/west comparison** showing ice appears only in west (PRL 2026 replication)
4. **Honest uncertainty** — Monte Carlo CI on ice volume, not fake precision
5. **Real terrain** — 93.3% LOLA DEM coverage for slopes, not synthetic proxy
6. **Complete pipeline** — single command produces all outputs

---

*Last updated: 2026-06-17*
