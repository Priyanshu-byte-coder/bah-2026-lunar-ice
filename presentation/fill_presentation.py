"""
Fill ISRO BAH 2026 Idea Submission Template with LunarIceNet content.
Run: python fill_presentation.py
Output: LunarIceNet_BAH2026_PS8.pptx
"""
from pptx import Presentation
from pptx.util import Pt, Emu
from pptx.dml.color import RGBColor
import copy

SRC = "[Pub] ISRO BAH 2026 _ Idea Submission Template.pptx"
OUT = "LunarIceNet_BAH2026_PS8.pptx"

# ── helpers ────────────────────────────────────────────────────────────────────

def set_textbox(slide, shape_id, lines, title_line=None, font_size_pt=None):
    """Replace text in a shape (by shape_id) while preserving background/layout."""
    for shape in slide.shapes:
        if shape.shape_id == shape_id:
            tf = shape.text_frame
            tf.word_wrap = True

            # Sample existing formatting from first run
            sample_font_size = None
            sample_font_name = "Google Sans"
            for para in tf.paragraphs:
                for run in para.runs:
                    if run.font.size:
                        sample_font_size = run.font.size
                    if run.font.name:
                        sample_font_name = run.font.name
                    break
                break

            use_size = Pt(font_size_pt) if font_size_pt else sample_font_size

            # Clear all paragraphs
            from pptx.oxml.ns import qn
            from lxml import etree
            txBody = tf._txBody
            for p in txBody.findall(qn('a:p')):
                txBody.remove(p)

            def add_para(text, bold=False, size_pt=None, color_rgb=None):
                p_elem = etree.SubElement(txBody, qn('a:p'))
                r_elem = etree.SubElement(p_elem, qn('a:r'))
                rPr = etree.SubElement(r_elem, qn('a:rPr'), lang='en-US', dirty='0')
                sz = size_pt if size_pt else (use_size // 12700 if use_size else 18)
                rPr.set('sz', str(int(sz * 100)))  # hundredths of a point
                if bold:
                    rPr.set('b', '1')
                if color_rgb:
                    solidFill = etree.SubElement(rPr, qn('a:solidFill'))
                    srgbClr = etree.SubElement(solidFill, qn('a:srgbClr'))
                    srgbClr.set('val', color_rgb)
                # font
                latin = etree.SubElement(rPr, qn('a:latin'))
                latin.set('typeface', sample_font_name)
                t_elem = etree.SubElement(r_elem, qn('a:t'))
                t_elem.text = text

            if title_line:
                add_para(title_line, bold=True, size_pt=22, color_rgb='FFFFFF')

            for line in lines:
                if line == "":
                    add_para("", size_pt=10)
                elif line.startswith("##"):
                    add_para(line[2:].strip(), bold=True, size_pt=16, color_rgb='FFD700')
                elif line.startswith("**") and line.endswith("**"):
                    add_para(line[2:-2], bold=True, size_pt=14)
                elif line.startswith("• "):
                    add_para(line, size_pt=13)
                else:
                    add_para(line, size_pt=13)
            return
    print(f"WARNING: shape_id {shape_id} not found on slide")


# ── content ────────────────────────────────────────────────────────────────────

prs = Presentation(SRC)

# ── SLIDE 1: Cover ─────────────────────────────────────────────────────────────
# shape_id 55 = Team Name, 56 = Problem Statement, 57 = Team Leader
slide1 = prs.slides[0]
for shape in slide1.shapes:
    if shape.shape_id == 55:
        tf = shape.text_frame
        for para in tf.paragraphs:
            for run in para.runs:
                run.text = "Team Name : LunarIceNet"
    elif shape.shape_id == 56:
        tf = shape.text_frame
        for para in tf.paragraphs:
            for run in para.runs:
                run.text = "Problem Statement : PS-8 — Subsurface Ice Detection in Lunar South Polar Regions"
    elif shape.shape_id == 57:
        tf = shape.text_frame
        for para in tf.paragraphs:
            for run in para.runs:
                run.text = "Team Leader Name : Priyanshu Doshi"

# ── SLIDE 2: Team Members ──────────────────────────────────────────────────────
# shape_id 64 = TABLE — leave as is (user fills manually)
# shape_id 63 = "Team Members" heading — keep

# ── SLIDE 3: Opportunity / USP ────────────────────────────────────────────────
set_textbox(prs.slides[2], 70, [
    "## OPPORTUNITY",
    "",
    "**Problem:**",
    "• Chandrayaan-2 DFSAR captured first full-polarimetric L-band SAR of lunar south pole",
    "• 55M+ radar pixels over -80° to -90° latitude — largest lunar SAR dataset ever",
    "• Classical thresholding (CPR > 1.0) produces too many false positives; no depth info",
    "• No automated pipeline exists: raw DFSAR → landing site → rover path → ice volume",
    "",
    "## HOW IT'S DIFFERENT",
    "",
    "• First deep learning system trained directly on Chandrayaan-2 DFSAR Level 3C mosaics",
    "• Physics-informed loss: penalizes predictions violating temperature/latitude constraints",
    "• Cross-attention fusion: radar features attend to physics context (lat, PSR membership)",
    "• Multi-task: ice probability + depth estimate + confidence in one forward pass",
    "• End-to-end: raw DFSAR → mission report in one command",
    "• Validated against PRL 2026 (Sinha et al.): west-look Faustini DPSR shows CPR > 1",
    "",
    "## USP",
    "",
    "• East model F1 = 0.843 | West model F1 = 0.938 on real DFSAR data",
    "• 93.3% real terrain coverage from NASA LOLA DEM (not synthetic slopes)",
    "• Monte Carlo ice volume uncertainty (1000 samples, Lichtenecker dielectric model)",
    "• A* rover path planning on 25 m/pixel grid with slope + solar + ice reward",
    "• Complete mission output: landing sites + traverse route + volume report in minutes",
])

# ── SLIDE 4: Features ─────────────────────────────────────────────────────────
set_textbox(prs.slides[3], 76, [
    "## CORE FEATURES",
    "",
    "**1. Subsurface Ice Detection (LunarIceNet CNN)**",
    "• 12.4M-parameter physics-informed neural network",
    "• Input: CPR, SERD, T-Ratio from Chandrayaan-2 DFSAR Level 3C",
    "• Multi-scale ResNet encoder + Squeeze-Excitation channel attention",
    "• Cross-attention fusion of radar features with physical priors",
    "• Output: per-pixel ice probability, depth (m), confidence maps",
    "",
    "**2. Real Terrain Slope Integration (LOLA DEM)**",
    "• Loads NASA LRO LOLA GDR binary DEMs (ldem_85s_40m, ldem_875s_20m)",
    "• 93.3% coverage of DFSAR grid at 40m resolution; inner pole at 20m",
    "• Slope computed via Gaussian-smoothed finite-difference gradient",
    "",
    "**3. Landing Site Scoring**",
    "• Weighted composite: 35% ice prob + 20% slope + 15% access + 15% illum + 15% conf",
    "• Top-10 candidates with real LOLA slopes — no synthetic proxies",
    "",
    "**4. Rover Traverse Planning (A*)**",
    "• 8-connected grid on 25m/pixel DFSAR extent",
    "• Cost = distance + slope_penalty + solar_penalty - ice_reward",
    "• Hard constraint: slope > 25° impassable; range budget 8 km",
    "",
    "**5. Ice Volume Estimation (Lichtenecker + Monte Carlo)**",
    "• CPR → ice fraction via Lichtenecker dielectric mixing model",
    "• Penetration depth: λ/(4π × Im(√ε_eff)), L-band λ=0.24 m",
    "• 1000-sample Monte Carlo → 90% confidence interval on total volume",
    "• Per-crater DPSR breakdown (16 known DPSRs in south polar region)",
])

# ── SLIDE 5: Process Flow ─────────────────────────────────────────────────────
set_textbox(prs.slides[4], 82, [
    "## PIPELINE PROCESS FLOW",
    "",
    "INPUT DATA",
    "• Chandrayaan-2 DFSAR Level 3C mosaics (east + west look) from ISRO PRADAN portal",
    "• NASA LRO LOLA GDR polar DEM (ldem_85s_40m.img, 110 MB, PDS3 binary)",
    "• Lunar PSR catalog (9 craters, 16 DPSRs at south pole)",
    "",
    "STAGE 1 — ICE DETECTION",
    "• Normalize CPR/SERD/T-Ratio → extract 64×64 patches → LunarIceNet inference",
    "• Output: ice_probability.npy, depth_estimate.npy, confidence.npy (600 MB each)",
    "",
    "STAGE 2 — LANDING SITE SELECTION",
    "• Load LOLA DEM → compute slope → regrid to DFSAR pixel coordinates",
    "• Score 4000+ candidate sites → rank top 10 by composite score",
    "",
    "STAGE 3 — ROVER TRAVERSE PLANNING",
    "• A* search: landing site → nearest ice-rich target within 8 km",
    "• Output: waypoint list, distance, time estimate, safety analysis",
    "",
    "STAGE 4 — ICE VOLUME ESTIMATION",
    "• CPR → ice fraction (Lichtenecker) → penetration depth → per-pixel volume",
    "• Monte Carlo uncertainty + per-DPSR crater breakdown",
    "",
    "STAGE 5 — VISUALIZATION & MISSION REPORT",
    "• Ice analysis maps, polar projection, rover traverse, LOLA DEM summary",
    "• Mission report: all key numbers in structured ASCII output",
    "",
    "OUTPUT: mission_report.txt + 4 PNG maps + landing_sites.txt + ice_volume.txt",
])

# ── SLIDE 6: Wireframes (optional) ───────────────────────────────────────────
set_textbox(prs.slides[5], 88, [
    "## OUTPUT VISUALIZATIONS",
    "",
    "**Panel 1 — Ice Detection Map (ice_analysis_east.png)**",
    "• 4-panel: ice probability | depth estimate | confidence | LOLA terrain slope",
    "• Custom colormap: transparent → blue → cyan → white (high ice probability)",
    "• Red zones = slope > 15° (unsafe for landing)",
    "",
    "**Panel 2 — Polar Ice Map (polar_ice_map_east.png)**",
    "• South polar stereographic projection centered at -90°",
    "• Ice probability overlaid on full DFSAR coverage footprint",
    "• Latitude rings at -82°, -84°, -86°, -88°, -90°",
    "• DPSR crater locations annotated",
    "",
    "**Panel 3 — Rover Traverse (rover_traverse_east.png)**",
    "• A* optimal path overlaid on slope map",
    "• Green = safe (<15°), yellow = caution (15-25°), red = impassable (>25°)",
    "• Landing site (star), target (diamond), ice sampling waypoints (circles)",
    "",
    "**Panel 4 — LOLA DEM Summary (lola_dem_summary.png)**",
    "• Left: elevation map (-5500 m to +7000 m range)",
    "• Right: slope map with contours at 15° and 25°",
    "• Coverage: -90° to -85° south polar region, 40 m/pixel",
])

# ── SLIDE 7: Architecture ─────────────────────────────────────────────────────
set_textbox(prs.slides[6], 94, [
    "## LUNARICENET ARCHITECTURE (12.4M PARAMETERS)",
    "",
    "INPUT: DFSAR patch (B, 3, 64, 64) + Physical params (B, 5)",
    "",
    "BRANCH 1 — Multi-Scale Radar Encoder",
    "• Stem: Conv7×7 → BN → ReLU → MaxPool  [→ B, 64, 16, 16]",
    "• Layer 1: 2× ResidualBlock(64→64)  with Squeeze-Excitation attention",
    "• Layer 2: 2× ResidualBlock(64→128, stride=2)",
    "• Layer 3: 2× ResidualBlock(128→256, stride=2)",
    "• Layer 4: 2× ResidualBlock(256→512, stride=2)",
    "• Feature Pyramid: all scales projected to embed_dim, fused → (B, 128, 16, 16)",
    "",
    "BRANCH 2 — Physics Encoder (MLP)",
    "• Inputs: [latitude, longitude, PSR_prob, dist_from_pole, PSR_flag]",
    "• Linear(5→64) → LayerNorm → Linear(64→128) → LayerNorm → Linear(128→128)",
    "• Output: (B, 128) physics embedding",
    "",
    "FUSION — Cross-Attention (2 layers × 4 heads)",
    "• Radar spatial tokens (B, H×W, 128) attend to physics vector (B, 1, 128)",
    "• Self-attention among spatial positions → FFN (GELU, 4× expansion)",
    "",
    "OUTPUT HEAD — 3× ConvTranspose2d decoder",
    "• Ice probability  : sigmoid → [0, 1] per pixel",
    "• Depth estimate   : ReLU → metres (non-negative)",
    "• Confidence score : sigmoid → [0, 1] uncertainty map",
    "",
    "LOSS = 1.0×BCE + 0.5×DepthMSE + 0.3×PhysicsPenalty + 0.2×TempPrior",
])

# ── SLIDE 8: Technologies ─────────────────────────────────────────────────────
set_textbox(prs.slides[7], 100, [
    "## TECHNOLOGIES & TOOLS",
    "",
    "**Deep Learning**",
    "• PyTorch 2.2 — model training, AMP mixed precision (fp16)",
    "• torchmetrics — F1, precision, recall, IoU tracking",
    "",
    "**Geospatial / Remote Sensing**",
    "• rasterio — GeoTIFF I/O, UPS projection handling",
    "• numpy — all array ops, chunked coordinate computation",
    "• scipy — Gaussian smoothing (LOLA slope), image filtering",
    "",
    "**Data Sources**",
    "• ISRO PRADAN portal — Chandrayaan-2 DFSAR Level 3C mosaics",
    "• NASA PDS Geosciences — LRO LOLA GDR polar DEMs",
    "",
    "**Mission Planning**",
    "• Custom A* implementation — 8-connected grid, composite cost function",
    "• Monte Carlo sampling — 1000-sample uncertainty quantification",
    "• Lichtenecker dielectric mixing model — CPR → ice fraction",
    "",
    "**Visualization**",
    "• matplotlib — polar stereographic maps, multi-panel figures",
    "• plotly — interactive 3D terrain + ice overlay",
    "",
    "**Infrastructure**",
    "• NVIDIA CUDA + cuDNN — GPU training (T4/RTX 3060)",
    "• Python 3.12, Git, GitHub for version control",
    "",
    "**Hardware Used**",
    "• GPU: NVIDIA (CUDA-enabled, 8+ GB VRAM for training)",
    "• RAM: 32 GB (600 MB arrays per DFSAR channel)",
    "• Storage: ~5 GB (data + checkpoints + outputs)",
])

# ── SLIDE 9: Cost ─────────────────────────────────────────────────────────────
set_textbox(prs.slides[8], 106, [
    "## ESTIMATED IMPLEMENTATION COST",
    "",
    "**Development (One-Time)**",
    "• Cloud GPU for training (30 epochs × 2 directions): ~8 hrs × ₹15/hr = ₹120",
    "• Or: personal GPU (RTX 3060) — electricity only ~₹50",
    "• Storage for DFSAR + LOLA data (~5 GB): negligible",
    "",
    "**Data (Free — Open Access)**",
    "• DFSAR Level 3C mosaics: ISRO PRADAN portal — FREE (registration required)",
    "• LOLA DEM: NASA PDS — FREE (public domain)",
    "• PSR catalog: published literature — FREE",
    "",
    "**Deployment (Per Mission Use)**",
    "• Inference on new DFSAR mosaic: ~2 hrs CPU or 15 min GPU",
    "• No internet required after data download — fully offline capable",
    "",
    "**Operational (For Chandrayaan-4 Integration)**",
    "• Integration with ISRO ground segment: engineering effort only",
    "• No proprietary software licenses required",
    "• All dependencies open-source (PyTorch, rasterio, numpy, scipy)",
    "",
    "**Total cost to reproduce full system: < ₹500**",
    "(excluding hardware already owned; data is free)",
])

# ── SAVE ──────────────────────────────────────────────────────────────────────
prs.save(OUT)
print(f"Saved: {OUT}")
print(f"Slides: {len(prs.slides)}")
print("NOTE: Fill in Slide 2 (Team Members table) manually in PowerPoint.")
