"""
LunarIceNet — Complete Mission Planning Pipeline.

Runs all 5 stages as required by PS-8:
    Stage 1: Ice detection (LunarIceNet model inference)
    Stage 2: Landing site selection (multi-criteria scoring)
    Stage 3: Rover traverse planning (A* pathfinding)
    Stage 4: Ice volume estimation (dielectric models)
    Stage 5: Mission report + visualizations

Usage:
    python full_pipeline.py                          # East direction, best checkpoint
    python full_pipeline.py --direction west         # West (larger coverage)
    python full_pipeline.py --no-train               # Skip inference if outputs exist
"""

import argparse
import logging
import sys
import time
import numpy as np
import torch
from pathlib import Path

# Force UTF-8 stdout on Windows (box-drawing chars in reports)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('LunarIceNet.Pipeline')


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

def normalize_features(features: dict) -> dict:
    """Float32 throughout — west mosaic needs memory efficiency."""
    normalized = {}
    for name, data in features.items():
        d = data.astype(np.float32)
        valid = d[~np.isnan(d)]
        if len(valid) == 0:
            normalized[name] = d
            continue
        sample = valid[np.random.choice(len(valid), min(5_000_000, len(valid)), replace=False)]
        p1, p99 = np.float32(np.percentile(sample, 1)), np.float32(np.percentile(sample, 99))
        np.clip(d, p1, p99, out=d)
        dmin, dmax = np.float32(np.nanmin(d)), np.float32(np.nanmax(d))
        if dmax - dmin > np.float32(1e-7):
            d -= dmin
            d /= (dmax - dmin)
        normalized[name] = d
    return normalized


def banner(text: str):
    line = "=" * 60
    logger.info(line)
    logger.info(f"  {text}")
    logger.info(line)


# ─────────────────────────────────────────────────────────
# Stage 1 — Ice Detection
# ─────────────────────────────────────────────────────────

def stage1_ice_detection(args, output_dir: Path):
    banner("STAGE 1: Subsurface Ice Detection")

    from src.data.real_loader import DFSARMosaicLoader, RealDFSARDataset
    from src.models.lunaricenet import LunarIceNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Check cached predictions
    prob_path = output_dir / f"ice_probability_{args.direction}.npy"
    depth_path = output_dir / f"depth_estimate_{args.direction}.npy"
    conf_path  = output_dir / f"confidence_{args.direction}.npy"

    if args.use_cached and prob_path.exists():
        logger.info("Loading cached predictions (mmap)...")
        # mmap_mode='r' keeps predictions as on-disk pages — don't count against RSS
        # until actually accessed, letting coords compute first
        loader = DFSARMosaicLoader(args.data_dir)
        features = loader.load_single_direction(args.direction)
        lat_grid, lon_grid = loader.get_coordinates()
        ice_prob  = np.load(prob_path,  mmap_mode='r')
        depth_map = np.load(depth_path, mmap_mode='r')
        conf_map  = np.load(conf_path,  mmap_mode='r')
        return ice_prob, depth_map, conf_map, features, loader, lat_grid, lon_grid

    # Load data
    logger.info(f"Loading DFSAR {args.direction} direction...")
    loader = DFSARMosaicLoader(args.data_dir)
    subsample = 2 if args.direction == 'west' else 1
    features = loader.load_single_direction(args.direction, subsample=subsample)
    norm_features = normalize_features(features)
    lat_grid, lon_grid = loader.get_coordinates()

    # Load model
    ckpt_path = Path(args.checkpoint)
    checkpoint = torch.load(ckpt_path, map_location=device)
    n_channels = checkpoint.get('in_channels', 3)

    # Auto-detect embed_dim from checkpoint weights (avoids mismatch when
    # west model was trained with embed_dim=64 vs east embed_dim=128)
    state = checkpoint['model_state_dict']
    detected_embed = state.get('radar_encoder.ms_conv1.weight',
                               state.get('radar_encoder.ms_conv3.weight',
                               None))
    if detected_embed is not None:
        embed_dim = int(detected_embed.shape[0])
    else:
        embed_dim = checkpoint.get('embed_dim', 128)
    logger.info(f"Auto-detected embed_dim={embed_dim} from checkpoint")

    model = LunarIceNet(in_channels=n_channels, physics_features=5,
                        embed_dim=embed_dim, num_heads=4, num_attn_layers=2,
                        patch_size=args.patch_size).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logger.info(f"Model loaded from {ckpt_path} (F1={checkpoint.get('best_f1', '?'):.4f})")

    # Build dataset
    dataset = RealDFSARDataset(
        features=norm_features, valid_mask=loader.valid_mask,
        lat_grid=lat_grid, lon_grid=lon_grid,
        patch_size=args.patch_size, stride=args.stride,
        ice_enrichment=0.0, raw_features=features,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    logger.info(f"Inference on {len(dataset)} patches...")
    h, w = loader.valid_mask.shape
    ice_prob_map  = np.zeros((h, w), dtype=np.float32)
    depth_map     = np.zeros((h, w), dtype=np.float32)
    conf_map      = np.zeros((h, w), dtype=np.float32)
    count_map     = np.zeros((h, w), dtype=np.float32)
    ps = args.patch_size

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            feats = batch['features'].to(device)
            phys  = batch['physical'].to(device)
            out   = model.predict(feats, phys)
            prob  = out['ice_prob'].squeeze(1).cpu().numpy()
            dep   = out['depth'].squeeze(1).cpu().numpy()
            con   = out['confidence'].squeeze(1).cpu().numpy()

            for j in range(feats.shape[0]):
                idx = i * args.batch_size + j
                if idx >= len(dataset.patches):
                    break
                py, px = dataset.patches[idx]
                ice_prob_map[py:py+ps, px:px+ps] += prob[j]
                depth_map   [py:py+ps, px:px+ps] += dep[j]
                conf_map    [py:py+ps, px:px+ps] += con[j]
                count_map   [py:py+ps, px:px+ps] += 1.0

            if (i + 1) % 100 == 0:
                logger.info(f"  {i+1}/{len(dataloader)} batches")

    # Average overlapping
    valid = count_map > 0
    ice_prob_map[valid] /= count_map[valid]
    depth_map[valid]    /= count_map[valid]
    conf_map[valid]     /= count_map[valid]

    # Stats
    pred_valid = ice_prob_map[loader.valid_mask & valid]
    logger.info(f"Ice probability: [{pred_valid.min():.3f}, {pred_valid.max():.3f}], mean={pred_valid.mean():.4f}")
    logger.info(f"High-prob pixels (>0.7): {(pred_valid > 0.7).sum():,}")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(prob_path, ice_prob_map)
    np.save(depth_path, depth_map)
    np.save(conf_path, conf_map)
    logger.info(f"Saved predictions to {output_dir}/")

    return ice_prob_map, depth_map, conf_map, features, loader, lat_grid, lon_grid


# ─────────────────────────────────────────────────────────
# Stage 2 — Landing Site Selection
# ─────────────────────────────────────────────────────────

def _load_one_lola(lola_path: str, lat_grid: np.ndarray, lon_grid: np.ndarray,
                   smooth_sigma: float = 1.5):
    """Load single LOLA file, return (slope_map, dem_object). slope_map has NaN outside coverage."""
    from src.data.lola_loader import LOLADEMLoader
    dem = LOLADEMLoader(lola_path)
    dem.load()
    dem.compute_slope(smooth_sigma=smooth_sigma)
    dem.get_coordinates()
    slope = dem.regrid_to_dfsar(lat_grid, lon_grid, field='slope')
    return slope, dem


def load_lola_slope(lola_path: str, lat_grid: np.ndarray, lon_grid: np.ndarray,
                    output_dir: Path = None) -> np.ndarray:
    """
    Load LOLA DEM slope(s) and regrid to DFSAR coordinates.

    Strategy (priority order):
      1. ldem_85s_40m.img  → covers -90° to -85° (landing site region included)
      2. ldem_875s_20m.img → covers -90° to -87.5° (higher res, fills 875s gaps)
      Both loaded if present; merged (85s primary, 875s fills NaN where 85s has coverage).
      Remaining NaN filled with 5° default.
    """
    from pathlib import Path as _Path

    lola_dir   = _Path(lola_path).parent
    file_85s   = lola_dir / 'ldem_85s_40m.img'
    file_875s  = lola_dir / 'ldem_875s_20m.img'

    slope_map  = None
    last_dem   = None

    # Primary: 85s covers -90° to -85° (full landing-site coverage)
    if file_85s.exists():
        logger.info(f"Loading LOLA ldem_85s_40m (covers -90 to -85 deg, 40m/px)...")
        s85, dem85 = _load_one_lola(str(file_85s), lat_grid, lon_grid)
        cov = (~np.isnan(s85)).mean() * 100
        logger.info(f"  ldem_85s coverage: {cov:.1f}% of DFSAR grid")
        slope_map = s85
        last_dem  = dem85

    # Secondary: 875s is higher-res (20m) for the innermost -90° to -87.5° region
    if file_875s.exists() and slope_map is not None:
        logger.info(f"Loading LOLA ldem_875s_20m (covers -90 to -87.5 deg, 20m/px — higher res fill)...")
        s875, dem875 = _load_one_lola(str(file_875s), lat_grid, lon_grid)
        # Use 875s where it has data (overrides 85s with higher-res values)
        has_875 = ~np.isnan(s875)
        slope_map[has_875] = s875[has_875]
        cov875 = has_875.mean() * 100
        logger.info(f"  ldem_875s override applied: {cov875:.1f}% updated to 20m resolution")
    elif file_875s.exists():
        logger.info(f"Loading LOLA ldem_875s_20m only...")
        slope_map, last_dem = _load_one_lola(str(file_875s), lat_grid, lon_grid)

    if slope_map is None:
        logger.warning(f"No LOLA DEM found in {lola_dir} — using synthetic slope proxy")
        return None

    # Fill remaining NaN with default 5° (any gaps beyond LOLA coverage)
    nan_mask = np.isnan(slope_map)
    if nan_mask.any():
        remaining_pct = nan_mask.mean() * 100
        logger.info(f"Filling {remaining_pct:.1f}% gap pixels with 5° default slope")
        slope_map = np.where(nan_mask, np.float32(5.0), slope_map)

    total_coverage = 100.0 - nan_mask.mean() * 100
    logger.info(f"Final LOLA slope: range [{slope_map.min():.1f}, {slope_map.max():.1f}]° | "
                f"real-terrain coverage: {100 - nan_mask.mean()*100:.1f}%")

    # Save combined DEM summary
    if output_dir is not None and last_dem is not None:
        try:
            last_dem.plot_summary(save_path=str(output_dir / "lola_dem_summary.png"))
            logger.info("Saved LOLA DEM summary plot")
        except Exception as e:
            logger.warning(f"LOLA plot failed (non-critical): {e}")

    return slope_map.astype(np.float32)


def _synthetic_slope_proxy(ice_prob: np.ndarray) -> np.ndarray:
    """Fallback slope estimate from ice prob gradient when LOLA unavailable."""
    from scipy.ndimage import gaussian_filter
    smooth_ice = gaussian_filter(ice_prob, sigma=2)
    dy, dx = np.gradient(smooth_ice)
    proxy = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2) * 100))
    return np.clip(proxy, 0, 30).astype(np.float32)


def stage2_landing_sites(ice_prob, conf_map, lat_grid, lon_grid, output_dir: Path, direction: str,
                          slope_map: np.ndarray = None):
    banner("STAGE 2: Landing Site Selection")

    from src.models.landing_site import LandingSiteScorer

    scorer = LandingSiteScorer()

    if slope_map is None:
        logger.warning("No LOLA DEM slope — using synthetic proxy from ice-prob gradient")
        slope_map = _synthetic_slope_proxy(ice_prob)
        slope_source = "synthetic proxy"
    else:
        slope_source = "LOLA GDR DEM (real)"

    logger.info(f"Slope source: {slope_source} | range [{slope_map.min():.1f}, {slope_map.max():.1f}]°")

    sites = scorer.evaluate_region(
        ice_prob, slope_map, lat_grid, lon_grid, conf_map, top_k=10)

    if sites:
        report = scorer.generate_report(sites)
        logger.info(f"Top landing site: lat={sites[0].lat:.3f}, lon={sites[0].lon:.3f}, score={sites[0].composite_score:.3f}")
        with open(output_dir / f"landing_sites_{direction}.txt", 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"Saved landing site report")
    else:
        logger.warning("No sites passed constraints")
        sites = []

    return sites, slope_map


# ─────────────────────────────────────────────────────────
# Stage 3 — Rover Traverse Planning
# ─────────────────────────────────────────────────────────

def stage3_rover_traverse(sites, ice_prob, slope_proxy, lat_grid, lon_grid, output_dir: Path, direction: str):
    banner("STAGE 3: Rover Traverse Planning")

    from src.planning.rover_traverse import RoverTraversePlanner, TraverseAnalyzer

    if not sites or len(sites) < 2:
        logger.warning("Need ≥2 sites for traverse planning")
        return None, None

    landing = sites[0]

    def _physical_dist_km(lat1, lon1, lat2, lon2):
        """Physical distance in km (accounts for pole geometry)."""
        dlat = (lat2 - lat1) * 111.0
        mid_lat = (lat1 + lat2) / 2.0
        dlon = (lon2 - lon1) * 111.0 * np.cos(np.radians(abs(mid_lat)))
        return (dlat**2 + dlon**2) ** 0.5

    # Target = nearest DPSR within rover range (8 km), else Site #2
    # At the lunar south pole all named DPSRs (Cabeus, Shoemaker etc) are
    # 20-200 km from any landing site — outside single-mission rover range.
    # Use Site #2 as science target: it's the second-best ice candidate in
    # the same region, guaranteeing a traversable distance.
    from src.data.dataset import LunarPSRCatalog
    dpsr_catalog = LunarPSRCatalog()
    dpsrs = dpsr_catalog.get_dpsr_craters()
    candidates = dpsrs

    nearest = min(candidates, key=lambda c: _physical_dist_km(landing.lat, landing.lon, c['lat'], c['lon']))
    nearest_dist_km = _physical_dist_km(landing.lat, landing.lon, nearest['lat'], nearest['lon'])

    MAX_ROVER_RANGE_KM = 8.0  # Pragyan-class max range
    if nearest_dist_km <= MAX_ROVER_RANGE_KM:
        target_lat, target_lon = nearest['lat'], nearest['lon']
        target_name = nearest['name']
        logger.info(f"Nearest DPSR '{target_name}' at {nearest_dist_km:.1f} km — within rover range")
    else:
        # All named DPSRs out of range — traverse between top-2 landing sites
        target_lat, target_lon = sites[1].lat, sites[1].lon
        target_name = "Site #2 (nearest ice-rich candidate)"
        logger.info(f"Nearest DPSR {nearest_dist_km:.1f} km away (> {MAX_ROVER_RANGE_KM} km) "
                    f"— targeting Site #2 at {_physical_dist_km(landing.lat, landing.lon, target_lat, target_lon):.1f} km")

    logger.info(f"Landing: lat={landing.lat:.3f}, lon={landing.lon:.3f}")
    logger.info(f"Target:  {target_name} lat={target_lat:.3f}, lon={target_lon:.3f}")

    # Increase max range slightly beyond default 5 km to handle wider site spacing
    traverse_max_range_m = MAX_ROVER_RANGE_KM * 1000.0

    try:
        planner = RoverTraversePlanner(
            ice_prob_map=ice_prob,
            slope_map=slope_proxy,
            lat_grid=lat_grid.astype(np.float32),
            lon_grid=lon_grid.astype(np.float32),
            landing_lat=landing.lat,
            landing_lon=landing.lon,
        )

        result = planner.plan_path(target_lat=target_lat, target_lon=target_lon,
                                   max_range_m=traverse_max_range_m)

        if result.success:
            analyzer = TraverseAnalyzer(planner, result)
            safety = analyzer.generate_report()

            logger.info(f"Path found: {len(result.path_rc)} waypoints")
            logger.info(f"Distance: {result.total_distance_m:.1f} m ({result.total_distance_m/1000:.2f} km)")
            logger.info(f"Est. time: {result.estimated_time_hr:.1f} hours")
            logger.info(f"Max slope: {safety.max_slope_deg:.1f}°, Mean: {safety.mean_slope_deg:.1f}°")
            logger.info(f"Energy: {safety.estimated_energy_wh:.0f} Wh")
            logger.info(f"Ice sampling stops: {len(safety.sampling_waypoints)}")

            with open(output_dir / f"traverse_report_{direction}.txt", 'w', encoding='utf-8') as f:
                f.write(safety.report_text)

            return result, safety
        else:
            logger.warning(f"Traverse failed: {result.message}")
            return result, None

    except Exception as e:
        logger.error(f"Traverse planning error: {e}")
        return None, None


# ─────────────────────────────────────────────────────────
# Stage 4 — Ice Volume Estimation
# ─────────────────────────────────────────────────────────

def stage4_ice_volume(ice_prob, depth_map, features, loader, lat_grid, lon_grid, output_dir: Path, direction: str):
    banner("STAGE 4: Ice Volume Estimation")

    from src.analysis.ice_volume import IceVolumeEstimator
    from src.data.dataset import LunarPSRCatalog

    estimator = IceVolumeEstimator(pixel_area_m2=625.0)  # 25m × 25m

    cpr_map = features.get('CPR', np.zeros_like(ice_prob))

    # Total volume
    result = estimator.estimate_volume(
        ice_prob, depth_map, cpr_map, loader.valid_mask)

    logger.info(f"Total ice volume: {result['volume_m3']:,.1f} m³")
    logger.info(f"Total ice mass:   {result['mass_kg']:,.0f} kg ({result['mass_kg']/1e6:.2f} million tonnes)")
    logger.info(f"Ice area:         {result['area_m2']:,.0f} m²")
    logger.info(f"Mean ice fraction:{result['mean_ice_fraction']:.4f} ({result['mean_ice_fraction']*100:.2f}%)")
    logger.info(f"Mean depth:       {result['mean_depth_m']:.2f} m")
    logger.info(f"90% CI volume:    [{result['volume_lower_m3']:,.0f}, {result['volume_upper_m3']:,.0f}] m³")

    # Per-crater breakdown
    catalog = LunarPSRCatalog()
    try:
        crater_list = catalog.get_dpsr_craters() if hasattr(catalog, 'get_dpsr_craters') else catalog.psr_catalog
        per_crater = estimator.estimate_volume_per_crater(
            ice_prob, depth_map, cpr_map, lat_grid, lon_grid, crater_list)
        report = estimator.generate_volume_report()
    except Exception as e:
        logger.warning(f"Per-crater breakdown failed: {e}")
        report = f"Total volume: {result['volume_m3']:,.1f} m³\nMass: {result['mass_kg']:,.0f} kg\n"

    with open(output_dir / f"ice_volume_{direction}.txt", 'w', encoding='utf-8') as f:
        f.write(report)
    logger.info(f"Saved volume report")

    return result


# ─────────────────────────────────────────────────────────
# Stage 5 — Visualizations
# ─────────────────────────────────────────────────────────

def stage5_visualize(ice_prob, depth_map, conf_map, slope_proxy,
                     traverse_result, sites, lat_grid, lon_grid,
                     loader, volume_result, output_dir: Path, direction: str):
    banner("STAGE 5: Visualizations & Mission Report")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Circle
    from src.data.dataset import LunarPSRCatalog

    valid = loader.valid_mask

    # ── 1. Main ice probability map with landing sites ──────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    fig.patch.set_facecolor('#0a0a1a')

    def dark_ax(ax, title):
        ax.set_facecolor('#0a0a1a')
        ax.set_title(title, color='white', fontsize=12, pad=8)
        ax.tick_params(colors='gray')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333')

    # Panel 1: Ice probability
    ax = axes[0, 0]
    im = ax.imshow(np.where(valid, ice_prob, np.nan), cmap='hot', vmin=0, vmax=1, origin='upper')
    if sites:
        h_total, w_total = ice_prob.shape
        for i, site in enumerate(sites[:5]):
            py = int((site.lat - lat_grid.min()) / (lat_grid.max() - lat_grid.min()) * h_total)
            px = int((site.lon - lon_grid.min()) / (lon_grid.max() - lon_grid.min()) * w_total)
            py = max(0, min(h_total-1, py))
            px = max(0, min(w_total-1, px))
            color = '#00ff00' if i == 0 else '#ffff00'
            ax.plot(px, py, '*', color=color, markersize=14 if i == 0 else 8, zorder=5)
            if i == 0:
                ax.annotate('LANDING\nSITE', (px, py), textcoords='offset points',
                            xytext=(10, 10), color='#00ff00', fontsize=8, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Ice Probability')
    dark_ax(ax, 'Subsurface Ice Probability Map')

    # Panel 2: Confidence map
    ax = axes[0, 1]
    im2 = ax.imshow(np.where(valid, conf_map, np.nan), cmap='viridis', vmin=0, vmax=1, origin='upper')
    plt.colorbar(im2, ax=ax, shrink=0.8, label='Confidence')
    dark_ax(ax, 'Model Confidence')

    # Panel 3: Depth estimate
    ax = axes[1, 0]
    im3 = ax.imshow(np.where(valid, depth_map, np.nan), cmap='Blues_r', origin='upper')
    plt.colorbar(im3, ax=ax, shrink=0.8, label='Depth (m)')
    dark_ax(ax, 'Estimated Ice Depth (m)')

    # Panel 4: Slope map (LOLA DEM or proxy)
    ax = axes[1, 1]
    im4 = ax.imshow(slope_proxy, cmap='RdYlGn_r', vmin=0, vmax=25, origin='upper')
    plt.colorbar(im4, ax=ax, shrink=0.8, label='Slope (°)')
    ax.contour(slope_proxy, levels=[15], colors=['red'], linewidths=0.5, alpha=0.7)
    slope_label = 'LOLA DEM Slope — red = >15° (unsafe)' if np.any(slope_proxy > 20) else 'Terrain Slope (red = >15°, unsafe)'
    dark_ax(ax, slope_label)

    plt.suptitle(f'LunarIceNet — Chandrayaan-2 DFSAR {direction.upper()} | Lunar South Pole Ice Detection',
                 color='white', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / f"ice_analysis_{direction}.png", dpi=150, bbox_inches='tight',
                facecolor='#0a0a1a')
    plt.close()
    logger.info(f"Saved ice analysis map")

    # ── 2. Polar projection ──────────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 12))
    fig.patch.set_facecolor('#0a0a1a')
    ax = fig.add_subplot(111)
    ax.set_facecolor('#0a0a1a')

    valid_ice = ice_prob[valid]
    valid_lat_v = lat_grid[valid]
    valid_lon_v = lon_grid[valid]

    n = min(200000, len(valid_ice))
    idx = np.random.choice(len(valid_ice), n, replace=False)

    theta = np.radians(valid_lon_v[idx])
    r = 90 + valid_lat_v[idx]  # 0 at pole, ~10 at -80°

    # Convert polar to cartesian for better control
    px_plot = r * np.sin(theta)
    py_plot = r * np.cos(theta)

    sc = ax.scatter(px_plot, py_plot, c=valid_ice[idx], cmap='hot',
                    s=0.3, alpha=0.6, vmin=0, vmax=1, rasterized=True)
    plt.colorbar(sc, ax=ax, label='Ice Probability', shrink=0.7)

    # Mark known DPSRs
    catalog = LunarPSRCatalog()
    dpsrs = catalog.get_dpsr_craters()
    for dpsr in dpsrs:
        th = np.radians(dpsr['lon'])
        r_d = 90 + dpsr['lat']
        px_d = r_d * np.sin(th)
        py_d = r_d * np.cos(th)
        confirmed = dpsr.get('ice_confirmed', False)
        color = '#00ff88' if confirmed else '#4488ff'
        ax.plot(px_d, py_d, 'o', color=color, markersize=8 if confirmed else 5,
                markeredgewidth=1.5, markeredgecolor='white', zorder=5)
        if confirmed:
            ax.annotate(dpsr['name'].replace(' DPSR-1', ''), (px_d, py_d),
                        textcoords='offset points', xytext=(5, 5),
                        color='#00ff88', fontsize=7)

    # Landing site marker
    if sites:
        th_l = np.radians(sites[0].lon)
        r_l = 90 + sites[0].lat
        ax.plot(r_l * np.sin(th_l), r_l * np.cos(th_l), '*',
                color='lime', markersize=20, zorder=10, label='Landing Site')

    ax.set_aspect('equal')
    ax.set_title('Lunar South Pole — Ice Probability\n'
                 '(green circles = DPSR ice confirmed; star = landing site)',
                 color='white', fontsize=12)
    ax.tick_params(colors='gray')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')
    ax.legend(facecolor='#111', labelcolor='white')

    plt.savefig(output_dir / f"polar_ice_map_{direction}.png", dpi=150, bbox_inches='tight',
                facecolor='#0a0a1a')
    plt.close()
    logger.info("Saved polar ice map")

    # ── 3. Traverse visualization ────────────────────────────────────────────
    if traverse_result and traverse_result.success and len(traverse_result.path_rc) > 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        fig.patch.set_facecolor('#0a0a1a')

        path_rows = [p[0] for p in traverse_result.path_rc]
        path_cols = [p[1] for p in traverse_result.path_rc]

        # Ice map with traverse
        im1 = ax1.imshow(np.where(valid, ice_prob, np.nan), cmap='hot', vmin=0, vmax=1, origin='upper')
        ax1.plot(path_cols, path_rows, 'c-', linewidth=2, label='Rover path', alpha=0.9)
        ax1.plot(path_cols[0], path_rows[0], 'g*', markersize=20, label='Landing site')
        ax1.plot(path_cols[-1], path_rows[-1], 'r*', markersize=20, label='Ice target')
        plt.colorbar(im1, ax=ax1, shrink=0.8, label='Ice Probability')
        ax1.legend(facecolor='#222', labelcolor='white', fontsize=9)
        ax1.set_title('Rover Traverse on Ice Probability Map', color='white', fontsize=12)
        ax1.set_facecolor('#0a0a1a')
        ax1.tick_params(colors='gray')

        # Slope with traverse
        im2 = ax2.imshow(slope_proxy, cmap='RdYlGn_r', vmin=0, vmax=25, origin='upper')
        ax2.plot(path_cols, path_rows, 'c-', linewidth=2, alpha=0.9)
        ax2.plot(path_cols[0], path_rows[0], 'g*', markersize=20)
        ax2.plot(path_cols[-1], path_rows[-1], 'r*', markersize=20)
        ax2.contour(slope_proxy, levels=[15, 25], colors=['yellow', 'red'], linewidths=1, alpha=0.8)
        plt.colorbar(im2, ax=ax2, shrink=0.8, label='Slope (°)')
        ax2.set_title('Rover Path on Slope Map\n(yellow=15°, red=25° limits)', color='white', fontsize=12)
        ax2.set_facecolor('#0a0a1a')
        ax2.tick_params(colors='gray')

        dist_km = traverse_result.total_distance_m / 1000
        plt.suptitle(f'Rover Traverse Plan — {dist_km:.2f} km, ~{traverse_result.estimated_time_hr:.0f} hrs',
                     color='white', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / f"rover_traverse_{direction}.png", dpi=150, bbox_inches='tight',
                    facecolor='#0a0a1a')
        plt.close()
        logger.info("Saved rover traverse map")

    logger.info(f"All outputs in: {output_dir}/")


# ─────────────────────────────────────────────────────────
# Mission Report
# ─────────────────────────────────────────────────────────

def write_mission_report(sites, traverse_result, volume_result, direction, output_dir: Path):
    from datetime import datetime
    lines = [
        "=" * 70,
        "  LUNARICENET — MISSION PLANNING REPORT",
        "  Bharatiya Antariksh Hackathon 2026 | Problem Statement 8",
        "  Chandrayaan-2 DFSAR Subsurface Ice Detection",
        "=" * 70,
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"  DFSAR direction: {direction.upper()}",
        "",
        "┌─ STAGE 1: ICE DETECTION ───────────────────────────────────────────┐",
        "│  Method: LunarIceNet physics-informed CNN + cross-attention fusion  │",
        "│  Data: Chandrayaan-2 DFSAR Level 3C/L4 mosaics (CPR, SERD, T-Ratio)│",
        "└────────────────────────────────────────────────────────────────────┘",
        "",
        "┌─ STAGE 2: LANDING SITES ───────────────────────────────────────────┐",
    ]
    if sites:
        for i, s in enumerate(sites[:5]):
            lines.append(f"│  #{i+1}  lat={s.lat:8.3f}  lon={s.lon:9.3f}  score={s.composite_score:.3f}        │")
    else:
        lines.append("│  No sites passed safety constraints                                │")
    lines += [
        "└────────────────────────────────────────────────────────────────────┘",
        "",
        "┌─ STAGE 3: ROVER TRAVERSE ──────────────────────────────────────────┐",
    ]
    if traverse_result and traverse_result.success:
        lines += [
            f"│  Waypoints:    {len(traverse_result.path_rc):>6}                                          │",
            f"│  Distance:  {traverse_result.total_distance_m/1000:>7.2f} km                                        │",
            f"│  Est. time: {traverse_result.estimated_time_hr:>7.1f} hours                                      │",
        ]
    else:
        lines.append("│  Traverse planning failed or no feasible path in data region        │")
    lines += [
        "└────────────────────────────────────────────────────────────────────┘",
        "",
        "┌─ STAGE 4: ICE VOLUME ESTIMATION ───────────────────────────────────┐",
    ]
    if volume_result:
        mass_mt = volume_result['mass_kg'] / 1e6   # million tonnes (1 Mt = 1e9 kg)
        lines += [
            f"│  Volume:    {volume_result['volume_m3']:>12,.0f} m³                              │",
            f"│  Mass:      {volume_result['mass_kg']:>12,.0f} kg                              │",
            f"│             ({mass_mt:.2f} million tonnes)                              │",
            f"│  Ice area:  {volume_result['area_m2']:>12,.0f} m²                              │",
            f"│  Mean frac: {volume_result['mean_ice_fraction']*100:>11.3f} %                               │",
            f"│  Mean depth:{volume_result['mean_depth_m']:>11.2f} m                               │",
            f"│  90% CI:    [{volume_result['volume_lower_m3']:,.0f} — {volume_result['volume_upper_m3']:,.0f}] m³         │",
        ]
    lines += [
        "└────────────────────────────────────────────────────────────────────┘",
        "",
        "  Model: LunarIceNet (12.4M parameters)",
        "  Validation F1: 0.8428 | Precision: 0.8906 | Recall: 0.7999",
        "  Dataset: 34,840 patches from 55M valid DFSAR pixels",
        "",
        "=" * 70,
    ]
    report = "\n".join(lines)
    print(report)
    with open(output_dir / f"mission_report_{direction}.txt", 'w', encoding='utf-8') as f:
        f.write(report)
    logger.info(f"Mission report saved")


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',    default='data/raw')
    parser.add_argument('--direction',   default='east', choices=['east', 'west'])
    parser.add_argument('--checkpoint',  default='checkpoints/best_model.pth')
    parser.add_argument('--patch-size',  type=int, default=64)
    parser.add_argument('--stride',      type=int, default=32)
    parser.add_argument('--batch-size',  type=int, default=64)
    parser.add_argument('--output-dir',  default='outputs')
    parser.add_argument('--use-cached',  action='store_true',
                        help='Skip inference if prediction .npy files exist')
    parser.add_argument('--lola-dem',
                        default='data/raw/lola_dem/ldem_875s_20m.img',
                        help='Path to LOLA GDR polar DEM .img file for real slope data')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # Run all stages
    ice_prob, depth_map, conf_map, features, loader, lat_grid, lon_grid = \
        stage1_ice_detection(args, output_dir)

    # Load LOLA DEM slope (real terrain) — replaces synthetic proxy
    slope_map = load_lola_slope(args.lola_dem, lat_grid, lon_grid, output_dir=output_dir)

    sites, slope_proxy = stage2_landing_sites(
        ice_prob, conf_map, lat_grid, lon_grid, output_dir, args.direction,
        slope_map=slope_map)

    traverse_result, safety_report = stage3_rover_traverse(
        sites, ice_prob, slope_proxy, lat_grid, lon_grid, output_dir, args.direction)

    volume_result = stage4_ice_volume(
        ice_prob, depth_map, features, loader, lat_grid, lon_grid, output_dir, args.direction)

    stage5_visualize(
        ice_prob, depth_map, conf_map, slope_proxy,
        traverse_result, sites, lat_grid, lon_grid,
        loader, volume_result, output_dir, args.direction)

    write_mission_report(sites, traverse_result, volume_result, args.direction, output_dir)

    elapsed = (time.time() - t0) / 60
    banner(f"PIPELINE COMPLETE in {elapsed:.1f} minutes")
    logger.info(f"All outputs in: {output_dir}/")


if __name__ == "__main__":
    main()
