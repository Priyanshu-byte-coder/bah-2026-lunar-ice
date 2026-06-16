"""
Run inference on real DFSAR data and generate ice probability maps + landing site analysis.

Usage:
    python predict_real.py                              # Use best checkpoint
    python predict_real.py --checkpoint checkpoints/epoch_20.pth
    python predict_real.py --direction west --stride 16  # Dense prediction on west
"""

import argparse
import logging
import numpy as np
import torch
from pathlib import Path

from src.data.real_loader import DFSARMosaicLoader, RealDFSARDataset
from src.models.lunaricenet import LunarIceNet
from src.models.landing_site import LandingSiteScorer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('LunarIceNet.Predict')


def normalize_features(features: dict) -> dict:
    """Keep float32 throughout — west mosaic (24794×24181) needs memory efficiency."""
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


def predict_real(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load data
    logger.info(f"Loading DFSAR {args.direction} mosaics...")
    loader = DFSARMosaicLoader(args.data_dir)
    features = loader.load_single_direction(args.direction)
    norm_features = normalize_features(features)
    lat_grid, lon_grid = loader.get_coordinates()

    # Load model
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        return

    checkpoint = torch.load(ckpt_path, map_location=device)
    n_channels = checkpoint.get('in_channels', 3)

    model = LunarIceNet(
        in_channels=n_channels,
        physics_features=5,
        embed_dim=128,
        num_heads=4,
        num_attn_layers=2,
        patch_size=args.patch_size,
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logger.info(f"Loaded checkpoint (epoch {checkpoint.get('epoch', '?')}, F1={checkpoint.get('best_f1', '?')})")

    # Build dataset for inference
    dataset = RealDFSARDataset(
        features=norm_features,
        valid_mask=loader.valid_mask,
        lat_grid=lat_grid,
        lon_grid=lon_grid,
        patch_size=args.patch_size,
        stride=args.stride,
        ice_enrichment=0.0,  # No enrichment for inference
        raw_features=features,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Run inference
    logger.info(f"Running inference on {len(dataset)} patches...")
    h, w = loader.valid_mask.shape
    ice_prob_map = np.zeros((h, w), dtype=np.float32)
    depth_map = np.zeros((h, w), dtype=np.float32)
    confidence_map = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    ps = args.patch_size
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            feats = batch['features'].to(device)
            phys = batch['physical'].to(device)

            outputs = model.predict(feats, phys)
            ice_prob = outputs['ice_prob'].squeeze(1).cpu().numpy()
            depth = outputs['depth'].squeeze(1).cpu().numpy()
            conf = outputs['confidence'].squeeze(1).cpu().numpy()

            # Map back to full mosaic
            batch_size_actual = feats.shape[0]
            start_idx = i * args.batch_size
            for j in range(batch_size_actual):
                idx = start_idx + j
                if idx >= len(dataset.patches):
                    break
                py, px = dataset.patches[idx]
                ice_prob_map[py:py+ps, px:px+ps] += ice_prob[j]
                depth_map[py:py+ps, px:px+ps] += depth[j]
                confidence_map[py:py+ps, px:px+ps] += conf[j]
                count_map[py:py+ps, px:px+ps] += 1.0

            if (i + 1) % 50 == 0:
                logger.info(f"  Batch {i+1}/{len(dataloader)}")

    # Average overlapping regions
    valid = count_map > 0
    ice_prob_map[valid] /= count_map[valid]
    depth_map[valid] /= count_map[valid]
    confidence_map[valid] /= count_map[valid]

    # Stats
    pred_valid = ice_prob_map[valid]
    logger.info(f"\nPrediction Statistics:")
    logger.info(f"  Ice probability: [{pred_valid.min():.3f}, {pred_valid.max():.3f}], mean={pred_valid.mean():.3f}")
    logger.info(f"  High-confidence ice (prob>0.7): {(pred_valid > 0.7).sum():,} pixels")
    logger.info(f"  Mean confidence: {confidence_map[valid].mean():.3f}")

    # Save outputs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / f"ice_probability_{args.direction}.npy", ice_prob_map)
    np.save(output_dir / f"depth_estimate_{args.direction}.npy", depth_map)
    np.save(output_dir / f"confidence_{args.direction}.npy", confidence_map)
    logger.info(f"Saved prediction maps to {output_dir}/")

    # Landing site analysis
    logger.info("\nRunning landing site analysis...")
    scorer = LandingSiteScorer()

    # Use prediction as slope proxy (no DEM yet — use flat estimate)
    slope_map = np.ones_like(ice_prob_map) * 5.0  # Assume 5 degrees (will replace with LOLA DEM)

    sites = scorer.evaluate_region(
        ice_prob_map, slope_map, lat_grid, lon_grid,
        confidence_map, top_k=10,
    )

    if sites:
        report = scorer.generate_report(sites)
        print(report)
        with open(output_dir / f"landing_site_report_{args.direction}.txt", 'w') as f:
            f.write(report)
    else:
        logger.info("No landing sites passed constraints (model may need more training)")

    # Generate visualizations
    try:
        generate_maps(ice_prob_map, depth_map, confidence_map,
                      lat_grid, lon_grid, loader.valid_mask, output_dir, args.direction)
    except Exception as e:
        logger.warning(f"Visualization failed: {e}")


def generate_maps(ice_prob, depth, confidence, lat, lon, valid_mask,
                  output_dir, direction):
    """Generate ice probability and analysis maps."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Ice probability
    im0 = axes[0].imshow(np.where(valid_mask, ice_prob, np.nan),
                          cmap='hot', vmin=0, vmax=1)
    axes[0].set_title('Ice Probability')
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    # Depth estimate
    im1 = axes[1].imshow(np.where(valid_mask, depth, np.nan),
                          cmap='Blues', vmin=0)
    axes[1].set_title('Depth Estimate (m)')
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    # Confidence
    im2 = axes[2].imshow(np.where(valid_mask, confidence, np.nan),
                          cmap='viridis', vmin=0, vmax=1)
    axes[2].set_title('Prediction Confidence')
    plt.colorbar(im2, ax=axes[2], shrink=0.8)

    for ax in axes:
        ax.set_xlabel('Pixel X')
        ax.set_ylabel('Pixel Y')

    plt.suptitle(f'LunarIceNet Predictions — DFSAR {direction.upper()} Direction',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f"prediction_maps_{direction}.png", dpi=150)
    plt.close()

    # Polar projection map
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})
    valid_ice = ice_prob[valid_mask]
    valid_lat = lat[valid_mask]
    valid_lon = lon[valid_mask]

    # Subsample for plotting
    n = min(100000, len(valid_ice))
    idx = np.random.choice(len(valid_ice), n, replace=False)

    theta = np.radians(valid_lon[idx])
    r = 90 + valid_lat[idx]  # Distance from pole

    scatter = ax.scatter(theta, r, c=valid_ice[idx], cmap='hot',
                         s=0.1, alpha=0.5, vmin=0, vmax=1)
    ax.set_title(f'Lunar South Pole — Ice Probability\n(DFSAR {direction})', pad=20)
    ax.set_theta_zero_location('N')
    ax.set_rlabel_position(0)
    plt.colorbar(scatter, ax=ax, label='Ice Probability', shrink=0.8)
    plt.savefig(output_dir / f"polar_ice_map_{direction}.png", dpi=150)
    plt.close()

    logger.info(f"Saved visualizations to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Run LunarIceNet inference on real DFSAR data")
    parser.add_argument('--data-dir', default='data/raw')
    parser.add_argument('--direction', default='east', choices=['east', 'west'])
    parser.add_argument('--checkpoint', default='checkpoints/best_model.pth')
    parser.add_argument('--patch-size', type=int, default=64)
    parser.add_argument('--stride', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--output-dir', default='outputs')
    args = parser.parse_args()
    predict_real(args)


if __name__ == "__main__":
    main()
