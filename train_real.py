"""
Train LunarIceNet on real Chandrayaan-2 DFSAR Level 3C mosaic data.

Usage:
    python train_real.py                    # Train on east direction (smaller, faster)
    python train_real.py --direction west   # Train on west (larger coverage, 340M pixels)
    python train_real.py --epochs 50 --batch-size 32
    python train_real.py --analyze-only     # Just print data statistics, no training
"""

import argparse
import logging
import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from pathlib import Path

from src.data.real_loader import DFSARMosaicLoader, RealDFSARDataset
from src.models.lunaricenet import LunarIceNet, PhysicsInformedLoss
from src.models.trainer import Metrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('LunarIceNet.RealTrain')


def analyze_data(loader: DFSARMosaicLoader, features: dict):
    """Print detailed data statistics."""
    print("\n" + "=" * 60)
    print("DFSAR MOSAIC DATA ANALYSIS")
    print("=" * 60)

    for name, data in features.items():
        valid = data[~np.isnan(data)]
        print(f"\n{name}:")
        print(f"  Shape: {data.shape}")
        print(f"  Valid pixels: {len(valid):,} / {data.size:,} ({100*len(valid)/data.size:.1f}%)")
        if len(valid) > 0:
            print(f"  Range: [{np.min(valid):.4f}, {np.max(valid):.4f}]")
            print(f"  Mean: {np.mean(valid):.4f}, Std: {np.std(valid):.4f}")
            print(f"  Median: {np.median(valid):.4f}")
            percentiles = np.percentile(valid, [1, 5, 25, 75, 95, 99])
            print(f"  Percentiles [1,5,25,75,95,99]: {[f'{p:.4f}' for p in percentiles]}")
            if name == 'CPR':
                print(f"  CPR > 1.0: {(valid > 1.0).sum():,} ({100*(valid > 1.0).mean():.3f}%)")
                print(f"  CPR > 1.5: {(valid > 1.5).sum():,} ({100*(valid > 1.5).mean():.4f}%)")
                print(f"  CPR > 2.0: {(valid > 2.0).sum():,} ({100*(valid > 2.0).mean():.4f}%)")

    # Coordinate analysis
    lat, lon = loader.get_coordinates()
    valid_lat = lat[loader.valid_mask]
    valid_lon = lon[loader.valid_mask]
    print(f"\nCoordinates (valid pixels):")
    print(f"  Lat range: [{np.min(valid_lat):.2f}, {np.max(valid_lat):.2f}]")
    print(f"  Lon range: [{np.min(valid_lon):.2f}, {np.max(valid_lon):.2f}]")
    print(f"  Total valid: {loader.valid_mask.sum():,}")

    # Ice candidates
    ice_mask = loader.get_ice_candidates(1.0)
    ice_lats = lat[ice_mask]
    print(f"\nIce Candidates (CPR > 1.0): {ice_mask.sum():,} pixels")
    if ice_mask.sum() > 0:
        print(f"  Lat range: [{np.min(ice_lats):.2f}, {np.max(ice_lats):.2f}]")

    print("=" * 60 + "\n")


def normalize_features(features: dict) -> dict:
    """Robust normalization per feature (clip outliers, scale to [0,1]). Keeps float32."""
    normalized = {}
    for name, data in features.items():
        d = data.astype(np.float32)          # ensure float32, avoid float64 OOM
        valid = d[~np.isnan(d)]
        if len(valid) == 0:
            normalized[name] = d
            continue
        # Clip to 1st-99th percentile (compute on subsample for large arrays)
        if len(valid) > 5_000_000:
            sample = valid[np.random.choice(len(valid), 5_000_000, replace=False)]
        else:
            sample = valid
        p1, p99 = np.float32(np.percentile(sample, 1)), np.float32(np.percentile(sample, 99))
        np.clip(d, p1, p99, out=d)           # in-place, no extra allocation
        dmin, dmax = np.float32(np.nanmin(d)), np.float32(np.nanmax(d))
        if dmax - dmin > np.float32(1e-7):
            d -= dmin
            d /= (dmax - dmin)
        normalized[name] = d
    return normalized


def train_real(args):
    """Full training pipeline on real DFSAR data."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # --- Load real data ---
    logger.info(f"Loading DFSAR {args.direction} direction mosaics...")
    mosaic_loader = DFSARMosaicLoader(args.data_dir)
    # West is 24794×24181 (2.2 GB/array @ float32) — subsample 2x to fit in RAM
    subsample = args.subsample if args.direction == 'west' else 1
    features = mosaic_loader.load_single_direction(args.direction, subsample=subsample)

    if args.analyze_only:
        analyze_data(mosaic_loader, features)
        return

    analyze_data(mosaic_loader, features)

    # Normalize (keep raw for labels)
    logger.info("Normalizing features...")
    norm_features = normalize_features(features)

    # Coordinates
    lat_grid, lon_grid = mosaic_loader.get_coordinates()

    # --- Build dataset ---
    logger.info(f"Building dataset (patch={args.patch_size}, stride={args.stride})...")
    dataset = RealDFSARDataset(
        features=norm_features,
        valid_mask=mosaic_loader.valid_mask,
        lat_grid=lat_grid,
        lon_grid=lon_grid,
        patch_size=args.patch_size,
        stride=args.stride,
        ice_enrichment=0.5,
        raw_features=features,  # Un-normalized for CPR > 1.0 labels
    )
    logger.info(f"Total patches: {len(dataset)}")

    if len(dataset) == 0:
        logger.error("No valid patches found. Check data.")
        return

    # Train/val split
    n_val = max(1, int(len(dataset) * 0.2))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                     generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)

    logger.info(f"Train: {n_train} patches, Val: {n_val} patches")
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # --- Build model ---
    n_channels = len(features)  # 3 (CPR, SERD, T-Ratio)
    model = LunarIceNet(
        in_channels=n_channels,
        physics_features=5,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_attn_layers=args.num_attn_layers,
        patch_size=args.patch_size,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"LunarIceNet: {total_params:,} parameters, {n_channels} input channels")

    # --- Loss, optimizer, scheduler ---
    criterion = PhysicsInformedLoss(
        bce_weight=1.0, depth_weight=0.5,
        physics_weight=0.3, temp_prior_weight=0.2,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # Mixed precision — halves VRAM usage for activations (critical for west mosaic)
    use_amp = device.type == 'cuda'
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)
    logger.info(f"Mixed precision (AMP): {'ON' if use_amp else 'OFF'}")

    # --- Training loop ---
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = 0.0
    history = {'train': [], 'val': []}

    logger.info("=" * 60)
    logger.info("TRAINING START")
    logger.info("=" * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # --- Train ---
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []
        for batch in train_loader:
            feats = batch['features'].to(device)
            phys = batch['physical'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(feats, phys)
                losses = criterion(outputs, labels, phys)
            scaler.scale(losses['total']).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += losses['total'].item()
            with torch.no_grad():
                preds = torch.sigmoid(outputs['ice_prob'].squeeze(1))
                train_preds.append(preds.cpu())
                train_targets.append(labels.cpu())

        train_loss /= len(train_loader)
        train_preds = torch.cat(train_preds)
        train_targets = torch.cat(train_targets)
        train_metrics = Metrics.compute(train_preds, train_targets)
        train_metrics['loss'] = train_loss

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                feats = batch['features'].to(device)
                phys = batch['physical'].to(device)
                labels = batch['label'].to(device)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = model(feats, phys)
                    losses = criterion(outputs, labels, phys)
                val_loss += losses['total'].item()

                preds = torch.sigmoid(outputs['ice_prob'].squeeze(1))
                val_preds.append(preds.cpu())
                val_targets.append(labels.cpu())

        val_loss /= len(val_loader)
        val_preds = torch.cat(val_preds)
        val_targets = torch.cat(val_targets)
        val_metrics = Metrics.compute(val_preds, val_targets)
        val_metrics['loss'] = val_loss

        scheduler.step()
        elapsed = time.time() - t0

        history['train'].append(train_metrics)
        history['val'].append(val_metrics)

        # Log
        logger.info(
            f"Epoch {epoch:3d}/{args.epochs} ({elapsed:.1f}s) | "
            f"Train Loss:{train_loss:.4f} F1:{train_metrics['f1']:.4f} | "
            f"Val Loss:{val_loss:.4f} F1:{val_metrics['f1']:.4f} P:{val_metrics['precision']:.4f} R:{val_metrics['recall']:.4f}"
        )

        # Save best
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'metrics': val_metrics,
                'in_channels': n_channels,
                'direction': args.direction,
            }, ckpt_dir / "best_model.pth")
            logger.info(f"  -> Best F1: {best_f1:.4f}")

        # Periodic save
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'metrics': val_metrics,
            }, ckpt_dir / f"epoch_{epoch}.pth")

    # Save history
    with open(ckpt_dir / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2, default=str)

    logger.info("=" * 60)
    logger.info(f"Training complete. Best Val F1: {best_f1:.4f}")
    logger.info(f"Checkpoint: {ckpt_dir / 'best_model.pth'}")
    logger.info("=" * 60)

    # --- Quick inference on sample ---
    logger.info("Running sample inference...")
    model.eval()
    sample = dataset[0]
    with torch.no_grad():
        feats = sample['features'].unsqueeze(0).to(device)
        phys = sample['physical'].unsqueeze(0).to(device)
        out = model.predict(feats, phys)

    ice_prob = out['ice_prob'].squeeze().cpu().numpy()
    depth = out['depth'].squeeze().cpu().numpy()
    conf = out['confidence'].squeeze().cpu().numpy()
    logger.info(f"Sample ice prob: [{ice_prob.min():.3f}, {ice_prob.max():.3f}], mean={ice_prob.mean():.3f}")
    logger.info(f"Sample depth: [{depth.min():.3f}, {depth.max():.3f}]m")
    logger.info(f"Sample confidence: {conf.mean():.3f}")


def main():
    parser = argparse.ArgumentParser(description="Train LunarIceNet on real DFSAR data")
    parser.add_argument('--data-dir', default='data/raw', help='Path to raw data')
    parser.add_argument('--direction', default='east', choices=['east', 'west'],
                        help='Look direction (east=smaller/faster, west=more coverage)')
    parser.add_argument('--patch-size', type=int, default=64, help='Patch size in pixels')
    parser.add_argument('--stride', type=int, default=32, help='Stride between patches')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--embed-dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--num-heads', type=int, default=4, help='Attention heads')
    parser.add_argument('--num-attn-layers', type=int, default=2, help='Attention layers')
    parser.add_argument('--checkpoint-dir', default='checkpoints', help='Checkpoint dir')
    parser.add_argument('--analyze-only', action='store_true', help='Only analyze data')
    parser.add_argument('--subsample', type=int, default=2,
                        help='Spatial subsampling for west direction (2=50m/px, avoids OOM)')
    args = parser.parse_args()
    train_real(args)


if __name__ == "__main__":
    main()
