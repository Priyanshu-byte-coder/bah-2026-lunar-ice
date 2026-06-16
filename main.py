"""
LunarIceNet — Main Entry Point.

Usage:
    python main.py train          # Train model on synthetic/real data
    python main.py predict        # Run inference on DFSAR data
    python main.py evaluate       # Evaluate model and generate metrics
    python main.py dashboard      # Launch Streamlit dashboard
    python main.py demo           # Run full demo pipeline
"""

import sys
import yaml
import logging
import numpy as np
import torch
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('LunarIceNet')


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def train(config: dict):
    """Train LunarIceNet on DFSAR data."""
    from src.data.dataset import DFSARDataModule
    from src.models.lunaricenet import LunarIceNet
    from src.models.trainer import LunarIceNetTrainer

    logger.info("=" * 60)
    logger.info("LunarIceNet Training Pipeline")
    logger.info("=" * 60)

    # Data
    data_module = DFSARDataModule(
        data_dir="data",
        config_path="configs/config.yaml",
        batch_size=config['training']['batch_size'],
        num_workers=0,  # 0 for Windows compatibility
    )
    data_module.setup()

    logger.info(f"Train samples: {len(data_module.train_dataset)}")
    logger.info(f"Val samples: {len(data_module.val_dataset)}")

    # Model
    model_cfg = config['model']
    model = LunarIceNet(
        in_channels=model_cfg['architecture']['radar_encoder']['in_channels'],
        physics_features=model_cfg['architecture']['physics_encoder']['in_features'],
        embed_dim=256,
        num_heads=model_cfg['architecture']['fusion']['num_heads'],
        num_attn_layers=model_cfg['architecture']['fusion']['num_layers'],
        patch_size=config['data']['preprocessing']['patch_size'],
    )

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")

    # Train
    trainer = LunarIceNetTrainer(model, config)
    trainer.train(
        data_module.train_dataloader(),
        data_module.val_dataloader(),
        checkpoint_dir="checkpoints",
    )


def predict(config: dict):
    """Run inference on DFSAR data."""
    from src.models.lunaricenet import LunarIceNet
    from src.models.landing_site import LandingSiteScorer

    logger.info("Running LunarIceNet Inference...")

    # Load model
    model = LunarIceNet(
        in_channels=config['model']['architecture']['radar_encoder']['in_channels'],
        physics_features=config['model']['architecture']['physics_encoder']['in_features'],
    )

    ckpt_path = Path("checkpoints/best_model.pth")
    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Loaded trained model checkpoint")
    else:
        logger.warning("No checkpoint found. Using random weights for demo.")

    model.eval()

    # Demo inference with synthetic data
    batch_size = 1
    dummy_radar = torch.randn(batch_size, config['model']['architecture']['radar_encoder']['in_channels'], 256, 256)
    dummy_physical = torch.tensor([[-87.3, 77.0, 0.7, 87.3, 1.0]])

    with torch.no_grad():
        outputs = model.predict(dummy_radar, dummy_physical)

    ice_prob = outputs['ice_prob'].squeeze().numpy()
    depth = outputs['depth'].squeeze().numpy()
    confidence = outputs['confidence'].squeeze().numpy()

    logger.info(f"Ice probability range: [{ice_prob.min():.3f}, {ice_prob.max():.3f}]")
    logger.info(f"Mean confidence: {confidence.mean():.3f}")

    # Landing site analysis
    scorer = LandingSiteScorer(config.get('landing_site', {}).get('scoring_weights'))

    # Create coordinate grids
    lat_grid = np.linspace(-90, -80, 256).reshape(-1, 1) * np.ones((1, 256))
    lon_grid = np.ones((256, 1)) * np.linspace(-180, 180, 256).reshape(1, -1)
    slope_map = np.random.uniform(0, 20, (256, 256))

    sites = scorer.evaluate_region(
        ice_prob, slope_map, lat_grid, lon_grid, confidence,
        top_k=5,
    )

    report = scorer.generate_report(sites)
    print(report)


def demo(config: dict):
    """Run full demo pipeline: synthetic data → features → model → visualization."""
    logger.info("=" * 60)
    logger.info("LunarIceNet — Full Demo Pipeline")
    logger.info("=" * 60)

    from src.features.polarimetric import extract_all_features, create_feature_stack
    from src.models.lunaricenet import LunarIceNet
    from src.models.landing_site import LandingSiteScorer

    # 1. Generate synthetic DFSAR-like data
    logger.info("Step 1: Generating synthetic DFSAR data...")
    size = 128
    np.random.seed(42)

    # Simulate SAR channels with ice signature
    base = np.random.randn(size, size) * 0.3
    ice_region = np.exp(-((np.arange(size).reshape(-1,1) - 60)**2 +
                          (np.arange(size).reshape(1,-1) - 80)**2) / (2*20**2))

    hh = (0.5 + base + ice_region * 0.8) + 1j * (np.random.randn(size, size) * 0.2)
    hv = (0.1 + np.random.randn(size, size) * 0.1 + ice_region * 0.3) + 1j * (np.random.randn(size, size) * 0.1)
    vh = hv.copy()  # Reciprocity
    vv = (0.4 + base * 0.8 + ice_region * 0.3) + 1j * (np.random.randn(size, size) * 0.2)

    # 2. Extract polarimetric features
    logger.info("Step 2: Extracting polarimetric features...")
    features = extract_all_features(hh, hv, vh, vv, spatial_filter_size=3)
    logger.info(f"  Extracted {len(features)} feature maps: {list(features.keys())}")

    # Key indicators
    cpr = features['cpr']
    dop = features['dop']
    logger.info(f"  CPR range: [{cpr.min():.2f}, {cpr.max():.2f}] (>1 = ice)")
    logger.info(f"  DOP range: [{dop.min():.2f}, {dop.max():.2f}] (<0.13 = ice)")

    ice_indicator = (cpr > 1.0) & (dop < 0.3)
    logger.info(f"  Classical ice indicator coverage: {ice_indicator.mean()*100:.1f}%")

    # 3. Run model inference
    logger.info("Step 3: Running LunarIceNet inference...")
    feature_stack, feature_names = create_feature_stack(features)
    n_features = feature_stack.shape[0]

    model = LunarIceNet(
        in_channels=n_features,
        physics_features=5,
        embed_dim=128,  # Smaller for demo
        num_heads=4,
        num_attn_layers=1,
        patch_size=size,
    )

    radar_tensor = torch.from_numpy(feature_stack).unsqueeze(0)  # (1, C, H, W)
    physical_tensor = torch.tensor([[-87.3, 77.0, 0.7, 87.3, 1.0]])

    outputs = model.predict(radar_tensor, physical_tensor)
    ice_prob = outputs['ice_prob'].squeeze().numpy()
    confidence = outputs['confidence'].squeeze().numpy()

    logger.info(f"  ML ice probability range: [{ice_prob.min():.3f}, {ice_prob.max():.3f}]")
    logger.info(f"  Mean confidence: {confidence.mean():.3f}")

    # 4. Landing site scoring
    logger.info("Step 4: Scoring landing sites...")
    scorer = LandingSiteScorer()

    lat_grid = np.linspace(-90, -80, size).reshape(-1, 1) * np.ones((1, size))
    lon_grid = np.ones((size, 1)) * np.linspace(60, 90, size).reshape(1, -1)
    slope_map = np.random.uniform(2, 18, (size, size)).astype(np.float32)

    sites = scorer.evaluate_region(
        ice_prob, slope_map, lat_grid, lon_grid, confidence, top_k=5
    )

    if sites:
        print("\n" + scorer.generate_report(sites))
    else:
        logger.info("  No sites passed constraint filters with random model weights.")
        logger.info("  (Expected — model needs training first)")

    # 5. Save visualizations
    logger.info("Step 5: Saving visualizations...")
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    try:
        from src.visualization.lunar_map import create_feature_visualization
        fig = create_feature_visualization(
            {k: features[k] for k in ['cpr', 'dop', 'entropy', 'pedestal_height',
                                        'mchi_volume', 'mchi_surface']
             if k in features},
            save_path=str(output_dir / "polarimetric_features.png"),
        )
        logger.info("  Saved feature visualization")
    except Exception as e:
        logger.warning(f"  Visualization failed (matplotlib may not be available): {e}")

    logger.info("=" * 60)
    logger.info("Demo complete!")
    logger.info(f"  Features extracted: {len(features)}")
    logger.info(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"  Landing sites found: {len(sites)}")
    logger.info("=" * 60)


def launch_dashboard():
    """Launch Streamlit dashboard."""
    import subprocess
    subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard/app.py"])


def main():
    config = load_config()

    if len(sys.argv) < 2:
        print(__doc__)
        print("Available commands: train, predict, evaluate, dashboard, demo")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "train":
        train(config)
    elif command == "train-real":
        import subprocess
        subprocess.run([sys.executable, "train_real.py"] + sys.argv[2:])
    elif command == "predict":
        predict(config)
    elif command == "demo":
        demo(config)
    elif command == "dashboard":
        launch_dashboard()
    else:
        print(f"Unknown command: {command}")
        print("Available: train, train-real, predict, demo, dashboard")
        sys.exit(1)


if __name__ == "__main__":
    main()
