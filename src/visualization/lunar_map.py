"""
Lunar Visualization Module.

Creates interactive 3D maps and 2D visualizations of:
    - Ice probability maps overlaid on lunar terrain
    - Landing site candidates with scoring breakdown
    - Polarimetric feature maps
    - Training progress charts
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def create_ice_probability_map(
    ice_prob: np.ndarray,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    title: str = "Subsurface Ice Probability — Lunar South Pole",
    save_path: Optional[str] = None,
):
    """
    Create 2D polar projection map of ice probability.

    Args:
        ice_prob: (H, W) ice probability values [0, 1]
        lat_grid: (H, W) latitude values
        lon_grid: (H, W) longitude values
        title: Plot title
        save_path: Optional path to save figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    fig, ax = plt.subplots(1, 1, figsize=(12, 12),
                           subplot_kw={'projection': 'polar'})

    # Convert to polar coordinates (south pole projection)
    theta = np.radians(lon_grid)
    r = 90 + lat_grid  # 0 at pole, 10 at -80°

    # Custom colormap: transparent → blue → cyan → white
    colors = ['#00000000', '#1a237e', '#0d47a1', '#01579b',
              '#00bcd4', '#4dd0e1', '#b2ebf2', '#ffffff']
    cmap = mcolors.LinearSegmentedColormap.from_list('ice', colors, N=256)

    c = ax.pcolormesh(theta, r, ice_prob, cmap=cmap, vmin=0, vmax=1, shading='auto')
    ax.set_ylim(0, 10)
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_theta_zero_location('N')

    # Add latitude rings
    for lat in [-82, -84, -86, -88]:
        ax.plot(np.linspace(0, 2*np.pi, 100),
                np.full(100, 90 + lat), 'k--', alpha=0.3, linewidth=0.5)
        ax.text(0, 90 + lat, f'{lat}°', fontsize=8, alpha=0.5)

    plt.colorbar(c, ax=ax, label='Ice Probability', shrink=0.6, pad=0.08)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='black')
        logger.info(f"Saved ice probability map to {save_path}")

    return fig


def create_3d_terrain_with_ice(
    dem: np.ndarray,
    ice_prob: np.ndarray,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    landing_sites: Optional[List] = None,
    title: str = "Lunar South Pole — 3D Terrain with Ice Detection",
    save_path: Optional[str] = None,
):
    """
    Create interactive 3D terrain visualization with ice overlay using Plotly.

    Args:
        dem: (H, W) elevation data in meters
        ice_prob: (H, W) ice probability
        lat_grid, lon_grid: Coordinate grids
        landing_sites: Optional list of LandingSite objects
        save_path: Optional HTML save path
    """
    import plotly.graph_objects as go

    # Subsample for performance
    step = max(1, dem.shape[0] // 200)
    dem_s = dem[::step, ::step]
    ice_s = ice_prob[::step, ::step]
    lat_s = lat_grid[::step, ::step]
    lon_s = lon_grid[::step, ::step]

    # Create ice-colored terrain
    ice_colors = np.zeros((*ice_s.shape, 4))
    ice_colors[..., 0] = 0.7 * (1 - ice_s) + 0.0 * ice_s    # R: gray→blue
    ice_colors[..., 1] = 0.7 * (1 - ice_s) + 0.8 * ice_s    # G: gray→cyan
    ice_colors[..., 2] = 0.7 * (1 - ice_s) + 1.0 * ice_s    # B: gray→white
    ice_colors[..., 3] = 1.0

    # Convert RGBA to plotly colorscale string
    surfacecolor = ice_s

    fig = go.Figure()

    # 3D surface
    fig.add_trace(go.Surface(
        x=lon_s, y=lat_s, z=dem_s,
        surfacecolor=surfacecolor,
        colorscale=[
            [0.0, 'rgb(50,50,50)'],      # No ice: dark gray
            [0.3, 'rgb(26,35,126)'],      # Low: dark blue
            [0.5, 'rgb(13,71,161)'],      # Medium: blue
            [0.7, 'rgb(0,188,212)'],      # High: cyan
            [1.0, 'rgb(255,255,255)'],    # Very high: white (ice)
        ],
        colorbar=dict(title='Ice Probability', x=1.05),
        opacity=0.95,
        name='Terrain + Ice',
    ))

    # Add landing site markers
    if landing_sites:
        for site in landing_sites[:5]:  # Top 5
            fig.add_trace(go.Scatter3d(
                x=[site.lon], y=[site.lat],
                z=[dem_s[
                    np.argmin(np.abs(lat_s[:, 0] - site.lat)),
                    np.argmin(np.abs(lon_s[0, :] - site.lon))
                ] + 500],  # Offset above terrain
                mode='markers+text',
                marker=dict(size=8, color='red', symbol='diamond'),
                text=[f"#{site.rank} ({site.composite_score:.2f})"],
                textposition='top center',
                name=f"Site #{site.rank}",
            ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='Longitude (°)',
            yaxis_title='Latitude (°)',
            zaxis_title='Elevation (m)',
            bgcolor='black',
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.2)),
        ),
        template='plotly_dark',
        width=1200,
        height=800,
    )

    if save_path:
        fig.write_html(save_path)
        logger.info(f"Saved 3D visualization to {save_path}")

    return fig


def create_feature_visualization(
    features: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
):
    """
    Visualize polarimetric feature maps in grid layout.

    Shows CPR, DOP, m-chi components, entropy, etc.
    """
    import matplotlib.pyplot as plt

    feature_names = list(features.keys())
    n = len(feature_names)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if rows == 1:
        axes = [axes] if cols == 1 else list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    for idx, name in enumerate(feature_names):
        ax = axes[idx]
        data = features[name]

        # Choose colormap based on feature
        if name == 'cpr':
            im = ax.imshow(data, cmap='hot', vmin=0, vmax=2)
            ax.set_title(f'CPR\n(>1 = ice indicator)', fontsize=10)
        elif name == 'dop':
            im = ax.imshow(data, cmap='viridis_r', vmin=0, vmax=1)
            ax.set_title(f'DOP\n(<0.13 = ice indicator)', fontsize=10)
        elif 'entropy' in name:
            im = ax.imshow(data, cmap='inferno', vmin=0, vmax=1)
            ax.set_title(f'{name}\n(High = random scatter)', fontsize=10)
        else:
            im = ax.imshow(data, cmap='viridis')
            ax.set_title(name, fontsize=10)

        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.axis('off')

    # Hide unused axes
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Polarimetric Feature Maps — DFSAR Analysis', fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def create_training_curves(
    history: Dict,
    save_path: Optional[str] = None,
):
    """Plot training and validation loss/metrics curves."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    epochs = range(1, len(history['train']) + 1)

    # Loss
    axes[0].plot(epochs, [h['loss'] for h in history['train']], 'b-', label='Train')
    axes[0].plot(epochs, [h['loss'] for h in history['val']], 'r-', label='Val')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # F1 Score
    axes[1].plot(epochs, [h['f1'] for h in history['train']], 'b-', label='Train')
    axes[1].plot(epochs, [h['f1'] for h in history['val']], 'r-', label='Val')
    axes[1].set_title('F1 Score')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Precision & Recall
    axes[2].plot(epochs, [h['precision'] for h in history['val']], 'g-', label='Precision')
    axes[2].plot(epochs, [h['recall'] for h in history['val']], 'm-', label='Recall')
    axes[2].set_title('Precision & Recall (Val)')
    axes[2].set_xlabel('Epoch')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('LunarIceNet Training Progress', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def create_landing_site_comparison(
    sites: List,
    save_path: Optional[str] = None,
):
    """
    Radar chart comparing top landing sites across all criteria.
    """
    import matplotlib.pyplot as plt

    categories = ['Ice Prob', 'Flat Terrain', 'Accessibility', 'Illumination', 'Confidence']
    n_cats = len(categories)

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]  # Close polygon

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

    for idx, site in enumerate(sites[:5]):
        values = [
            site.details.get('ice_probability', site.ice_probability),
            site.details.get('terrain_slope', 1 - site.terrain_slope / 30),
            site.details.get('accessibility', site.accessibility),
            site.details.get('illumination', site.illumination),
            site.details.get('confidence', site.confidence),
        ]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, color=colors[idx % len(colors)],
                label=f"#{site.rank} ({site.composite_score:.2f})")
        ax.fill(angles, values, alpha=0.1, color=colors[idx % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title('Landing Site Comparison', fontsize=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
