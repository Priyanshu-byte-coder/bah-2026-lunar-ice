"""
Generate presentation visual assets for LunarIceNet BAH 2026 PS-8.
All figures use matplotlib only, dark theme, professional styling.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np
import os

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

BG = '#0a0a1a'
BOX_BLUE = '#1a237e'
CYAN = '#00bcd4'
WHITE = '#ffffff'
LIGHT_GRAY = '#b0bec5'
DARK_GRAY = '#263238'

SAVE_KW = dict(dpi=150, bbox_inches='tight', facecolor=BG, edgecolor='none')


# ─────────────────────────────────────────────────────────────
# 1. Pipeline Flow Diagram
# ─────────────────────────────────────────────────────────────
def make_pipeline_flow():
    fig, ax = plt.subplots(figsize=(12, 4), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-1.5, 2.5)
    ax.axis('off')

    # Title
    ax.text(5, 2.2, 'LunarIceNet — End-to-End Pipeline',
            ha='center', va='center', fontsize=18, fontweight='bold',
            color=WHITE, fontfamily='sans-serif')

    stages = [
        ("DFSAR\nData Input", "Raw SAR imagery"),
        ("Ice\nDetection", "LunarIceNet CNN"),
        ("Landing\nSites", "Multi-criteria scoring"),
        ("Rover\nTraverse", "A* pathfinding"),
        ("Ice Volume\n& Report", "Lichtenecker model"),
    ]

    n = len(stages)
    box_w, box_h = 1.6, 1.2
    gap = 0.5
    total_w = n * box_w + (n - 1) * gap
    x_start = (10 - total_w) / 2 + box_w / 2

    positions = []
    for i in range(n):
        cx = x_start + i * (box_w + gap)
        positions.append(cx)

    # Gradient-like effect via layered boxes
    for i, (label, sub) in enumerate(stages):
        cx = positions[i]
        cy = 0.8

        # Outer glow
        glow = FancyBboxPatch((cx - box_w/2 - 0.04, cy - box_h/2 - 0.04),
                              box_w + 0.08, box_h + 0.08,
                              boxstyle="round,pad=0.12",
                              facecolor=CYAN, alpha=0.12, edgecolor='none',
                              zorder=1)
        ax.add_patch(glow)

        # Main box
        box = FancyBboxPatch((cx - box_w/2, cy - box_h/2),
                             box_w, box_h,
                             boxstyle="round,pad=0.1",
                             facecolor=BOX_BLUE, edgecolor=CYAN,
                             linewidth=1.5, alpha=0.95, zorder=2)
        ax.add_patch(box)

        # Stage number badge
        badge_r = 0.18
        badge = plt.Circle((cx - box_w/2 + 0.22, cy + box_h/2 - 0.18),
                           badge_r, facecolor=CYAN, edgecolor='none',
                           zorder=3, alpha=0.9)
        ax.add_patch(badge)
        ax.text(cx - box_w/2 + 0.22, cy + box_h/2 - 0.18, str(i + 1),
                ha='center', va='center', fontsize=9, fontweight='bold',
                color=BG, zorder=4)

        # Stage label
        ax.text(cx, cy + 0.05, label, ha='center', va='center',
                fontsize=11, fontweight='bold', color=WHITE, zorder=3,
                fontfamily='sans-serif', linespacing=1.15)

        # Sub-label below box
        ax.text(cx, cy - box_h/2 - 0.28, sub, ha='center', va='top',
                fontsize=8.5, color=LIGHT_GRAY, style='italic', zorder=3,
                fontfamily='sans-serif')

    # Arrows between boxes
    for i in range(n - 1):
        x1 = positions[i] + box_w / 2 + 0.02
        x2 = positions[i + 1] - box_w / 2 - 0.02
        ax.annotate('', xy=(x2, 0.8), xytext=(x1, 0.8),
                    arrowprops=dict(arrowstyle='->', color=CYAN,
                                    lw=2.5, mutation_scale=18),
                    zorder=3)

    fig.savefig(os.path.join(ASSETS_DIR, 'pipeline_flow.png'), **SAVE_KW)
    plt.close(fig)
    print("[OK] pipeline_flow.png")


# ─────────────────────────────────────────────────────────────
# 2. Architecture Diagram
# ─────────────────────────────────────────────────────────────
def make_architecture():
    fig, ax = plt.subplots(figsize=(14, 7), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis('off')

    def draw_box(x, y, w, h, label, color=BOX_BLUE, edge=CYAN,
                 fontsize=10, alpha=0.92, lw=1.5):
        box = FancyBboxPatch((x, y), w, h,
                             boxstyle="round,pad=0.12",
                             facecolor=color, edgecolor=edge,
                             linewidth=lw, alpha=alpha, zorder=2)
        ax.add_patch(box)
        if label:
            ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                    fontsize=fontsize, fontweight='bold', color=WHITE, zorder=3,
                    linespacing=1.2)

    def arrow(x1, y1, x2, y2, color=CYAN):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color,
                                    lw=2, mutation_scale=15,
                                    connectionstyle='arc3,rad=0'),
                    zorder=3)

    # Title
    ax.text(7, 6.65, 'LunarIceNet Architecture', ha='center', va='center',
            fontsize=20, fontweight='bold', color=WHITE)
    ax.text(7, 6.25, '12.4M Parameters  |  Multi-Scale ResNet + Cross-Attention Fusion',
            ha='center', va='center', fontsize=11, color=CYAN,
            fontfamily='sans-serif')

    # ─── Input: DFSAR Patch ───
    draw_box(0.3, 3.2, 2.4, 2.4, '', fontsize=12)
    ax.text(1.5, 4.9, 'DFSAR Patch', ha='center', va='center',
            fontsize=12, fontweight='bold', color=WHITE, zorder=3)
    ax.text(1.5, 4.5, '3 × 64 × 64', ha='center', va='center',
            fontsize=10, color=LIGHT_GRAY, zorder=3)
    # Channel labels — positioned below the title text with clear spacing
    channels = ['CPR', 'SERD', 'T-Ratio']
    colors_ch = ['#42a5f5', '#ab47bc', '#66bb6a']
    for j, (ch, cc) in enumerate(zip(channels, colors_ch)):
        yy = 4.0 - j * 0.35
        ax.plot([0.6, 0.85], [yy, yy], color=cc, lw=3, zorder=3)
        ax.text(0.95, yy, ch, fontsize=9, color=cc, va='center', zorder=3)

    # ─── Physics Encoder ───
    draw_box(0.3, 0.5, 2.4, 1.8, '', fontsize=10)
    ax.text(1.5, 1.85, 'Physics Encoder', ha='center', va='center',
            fontsize=11, fontweight='bold', color=WHITE, zorder=3)
    ax.text(1.5, 1.5, '(MLP)', ha='center', va='center',
            fontsize=9, color=LIGHT_GRAY, zorder=3)
    # Physics input labels — row below the title
    phys_inputs = ['lat', 'lon', 'PSR', 'pole_dist']
    total_w = len(phys_inputs) * 0.55
    x_start = 1.5 - total_w / 2 + 0.275
    for j, pi in enumerate(phys_inputs):
        ax.text(x_start + j * 0.55, 0.85, pi, fontsize=7.5, color='#ffab40',
                ha='center', va='center', zorder=3,
                bbox=dict(boxstyle='round,pad=0.12', facecolor='#1a1a3a',
                          edgecolor='#ffab40', lw=0.8))

    # ─── Multi-Scale ResNet Encoder ───
    draw_box(3.6, 3.2, 3.0, 2.4, '', fontsize=12)
    ax.text(5.1, 5.1, 'Multi-Scale ResNet', ha='center', va='center',
            fontsize=12, fontweight='bold', color=WHITE, zorder=3)
    ax.text(5.1, 4.7, 'Encoder', ha='center', va='center',
            fontsize=12, fontweight='bold', color=WHITE, zorder=3)
    ax.text(5.1, 4.3, '3 scales × residual blocks', ha='center', va='center',
            fontsize=8.5, color=LIGHT_GRAY, zorder=3, style='italic')

    # Scale blocks inside — positioned below sublabel
    for j in range(3):
        bx = 3.95 + j * 0.95
        by = 3.55
        small = FancyBboxPatch((bx, by), 0.7, 0.45,
                               boxstyle="round,pad=0.05",
                               facecolor='#283593', edgecolor='#5c6bc0',
                               lw=1, zorder=3)
        ax.add_patch(small)
        ax.text(bx + 0.35, by + 0.225, f'S{j+1}', ha='center', va='center',
                fontsize=9, color='#9fa8da', fontweight='bold', zorder=4)

    # ─── Cross-Attention Fusion ───
    draw_box(7.5, 1.8, 2.8, 3.4, '', color='#1b2838', edge='#4dd0e1',
             fontsize=10, lw=2)
    ax.text(8.9, 4.8, 'Cross-Attention', ha='center', va='center',
            fontsize=12, fontweight='bold', color=WHITE, zorder=3)
    ax.text(8.9, 4.4, 'Fusion', ha='center', va='center',
            fontsize=12, fontweight='bold', color=WHITE, zorder=3)
    ax.text(8.9, 4.0, '2 layers × 4 heads', ha='center', va='center',
            fontsize=9, color='#4dd0e1', zorder=3, style='italic')

    # Fusion internals — two attention layer blocks
    for j in range(2):
        by = 2.2 + j * 1.0
        small = FancyBboxPatch((7.85, by), 2.1, 0.65,
                               boxstyle="round,pad=0.06",
                               facecolor='#0d47a1', edgecolor='#4dd0e1',
                               lw=0.8, alpha=0.7, zorder=3)
        ax.add_patch(small)
        ax.text(8.9, by + 0.325, f'Attention Layer {j+1}', ha='center',
                va='center', fontsize=9, color='#b2ebf2', zorder=4)

    # ─── Output Heads ───
    heads = [
        ("Ice\nProbability", CYAN, 4.6),
        ("Depth\nEstimate", '#ff9800', 3.5),
        ("Confidence", '#4caf50', 2.4),
    ]
    for label, color, yy in heads:
        draw_box(11.2, yy - 0.4, 2.0, 0.9, label, color='#1a1a3a',
                 edge=color, fontsize=10, lw=2)

    # ─── Arrows ───
    # DFSAR → ResNet
    arrow(2.7, 4.4, 3.6, 4.4)
    # Physics → Fusion (diagonal)
    arrow(2.7, 1.4, 7.5, 2.5)
    # ResNet → Fusion
    arrow(6.6, 4.4, 7.5, 4.0)
    # Fusion → heads
    for _, color, yy in heads:
        arrow(10.3, yy, 11.2, yy, color=color)

    fig.savefig(os.path.join(ASSETS_DIR, 'architecture_diagram.png'), **SAVE_KW)
    plt.close(fig)
    print("[OK] architecture_diagram.png")


# ─────────────────────────────────────────────────────────────
# 3. Results Comparison
# ─────────────────────────────────────────────────────────────
def make_results_comparison():
    fig = plt.figure(figsize=(10, 5), facecolor=BG)
    fig.suptitle('Model Performance & Key Results', fontsize=17,
                 fontweight='bold', color=WHITE, y=0.96)

    # ─── Left: Grouped bar chart ───
    ax1 = fig.add_axes([0.07, 0.12, 0.48, 0.72])
    ax1.set_facecolor('#0d0d24')

    metrics = ['F1 Score', 'Precision', 'Recall']
    east = [0.8428, 0.8906, 0.7999]
    west = [0.9383, 0.9412, 0.9353]

    x = np.arange(len(metrics))
    width = 0.32

    bars1 = ax1.bar(x - width/2, east, width, label='East',
                    color='#1565c0', edgecolor='#42a5f5', linewidth=1.2,
                    zorder=3)
    bars2 = ax1.bar(x + width/2, west, width, label='West',
                    color='#e65100', edgecolor='#ff9800', linewidth=1.2,
                    zorder=3)

    # Value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, h + 0.012,
                     f'{h:.3f}', ha='center', va='bottom',
                     fontsize=9, color=WHITE, fontweight='bold', zorder=4)

    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, fontsize=11, color=LIGHT_GRAY)
    ax1.set_ylim(0, 1.08)
    ax1.set_ylabel('Score', fontsize=11, color=LIGHT_GRAY)
    ax1.tick_params(axis='y', colors=LIGHT_GRAY, labelsize=9)
    ax1.spines['bottom'].set_color('#37474f')
    ax1.spines['left'].set_color('#37474f')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.yaxis.grid(True, color='#1a2a3a', linewidth=0.5, zorder=0)
    ax1.set_axisbelow(True)

    leg = ax1.legend(fontsize=10, loc='upper left',
                     facecolor='#0d0d24', edgecolor='#37474f',
                     labelcolor=WHITE)

    # ─── Right: Key results text box ───
    ax2 = fig.add_axes([0.60, 0.12, 0.37, 0.72])
    ax2.set_facecolor('#0d0d24')
    ax2.axis('off')

    # Draw border
    border = FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                            boxstyle="round,pad=0.03",
                            facecolor='#0d0d24', edgecolor='#37474f',
                            linewidth=1.5, transform=ax2.transAxes,
                            zorder=1)
    ax2.add_patch(border)

    ax2.text(0.5, 0.92, 'Key Mission Results', ha='center', va='center',
             fontsize=13, fontweight='bold', color=CYAN,
             transform=ax2.transAxes, zorder=2)

    # Divider
    ax2.plot([0.1, 0.9], [0.84, 0.84], color='#37474f', lw=1,
             transform=ax2.transAxes, zorder=2)

    results_text = [
        ('EAST POLE', '#42a5f5', [
            '8.4M m\u00b3  ice volume',
            '1.76 km  rover traverse',
            '93.3%   LOLA coverage',
        ]),
        ('WEST POLE', '#ff9800', [
            '5.6M m\u00b3  ice volume',
            '0.12 km  rover traverse',
            '24.6%   LOLA coverage',
        ]),
    ]

    y_pos = 0.75
    for title, color, items in results_text:
        ax2.text(0.12, y_pos, title, fontsize=11, fontweight='bold',
                 color=color, transform=ax2.transAxes, zorder=2)
        y_pos -= 0.06
        for item in items:
            ax2.text(0.15, y_pos, item, fontsize=9.5, color=LIGHT_GRAY,
                     transform=ax2.transAxes, zorder=2,
                     fontfamily='monospace')
            y_pos -= 0.065
        y_pos -= 0.04

    # Combined total
    ax2.plot([0.1, 0.9], [y_pos + 0.02, y_pos + 0.02], color='#37474f', lw=1,
             transform=ax2.transAxes, zorder=2)
    y_pos -= 0.04
    ax2.text(0.12, y_pos, 'TOTAL', fontsize=11, fontweight='bold',
             color='#4caf50', transform=ax2.transAxes, zorder=2)
    y_pos -= 0.06
    ax2.text(0.15, y_pos, '14.0M m\u00b3  combined ice volume',
             fontsize=9.5, color=WHITE, transform=ax2.transAxes, zorder=2,
             fontfamily='monospace', fontweight='bold')

    fig.savefig(os.path.join(ASSETS_DIR, 'results_comparison.png'), **SAVE_KW)
    plt.close(fig)
    print("[OK] results_comparison.png")


# ─────────────────────────────────────────────────────────────
# 4. Key Metrics Infographic
# ─────────────────────────────────────────────────────────────
def make_key_metrics():
    fig, ax = plt.subplots(figsize=(10, 3), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis('off')

    # Title
    ax.text(5, 2.7, 'LunarIceNet — At a Glance', ha='center', va='center',
            fontsize=16, fontweight='bold', color=WHITE)

    metrics = [
        ('12.4M',  'Model\nParameters',       CYAN,     1.25),
        ('140M+',  'DFSAR Pixels\nAnalyzed',   '#ffc107', 3.75),
        ('0.94',   'Best F1\nScore',           '#4caf50', 6.25),
        ('14M m\u00b3', 'Total Ice\nVolume',   '#e040fb', 8.75),
    ]

    for value, label, color, cx in metrics:
        # Glow circle behind number
        circle = plt.Circle((cx, 1.5), 0.85, facecolor=color, alpha=0.06,
                            edgecolor=color, linewidth=1.2, linestyle='--',
                            zorder=1)
        ax.add_patch(circle)

        # Big number
        ax.text(cx, 1.6, value, ha='center', va='center',
                fontsize=30, fontweight='bold', color=color, zorder=2,
                fontfamily='sans-serif',
                path_effects=[pe.withStroke(linewidth=1, foreground=BG)])

        # Label below
        ax.text(cx, 0.65, label, ha='center', va='center',
                fontsize=10, color=LIGHT_GRAY, zorder=2,
                fontfamily='sans-serif', linespacing=1.3)

    # Subtle separators
    for sx in [2.5, 5.0, 7.5]:
        ax.plot([sx, sx], [0.5, 2.3], color='#1a2a3a', lw=1, zorder=0)

    fig.savefig(os.path.join(ASSETS_DIR, 'key_metrics.png'), **SAVE_KW)
    plt.close(fig)
    print("[OK] key_metrics.png")


# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    make_pipeline_flow()
    make_architecture()
    make_results_comparison()
    make_key_metrics()
    print(f"\nAll assets saved to {ASSETS_DIR}")
