#!/usr/bin/env python3
"""
Publication-quality plot: Excess Kurtosis of Attention Distributions
Figure: Early-layer heads dominate top kurtosis ranks with sharp peaks on anchors
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec

# ----------------------------------------------------------------------
# Publication Style Fallback (consistent with your previous plots)
# ----------------------------------------------------------------------
try:
    import publication_plot_style as pps
except ImportError:
    print("Warning: publication_plot_style.py not found. Using fallback.")
    import matplotlib as mpl
    class pps:
        @staticmethod
        def setup_publication_style():
            try:
                plt.style.use('seaborn-v0_8-paper')
            except Exception:
                pass
            mpl.rcParams.update({
                'font.family': 'sans-serif',
                'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
                'font.size': 11,
                'axes.labelsize': 12.5,
                'axes.titlesize': 14.5,
                'xtick.labelsize': 10.8,
                'ytick.labelsize': 10.8,
                'legend.fontsize': 11.2,
                'figure.titlesize': 15.5,
                'axes.linewidth': 1.25,
                'grid.alpha': 0.35,
                'grid.linewidth': 0.6,
            })

        @staticmethod
        def create_figure(figsize=(11.5, 7.2), dpi=300):
            return plt.subplots(figsize=figsize, dpi=dpi)

        @staticmethod
        def style_axes(ax, fig):
            ax.set_facecolor('#FAFAFA')
            fig.patch.set_facecolor('white')
            for spine in ax.spines.values():
                spine.set_edgecolor('#333333')
                spine.set_linewidth(1.25)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.yaxis.grid(True, alpha=0.35, linestyle='--', linewidth=0.7)
            ax.set_axisbelow(True)

        @staticmethod
        def save_publication_figure(fig, output_path, filename, formats=('pdf', 'png')):
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            for fmt in formats:
                path = output_path / f"{filename}.{fmt}"
                fig.savefig(path, format=fmt, dpi=400, bbox_inches='tight',
                            facecolor='white', edgecolor='none')
                print(f"Saved: {path}")


# ----------------------------------------------------------------------
# Main Plot Function
# ----------------------------------------------------------------------
def create_phase1_attn_kurtosis_plot(
    figsize=(11.8, 7.8),
    dpi=300,
    output_path="./publication_plots",
    filename="fig_phase1_attn_heads_kurtosis",
    formats=('pdf', 'png')
):
    pps.setup_publication_style()
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = gridspec.GridSpec(1, 2, width_ratios=[3.2, 1], wspace=0.12)

    # Main bar plot (left)
    ax = fig.add_subplot(gs[0])

    # Example data: Top-ranked heads (realistic pattern for your story)
    # Early layers dominate top ranks, then sharp drop-off
    head_labels = [
        'L2H3', 'L3H1', 'L1H7', 'L4H2', 'L5H4', 'L2H8',
        'L3H5', 'L6H0', 'L7H3', 'L4H9', 'L8H1', 'L9H2',
        'L11H4', 'L12H6', 'L14H0'
    ]

    # Excess kurtosis values (high in early layers → sharp drop)
    kurtosis_values = [18.4, 16.7, 15.2, 13.9, 12.1, 10.8,
                       9.3, 7.2, 5.8, 4.9, 3.1, 2.4, 1.2, 0.8, 0.3]

    # Layer-based coloring: Early (blue), Mid (purple-gray), Late (red)
    colors = []
    for lbl in head_labels:
        layer = int(lbl[1:lbl.find('H')])
        if layer <= 5:
            colors.append('#4A90E2')      # Early - light/medium blue
        elif layer <= 10:
            colors.append('#8E44AD')      # Mid - purple
        else:
            colors.append('#E74C3C')      # Late - red

    # Plot bars (horizontal for better label readability with long names)
    y_pos = np.arange(len(head_labels))
    bars = ax.barh(y_pos, kurtosis_values, color=colors, edgecolor='#2C3E50',
                   linewidth=1.1, height=0.72, alpha=0.93, zorder=3)

    # Value labels
    for bar, val in zip(bars, kurtosis_values):
        ax.text(val + 0.6, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}', va='center', ha='left',
                fontsize=10.5, fontweight='bold', color='#222222')

    # Baselines
    ax.axvline(x=0, color='#7F8C8D', linestyle='--', linewidth=1.4, alpha=0.85, zorder=2,
               label='Mesokurtic (normal)')
    ax.axvline(x=3, color='#C0392B', linestyle='--', linewidth=1.4, alpha=0.85, zorder=2,
               label='Leptokurtic threshold')

    # Axes & labels
    ax.set_xlabel('Excess Kurtosis of Attention Distribution\n(Early CoT Queries)', 
                  fontsize=13.2, labelpad=14)
    ax.set_ylabel('Ranked Attention Heads (highest → lowest kurtosis)', 
                  fontsize=13.2, labelpad=12)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(head_labels, fontsize=11.2, fontfamily='monospace')
    
    ax.set_xlim(-1, 21)
    ax.set_title('Early-Layer Heads Exhibit Sharply Peaked Attention\nto Hierarchical Anchors in Initial CoT Steps',
                 fontsize=15.2, fontweight='bold', pad=22)

    # Legend for color groups
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4A90E2', edgecolor='black', label='Early Layers (L1–5)'),
        Patch(facecolor='#8E44AD', edgecolor='black', label='Mid Layers (L6–10)'),
        Patch(facecolor='#E74C3C', edgecolor='black', label='Late Layers (L11+)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11,
              frameon=True, edgecolor='#333333', facecolor='white')

    pps.style_axes(ax, fig)

    # Final layout
    plt.tight_layout(pad=1.8)
    
    pps.save_publication_figure(fig, output_path, filename, formats=formats)
    plt.close(fig)
    print("\n✅ Phase 1 Attention Kurtosis figure generated successfully!")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Excess Kurtosis of Attention Heads (Mechanistic Reasoning)")
    parser.add_argument('--output-path', type=str, default='publication_plots')
    parser.add_argument('--filename', type=str, default='fig_phase1_attn_heads_kurtosis')
    parser.add_argument('--figsize', nargs=2, type=float, default=[11.8, 7.8])
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--formats', nargs='+', default=['pdf', 'png'],
                        choices=['pdf', 'png', 'svg', 'eps'])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_phase1_attn_kurtosis_plot(
        figsize=tuple(args.figsize),
        dpi=args.dpi,
        output_path=args.output_path,
        filename=args.filename,
        formats=args.formats
    )   