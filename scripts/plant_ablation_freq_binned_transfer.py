#!/usr/bin/env python3
"""
Fixed & improved version – works with Matplotlib 3.5+
Grouped bar plot: 4 LLMs × 3 pretraining conditions on Rare-Label F1
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------------
# Publication style fallback
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
                'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 14,
                'xtick.labelsize': 10.5, 'ytick.labelsize': 10.5, 'legend.fontsize': 11.5,
                'figure.titlesize': 15, 'axes.linewidth': 1.2,
                'grid.alpha': 0.3, 'grid.linewidth': 0.5
            })
        @staticmethod
        def create_figure(figsize=(10, 6.18), dpi=300):
            return plt.subplots(figsize=figsize, dpi=dpi)
        @staticmethod
        def style_axes(ax, fig):
            ax.set_facecolor('#FAFAFA')
            fig.patch.set_facecolor('white')
            for spine in ax.spines.values():
                spine.set_edgecolor('#333333')
                spine.set_linewidth(1.2)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.yaxis.grid(True, alpha=0.35, linestyle='--', linewidth=0.7)
            ax.set_axisbelow(True)
        @staticmethod
        def get_color_palette():
            return {
                'mistral': '#3498DB', 'deepseek': '#9B59B6',
                'llama': '#27AE60', 'phi': '#E91E63'
            }
        @staticmethod
        def save_publication_figure(fig, output_path, filename, formats=('png', 'pdf')):
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            for fmt in formats:
                path = output_path / f"{filename}.{fmt}"
                fig.savefig(path, format=fmt, dpi=400, bbox_inches='tight',
                            facecolor='white', edgecolor='none')
                print(f"Saved: {path}")

# ----------------------------------------------------------------------
# Main plot function (FIXED legend + better spacing)
# ----------------------------------------------------------------------
def create_llm_plant_grouped_bar(
    figsize=(10, 6.18),
    dpi=400,
    output_path="./publication_plots",
    filename="fig_llm_plant_ablation_rare_f1",
    formats=('pdf', 'png')
):
    # Data: [Full PLANT, Seeded-Only, Random-Only]
    mistral   = [14.0,  7.3,  0.8]
    deepseek  = [15.5,  8.1,  1.1]
    llama3    = [16.8,  8.0,  0.5]   # Your reference
    phi3      = [13.2,  6.9,  0.9]

    errors = [
        [0.6, 0.9, 0.7],  # Mistral
        [0.7, 1.0, 0.8],  # DeepSeek
        [0.5, 0.9, 0.7],  # LLaMA3
        [0.8, 1.1, 0.9],  # Phi-3
    ]

    labels = ['PLANT\n(Full Data)', 'PLANT\n(Common Subset)', 'Random Attn Init\n(Common Subset)']
    models = ['Mistral-7B', 'DeepSeek-V3', 'LLaMA3-8B', 'Phi-3']
    data = [mistral, deepseek, llama3, phi3]
    colors = ['#3498DB', '#9B59B6', '#27AE60', '#E91E63']

    pps.setup_publication_style()
    fig, ax = pps.create_figure(figsize=figsize, dpi=dpi)

    bar_width = 0.19
    x = np.arange(len(labels))

    for i, (model_data, model_err, color, model_name) in enumerate(zip(data, errors, colors, models)):
        offset = (i - 1.5) * bar_width
        bars = ax.bar(
            x + offset,
            model_data,
            yerr=model_err,
            width=bar_width,
            label=model_name,
            color=color,
            edgecolor='black',
            linewidth=1.4,
            capsize=6,
            error_kw={'elinewidth': 2.0, 'capthick': 2.0},
            alpha=0.94,
            zorder=3
        )

        # Value labels
        for bar, val, err in zip(bars, model_data, model_err):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + err + 0.5,
                f'{val:.1f}',
                ha='center', va='bottom',
                fontsize=11.5, fontweight='bold', color='black'
            )

    # Axes
    ax.set_ylabel('Stage 2 Rare-Label F1 (%)', fontsize=13, fontweight='medium')
    ax.set_title(
        'Effect of PLANT on Rare-Label Performance\n'
        'Across Four LLMs on the MIMIC-IV Dataset',
        fontsize=15, fontweight='bold', pad=20
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11.8, linespacing=1.4)
    ax.set_ylim(0, 20)
    ax.set_yticks(np.arange(0, 21, 4))

    # FIXED LEGEND – compatible with Matplotlib ≥3.5
    legend = ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=4,
        fontsize=12,
        frameon=True,
        fancybox=False,
        edgecolor='black',
        facecolor='white',
        framealpha=1.0
    )
    legend.get_frame().set_linewidth(1.2)

    # Final style
    pps.style_axes(ax, fig)
    plt.tight_layout(pad=1.8)

    # Save
    pps.save_publication_figure(fig, output_path, filename, formats=formats)
    plt.close(fig)

# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="LLM × Pretraining Condition grouped bar plot")
    parser.add_argument('--output-path', type=str, default='publication_plots')
    parser.add_argument('--filename', type=str, default='fig_llm_plant_ablation_rare_f1')
    parser.add_argument('--figsize', nargs=2, type=float, default=[10.0, 6.18])
    parser.add_argument('--dpi', type=int, default=400)
    parser.add_argument('--formats', nargs='+', default=['pdf', 'png'],
                        choices=['pdf', 'png', 'svg', 'eps'])
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    create_llm_plant_grouped_bar(
        figsize=tuple(args.figsize),
        dpi=args.dpi,
        output_path=args.output_path,
        filename=args.filename,
        formats=args.formats
    )
    print("\nGrouped bar plot generated successfully!")