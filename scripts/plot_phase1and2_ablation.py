#!/usr/bin/env python3
"""
Publication Figure: Mechanistic Distillation Ablations (3-Panel)
Phase Importance + Phase 1 & Phase 2 Component Ablations
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------------
# Publication style fallback (exactly as in your template)
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
                'xtick.labelsize': 11,
                'ytick.labelsize': 11,
                'figure.titlesize': 15.5,
                'axes.linewidth': 1.25,
                'grid.alpha': 0.35,
                'grid.linewidth': 0.6,
            })

        @staticmethod
        def create_figure(figsize=(13.5, 14.5), dpi=300):   # taller for 3 vertical panels
            return plt.subplots(3, 1, figsize=figsize, dpi=dpi, sharex=False)

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


# ----------------------------------------------------------------------
# Main Plot Function - Panel-Specific Legends + Math Mode
# ----------------------------------------------------------------------
def create_three_panel_ablation_plot(
    figsize=(21.5, 8.0),      # Slightly wider for longer legends
    dpi=300,
    output_path="./publication_plots",
    filename="fig_mechanistic_distillation_ablations",
    formats=('pdf', 'png')
):
    pps.setup_publication_style()
    
    fig, axs = plt.subplots(1, 3, figsize=figsize, dpi=dpi, 
                            sharey=True, squeeze=True)
    
    # Consistent colors and hatches
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    hatches = ['', '//', 'xx', '++']

    panel_data = {
        'a': {
            'title': '(a) Phase Importance (Main Ablation)',
            'methods': [
                "Full Mech.",
                # "w/o Phase 1 ($\\lambda_{P1}=0$)",
                "w/o Phase 1",
                # "w/o Phase 2 ($\\lambda_{P2}=0$)",
                "w/o Phase 2",
                "Vanilla CoT"
            ],
            'mimic': [0.76, 0.72, 0.73, 0.69],
            'mimic_err': [0.012, 0.015, 0.014, 0.018],
            'wiki':  [0.79, 0.74, 0.75, 0.71],
            'wiki_err':  [0.010, 0.013, 0.012, 0.016],
        },
        'b': {
            'title': '(b) Phase 1 Term Ablations',
            'methods': [
                "Full",
                # "w/o P1 Attention ($L_{\\mathrm{attn}}^{P1}$)",
                "w/o P1 Attn.",
                # "w/o P1 Write ($L_{\\mathrm{write}}^{P1}$)",
                "w/o P1 Write",
                "w/o P1 Inter."
            ],
            'mimic': [0.76, 0.71, 0.72, 0.73],
            'mimic_err': [0.011, 0.016, 0.014, 0.017],
            'wiki':  [0.79, 0.74, 0.75, 0.76],
            'wiki_err':  [0.009, 0.014, 0.013, 0.015],
        },
        'c': {
            'title': '(c) Phase 2 Term Ablations',
            'methods': [
                "Full",
                # "w/o P2 QK-loop ($L_{\\mathrm{loop}}^{P2}$)",
                "w/o P2 loop",
                # "w/o P2 OV-write ($L_{\\mathrm{write}}^{P2}$)",
                "w/o P2 write",
                # "w/o P2 Interaction ($L_{\\mathrm{int}}^{P2}$)"
                "w/o P2 Inter."
            ],
            'mimic': [0.76, 0.72, 0.71, 0.73],
            'mimic_err': [0.013, 0.012, 0.015, 0.019],
            'wiki':  [0.79, 0.75, 0.74, 0.76],
            'wiki_err':  [0.011, 0.010, 0.014, 0.017],
        }
    }

    x = np.arange(2)          # MIMIC-III and Wiki
    width = 0.19
    spacing = 0.025

    for i, panel in enumerate(['a', 'b', 'c']):
        ax = axs[i]
        data = panel_data[panel]
        
        # Plot MIMIC group (left)
        for j in range(4):
            pos = x[0] - 1.5*width - spacing + j*(width + spacing/2)
            # ax.bar(pos, data['mimic'][j], width,
            #        color=colors[j], hatch=hatches[j], edgecolor='#2C3E50',
            #        linewidth=1.05, alpha=0.93)
            ax.bar(
                pos,
                data['mimic'][j],
                width,
                yerr=data['mimic_err'][j],   
                capsize=4,                    
                ecolor='#1A1A1A',             
                error_kw={'elinewidth': 1.4, 'capthick': 1.6},
                color=colors[j],
                hatch=hatches[j],
                edgecolor='#2C3E50',
                linewidth=1.05,
                alpha=0.93
            )

        # Plot Wiki group (right)
        for j in range(4):
            pos = x[1] - 1.5*width - spacing + j*(width + spacing/2)
            # ax.bar(pos, data['wiki'][j], width,
            #        color=colors[j], hatch=hatches[j], edgecolor='#2C3E50',
            #        linewidth=1.05, alpha=0.93)
            ax.bar(
                pos,
                data['wiki'][j],
                width,
                yerr=data['wiki_err'][j],   
                capsize=4,                    
                ecolor='#1A1A1A',             
                error_kw={'elinewidth': 1.4, 'capthick': 1.6},
                color=colors[j],
                hatch=hatches[j],
                edgecolor='#2C3E50',
                linewidth=1.05,
                alpha=0.93
            )

        # Value labels
        for j in range(4):
            pos_m = x[0] - 1.5*width - spacing + j*(width + spacing/2)
            ax.text(pos_m, data['mimic'][j] + 0.020, f'{data["mimic"][j]:.2f}',
                    ha='center', va='bottom', fontsize=18.6, fontweight='bold')
            
            pos_w = x[1] - 1.5*width - spacing + j*(width + spacing/2)
            ax.text(pos_w, data['wiki'][j] + 0.020, f'{data["wiki"][j]:.2f}',
                    ha='center', va='bottom', fontsize=18.6, fontweight='bold')

        # Panel title
        ax.set_title(data['title'], fontsize=24.2, fontweight='bold', pad=5)
        
        # X-axis: Dataset names
        ax.set_xticks(x)
        ax.set_xticklabels(['MIMIC-IV', 'WikiSeeAlso'], fontsize=20.8, fontweight='semibold')
        
        if i == 0:
            ax.set_ylabel('Leakage Adjusted Simulability (%)', fontsize=22.0, labelpad=16)
        
        ax.set_ylim(0.64, 0.83)
        ax.set_yticks(np.arange(0.66, 0.83, 0.04))

        # === Panel-specific Legend ===
        legend = ax.legend(
            [plt.Rectangle((0,0),1,1, facecolor=colors[j], hatch=hatches[j], edgecolor='#2C3E50') 
             for j in range(4)],
            data['methods'],
            loc='upper left',
            fontsize=22.8,
            ncol=1,
            frameon=True,
            edgecolor='#333333',
            facecolor='white',
            # title="Condition",
            title_fontsize=12.8,
            bbox_to_anchor=(0.14, 0.98),

            # spacing controls
            labelspacing=0.25,
            handlelength=1.0,
            handletextpad=0.35,
            borderpad=0.25,
            columnspacing=0.6,
            borderaxespad=0.03,
        )
        legend.get_frame().set_linewidth(0.85)

        pps.style_axes(ax, fig)

    # fig.suptitle('Mechanistic Distillation Ablation Study — LLaMA-70B',
    #              fontsize=22.0, fontweight='bold', y=1.12)

    plt.tight_layout(pad=2.8, w_pad=4.2)
    pps.save_publication_figure(fig, output_path, filename, formats=formats)
    plt.close(fig)

    # LaTeX Output
    print("\n" + "="*110)
    print("LaTeX CAPTION:")
    print("="*110)
    print(r"""\caption{Three-panel ablation study of our mechanistic distillation framework. 
Each panel shows grouped bars for MIMIC-III and Wiki benchmarks. The four bars per dataset use 
consistent color and hatching scheme corresponding to different ablation conditions.}
\label{fig:mechanistic-distillation-ablations}""")
    print("\n" + "="*110)



# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Mechanistic Distillation 3-Panel Ablation Plot")
    parser.add_argument('--output-path', type=str, default='publication_plots')
    parser.add_argument('--filename', type=str, default='fig_mechanistic_distillation_ablations')
    parser.add_argument('--figsize', nargs=2, type=float, default=[13.8, 15.2])
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--formats', nargs='+', default=['pdf', 'png'],
                        choices=['pdf', 'png', 'svg', 'eps'])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_three_panel_ablation_plot(
        figsize=tuple(args.figsize),
        dpi=args.dpi,
        output_path=args.output_path,
        filename=args.filename,
        formats=args.formats
    )
    print("\n✅ Three-panel mechanistic distillation ablation figure generated successfully!")