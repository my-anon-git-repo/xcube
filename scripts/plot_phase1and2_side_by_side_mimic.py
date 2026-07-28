#!/usr/bin/env python3
"""
RQ3 - Reasoning Focus vs Near-Miss Confusion (Coarse-to-Fine Dynamics)
MIMIC-IV dataset
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------------
# Publication style fallback (exact same as your template)
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
        def create_figure(figsize=(12.5, 7.2), dpi=300):
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
        def save_publication_figure(fig, output_path, filename, formats=('png', 'pdf')):
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
def create_reasoning_dynamics_plot(
    figsize=(13.8, 6.8),
    dpi=300,
    output_path="./publication_plots",
    filename="fig_reasoning_focus_confusion",
    formats=('pdf', 'png')
):
    pps.setup_publication_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=dpi, sharex=True)
    
    # Data (MIMIC-IV)

    t = np.array([0.00,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,
                  0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00])
    
    focus_teacher = np.array([0.460,0.664,0.798,0.881,0.882,0.901,0.948,0.938,0.917,0.939,
                              0.920,0.920,0.934,0.892,0.895,0.919,0.910,0.936,0.912,0.902,0.950])
    
    focus_ours    = np.array([0.392,0.565,0.638,0.712,0.765,0.768,0.812,0.799,0.815,0.812,
                              0.855,0.821,0.799,0.837,0.797,0.825,0.783,0.795,0.826,0.837,0.825])
    
    focus_p1      = np.array([0.373,0.522,0.598,0.675,0.719,0.773,0.776,0.746,0.792,0.783,
                              0.780,0.806,0.816,0.814,0.781,0.791,0.803,0.816,0.788,0.794,0.777])
    
    focus_p2      = np.array([0.356,0.451,0.506,0.514,0.565,0.577,0.577,0.613,0.650,0.630,
                              0.671,0.594,0.669,0.659,0.656,0.667,0.628,0.666,0.679,0.703,0.664])
    
    focus_vanilla = np.array([0.334,0.378,0.438,0.454,0.461,0.503,0.512,0.545,0.525,0.543,
                              0.552,0.539,0.581,0.587,0.587,0.587,0.567,0.591,0.595,0.588,0.603])

    conf_teacher = np.array([0.626,0.648,0.623,0.624,0.619,0.591,0.620,0.621,0.657,0.617,
                             0.625,0.435,0.300,0.259,0.206,0.176,0.131,0.153,0.103,0.128,0.149])
    
    conf_ours    = np.array([0.658,0.671,0.679,0.671,0.658,0.679,0.662,0.684,0.664,0.699,
                             0.664,0.512,0.423,0.321,0.295,0.280,0.214,0.227,0.218,0.220,0.185])
    
    conf_p1      = np.array([0.620,0.648,0.644,0.644,0.645,0.630,0.643,0.644,0.629,0.668,
                             0.647,0.607,0.621,0.584,0.599,0.594,0.555,0.574,0.558,0.557,0.567])
    
    conf_p2      = np.array([0.691,0.684,0.682,0.683,0.694,0.700,0.699,0.707,0.695,0.717,
                             0.691,0.621,0.508,0.429,0.385,0.380,0.349,0.349,0.336,0.321,0.304])
    
    conf_vanilla = np.array([0.627,0.643,0.663,0.653,0.631,0.653,0.656,0.637,0.652,0.651,
                             0.633,0.646,0.640,0.640,0.632,0.589,0.589,0.605,0.599,0.594,0.639])

    t_star = 0.50

    # Colors (publication-friendly)
    colors = {
        'Teacher': '#1C1C1C',
        'Full (Ours)': '#1E4D9C',      # deep blue
        'Phase 1 only': '#E67E22',     # orange
        'Phase 2 only': '#9B59B6',     # purple
        'Vanilla CoT': '#7F8C8D'       # gray
    }

    linewidths = {'Teacher': 3.2, 'Full (Ours)': 2.8, 'others': 2.1}
    linestyles = {'Teacher': '--', 'others': '-'}

    # === Left Panel: Reasoning Focus ===
    ax1.plot(t, focus_teacher, color=colors['Teacher'], lw=linewidths['Teacher'],
             linestyle=linestyles['Teacher'], label='Teacher (70B)', zorder=5)
    ax1.plot(t, focus_ours,    color=colors['Full (Ours)'], lw=linewidths['Full (Ours)'],
             linestyle=linestyles['others'], label='Full (Ours)', zorder=4)
    ax1.plot(t, focus_p1,      color=colors['Phase 1 only'], lw=linewidths['others'],
             linestyle=linestyles['others'], label='CoT + Phase 1 only', zorder=3)
    ax1.plot(t, focus_p2,      color=colors['Phase 2 only'], lw=linewidths['others'],
             linestyle=linestyles['others'], label='CoT + Phase 2 only', zorder=3)
    ax1.plot(t, focus_vanilla, color=colors['Vanilla CoT'], lw=linewidths['others'],
             linestyle=linestyles['others'], label='SemCoT', zorder=2)

    ax1.set_ylabel('Reasoning Focus\n(higher ↑ better)', fontsize=20.5, labelpad=12)
    ax1.set_ylim(0.25, 1.02)

    # === Right Panel: Near-Miss Confusion ===
    ax2.plot(t, conf_teacher, color=colors['Teacher'], lw=linewidths['Teacher'],
             linestyle=linestyles['Teacher'], label='Teacher (70B)', zorder=5)
    ax2.plot(t, conf_ours,    color=colors['Full (Ours)'], lw=linewidths['Full (Ours)'],
             linestyle=linestyles['others'], label='Full (Ours)', zorder=4)
    ax2.plot(t, conf_p1,      color=colors['Phase 1 only'], lw=linewidths['others'],
             linestyle=linestyles['others'], label='Phase 1 only (Ours)', zorder=3)
    ax2.plot(t, conf_p2,      color=colors['Phase 2 only'], lw=linewidths['others'],
             linestyle=linestyles['others'], label='Phase 2 only (Ours)', zorder=3)
    ax2.plot(t, conf_vanilla, color=colors['Vanilla CoT'], lw=linewidths['others'],
             linestyle=linestyles['others'], label='SemCoT', zorder=2)

    ax2.set_ylabel('Near-Miss Confusion\n(lower ↓ better)', fontsize=20.5, labelpad=12)
    ax2.set_ylim(0.05, 0.72)

    # Shared X-axis
    for ax in (ax1, ax2):
        ax.set_xlabel('Normalized CoT Position $t \\in [0,1]$', fontsize=20.5, labelpad=14)
        ax.set_xlim(-0.02, 1.02)
        ax.set_xticks([0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0])
        ax.set_xticklabels(['0.0', '0.2', '0.4', '$t^*$', '0.6', '0.8', '1.0'])
        
        # Phase boundary
        ax.axvline(x=t_star, color='#2C3E50', linestyle='--', lw=1.8, alpha=0.85, zorder=1)
        
        # Phase shading
        ax.axvspan(0, t_star, alpha=0.08, color='#3498DB', zorder=0)   # Phase 1 light blue
        ax.axvspan(t_star, 1, alpha=0.08, color='#E74C3C', zorder=0)   # Phase 2 light red

    # Legend (shared, placed in right panel)
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=5, fontsize=17.2,
               bbox_to_anchor=(0.525, 1.02), frameon=True, edgecolor='#333333', facecolor='white')

    # Style both axes
    pps.style_axes(ax1, fig)
    pps.style_axes(ax2, fig)

    plt.tight_layout(pad=2.8, rect=[0, 0, 1, 0.96])  # leave space for legend
    pps.save_publication_figure(fig, output_path, filename, formats=formats)
    plt.close(fig)

    # ====================== LaTeX Output ======================
    print("\n" + "="*95)
    print("LaTeX CAPTION:")
    print("="*95)
    print(r"""\caption{\textbf{RQ3: How do phase-specific losses shape the temporal dynamics of reasoning?}
Figure shows cumulative metrics along the normalized Chain-of-Thought trajectory \( t \in [0,1] \).
(a) \emph{Reasoning Focus} (Phase~1, \( t \le t^* \)) measures alignment with dominant semantic anchors \( c(x) \).
(b) \emph{Near-Miss Confusion} (Phase~2, \( t \ge t^* \)) measures alignment with plausible but incorrect alternatives \( \mathcal{N}(x) \).
At each position \( t \), both metrics are computed on the cumulative CoT prefix up to token \( t \).
The teacher exhibits clear coarse-to-fine structure. Only our full mechanistic distillation successfully reproduces both phases.}
\label{fig:reasoning_focus_confusion}""")
    print("\n" + "="*95)
    print("LaTeX PARAGRAPH (you provided earlier - ready to paste):")
    print("="*95)
    print(r"""\textbf{RQ3: How do phase-specific losses shape the temporal dynamics of reasoning?}
Figure~\ref{fig:reasoning_focus_confusion} dissects the coarse-to-fine structure of reasoning...""")  # (truncated for brevity - full text from your message)
    print("="*95)

# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="RQ3 Reasoning Dynamics Plot")
    parser.add_argument('--output-path', type=str, default='publication_plots')
    parser.add_argument('--filename', type=str, default='fig_reasoning_focus_confusion')
    parser.add_argument('--figsize', nargs=2, type=float, default=[13.8, 6.8])
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--formats', nargs='+', default=['pdf', 'png'],
                        choices=['pdf', 'png', 'svg', 'eps'])
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    create_reasoning_dynamics_plot(
        figsize=tuple(args.figsize),
        dpi=args.dpi,
        output_path=args.output_path,
        filename=args.filename,
        formats=args.formats
    )
    print("\n✅ RQ3 Reasoning Focus vs Confusion plot generated successfully!")