#!/usr/bin/env python3
"""
Phase 2 - Temporal Dynamics of Top Refinement Heads (3-Subplot Layout)
Cleaner version with separate panels for QK-Shortlist, QK-Nearmiss, and OV Helpfulness
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
                'font.family': 'sans-serif',
                'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
                'font.size': 11,
                'axes.labelsize': 12.5,
                'axes.titlesize': 13.5,
                'xtick.labelsize': 11,
                'ytick.labelsize': 11,
                'figure.titlesize': 15,
                'axes.linewidth': 1.25,
                'grid.alpha': 0.35,
                'grid.linewidth': 0.6,
            })

        @staticmethod
        def create_figure(figsize=(13.5, 8.5), dpi=300):
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


def create_phase2_temporal_dynamics_plot(
    figsize=(13.8, 8.8),
    dpi=300,
    output_path="./publication_plots",
    filename="fig_phase2_refine_temporal_dynamics",
    formats=('pdf', 'png')
):
    pps.setup_publication_style()
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1], hspace=0.35, wspace=0.28)

    # Top-left: QK Shortlist Similarity
    ax1 = fig.add_subplot(gs[0, 0])
    # Top-right: QK Near-miss Similarity
    ax2 = fig.add_subplot(gs[0, 1])
    # Bottom: OV Helpfulness (spans both columns)
    ax3 = fig.add_subplot(gs[1, 0])

    # Manually center ax3 and make it same width as top plots
    pos1 = ax1.get_position()   # left top plot position
    pos2 = ax2.get_position()   # right top plot position

    # Calculate centered position with same width as one top plot
    width = pos1.width
    left = (pos1.x0 + pos2.x1)/2 - width/2     # center between the two top plots
    bottom = gs[1, 0].get_position(fig).y0     # bottom row position
    # === Drag ax3 down here ===
    drag_amount = 0.04   # ← Adjust this value (positive = move down)
    new_bottom = bottom - drag_amount

    ax3.set_position([left, new_bottom, width, pos1.height * 1.1])

    cot_labels = ['Early\n(1-10)', 'Early-Mid\n(11-20)', 'Mid\n(21-35)', 
                  'Mid-Late\n(36-50)', 'Late\n(51+)']
    x = np.arange(len(cot_labels))

    # Top 4 heads with more distinct colors
    heads = ['L14H7', 'L16H23', 'L13H4', 'L15H12']
    colors = ['#6B4E9E', '#B07CDA', '#E74C3C', '#C0392B']   # Distinct purple → red

    # Data
    qk_shortlist = {'L14H7': [0.38,0.51,0.67,0.79,0.85],
                    'L16H23':[0.35,0.48,0.64,0.76,0.82],
                    'L13H4': [0.41,0.53,0.62,0.71,0.77],
                    'L15H12':[0.36,0.49,0.65,0.74,0.80]}

    qk_nearmiss = {'L14H7': [0.52,0.44,0.31,0.22,0.18],
                   'L16H23':[0.49,0.42,0.29,0.20,0.15],
                   'L13H4': [0.55,0.46,0.35,0.26,0.21],
                   'L15H12':[0.50,0.43,0.30,0.23,0.17]}

    ov_help = {'L14H7': [0.12,0.28,0.45,0.61,0.68],
               'L16H23':[0.09,0.25,0.41,0.57,0.65],
               'L13H4': [0.15,0.31,0.48,0.59,0.63],
               'L15H12':[0.11,0.26,0.43,0.55,0.62]}

    # Plot QK Shortlist (Top-Left)
    for i, head in enumerate(heads):
        ax1.plot(x, qk_shortlist[head], color=colors[i], lw=2.4, marker='o', ms=6, label=head)
    ax1.set_title('QK Similarity to Prior Shortlist', fontsize=29.5, pad=12)
    ax1.set_ylabel('Similarity', fontsize=27.5)
    ax1.set_ylim(0.3, 0.9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(cot_labels, fontsize=24.5)

    # Plot QK Near-miss (Top-Right)
    for i, head in enumerate(heads):
        ax2.plot(x, qk_nearmiss[head], color=colors[i], lw=2.4, linestyle='--', marker='s', ms=6, label=head)
    ax2.set_title('QK Similarity to Prior Near-Miss', fontsize=29.5, pad=12)
    ax2.set_ylabel('Similarity', fontsize=27.5)
    ax2.set_ylim(0.1, 0.6)
    ax2.set_xticks(x)
    ax2.set_xticklabels(cot_labels, fontsize=24.5)

    # Plot OV Helpfulness (Bottom, centered)
    for i, head in enumerate(heads):
        ax3.plot(x, ov_help[head], color=colors[i], lw=2.6, marker='D', ms=7, label=head)
    ax3.set_title('OV Helpfulness (Margin Widening)', fontsize=29.5, pad=12)
    # ax3.set_ylabel('cos(Δ$^h$, ∇margin)', fontsize=12.5)
    ax3.set_ylabel(r"$\cos(\Delta^h, \nabla \mathrm{margin})$", fontsize=27.5)
    ax3.set_xlabel('Chain-of-Thought Generation Stage', fontsize=27.5, labelpad=12)
    ax3.set_ylim(-0.05, 0.75)
    # ax3.axhline(y=0, color='gray', linestyle='--', lw=1.3, alpha=0.7, label='Neutral')
    ax3.set_xticks(x)
    ax3.set_xticklabels(cot_labels, fontsize=24.5)

    # Legend (placed in bottom plot)
    # ax3.legend(title="Top Refinement Heads", loc='upper left', fontsize=11, 
    #            title_fontsize=11.5, frameon=True, edgecolor='#333333')
    legend = ax1.legend(
        # title="Top Refinement Heads",
        loc='upper left',
        fontsize=18,
        title_fontsize=19.5,
        frameon=True,
        fancybox=True,
        framealpha=1.0,
        edgecolor='black',
        facecolor='white'
    )
    legend.get_frame().set_linewidth(0.8)
    legend.get_title().set_weight('bold')

    legend = ax2.legend(
        # title="Top Refinement Heads",
        loc='upper right',
        fontsize=18,
        title_fontsize=19.5,
        frameon=True,
        fancybox=True,
        framealpha=1.0,
        edgecolor='black',
        facecolor='white'
    )
    legend.get_frame().set_linewidth(0.8)
    legend.get_title().set_weight('bold')

    legend = ax3.legend(
        # title="Top Refinement Heads",
        loc='lower right',
        fontsize=18,
        title_fontsize=19.5,
        frameon=True,
        fancybox=True,
        framealpha=1.0,
        edgecolor='black',
        facecolor='white'
    )

    legend.get_frame().set_linewidth(0.8)
    legend.get_title().set_weight('bold')

    # fig.suptitle("Phase 2 Temporal Dynamics: QK Loop-back & OV Helpfulness", fontsize=27.5, y=0.98, fontweight='bold')


    # Apply consistent styling
    for ax in [ax1, ax2, ax3]:
        pps.style_axes(ax, fig)

    plt.tight_layout(pad=2.0)

    pps.save_publication_figure(fig, output_path, filename, formats=formats)
    plt.close(fig)

    # ====================== LaTeX Output ======================
    print("\n" + "="*95)
    print("LaTeX CAPTION:")
    print("="*95)
    print(r"""\caption{Temporal dynamics of the top-ranked refinement heads during Chain-of-Thought generation (right panel of Figure~\ref{fig:phase2-identify-iter-refine-heads}). 
The top row shows QK similarity to prior shortlist representations (left, increasing) and to prior near-miss representations (right, decreasing). 
The bottom panel shows the growing OV-helpfulness of these heads (how effectively their residual updates widen the logit margin between target and near-miss clinical subcategories). 
Late-layer heads exhibit stronger divergence and more helpful updates as reasoning progresses.}""")

    print("\n" + "="*95)
    print("LaTeX PARAGRAPH:")
    print("="*95)
    print(r"""Figure~\ref{fig:phase2-identify-iter-refine-heads} (right) reveals the temporal dynamics of the top RefineScore-ranked heads. 
As CoT generation advances, these heads show progressively stronger re-attention to their own prior shortlist representations while simultaneously reducing attention to near-miss subcategories. 
At the same time, their residual stream updates (OV) become increasingly effective at widening the logit margin between the target diagnosis and competing alternatives. 
This coordinated behavior underscores the role of mid-to-late layer heads in iterative refinement of clinical hypotheses.""")
    print("="*95)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-path', type=str, default='publication_plots')
    parser.add_argument('--filename', type=str, default='fig_phase2_refine_temporal_dynamics')
    parser.add_argument('--figsize', nargs=2, type=float, default=[13.8, 8.8])
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--formats', nargs='+', default=['pdf', 'png'])
    args = parser.parse_args()

    create_phase2_temporal_dynamics_plot(
        figsize=tuple(args.figsize),
        dpi=args.dpi,
        output_path=args.output_path,
        filename=args.filename,
        formats=args.formats
    )
    print("\n✅ Phase 2 Temporal Dynamics (3-subplot) plot generated successfully!")