#!/usr/bin/env python3
"""
Create publication-quality plots for MIMIC dataset analysis.
Shows the effect of varying Stage 1 pretraining length on Stage 2 performance.
Usage: python stage1_plant_pretraining_effect.py --data-path ./mimic_low_data/ --filename stage1_epoch_sweep2.csv --output_path ../tmp/publication_plots/ --verbose --layout stacked
"""
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
# Import the publication style module
try:
    import publication_plot_style as pps
except ImportError:
    print("Warning: publication_plot_style.py not found. Using fallback that emulates it.")
    import matplotlib as mpl
    class pps: # minimal emulation of your API (golden ratio, legend, saver)
        @staticmethod
        def setup_publication_style():
            try:
                plt.style.use('seaborn-v0_8-paper')
            except Exception:
                pass
            # Serif-ish defaults similar to your module
            mpl.rcParams['font.size'] = 11
            mpl.rcParams['axes.labelsize'] = 12
            mpl.rcParams['axes.titlesize'] = 14
            mpl.rcParams['xtick.labelsize'] = 10
            mpl.rcParams['ytick.labelsize'] = 10
            mpl.rcParams['legend.fontsize'] = 10.5
            mpl.rcParams['figure.titlesize'] = 14
            mpl.rcParams['axes.linewidth'] = 1.2
            mpl.rcParams['lines.linewidth'] = 2.5
            mpl.rcParams['lines.markersize'] = 8
            mpl.rcParams['grid.alpha'] = 0.3
            mpl.rcParams['grid.linewidth'] = 0.5
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
            ax.yaxis.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            ax.set_axisbelow(True)
        @staticmethod
        def get_color_palette():
            return {
                'primary': '#E74C3C',
                'secondary': '#3498DB',
                'tertiary': '#27AE60',
                'quaternary': '#8B4513',
                'accent1': '#9B59B6',
                'accent2': '#F39C12',
                'accent3': '#1ABC9C',
                'neutral': '#7F8C8D',
            }
        @staticmethod
        def save_publication_figure(fig, output_path, filename, formats=('png', 'pdf')):
            output_path = Path(output_path)
            output_path.mkdir(exist_ok=True, parents=True)
            for fmt in formats:
                save_path = output_path / f'{filename}.{fmt}'
                fig.savefig(
                    save_path,
                    format=fmt,
                    dpi=300,
                    bbox_inches='tight',
                    facecolor='white',
                    edgecolor='none'
                )
                print(f"✓ Saved: {save_path}")
        @staticmethod
        def format_legend(ax, loc='best', **kwargs):
            params = dict(frameon=True, fancybox=True, shadow=True, framealpha=0.95, borderpad=1.0)
            params.update(kwargs)
            return ax.legend(loc=loc, **params)
def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Create publication-quality plots for MIMIC dataset analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --data-path ./data --filename stage1_epoch_sweep.csv
  %(prog)s --output-path ./figures --formats png pdf svg
  %(prog)s --no-annotations --no-combined
  %(prog)s --figsize 10 6.18 --dpi 600
        """
    )
    # Data input
    parser.add_argument('--data-path', '--data_path', type=str, default='mimic_low_data',
                        help='Directory containing the data file (default: mimic_low_data)')
    parser.add_argument('--filename', type=str, default='stage1_epoch_sweep.csv',
                        help='Name of the CSV file to load (default: stage1_epoch_sweep.csv)')
    # Output
    parser.add_argument('--output-path', '--output_path', type=str, default='publication_plots',
                        help='Directory to save output plots (default: publication_plots)')
    parser.add_argument('--formats', nargs='+', default=['png', 'pdf'],
                        choices=['png', 'pdf', 'svg', 'eps'],
                        help='Output formats for figures (default: png pdf)')
    # Plot configuration (golden ratio defaults)
    parser.add_argument('--dpi', type=int, default=300, help='DPI for raster outputs (default: 300)')
    parser.add_argument('--figsize', nargs=2, type=float, default=[10, 6.18],
                        metavar=('WIDTH', 'HEIGHT'),
                        help='Figure size in inches (default: 10 6.18)')
    parser.add_argument('--combined-figsize', '--combined_figsize', nargs=2, type=float,
                        default=[10, 6.18], metavar=('WIDTH', 'HEIGHT'),
                        help='Figure size for combined plots (default: 10 6.18)')
    # Feature toggles
    parser.add_argument('--no-annotations', action='store_true', help='Disable statistical annotations')
    parser.add_argument('--no-combined', action='store_true', help='Skip creating combined plots')
    parser.add_argument('--no-latex', action='store_true', help='Skip generating LaTeX code')
    parser.add_argument('--no-summary', action='store_true', help='Skip printing summary statistics')
    parser.add_argument(
    '--single-y',
    action='store_true',
    help='Use a single shared y-axis in combined plots (default: dual y-axes)'
    )
    parser.add_argument(
    '--layout',
    type=str,
    default='dualy',
    choices=['dualy', 'singley', 'stacked'],
    help='Layout for combined plots: "dualy" (left/right y-axes), "singley" (shared y-axis), or "stacked" (top/bottom panels). Default: dualy'
    )
    # Style options (default ties to publication_plot_style palette)
    parser.add_argument('--style', type=str, default='publication',
                        choices=['publication', 'presentation', 'minimal'],
                        help='Plot style preset (default: publication)')
    parser.add_argument('--color-scheme', '--color_scheme', type=str, default='default',
                        choices=['default', 'colorblind', 'grayscale'],
                        help='Color scheme override (default: default)')
    # Verbose/quiet
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress all output except errors')
    args = parser.parse_args()
    if args.quiet and args.verbose:
        parser.error("Cannot use both --quiet and --verbose")
    return args
# Optional alternate schemes (kept for convenience)
def get_color_schemes():
    return {
        'default': None, # tells us to use pps.get_color_palette()
        'colorblind': {
            'primary': '#D55E00', 'secondary': '#0072B2',
            'tertiary': '#009E73', 'quaternary': '#CC79A7'
        },
        'grayscale': {
            'primary': '#000000', 'secondary': '#666666',
            'tertiary': '#999999', 'quaternary': '#CCCCCC'
        }
    }
REQUIRED_COLS = {"Dataset", "LLM", "S1_Epochs", "nDCG@k_endS1", "MacroF1_endS2", "P@15_endS2"}
OPTIONAL_COLS = {"MacroF1_endS2_std", "P@15_endS2_std"}  # Added for error bars
def load_and_prepare_data(csv_path: Path):
    df = pd.read_csv(csv_path)
    missing_required = REQUIRED_COLS - set(df.columns)
    if missing_required:
        raise ValueError(f"Missing required columns in {csv_path.name}: {sorted(missing_required)}")
    for col in ["S1_Epochs", "nDCG@k_endS1", "MacroF1_endS2", "P@15_endS2"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Handle optional standard error columns
    missing_optional = OPTIONAL_COLS - set(df.columns)
    if missing_optional:
        print(f"Warning: Missing optional columns for error bars: {sorted(missing_optional)}. Error bars will not be plotted.")
        for col in missing_optional:
            df[col] = 0.0  # Set to zero to allow plotting without error bars
    for col in OPTIONAL_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)  # Zero errors if missing
    df = df.dropna(subset=["S1_Epochs", "nDCG@k_endS1", "MacroF1_endS2", "P@15_endS2"]).copy()
    m3 = df[(df['Dataset'] == 'mimicfull') & (df['LLM'].str.lower() == 'mistral')].sort_values("nDCG@k_endS1")
    m4 = df[(df['Dataset'] == 'mimicfour') & (df['LLM'].str.lower().str.contains('llama'))].sort_values("nDCG@k_endS1")
    return m3.reset_index(drop=True), m4.reset_index(drop=True)
def safe_corr(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 2 or x.nunique() < 2 or y.nunique() < 2:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])
def create_single_plot_old(
    data, x_col, y_col, color, marker, label,
    xlabel, ylabel, title, add_annotations=True,
    figsize=(10, 6.18), dpi=300
):
    fig, ax = pps.create_figure(figsize=figsize, dpi=dpi)
    ax.plot(
        data[x_col], data[y_col],
        color=color, linewidth=2.5,
        marker=marker, markersize=8,
        markeredgewidth=1.5, markeredgecolor='black',
        label=label, alpha=0.9
    )
    if add_annotations and len(data) >= 1:
        corr = safe_corr(data[x_col], data[y_col])
        txt = f'Pearson r = {corr:.3f}' if not np.isnan(corr) else 'Pearson r = N/A'
        ax.text(
            0.05, 0.95, txt, transform=ax.transAxes, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=10
        )
        if y_col == 'MacroF1_endS2':
            convergence_val = float(data[y_col].iloc[-1])
            ax.axhline(y=convergence_val, color='gray', linestyle='--', alpha=0.5)
            x_pos_index = max(0, len(data) - 1 - 2)
            ax.text(
                data[x_col].iloc[x_pos_index], convergence_val,
                f'Converges to {convergence_val:.1f}',
                fontsize=9, ha='center', va='bottom'
            )
    ax.set_xlabel(xlabel, fontsize=13, fontweight='medium')
    ax.set_ylabel(ylabel, fontsize=13, fontweight='medium')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=16)
    # axis formatting
    x_min, x_max = data[x_col].min(), data[x_col].max()
    y_min, y_max = data[y_col].min(), data[y_col].max()
    x_range = max(1e-9, x_max - x_min); y_range = max(1e-9, y_max - y_min)
    ax.set_xlim(x_min - 0.05 * x_range, x_max + 0.05 * x_range)
    ax.set_ylim(y_min - 0.10 * y_range, y_max + 0.10 * y_range)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.2f}'))
    # apply your module’s axes style
    pps.style_axes(ax, fig)
    # tidy layout
    plt.tight_layout()
    return fig, ax
def create_single_plot(
    data, x_col, y_col, color, marker, label,
    xlabel, ylabel, title, add_annotations=True,
    figsize=(10, 6.18), dpi=300
):
    fig, ax = pps.create_figure(figsize=figsize, dpi=dpi)
    yerr = data[f"{y_col}_std"] if f"{y_col}_std" in data.columns else None
    ax.errorbar(
        data[x_col], data[y_col], yerr=yerr,
        color=color, linewidth=2.5,
        marker=marker, markersize=8,
        markeredgewidth=1.5, markeredgecolor='black',
        label=label, alpha=0.9,
        elinewidth=1.0, capsize=3, capthick=1.0  # Thin error bars with small caps
    )
    if add_annotations and len(data) >= 1:
        corr = safe_corr(data[x_col], data[y_col])
        txt = f'Pearson r = {corr:.3f}' if not np.isnan(corr) else 'Pearson r = N/A'
        ax.text(
            0.05, 0.93, txt, transform=ax.transAxes, va='top',
            bbox=dict(boxstyle='round,pad=1.2', facecolor='white', alpha=0.8),
            fontsize=12
        )
        convergence_val = float(data[y_col].iloc[-1])
        ax.axhline(y=convergence_val, color='gray', linestyle='--', alpha=0.5)
        x_pos_index = max(0, len(data) - 1 - 1)
        # Dynamically position convergence text above error bar
        y_pos = convergence_val
        if f"{y_col}_std" in data.columns and yerr is not None:
            y_pos = max(y_pos, data[y_col].iloc[x_pos_index] + data[f"{y_col}_std"].iloc[x_pos_index])
        y_pos += 0.02 * (data[y_col].max() - data[y_col].min())  # Add small offset
        ax.text(
            data[x_col].iloc[x_pos_index], y_pos,
            f'Converges to {convergence_val:.1f}',
            fontsize=11, ha='center', va='bottom'
        )
    ax.set_xlabel(xlabel, fontsize=13, fontweight='medium')
    ax.set_ylabel(ylabel, fontsize=13, fontweight='medium')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
    
    # axis formatting
    x_min, x_max = data[x_col].min(), data[x_col].max()
    y_min, y_max = data[y_col].min(), data[y_col].max()
    x_range = max(1e-9, x_max - x_min)
    y_range = max(1e-9, y_max - y_min)
    
    # Expand y-axis to accommodate error bars
    yerr_max = data[f"{y_col}_std"].max() if f"{y_col}_std" in data.columns and yerr is not None else 0.0
    ax.set_xlim(x_min - 0.05 * x_range, x_max + 0.05 * x_range)
    ax.set_ylim(y_min - (0.15 * y_range + yerr_max), y_max + (0.15 * y_range + yerr_max))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.2f}'))
    
    # apply your module’s axes style
    pps.style_axes(ax, fig)

    # tidy layout
    plt.tight_layout(pad=2.0)
    # fig.subplots_adjust(bottom=0.25)  # Added to standardize bottom padding
    return fig, ax
def create_combined_plot_dualy(
    data1, data2, x_col, y_col,
    xlabel, ylabel, title,
    colors, figsize=(10, 6.18), dpi=300
):
    """Combined plot with dual y-axes (left = data1, right = data2)."""
    fig, ax_left = pps.create_figure(figsize=figsize, dpi=dpi)
    # X label/format on the left axis
    ax_left.set_xlabel(xlabel, fontsize=13, fontweight='medium')
    from matplotlib.ticker import FuncFormatter
    ax_left.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.2f}'))
    # Plot data1 on left
    lines, labels = [], []
    if len(data1):
        l1, = ax_left.plot(
            data1[x_col], data1[y_col],
            color=colors['primary'], linewidth=2.5,
            marker='o', markersize=8, markeredgewidth=1.5, markeredgecolor='black',
            label='MIMIC-III-full + Mistral-7B', alpha=0.9
        )
        lines.append(l1); labels.append(l1.get_label())
        ax_left.axhline(float(data1[y_col].iloc[-1]), color=colors['primary'], linestyle='--', alpha=0.3)
    # Right axis for data2
    ax_right = ax_left.twinx()
    if len(data2):
        l2, = ax_right.plot(
            data2[x_col], data2[y_col],
            color=colors['secondary'], linewidth=2.5,
            marker='s', markersize=8, markeredgewidth=1.5, markeredgecolor='black',
            label='MIMIC-IV-full + LLaMA3-8B', alpha=0.9
        )
        lines.append(l2); labels.append(l2.get_label())
        ax_right.axhline(float(data2[y_col].iloc[-1]), color=colors['secondary'], linestyle='--', alpha=0.3)
    # Labels/titles
    ax_left.set_ylabel(f"{ylabel} (MIMIC-III + Mistral-7B)",
                       fontsize=13, fontweight='medium', color=colors['primary'])
    ax_right.set_ylabel(f"{ylabel} (MIMIC-IV + LLaMA3-8B)",
                        fontsize=13, fontweight='medium', color=colors['secondary'])
    ax_right.tick_params(axis='y', colors=colors['secondary'])
    ax_left.tick_params(axis='y', colors=colors['primary'])
    ax_left.set_title(title, fontsize=14, fontweight='bold', pad=16)
    # Style both axes (keep grid only on left)
    pps.style_axes(ax_left, fig)
    pps.style_axes(ax_right, fig)
    ax_right.spines['right'].set_visible(True)
    ax_right.spines['right'].set_color(colors['secondary'])
    ax_right.yaxis.grid(False)
    # Single legend
    pps.format_legend(ax_left, loc='lower right', handles=lines, labels=labels)
    plt.tight_layout()
    return fig, (ax_left, ax_right)
def create_combined_plot_singley(
    data1, data2, x_col, y_col,
    xlabel, ylabel, title,
    colors, figsize=(10, 6.18), dpi=300
):
    """Combined plot with a single shared y-axis (both series on the left axis)."""
    fig, ax = pps.create_figure(figsize=figsize, dpi=dpi)
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.2f}'))
    ax.set_xlabel(xlabel, fontsize=13, fontweight='medium')
    ax.set_ylabel(ylabel, fontsize=13, fontweight='medium')
    lines, labels = [], []
    if len(data1):
        l1, = ax.plot(
            data1[x_col], data1[y_col],
            color=colors['primary'], linewidth=2.5,
            marker='o', markersize=8, markeredgewidth=1.5, markeredgecolor='black',
            label='MIMIC-III-full + Mistral-7B', alpha=0.9
        )
        lines.append(l1); labels.append(l1.get_label())
        ax.axhline(float(data1[y_col].iloc[-1]), color=colors['primary'], linestyle='--', alpha=0.3)
    if len(data2):
        l2, = ax.plot(
            data2[x_col], data2[y_col],
            color=colors['secondary'], linewidth=2.5,
            marker='s', markersize=8, markeredgewidth=1.5, markeredgecolor='black',
            label='MIMIC-IV-full + LLaMA3-8B', alpha=0.9
        )
        lines.append(l2); labels.append(l2.get_label())
        ax.axhline(float(data2[y_col].iloc[-1]), color=colors['secondary'], linestyle='--', alpha=0.3)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=16)
    pps.style_axes(ax, fig)
    pps.format_legend(ax, loc='lower right', handles=lines, labels=labels)
    plt.tight_layout()
    return fig, ax
def create_combined_plot_stacked(
    data1, data2, x_col, y_col,
    xlabel, ylabel, title,
    colors, figsize=(10, 8), dpi=300
):
    """Combined plot in a vertical layout (top = data1, bottom = data2) with error bars."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, sharex=True, figsize=figsize, dpi=dpi,
        gridspec_kw={'hspace': 0.15}
    )
    # --- Top panel (MIMIC-III + Mistral) ---
    if len(data1):
        yerr = data1[f"{y_col}_std"] if f"{y_col}_std" in data1.columns else None
        ax_top.errorbar(
            data1[x_col], data1[y_col], yerr=yerr,
            color=colors['primary'], linewidth=2.5,
            marker='o', markersize=8, markeredgewidth=1.5, markeredgecolor='black',
            label='MIMIC-III-full + Mistral-7B', alpha=0.9,
            elinewidth=1.0, capsize=3, capthick=1.0  # Thin error bars with small caps
        )
        ax_top.axhline(float(data1[y_col].iloc[-1]), color=colors['primary'], linestyle='--', alpha=0.3)
    ax_top.set_ylabel(f"{ylabel}\n(MIMIC-III + Mistral-7B)",
                      fontsize=12, fontweight='medium', color=colors['primary'])
    pps.style_axes(ax_top, fig)
    pps.format_legend(ax_top, loc='lower right')
    # --- Bottom panel (MIMIC-IV + LLaMA) ---
    if len(data2):
        yerr = data2[f"{y_col}_std"] if f"{y_col}_std" in data2.columns else None
        ax_bottom.errorbar(
            data2[x_col], data2[y_col], yerr=yerr,
            color=colors['secondary'], linewidth=2.5,
            marker='s', markersize=8, markeredgewidth=1.5, markeredgecolor='black',
            label='MIMIC-IV-full + LLaMA3-8B', alpha=0.9,
            elinewidth=1.0, capsize=3, capthick=1.0  # Thin error bars with small caps
        )
        ax_bottom.axhline(float(data2[y_col].iloc[-1]), color=colors['secondary'], linestyle='--', alpha=0.3)
    ax_bottom.set_ylabel(f"{ylabel}\n(MIMIC-IV + LLaMA3-8B)",
                         fontsize=12, fontweight='medium', color=colors['secondary'])
    ax_bottom.set_xlabel(xlabel, fontsize=13, fontweight='medium')
    ax_bottom.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.2f}'))
    pps.style_axes(ax_bottom, fig)
    pps.format_legend(ax_bottom, loc='lower right')
    # Title
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig, (ax_top, ax_bottom)
def save_figure(fig, output_path: Path, filename: str, formats, dpi=300):
    """
    Save figure using pps.save_publication_figure (prints ✓ lines),
    but also return the list of saved file paths so the script can summarize.
    """
    # Use the pps saver for consistent params & logging
    pps.save_publication_figure(fig, output_path, filename, formats=formats)
    # Construct returned paths
    return [Path(output_path) / f"{filename}.{fmt}" for fmt in formats]
def generate_latex_table(m3: pd.DataFrame, m4: pd.DataFrame) -> str:
    left = m3[["S1_Epochs", "MacroF1_endS2", "P@15_endS2"]].rename(
        columns={"MacroF1_endS2": "M3_MacroF1", "P@15_endS2": "M3_P15"}
    )
    right = m4[["S1_Epochs", "MacroF1_endS2", "P@15_endS2"]].rename(
        columns={"MacroF1_endS2": "M4_MacroF1", "P@15_endS2": "M4_P15"}
    )
    merged = pd.merge(left, right, on="S1_Epochs", how="outer").sort_values("S1_Epochs")
    def fmt(x): return "" if pd.isna(x) else f"{x:.1f}"
    rows = []
    for _, r in merged.iterrows():
        rows.append(
            f"{int(r['S1_Epochs']) if not pd.isna(r['S1_Epochs']) else ''} & "
            f"{fmt(r.get('M3_MacroF1'))} & {fmt(r.get('M3_P15'))} & "
            f"{fmt(r.get('M4_MacroF1'))} & {fmt(r.get('M4_P15'))} \\\\"
        )
    return "\n".join([
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Effect of Stage 1 pretraining length on final Stage 2 performance metrics}",
        r"\label{tab:stage1_effect}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\multirow{2}{*}{S1 Epochs} & \multicolumn{2}{c}{MIMIC-III + Mistral-7B} & \multicolumn{2}{c}{MIMIC-IV + LLaMA3-8B} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}",
        r" & Macro-F1 & P@15 & Macro-F1 & P@15 \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

def generate_latex_figures():
    return r"""
% Individual plots
\begin{figure}[htbp]
    \centering
    \begin{subfigure}[b]{0.48\textwidth}
        \includegraphics[width=\textwidth]{mimic3_mistral_macrof1.pdf}
        \caption{Macro-F1 scores for the MIMIC-III-full dataset using the Mistral-7B model, showing the impact of Stage 1 pretraining performance ($\mathsf{nDCG}@k$) on final Stage 2 classification performance.}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.48\textwidth}
        \includegraphics[width=\textwidth]{mimic3_mistral_p15.pdf}
        \caption{P@15 scores for the MIMIC-III-full dataset using the Mistral-7B model, illustrating the effect of Stage 1 pretraining on final Stage 2 ranking performance.}
    \end{subfigure}
    \vspace{0.5cm}
    \begin{subfigure}[b]{0.48\textwidth}
        \includegraphics[width=\textwidth]{mimic4_llama_macrof1.pdf}
        \caption{Macro-F1 scores for the MIMIC-IV-full dataset using the LLaMA3-8B model, demonstrating the influence of Stage 1 pretraining on Stage 2 classification performance.}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.48\textwidth}
        \includegraphics[width=\textwidth]{mimic4_llama_p15.pdf}
        \caption{P@15 scores for the MIMIC-IV-full dataset using the LLaMA3-8B model, highlighting the effect of Stage 1 pretraining on Stage 2 ranking performance.}
    \end{subfigure}
    \caption{Effect of varying Stage 1 pretraining length (epochs) on Stage 2 performance for the MIMIC-III-full dataset with Mistral-7B and the MIMIC-IV-full dataset with LLaMA3-8B. As Stage 1 $\mathsf{nDCG}@k$ improves, both Macro-F1 (classification) and P@15 (ranking) metrics increase and stabilize, with error bars indicating variability.}
    \label{fig:stage1_effect}
\end{figure}
% Combined comparison plots
\begin{figure}[htbp]
    \centering
    \begin{subfigure}[b]{0.48\textwidth}
        \includegraphics[width=\textwidth]{combined_macrof1.pdf}
        \caption{Comparison of Macro-F1 scores between MIMIC-III-full with Mistral-7B and MIMIC-IV-full with LLaMA3-8B, showing the impact of Stage 1 pretraining ($\mathsf{nDCG}@k$) on Stage 2 classification performance.}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.48\textwidth}
        \includegraphics[width=\textwidth]{combined_p15.pdf}
        \caption{Comparison of P@15 scores between MIMIC-III-full with Mistral-7B and MIMIC-IV-full with LLaMA3-8B, illustrating the effect of Stage 1 pretraining on Stage 2 ranking performance.}
    \end{subfigure}
    \caption{Comparison of Stage 1 pretraining effects on Stage 2 performance across two dataset--LLM pairs: MIMIC-III-full with Mistral-7B and MIMIC-IV-full with LLaMA3-8B. Plots show Macro-F1 and P@15 metrics as a function of Stage 1 $\mathsf{nDCG}@k$, with error bars indicating variability.}
    \label{fig:combined_comparison}
\end{figure}
"""

def print_summary(m3, m4):
    def rng(s): return f"{s.min():.1f} → {s.max():.1f}" if len(s) else "N/A"
    print("\n📈 Summary Statistics:")
    print("\nMIMIC-III + Mistral-7B:")
    print(f" Macro-F1: {rng(m3['MacroF1_endS2'])}")
    print(f" P@15: {rng(m3['P@15_endS2'])}")
    print("\nMIMIC-IV + LLaMA3-8B:")
    print(f" Macro-F1: {rng(m4['MacroF1_endS2'])}")
    print(f" P@15: {rng(m4['P@15_endS2'])}")
def main():
    args = parse_arguments()
    # Logger
    if args.quiet:
        def verbose(*_a, **_k): ...
    else:
        verbose = print
    # Style (your module)
    pps.setup_publication_style()
    # Data
    data_path = Path(args.data_path)
    csv_path = data_path / args.filename
    verbose(f"Loading MIMIC dataset from {csv_path}...")
    try:
        m3, m4 = load_and_prepare_data(csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_path}")
        print("Please ensure the CSV file is in the correct location.")
        print("You can specify a different path with --data-path and --filename")
        return 1
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    verbose(f"Loaded {len(m3)} rows for MIMIC-III + Mistral-7B")
    verbose(f"Loaded {len(m4)} rows for MIMIC-IV + LLaMA3-8B")
    # Output dir
    output_dir = Path(args.output_path)
    output_dir.mkdir(exist_ok=True, parents=True)
    # Colors (default = your palette)
    scheme = get_color_schemes()[args.color_scheme]
    colors = scheme if scheme is not None else pps.get_color_palette()
    # Track saved files
    all_saved_files = []
    # Individual plots
    verbose("\nCreating individual plots...")
    plots_config = [
        {
            'data': m3, 'x_col': 'nDCG@k_endS1', 'y_col': 'MacroF1_endS2',
            'color': colors['primary'], 'marker': 'o',
            'label': 'MIMIC-III-full + Mistral-7B',
            'xlabel': 'Stage 1 nDCG@k', 'ylabel': 'Stage 2 Macro-F1 (%)',
            'title': 'Effect of Stage 1 Performance on Final Macro-F1\n(MIMIC-III-full + Mistral-7B)',
            'filename': 'mimic3_mistral_macrof1'
        },
        {
            'data': m3, 'x_col': 'nDCG@k_endS1', 'y_col': 'P@15_endS2',
            'color': colors['primary'], 'marker': 'o',
            'label': 'MIMIC-III-full + Mistral-7B',
            'xlabel': 'Stage 1 nDCG@k', 'ylabel': 'Stage 2 P@15 (%)',
            'title': 'Effect of Stage 1 Performance on Final P@15\n(MIMIC-III-full + Mistral-7B)',
            'filename': 'mimic3_mistral_p15'
        },
        {
            'data': m4, 'x_col': 'nDCG@k_endS1', 'y_col': 'MacroF1_endS2',
            'color': colors['secondary'], 'marker': 's',
            'label': 'MIMIC-IV-full + LLaMA3-8B',
            'xlabel': 'Stage 1 nDCG@k', 'ylabel': 'Stage 2 Macro-F1 (%)',
            'title': 'Effect of Stage 1 Performance on Final Macro-F1\n(MIMIC-IV-full + LLaMA3-8B)',
            'filename': 'mimic4_llama_macrof1'
        },
        {
            'data': m4, 'x_col': 'nDCG@k_endS1', 'y_col': 'P@15_endS2',
            'color': colors['secondary'], 'marker': 's',
            'label': 'MIMIC-IV-full + LLaMA3-8B',
            'xlabel': 'Stage 1 nDCG@k', 'ylabel': 'Stage 2 P@15 (%)',
            'title': 'Effect of Stage 1 Performance on Final P@15\n(MIMIC-IV-full + LLaMA3-8B)',
            'filename': 'mimic4_llama_p15'
        }
    ]
    for cfg in plots_config:
        if args.verbose:
            verbose(f" Creating {cfg['filename']}...")
        fig, _ = create_single_plot(
            **{k: v for k, v in cfg.items() if k != 'filename'},
            add_annotations=not args.no_annotations,
            figsize=tuple(args.figsize), dpi=args.dpi
        )
        saved = save_figure(fig, output_dir, cfg['filename'], args.formats, args.dpi)
        all_saved_files.extend(saved)
        plt.close(fig)
    # Combined plots
    if not args.no_combined:
        verbose("\nCreating combined comparison plots...")
    # Choose a sensible default height for stacked layout if user left default size
    combined_size = tuple(args.combined_figsize)
    if args.layout == 'stacked' and list(map(float, args.combined_figsize)) == [10.0, 6.18]:
        combined_size = (10, 8) # taller for two rows
    # Combined Macro-F1
    if args.layout == 'dualy':
        fig1, _ = create_combined_plot_dualy(
            m3, m4, 'nDCG@k_endS1', 'MacroF1_endS2',
            'Stage 1 nDCG@k', 'Stage 2 Macro-F1 (%)',
            'Effect of Stage 1 Pretraining on Final Macro-F1 Performance',
            colors=colors, figsize=combined_size, dpi=args.dpi
        )
    elif args.layout == 'singley':
        fig1, _ = create_combined_plot_singley(
            m3, m4, 'nDCG@k_endS1', 'MacroF1_endS2',
            'Stage 1 nDCG@k', 'Stage 2 Macro-F1 (%)',
            'Effect of Stage 1 Pretraining on Final Macro-F1 Performance',
            colors=colors, figsize=combined_size, dpi=args.dpi
        )
    else: # stacked
        fig1, _ = create_combined_plot_stacked(
            m3, m4, 'nDCG@k_endS1', 'MacroF1_endS2',
            'Stage 1 nDCG@k', 'Stage 2 Macro-F1 (%)',
            'Effect of Stage 1 Pretraining on Final Macro-F1 Performance',
            colors=colors, figsize=combined_size, dpi=args.dpi
        )
    saved = save_figure(fig1, output_dir, 'combined_macrof1', args.formats, args.dpi)
    all_saved_files.extend(saved)
    plt.close(fig1)
    # Combined P@15 — repeat with the same branching
    if args.layout == 'dualy':
        fig2, _ = create_combined_plot_dualy(
            m3, m4, 'nDCG@k_endS1', 'P@15_endS2',
            'Stage 1 nDCG@k', 'Stage 2 P@15 (%)',
            'Effect of Stage 1 Pretraining on Final P@15 Performance',
            colors=colors, figsize=combined_size, dpi=args.dpi
        )
    elif args.layout == 'singley':
        fig2, _ = create_combined_plot_singley(
            m3, m4, 'nDCG@k_endS1', 'P@15_endS2',
            'Stage 1 nDCG@k', 'Stage 2 P@15 (%)',
            'Effect of Stage 1 Pretraining on Final P@15 Performance',
            colors=colors, figsize=combined_size, dpi=args.dpi
        )
    else:
        fig2, _ = create_combined_plot_stacked(
            m3, m4, 'nDCG@k_endS1', 'P@15_endS2',
            'Stage 1 nDCG@k', 'Stage 2 P@15 (%)',
            'Effect of Stage 1 Pretraining on Final P@15 Performance',
            colors=colors, figsize=combined_size, dpi=args.dpi
        )
    saved = save_figure(fig2, output_dir, 'combined_p15', args.formats, args.dpi)
    all_saved_files.extend(saved)
    plt.close(fig2)
    # LaTeX
    latex_files = []
    if not args.no_latex:
        verbose("\nGenerating LaTeX code...")
        latex_table = generate_latex_table(m3, m4)
        latex_file = output_dir / 'mimic_results_table.tex'
        latex_file.write_text(latex_table)
        latex_files.append(latex_file)
        verbose(f" Saved: {latex_file}")
        latex_figures = generate_latex_figures()
        latex_figures_file = output_dir / 'mimic_figures.tex'
        latex_figures_file.write_text(latex_figures)
        latex_files.append(latex_figures_file)
        verbose(f" Saved: {latex_figures_file}")
    # Completion message
    if not args.quiet:
        print(f"\n✅ All plots created successfully!")
        print(f"📁 Output directory: {output_dir.absolute()}")
        num_plots = 4 if args.no_combined else 6
        print(f"📊 Created {num_plots} plots", end="")
        print(" (4 individual + 2 combined)" if not args.no_combined else " (individual only)")
        print(f"\n📄 Files generated:")
        print(f" Plot files: {len(all_saved_files)} files")
        print(" Plot paths:")
        for fp in sorted(all_saved_files, key=lambda p: (p.suffix, str(p))):
            print(f" - {fp}")
        for fmt in args.formats:
            count = sum(1 for f in all_saved_files if f.suffix == f'.{fmt}')
            print(f" - {count} {fmt.upper()} files")
        if not args.no_latex:
            print(f" LaTeX files: {len(latex_files)} files")
            for lf in latex_files:
                print(f" - {lf}")
    # Summary
    if not args.no_summary and not args.quiet:
        print_summary(m3, m4)
    return 0
if __name__ == "__main__":
    sys.exit(main())