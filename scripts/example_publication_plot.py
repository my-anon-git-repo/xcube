#!/usr/bin/env python3
"""
Template for creating publication-quality plots using matplotlib.
This template provides a structured approach to data visualization for academic papers.

Usage:
    python example_publication_plot.py --data_path ./data --plot_type line
    python example_publication_plot.py --help
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Optional

# Import our publication style module
try:
    import publication_plot_style as pps
except ImportError:
    print("Error: publication_plot_style.py not found in the current directory.")
    print("Please ensure the style module is in the same directory as this script.")
    sys.exit(1)

import matplotlib.pyplot as plt
from scipy import stats


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create publication-quality plots with customizable options.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data",
        help="Path containing the input data files"
    )
    
    parser.add_argument(
        "--log_path",
        type=str,
        default="../tmp",
        help="Path to save publication plots"
    )
    
    # Plot configuration
    parser.add_argument(
        "--plot_type",
        type=str,
        default="line",
        choices=["line", "bar", "scatter", "violin", "box", "heatmap"],
        help="Type of plot to create"
    )
    
    parser.add_argument(
        "--title",
        type=str,
        default="",
        help="Plot title (leave empty for default)"
    )
    
    parser.add_argument(
        "--xlabel",
        type=str,
        default="X-axis",
        help="X-axis label"
    )
    
    parser.add_argument(
        "--ylabel",
        type=str,
        default="Y-axis",
        help="Y-axis label"
    )
    
    # Data processing options
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize data before plotting"
    )
    
    parser.add_argument(
        "--aggregate",
        type=str,
        choices=["mean", "median", "sum", "none"],
        default="none",
        help="Aggregation method for grouped data"
    )
    
    # Output options
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf"],
        choices=["png", "pdf", "svg", "eps"],
        help="Output file formats"
    )
    
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for raster formats"
    )
    
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=[10, 6.18],
        help="Figure size in inches (width height)"
    )
    
    # Statistical options
    parser.add_argument(
        "--show_stats",
        action="store_true",
        help="Show statistical annotations on plot"
    )
    
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence interval for error bars"
    )
    
    return parser.parse_args()


def load_data(data_path: Path) -> Dict[str, pd.DataFrame]:
    """
    Load data from files in the specified directory.
    
    Parameters:
    -----------
    data_path : Path
        Directory containing data files
    
    Returns:
    --------
    Dict[str, pd.DataFrame] : Dictionary mapping dataset names to dataframes
    """
    data = {}
    
    # Define supported file extensions and their readers
    readers = {
        '.csv': lambda f: pd.read_csv(f),
        '.tsv': lambda f: pd.read_csv(f, sep='\t'),
        '.txt': lambda f: pd.read_csv(f, sep='\t'),
        '.xlsx': lambda f: pd.read_excel(f),
        '.json': lambda f: pd.read_json(f)
    }
    
    # Load all compatible files
    for file_path in data_path.iterdir():
        if file_path.suffix in readers:
            try:
                dataset_name = file_path.stem
                data[dataset_name] = readers[file_path.suffix](file_path)
                print(f"✓ Loaded {dataset_name} from {file_path.name}")
            except Exception as e:
                print(f"✗ Failed to load {file_path.name}: {e}")
    
    if not data:
        raise ValueError(f"No data files found in {data_path}")
    
    return data


def preprocess_data(data: Dict[str, pd.DataFrame], 
                   normalize: bool = False,
                   aggregate: str = "none") -> Dict[str, pd.DataFrame]:
    """
    Preprocess data according to specified options.
    
    Parameters:
    -----------
    data : Dict[str, pd.DataFrame]
        Raw data dictionary
    normalize : bool
        Whether to normalize the data
    aggregate : str
        Aggregation method to apply
    
    Returns:
    --------
    Dict[str, pd.DataFrame] : Processed data
    """
    processed = {}
    
    for name, df in data.items():
        # Make a copy to avoid modifying original
        df_processed = df.copy()
        
        # Normalize if requested
        if normalize:
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
            df_processed[numeric_cols] = (df_processed[numeric_cols] - df_processed[numeric_cols].mean()) / df_processed[numeric_cols].std()
        
        # Aggregate if requested
        if aggregate != "none" and len(df_processed.columns) > 1:
            # Assume first column is grouping variable
            group_col = df_processed.columns[0]
            if aggregate == "mean":
                df_processed = df_processed.groupby(group_col).mean().reset_index()
            elif aggregate == "median":
                df_processed = df_processed.groupby(group_col).median().reset_index()
            elif aggregate == "sum":
                df_processed = df_processed.groupby(group_col).sum().reset_index()
        
        processed[name] = df_processed
    
    return processed


def create_line_plot(ax, data: Dict[str, pd.DataFrame], args):
    """Create a line plot."""
    colors = pps.get_color_palette()
    color_list = list(colors.values())
    
    for i, (name, df) in enumerate(data.items()):
        # Assume first column is x, second is y
        if len(df.columns) >= 2:
            x = df.iloc[:, 0]
            y = df.iloc[:, 1]
            
            # Calculate error bars if more columns exist
            if len(df.columns) >= 3 and args.show_stats:
                yerr = df.iloc[:, 2]
                ax.errorbar(x, y, yerr=yerr, label=name, 
                           color=color_list[i % len(color_list)],
                           capsize=5, capthick=2, alpha=0.8)
            else:
                ax.plot(x, y, label=name, 
                       color=color_list[i % len(color_list)],
                       linewidth=2.5, marker='o', markersize=6)


def create_bar_plot(ax, data: Dict[str, pd.DataFrame], args):
    """Create a bar plot."""
    colors = pps.get_color_palette()
    color_list = list(colors.values())
    
    # Simple implementation for single dataset
    if len(data) == 1:
        name, df = list(data.items())[0]
        if len(df.columns) >= 2:
            x = range(len(df))
            y = df.iloc[:, 1]
            ax.bar(x, y, color=color_list[0], alpha=0.8, edgecolor='black', linewidth=1.2)
            ax.set_xticks(x)
            ax.set_xticklabels(df.iloc[:, 0], rotation=45, ha='right')
    else:
        # Multiple datasets - grouped bars
        n_groups = len(data)
        bar_width = 0.8 / n_groups
        
        for i, (name, df) in enumerate(data.items()):
            if len(df.columns) >= 2:
                x = np.arange(len(df))
                y = df.iloc[:, 1]
                offset = (i - n_groups/2) * bar_width + bar_width/2
                ax.bar(x + offset, y, bar_width, label=name,
                      color=color_list[i % len(color_list)], alpha=0.8,
                      edgecolor='black', linewidth=1.2)


def create_scatter_plot(ax, data: Dict[str, pd.DataFrame], args):
    """Create a scatter plot."""
    colors = pps.get_color_palette()
    color_list = list(colors.values())
    
    for i, (name, df) in enumerate(data.items()):
        if len(df.columns) >= 2:
            x = df.iloc[:, 0]
            y = df.iloc[:, 1]
            
            # Size based on third column if available
            if len(df.columns) >= 3:
                sizes = df.iloc[:, 2] * 100
            else:
                sizes = 50
            
            ax.scatter(x, y, s=sizes, label=name,
                      color=color_list[i % len(color_list)],
                      alpha=0.6, edgecolors='black', linewidth=1)


def create_violin_plot(ax, data: Dict[str, pd.DataFrame], args):
    """Create a violin plot."""
    colors = pps.get_color_palette()
    color_list = list(colors.values())
    
    # Prepare data for violin plot
    plot_data = []
    labels = []
    
    for name, df in data.items():
        if len(df.columns) >= 1:
            # Use first numeric column
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                plot_data.append(df[numeric_cols[0]].dropna().values)
                labels.append(name)
    
    if plot_data:
        parts = ax.violinplot(plot_data, positions=range(len(labels)),
                             widths=0.7, showmeans=True, showmedians=True)
        
        # Style violins
        pps.style_violin_plot(parts, color_list[:len(labels)])
        
        # Add box plot overlay
        bp = ax.boxplot(plot_data, positions=range(len(labels)),
                       widths=0.15, patch_artist=True, showfliers=False)
        pps.style_box_plot(bp, color_list[:len(labels)], alpha=0.3)
        
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)


def create_box_plot(ax, data: Dict[str, pd.DataFrame], args):
    """Create a box plot."""
    colors = pps.get_color_palette()
    color_list = list(colors.values())
    
    # Prepare data for box plot
    plot_data = []
    labels = []
    
    for name, df in data.items():
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            plot_data.append(df[numeric_cols[0]].dropna().values)
            labels.append(name)
    
    if plot_data:
        bp = ax.boxplot(plot_data, labels=labels, patch_artist=True,
                       notch=True, showmeans=True)
        pps.style_box_plot(bp, color_list[:len(labels)])


def create_heatmap(ax, data: Dict[str, pd.DataFrame], args):
    """Create a heatmap."""
    # Use first dataset for heatmap
    if data:
        name, df = list(data.items())[0]
        
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        if not numeric_df.empty:
            im = ax.imshow(numeric_df.values, cmap='RdBu_r', aspect='auto')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Value', rotation=270, labelpad=15)
            
            # Set ticks
            ax.set_xticks(range(len(numeric_df.columns)))
            ax.set_xticklabels(numeric_df.columns, rotation=45, ha='right')
            ax.set_yticks(range(len(numeric_df)))
            ax.set_yticklabels(range(len(numeric_df)))


def add_statistical_annotations(ax, data: Dict[str, pd.DataFrame], plot_type: str):
    """Add statistical annotations to the plot."""
    # Calculate and display relevant statistics
    stats_text = []
    
    for name, df in data.items():
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            values = df[numeric_cols[0]].dropna()
            mean_val = values.mean()
            std_val = values.std()
            stats_text.append(f"{name}: μ={mean_val:.3f}, σ={std_val:.3f}")
    
    if stats_text:
        # Place text box in upper right
        text = '\n'.join(stats_text)
        ax.text(0.98, 0.98, text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=9)


def create_plot(data: Dict[str, pd.DataFrame], args) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create the specified type of plot.
    
    Parameters:
    -----------
    data : Dict[str, pd.DataFrame]
        Processed data to plot
    args : argparse.Namespace
        Command line arguments
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes
    """
    # Initialize publication style
    pps.setup_publication_style()
    
    # Create figure
    fig, ax = pps.create_figure(figsize=tuple(args.figsize), dpi=args.dpi)
    
    # Create the appropriate plot type
    plot_functions = {
        'line': create_line_plot,
        'bar': create_bar_plot,
        'scatter': create_scatter_plot,
        'violin': create_violin_plot,
        'box': create_box_plot,
        'heatmap': create_heatmap
    }
    
    if args.plot_type in plot_functions:
        plot_functions[args.plot_type](ax, data, args)
    
    # Set labels and title
    ax.set_xlabel(args.xlabel, fontsize=13, fontweight='medium')
    ax.set_ylabel(args.ylabel, fontsize=13, fontweight='medium')
    
    if args.title:
        ax.set_title(args.title, fontsize=15, fontweight='bold', pad=20)
    else:
        # Default title based on plot type
        ax.set_title(f'{args.plot_type.capitalize()} Plot of {", ".join(data.keys())}',
                    fontsize=15, fontweight='bold', pad=20)
    
    # Apply publication styling
    pps.style_axes(ax, fig)
    
    # Add legend if multiple datasets
    if len(data) > 1 or args.plot_type in ['violin', 'box']:
        pps.format_legend(ax, loc='best')
    
    # Add statistical annotations if requested
    if args.show_stats:
        add_statistical_annotations(ax, data, args.plot_type)
    
    # Tight layout
    plt.tight_layout()
    
    return fig, ax


def generate_latex_output(data: Dict[str, pd.DataFrame], args, output_path: Path):
    """Generate LaTeX code for including the figure and caption."""
    
    # Calculate summary statistics
    stats = {}
    for name, df in data.items():
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            values = df[numeric_cols[0]].dropna()
            stats[name] = {
                'mean': values.mean(),
                'std': values.std(),
                'median': values.median(),
                'n': len(values)
            }
    
    # Generate caption
    caption = f"{args.plot_type.capitalize()} plot showing "
    
    if stats:
        stat_strings = []
        for name, stat in stats.items():
            stat_strings.append(f"{name} (n={stat['n']}, mean={stat['mean']:.3f}±{stat['std']:.3f})")
        caption += ", ".join(stat_strings) + ". "
    
    caption += f"Error bars represent {args.confidence*100:.0f}% confidence intervals." if args.show_stats else ""
    
    # Generate LaTeX code
    latex_code = f"""
% LaTeX code for including the figure
\\begin{{figure}}[htbp]
    \\centering
    \\includegraphics[width=\\textwidth]{{{output_path.name}}}
    \\caption{{{caption}}}
    \\label{{fig:{output_path.stem}}}
\\end{{figure}}
"""
    
    # Save LaTeX code
    latex_file = output_path.parent / f"{output_path.stem}_latex.tex"
    with open(latex_file, 'w') as f:
        f.write(latex_code)
    
    print(f"\n✓ LaTeX code saved to: {latex_file}")
    print("\nLaTeX caption preview:")
    print(f"\\caption{{{caption}}}")


def main():
    """Main function to create publication-quality plots."""
    # Parse arguments
    args = parse_arguments()
    
    # Print header
    print("\n" + "="*60)
    print("📊 Publication Plot Generator")
    print("="*60)
    
    # Setup paths
    data_path = Path(args.data_path)
    log_path = Path(args.log_path)
    
    # Verify data path exists
    if not data_path.exists():
        print(f"\n❌ Error: Data path '{data_path}' does not exist!")
        return 1
    
    # Load data
    print(f"\n📁 Loading data from: {data_path}")
    try:
        data = load_data(data_path)
    except Exception as e:
        print(f"\n❌ Error loading data: {e}")
        return 1
    
    # Preprocess data
    print("\n🔧 Preprocessing data...")
    data = preprocess_data(data, normalize=args.normalize, aggregate=args.aggregate)
    
    # Create plot
    print(f"\n🎨 Creating {args.plot_type} plot...")
    fig, ax = create_plot(data, args)
    
    # Save plot
    output_dir = log_path / 'publication_plots'
    output_dir.mkdir(exist_ok=True, parents=True)
    
    base_filename = f"{args.plot_type}_plot_{'_'.join(data.keys())}"
    
    print(f"\n💾 Saving plots to: {output_dir}")
    for fmt in args.formats:
        output_path = output_dir / f"{base_filename}.{fmt}"
        pps.save_publication_figure(fig, output_dir, base_filename, formats=[fmt])
    
    # Generate LaTeX output
    generate_latex_output(data, args, output_dir / f"{base_filename}.pdf")
    
    # Print summary
    print("\n" + "="*60)
    print("✅ Plot generation complete!")
    print("="*60)
    print(f"\nFiles saved in: {output_dir}")
    print(f"\nUsage examples:")
    print(f"  python {Path(__file__).name} --plot_type violin --show_stats")
    print(f"  python {Path(__file__).name} --plot_type scatter --normalize")
    print(f"  python {Path(__file__).name} --plot_type line --aggregate mean")
    
    # Show plot if running interactively
    if hasattr(sys, 'ps1'):
        plt.show()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())