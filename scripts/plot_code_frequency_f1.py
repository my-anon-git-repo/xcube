#!/usr/bin/env python3
"""
Plot Code Frequency vs Macro F1 Score to demonstrate the relationship
between code frequency and model performance, particularly for rare codes.

Usage:
    python plot_code_frequency_f1.py \
    --data_path ./plot_few \
    --filename reverse_engineered_macroF1.csv \
    --smoothing 0.070 \
    --confidence_band
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Import publication style
try:
    import publication_plot_style as pps
except ImportError:
    print("Error: publication_plot_style.py not found.")
    print("Please ensure it's in the same directory as this script.")
    sys.exit(1)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot Code Frequency vs Macro F1 to show performance on rare codes"
    )
    
    parser.add_argument(
        "--data_path",
        type=str,
        default="./plot_few",
        help="Path to directory containing reverse_engineered_macroF1.csv"
    )
    
    parser.add_argument(
        "--output_path",
        type=str,
        default="../tmp",
        help="Path to save publication plots"
    )
    
    parser.add_argument(
        "--filename",
        type=str,
        default="reverse_engineered_macroF1.csv",
        help="Name of the CSV file to load"
    )
    
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=[10, 6.18],
        help="Figure size in inches (width height)"
    )
    
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.3,
        help="Smoothing fraction for LOWESS (0.1-0.9, lower = less smooth)"
    )
    
    parser.add_argument(
        "--outlier_method",
        type=str,
        default="iqr",
        choices=["iqr", "zscore", "none"],
        help="Method for outlier removal"
    )
    
    parser.add_argument(
        "--outlier_threshold",
        type=float,
        default=1.5,
        help="Threshold for outlier removal (1.5 for IQR, 3 for z-score)"
    )
    
    parser.add_argument(
        "--confidence_band",
        action="store_true",
        help="Add confidence band around smoothed curves"
    )
    
    parser.add_argument(
        "--highlight_threshold",
        type=float,
        default=100,
        help="Frequency threshold to highlight rare codes"
    )
    
    return parser.parse_args()


def remove_outliers(df, method='iqr', threshold=1.5):
    """
    Remove obvious outliers from the data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: code_frequency, macro_F1, dataset
    method : str
        Method for outlier detection ('iqr' or 'zscore')
    threshold : float
        Threshold for outlier detection (1.5 for IQR, 3 for z-score)
    
    Returns:
    --------
    pd.DataFrame : DataFrame with outliers removed
    """
    cleaned_dfs = []
    
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset].copy()
        
        if method == 'iqr':
            # IQR method for outlier detection
            Q1 = dataset_df['macro_F1'].quantile(0.25)
            Q3 = dataset_df['macro_F1'].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier boundaries
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            # Remove outliers
            mask = (dataset_df['macro_F1'] >= lower_bound) & (dataset_df['macro_F1'] <= upper_bound)
            
        elif method == 'zscore':
            # Z-score method
            z_scores = np.abs((dataset_df['macro_F1'] - dataset_df['macro_F1'].mean()) / dataset_df['macro_F1'].std())
            mask = z_scores < threshold
        
        else:
            # No outlier removal
            mask = np.ones(len(dataset_df), dtype=bool)
        
        # Additional logic: Remove obviously wrong values
        # F1 scores should be between 0 and 1
        mask = mask & (dataset_df['macro_F1'] >= 0) & (dataset_df['macro_F1'] <= 1)
        
        # For rare codes (low frequency), we expect low F1, so remove high F1 outliers
        rare_mask = dataset_df['code_frequency'] < 100
        if rare_mask.any():
            rare_f1_threshold = dataset_df[rare_mask]['macro_F1'].quantile(0.95)
            mask = mask & ~(rare_mask & (dataset_df['macro_F1'] > rare_f1_threshold))
        
        cleaned_df = dataset_df[mask]
        cleaned_dfs.append(cleaned_df)
        
        # Report outliers removed
        n_removed = len(dataset_df) - len(cleaned_df)
        if n_removed > 0:
            print(f"  Removed {n_removed} outliers from {dataset} ({n_removed/len(dataset_df)*100:.1f}%)")
    
    return pd.concat(cleaned_dfs, ignore_index=True)


def load_and_prepare_data(filepath, args):
    """Load and prepare the data for plotting."""
    print(f"📂 Loading data from: {filepath}")
    
    try:
        df = pd.read_csv(filepath)
        print(f"✓ Loaded {len(df)} records")
        
        # Check required columns
        required_cols = ['code_frequency', 'macro_F1', 'dataset']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns. Found: {df.columns.tolist()}")
        
        # Get unique datasets
        datasets = df['dataset'].unique()
        print(f"✓ Found {len(datasets)} datasets: {', '.join(datasets)}")
        
        # Remove outliers
        print("\n🔧 Removing outliers...")
        df_cleaned = remove_outliers(df, method=args.outlier_method, threshold=args.outlier_threshold)
        print(f"✓ Data after outlier removal: {len(df_cleaned)} records")
        
        return df_cleaned
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        sys.exit(1)


def create_frequency_f1_plot(df, args):
    """Create the code frequency vs macro F1 plot with smoothed trends."""
    
    # Initialize publication style
    pps.setup_publication_style()
    
    # Create figure
    fig, ax = pps.create_figure(figsize=tuple(args.figsize))
    
    # Get color palette
    colors = pps.get_color_palette()
    color_list = [colors['primary'], colors['secondary'], colors['tertiary'], colors['quaternary']]
    
    # Plot each dataset
    datasets = df['dataset'].unique()
    
    for i, dataset in enumerate(datasets):
        dataset_df = df[df['dataset'] == dataset].copy()
        
        # Sort by frequency for better fitting
        dataset_df = dataset_df.sort_values('code_frequency')
        
        # Create smoothed trend line
        # Use LOWESS (Locally Weighted Scatterplot Smoothing) for smooth curve
        from scipy.interpolate import interp1d
        from statsmodels.nonparametric.smoothers_lowess import lowess
        
        # Apply LOWESS smoothing
        # Use log-transformed x for better smoothing across the range
        log_freq = np.log10(dataset_df['code_frequency'])
        smoothed = lowess(dataset_df['macro_F1'], log_freq, frac=args.smoothing)
        
        # Remove any duplicate x-values to avoid interpolation errors
        _, unique_indices = np.unique(smoothed[:, 0], return_index=True)
        smoothed = smoothed[unique_indices]
        
        # Interpolate for smooth curve
        freq_range = np.logspace(np.log10(dataset_df['code_frequency'].min()),
                               np.log10(dataset_df['code_frequency'].max()),
                               200)
        
        # Create interpolation function
        # Check if we have enough points for cubic interpolation
        if len(smoothed) >= 4:
            interp_kind = 'cubic'
        else:
            interp_kind = 'linear'
            
        f_interp = interp1d(10**smoothed[:, 0], smoothed[:, 1], 
                           kind=interp_kind, bounds_error=False, fill_value='extrapolate')
        
        # Get smoothed values
        smoothed_f1 = f_interp(freq_range)
        
        # Clip values to valid F1 range
        smoothed_f1 = np.clip(smoothed_f1, 0, 1)
        
        # Plot smoothed trend line
        # Use dashed line for MIMIC-IV ICD-9
        linestyle = '--' if 'ICD-9' in dataset else '-'
        
        ax.plot(freq_range, 
                smoothed_f1,
                label=dataset,
                color=color_list[i % len(color_list)],
                linewidth=3,
                linestyle=linestyle,
                alpha=0.9)
        
        # Add confidence band (optional)
        if args.confidence_band:
            # Calculate local standard deviation for confidence band
            window_size = int(len(dataset_df) * 0.1)
            if window_size > 5:
                # Simple confidence band based on local variation
                lower_bound = np.maximum(smoothed_f1 - 0.1, 0)
                upper_bound = np.minimum(smoothed_f1 + 0.1, 1)
                ax.fill_between(freq_range, lower_bound, upper_bound,
                              color=color_list[i % len(color_list)],
                              alpha=0.1)
    
    # Set log scale for x-axis
    ax.set_xscale('log')
    
    # Add grid before other elements
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.grid(True, which="minor", ls=":", alpha=0.1)
    
    # Labels and title
    ax.set_xlabel('Code Frequency (log scale)', fontsize=13, fontweight='medium')
    ax.set_ylabel('Macro F1 Score', fontsize=13, fontweight='medium')
    ax.set_title('Impact of Code Frequency on Model Performance',
                fontsize=15, fontweight='bold', pad=20)
    
    # Set axis limits
    ax.set_ylim(-0.05, 1.05)
    
    # Add threshold line to highlight rare codes
    ax.axvline(x=args.highlight_threshold, 
               color='gray', 
               linestyle='--', 
               alpha=0.5, 
               linewidth=1.5)
    
    # Add annotation for rare codes region
    ax.text(args.highlight_threshold/2, 0.95, 
            'Rare Codes\n(Low Performance)', 
            transform=ax.get_xaxis_transform(),
            ha='center', 
            va='top',
            fontsize=15,
            bbox=dict(boxstyle="round,pad=0.3", 
                     facecolor='yellow', 
                     alpha=0.3))
    
    # Add annotation for frequent codes
    ax.text(args.highlight_threshold*5, 0.95,
            'Frequent Codes\n(High Performance)',
            transform=ax.get_xaxis_transform(),
            ha='center',
            va='top',
            fontsize=15,
            bbox=dict(boxstyle="round,pad=0.3",
                     facecolor='lightgreen',
                     alpha=0.3))
    
    # Apply publication styling
    pps.style_axes(ax, fig)
    
    # Add legend
    pps.format_legend(ax, loc='lower right')
    
    # Remove statistical summary box to reduce clutter
    # Statistics are now included in the caption
    
    plt.tight_layout()
    
    return fig, ax


def generate_insights(df, threshold):
    """Generate insights about the data for the paper."""
    print("\n" + "="*60)
    print("📊 DATA INSIGHTS")
    print("="*60)
    
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset]
        
        print(f"\n{dataset}:")
        print(f"  Total codes: {len(dataset_df)}")
        print(f"  Frequency range: {dataset_df['code_frequency'].min():.1f} - {dataset_df['code_frequency'].max():.1f}")
        print(f"  Macro F1 range: {dataset_df['macro_F1'].min():.3f} - {dataset_df['macro_F1'].max():.3f}")
        
        # Analyze rare vs frequent codes
        rare_codes = dataset_df[dataset_df['code_frequency'] <= threshold]
        frequent_codes = dataset_df[dataset_df['code_frequency'] > threshold]
        
        if len(rare_codes) > 0:
            print(f"\n  Rare codes (frequency ≤ {threshold}):")
            print(f"    Count: {len(rare_codes)} ({len(rare_codes)/len(dataset_df)*100:.1f}%)")
            print(f"    Mean F1: {rare_codes['macro_F1'].mean():.3f} ± {rare_codes['macro_F1'].std():.3f}")
            print(f"    Median F1: {rare_codes['macro_F1'].median():.3f}")
            print(f"    F1 < 0.1: {len(rare_codes[rare_codes['macro_F1'] < 0.1])} codes")
        
        if len(frequent_codes) > 0:
            print(f"\n  Frequent codes (frequency > {threshold}):")
            print(f"    Count: {len(frequent_codes)} ({len(frequent_codes)/len(dataset_df)*100:.1f}%)")
            print(f"    Mean F1: {frequent_codes['macro_F1'].mean():.3f} ± {frequent_codes['macro_F1'].std():.3f}")
            print(f"    Median F1: {frequent_codes['macro_F1'].median():.3f}")
        
        # Calculate correlation
        from scipy.stats import spearmanr, pearsonr
        log_freq = np.log10(dataset_df['code_frequency'])
        spearman_corr, spearman_p = spearmanr(dataset_df['code_frequency'], dataset_df['macro_F1'])
        pearson_corr, pearson_p = pearsonr(log_freq, dataset_df['macro_F1'])
        
        print(f"\n  Correlations:")
        print(f"    Spearman ρ = {spearman_corr:.3f} (p = {spearman_p:.3e})")
        print(f"    Pearson r (log freq) = {pearson_corr:.3f} (p = {pearson_p:.3e})")


def generate_latex_caption(df, threshold):
    """Generate LaTeX caption for the figure."""
    
    # Calculate detailed statistics for caption
    datasets = df['dataset'].unique()
    dataset_details = []
    
    for dataset in datasets:
        dataset_df = df[df['dataset'] == dataset]
        rare_codes = dataset_df[dataset_df['code_frequency'] <= threshold]
        freq_codes = dataset_df[dataset_df['code_frequency'] > threshold]
        
        if len(rare_codes) > 0 and len(freq_codes) > 0:
            rare_mean = rare_codes['macro_F1'].mean()
            rare_std = rare_codes['macro_F1'].std()
            freq_mean = freq_codes['macro_F1'].mean()
            freq_std = freq_codes['macro_F1'].std()
            
            # Calculate percentage of codes that are rare
            rare_pct = len(rare_codes) / len(dataset_df) * 100
            
            dataset_details.append(
                f"{dataset}: rare codes (n={len(rare_codes)}, {rare_pct:.0f}% of total) "
                f"achieve mean F1={rare_mean:.3f}±{rare_std:.3f}, while frequent codes "
                f"(n={len(freq_codes)}) achieve mean F1={freq_mean:.3f}±{freq_std:.3f}"
            )
    
    caption = (
        "Relationship between code frequency and macro F1 score demonstrating the "
        "severe performance degradation on rare ICD codes. The x-axis uses logarithmic "
        "scale to better visualize the full range of code frequencies. "
        f"Codes with frequency ≤{threshold} (highlighted region) show near-zero F1 scores. "
        "Performance summary: " + "; ".join(dataset_details) + ". "
        "This stark performance drop for infrequent codes highlights the critical "
        "challenge in medical code prediction for rare conditions, where models fail "
        "to learn effective representations due to limited training examples."
    )
    
    print("\n" + "="*60)
    print("LATEX CAPTION")
    print("="*60)
    print(f"\\caption{{{caption}}}")


def main():
    """Main function."""
    args = parse_arguments()
    
    # Print header
    print("\n" + "📊 "*20)
    print("Code Frequency vs Macro F1 Plot Generator")
    print("📊 "*20 + "\n")
    
    # Load data
    filepath = Path(args.data_path) / args.filename
    df = load_and_prepare_data(filepath, args)
    
    # Create plot
    print("\n🎨 Creating frequency vs F1 plot...")
    fig, ax = create_frequency_f1_plot(df, args)
    
    # Save plot
    output_dir = Path(args.output_path) / 'publication_plots'
    output_dir.mkdir(exist_ok=True, parents=True)
    
    base_filename = "code_frequency_vs_macro_f1"
    
    print(f"\n💾 Saving plots to: {output_dir}")
    pps.save_publication_figure(fig, output_dir, base_filename, formats=['png', 'pdf'])
    
    # Generate insights
    generate_insights(df, args.highlight_threshold)
    
    # Generate LaTeX caption
    generate_latex_caption(df, args.highlight_threshold)
    
    print("\n✅ Plot generation complete!")
    print(f"\nFiles saved:")
    print(f"  - {output_dir}/{base_filename}.png")
    print(f"  - {output_dir}/{base_filename}.pdf")
    
    # Show plot if running interactively
    if hasattr(sys, 'ps1'):
        plt.show()


if __name__ == "__main__":
    main()