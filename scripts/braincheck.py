# =============================================================================
# Utility Functions
# =============================================================================
import os
import logging
from pathlib import Path
import pandas as pd
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union


def setup_logging(logfile="l2r_load", **kwargs):
    """Set up logging configuration"""
    log_dir = os.path.join(kwargs.get('output_dir', '.'), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    logfile_path = os.path.join(log_dir, f"{logfile}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(logfile_path),
            logging.StreamHandler()
        ]
    )
    return logfile_path


def join_path_file(fname, path, ext=''):
    """Join path and filename with extension"""
    return Path(path) / f"{fname}{ext}"


def print_args(**kwargs):
    """Print function arguments"""
    logging.info("Arguments:")
    for key, value in kwargs.items():
        logging.info(f"  {key}: {value}")


# =============================================================================
# Loading Function
# =============================================================================
def load_l2r_data(
    output_dir: str,
    fname: str
) -> Dict[str, Union[pd.DataFrame, List]]:
    """
    Load Learning-to-Rank data from saved files
    
    Args:
        output_dir: Directory containing the saved files
        fname: Base filename used when saving the data
    
    Returns:
        Dictionary containing loaded dataframes and related data
    """
    global generated_files
    generated_files = []
    
    logging.info(f"Loading Learning-to-Rank data from {output_dir}")
    
    # Create paths for the files
    token_label_rankings_file = join_path_file(fname + '_tok_lbl', Path(output_dir), ext='.ft')
    tokens_file = join_path_file(fname + '_tok', Path(output_dir), ext='.ft')
    labels_file = join_path_file(fname + '_lbl', Path(output_dir), ext='.ft')
    
    # Check that all required files exist
    files_to_check = [
        (token_label_rankings_file, "Token-label rankings"),
        (tokens_file, "Token mappings"),
        (labels_file, "Label mappings")
    ]
    
    for file_path, desc in files_to_check:
        if not file_path.exists():
            logging.error(f"Required file not found: {file_path} ({desc})")
            raise FileNotFoundError(f"Required file not found: {file_path}")
        else:
            logging.info(f"Found {desc} file: {file_path}")
    
    # Load the dataframes
    logging.info("Loading token-label rankings...")
    df_l2r = pd.read_feather(token_label_rankings_file)
    logging.info(f"Loaded token-label rankings with shape: {df_l2r.shape}")
    
    logging.info("Loading token mappings...")
    df_toks = pd.read_feather(tokens_file)
    logging.info(f"Loaded {len(df_toks)} token mappings")
    
    logging.info("Loading label mappings...")
    df_lbs = pd.read_feather(labels_file)
    logging.info(f"Loaded {len(df_lbs)} label mappings")
    
    # Extract token and label lists
    toks = df_toks['tok_val'].tolist()
    lbs = df_lbs['lbl_val'].tolist()
    
    # Create summary of loaded data
    loaded_data = {
        'df_l2r': df_l2r,
        'df_toks': df_toks,
        'df_lbs': df_lbs,
        'toks': toks,
        'lbs': lbs
    }
    
    # Add to generated files list for reporting
    generated_files.extend([
        (token_label_rankings_file, "DataFrame containing token-label pairs with mutual information scores and rankings"),
        (tokens_file, "DataFrame mapping token IDs to their string values"),
        (labels_file, "DataFrame mapping label IDs to their string values")
    ])
    
    # Print summary of loaded files
    print("\n" + "="*80)
    print("LOADED FILES SUMMARY")
    print("="*80)
    for file_path, description in generated_files:
        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_name)[1]
        file_size = os.path.getsize(file_path)
        
        # Format file size for human readability
        if file_size < 1024:
            size_str = f"{file_size} bytes"
        elif file_size < 1024*1024:
            size_str = f"{file_size/1024:.2f} KB"
        else:
            size_str = f"{file_size/(1024*1024):.2f} MB"
            
        print(f"File: {file_path}")
        print(f"  - Type: {file_ext[1:].upper()} file")
        print(f"  - Size: {size_str}")
        print(f"  - Description: {description}")
        print("-"*80)
    
    return loaded_data


# =============================================================================
# Analysis Functions
# =============================================================================
def analyze_token_label_rankings(df_l2r: pd.DataFrame, df_toks: pd.DataFrame, df_lbs: pd.DataFrame) -> Dict:
    """
    Analyze the token-label rankings dataframe
    
    Args:
        df_l2r: Token-label rankings dataframe
        df_toks: Token mappings dataframe
        df_lbs: Label mappings dataframe
        
    Returns:
        Dictionary with analysis results
    """
    logging.info("Analyzing token-label rankings data...")
    
    # Create a merged dataframe with token and label values
    df_merged = df_l2r.merge(df_toks, on='token').merge(df_lbs, left_on='label', right_on='lbl')
    
    # Get top ranked tokens for each label
    results = {}
    
    # Find labels with highest mutual information scores
    top_labels = df_merged.groupby('lbl_val')['bcx_mutual_info'].mean().sort_values(ascending=False).head(10)
    results['top_labels'] = top_labels.to_dict()
    
    # Find tokens with highest mutual information across all labels
    top_tokens = df_merged.groupby('tok_val')['bcx_mutual_info'].mean().sort_values(ascending=False).head(10)
    results['top_tokens'] = top_tokens.to_dict()
    
    # Get summary statistics for mutual information scores
    info_stats = {
        'mean': float(df_l2r['bcx_mutual_info'].mean()),
        'median': float(df_l2r['bcx_mutual_info'].median()),
        'min': float(df_l2r['bcx_mutual_info'].min()),
        'max': float(df_l2r['bcx_mutual_info'].max()),
        'std': float(df_l2r['bcx_mutual_info'].std())
    }
    results['info_stats'] = info_stats
    
    logging.info("Analysis complete")
    
    return results


# =============================================================================
# Main Function
# =============================================================================
def main(
    output_dir: str = "task2_output",
    data: str = "mimic4_icd10_full",
    base_dir: str = None,
    source_url: str = "XURLs.MIMIC4_L2R",
    logfile: str = "l2r_load",
    run_analysis: bool = True
):
    """
    Main function to load L2R data from previously generated files
    
    Args:
        output_dir: Directory containing the saved files
        data: Dataset name
        base_dir: Base directory for the project
        source_url: Source URL for the data
        logfile: Name of the log file
        run_analysis: Whether to run analysis on the loaded data
    """
    # -----------------------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------------------
    global generated_files
    generated_files = []
    
    # Setup path
    fname = '_'.join(data.split('_')[:-1])
    if base_dir is not None:
        base_dir = base_dir + '/' + source_url.split('.')[1]
        output_dir = str(Path(base_dir) / output_dir)
    
    # Setup logging
    logfile_path = setup_logging(logfile=logfile, output_dir=output_dir)
    print(f"You can tail the logfile: {logfile_path} to check logs")
    print_args(**locals())
    
    logging.info("****L2R Data Loading Started****")
    
    # -----------------------------------------------------------------------------
    # Load Data
    # -----------------------------------------------------------------------------
    try:
        loaded_data = load_l2r_data(output_dir, fname)
        logging.info("Successfully loaded all L2R data")
        
        # Basic validation
        df_l2r = loaded_data['df_l2r']
        df_toks = loaded_data['df_toks']
        df_lbs = loaded_data['df_lbs']
        
        logging.info(f"Token-label rankings shape: {df_l2r.shape}")
        logging.info(f"Number of unique tokens: {df_toks.shape[0]}")
        logging.info(f"Number of unique labels: {df_lbs.shape[0]}")
        
        # Validate relationships
        expected_pairs = len(loaded_data['toks']) * len(loaded_data['lbs'])
        actual_pairs = len(df_l2r)
        
        if expected_pairs == actual_pairs:
            logging.info(f"Data validation successful: Found expected {actual_pairs} token-label pairs")
        else:
            logging.warning(f"Data validation issue: Expected {expected_pairs} token-label pairs, found {actual_pairs}")
        
        # -----------------------------------------------------------------------------
        # Analysis (Optional)
        # -----------------------------------------------------------------------------
        if run_analysis:
            analysis_results = analyze_token_label_rankings(df_l2r, df_toks, df_lbs)
            
            print("\n" + "="*80)
            print("ANALYSIS RESULTS")
            print("="*80)
            
            print("Top 10 labels by mutual information:")
            for label, score in list(analysis_results['top_labels'].items())[:10]:
                print(f"  {label}: {score:.6f}")
            
            print("\nTop 10 tokens by mutual information:")
            for token, score in list(analysis_results['top_tokens'].items())[:10]:
                print(f"  {token}: {score:.6f}")
            
            print("\nMutual information statistics:")
            for stat, value in analysis_results['info_stats'].items():
                print(f"  {stat}: {value:.6f}")
            
            print("-"*80)
        
        logging.info("****L2R Data Loading Completed Successfully****")
        return loaded_data
        
    except Exception as e:
        logging.error(f"Error loading L2R data: {e}")
        raise


if __name__ == "__main__":
    # Example usage
    main(
        output_dir="task2_output",
        data="mimic4_icd10_full",
        base_dir="/path/to/base/dir",
        run_analysis=True
    )