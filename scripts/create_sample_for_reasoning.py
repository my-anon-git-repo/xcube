import argparse
import logging
from pathlib import Path
import pandas as pd
from datetime import datetime
import sys
from typing import Optional
import ast   # for safe-ish string → dict conversion
from collections import defaultdict

# Import xcube libraries
from xcube.imports import *

def sample_dset(
    source_url: str,
    fraction: float = 0.2,
    output_dir: Optional[Path] = None,
    stratify_by: str = "split",
    random_state: int = 42,
) -> Path:
    """
    Sample a fraction of the dataset while trying to preserve split proportions.

    Parameters:
        source_url    : URL or identifier of the source dataset (passed to untar_xxx)
        fraction      : Fraction of rows to sample (0.0 < fraction ≤ 1.0)
        output_dir    : Optional base directory where to save sampled data
                        (default: ./sampled_datasets/)
        stratify_by   : Column to stratify sampling by (usually 'split')
        random_state  : Random seed for reproducibility

    Returns:
        Path to the directory containing the sampled dataset files
    """
    if not (0 < fraction <= 1):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    logging.info(f"Starting sampling from {source_url} — target fraction: {fraction:.4f}")

    # 1. Download / extract dataset
    try:
        if not source_url.startswith("XURLs."):
            raise ValueError(f"Invalid source_url format: expected 'XURLs.XXX', got {source_url}")
        source_path = untar_xxx(eval(source_url))
        logging.info(f"Dataset extracted to: {source_path}")
    except Exception as e:
        logging.error(f"Failed to extract dataset from {source_url}: {e}")
        sys.exit(1)

    # 2. Find the main CSV file
    csv_files = list(source_path.glob("*.csv"))
    if not csv_files:
        logging.error("No .csv file found in extracted dataset directory")
        sys.exit(1)
    if len(csv_files) > 1:
        logging.warning(f"Multiple CSV files found, using first one: {csv_files[0].name}")

    dset_file = csv_files[0]
    logging.info(f"Reading dataset: {dset_file.name}")

    # 3. Load only needed columns
    needed_cols = ['uid', 'title', 'content', 'target_ind', 'labels', 'split']
    try:
        df = pd.read_csv(dset_file, usecols=lambda c: c in needed_cols)
        logging.info(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
    except Exception as e:
        logging.error(f"Failed to read CSV: {e}")
        sys.exit(1)

    if 'split' not in df.columns:
        logging.warning("Column 'split' not found — will do simple random sampling")
        stratify_by = None

    # 4. Sampling
    if stratify_by and stratify_by in df.columns:
        logging.info(f"Stratified sampling by column: {stratify_by}")
        df_sample = df.groupby(stratify_by, group_keys=False).apply(
            lambda x: x.sample(frac=fraction, random_state=random_state)
        )
    else:
        logging.info("Performing simple random sampling")
        df_sample = df.sample(frac=fraction, random_state=random_state)

    logging.info(f"Sampled {len(df_sample):,} rows ({len(df_sample)/len(df):.2%} of original)")

    # 5. Prepare output directory
    out_dir = source_path.with_name(source_path.name + "_sample")
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory: {out_dir}") 

    # 6. Save sampled dataset
    sampled_filename = dset_file.with_stem(dset_file.stem + "_sample").name
    output_csv = out_dir / sampled_filename
    df_sample.to_csv(output_csv, index=False)
    logging.info(f"Saved sampled dataset → {output_csv}")

    # 7. Extract unique labels from sampled data 
    logging.info("Extracting unique labels from sampled dataset...")

    unique_labels = set()
    label_to_values = defaultdict(set)  # key → set of display values (in case they vary)

    for label_str in df_sample['labels']:
        if pd.isna(label_str) or not label_str.strip():
            continue
        try:
            # Safely parse the string representation of dict
            label_dict = ast.literal_eval(label_str)
            if not isinstance(label_dict, dict):
                continue
                
            for key, value in label_dict.items():
                if key and value:  # skip empty
                    unique_labels.add(key)
                    label_to_values[key].add(value)
        except (ValueError, SyntaxError, TypeError) as e:
            logging.warning(f"Failed to parse labels string: {label_str[:80]}... → {e}")
            continue

    # Convert to sorted list for nicer output
    unique_labels_sorted = sorted(unique_labels)
    logging.info(f"Found {len(unique_labels_sorted):,} unique label keys in sample")

    # Prepare DataFrame with keys and (joined) values
    labels_df = pd.DataFrame({
        'label_key': unique_labels_sorted,
        'label_values': [', '.join(sorted(label_to_values[k])) for k in unique_labels_sorted]
    })

    # Save unique labels CSV
    labels_csv = out_dir / f"{dset_file.stem}_sample_unique_labels.csv"
    labels_df.to_csv(labels_csv, index=False)
    logging.info(f"Saved unique labels → {labels_csv}")

    # Optional: save small report
    (out_dir / "report.txt").write_text(
        f"""Sampling report
    ---------------
    Source:              {source_url}
    Original rows:       {len(df):,}
    Sampled rows:        {len(df_sample):,} ({len(df_sample)/len(df):.3%})
    Unique label keys:   {len(unique_labels_sorted):,}
    Output directory:    {out_dir}
    Sampled CSV:         {output_csv}
    Unique labels CSV:   {labels_csv}
    Date:                {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    Seed:                {random_state}
    Stratified:          {stratify_by or 'No'}
    """
    )

    logging.info("Sampling completed successfully")
    return out_dir


def main():
    parser = argparse.ArgumentParser(
        description="Sample a fraction from a large dataset (Wikipedia-style classification/linking)"
    )
    parser.add_argument(
        "--source_url",
        type=str,
        required=True,
        help="URL or identifier of the source dataset archive"
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.10,
        help="Fraction of the dataset to keep (default: 0.20)"
    )
    parser.add_argument(
        "--no_stratify",
        action="store_true",
        help="Disable stratified sampling by 'split' column"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    # Configure logging for the whole script
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-7s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logging.info("=== Dataset Sampling Script ===")

    output_path = sample_dset(
        source_url=args.source_url,
        fraction=args.fraction,
        stratify_by=None if args.no_stratify else "split",
        random_state=args.seed
    )

    print(f"\nDone! Sampled dataset saved to:")
    print(f"  {output_path}")


if __name__ == "__main__":
    main()