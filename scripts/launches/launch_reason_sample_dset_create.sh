#!/usr/bin/env bash
#
# create-sample.sh
# Wrapper to easily create a 5% sample of LF-WikiSeeAlso-320K
# with fixed seed for reproducibility
#
# Usage:
#   ./launch_reason_sample_dset_create.sh           # uses defaults: 0.05 fraction, seed 123
#   ./launch_reason_sample_dset_create.sh 0.02 456  # custom fraction and seed
#   ./launch_reason_sample_dset_create.sh 0.10      # custom fraction, default seed
#

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/../create_sample_for_reasoning.py"
SOURCE_URL="XURLs.LF_WIKISEEALSO_320K"

DEFAULT_FRACTION=0.0002
DEFAULT_SEED=123

# ── Argument parsing ─────────────────────────────────────────────

FRACTION="${1:-$DEFAULT_FRACTION}"
SEED="${2:-$DEFAULT_SEED}"

# Optional: you can add --no-stratify as third argument if needed
STRATIFY_FLAG=""
if [[ "${3:-}" == "--no-stratify" ]]; then
    STRATIFY_FLAG="--no-stratify"
fi

# ── Validation ───────────────────────────────────────────────────

if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found" >&2
    exit 1
fi

if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    echo "Error: $PYTHON_SCRIPT not found in current directory" >&2
    exit 1
fi

# Basic float check for fraction
if ! [[ "$FRACTION" =~ ^0\.[0-9]+$|^1(\.0+)?$ ]]; then
    echo "Error: fraction should be between 0.0 and 1.0 (got: $FRACTION)" >&2
    exit 1
fi

# ── Run ──────────────────────────────────────────────────────────

echo "=============================================================="
echo " Creating sample"
echo "=============================================================="
echo "Script     : $PYTHON_SCRIPT"
echo "Source     : $SOURCE_URL"
echo "Fraction   : $FRACTION"
echo "Seed       : $SEED"
echo "Stratify   : ${STRATIFY_FLAG:-(yes, by split)}"
echo ""

python3 "$PYTHON_SCRIPT" \
    --source_url "$SOURCE_URL" \
    --fraction "$FRACTION" \
    --seed "$SEED" \
    $STRATIFY_FLAG

echo ""