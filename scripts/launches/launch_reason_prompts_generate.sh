#!/usr/bin/env bash
#
# launch_reason_prompts_generate.sh
# Wrapper to generate classification prompts from the already created sample
# of LF-WikiSeeAlso-320K (or any other dataset using the same source_url pattern)
#
# Usage:
#   ./launch_reason_prompts_generate.sh
#       → uses default: XURLs.LF_WIKISEEALSO_320K
#
#   ./launch_reason_prompts_generate.sh "XURLs.OTHER_DATASET"
#       → generate prompts for a different dataset (if sampled already)
#
# Prerequisites:
#   - The sampling step must have been run first with the same source_url
#   - The *_sample.csv and *_unique_labels.csv files must exist in the _sample folder
#

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/../create_reasoning_prompts_dset.py"   # adjust filename if different

DEFAULT_SOURCE_URL="XURLs.LF_WIKISEEALSO_320K_sample"

# ── Argument parsing ─────────────────────────────────────────────

SOURCE_URL="${1:-$DEFAULT_SOURCE_URL}"

# Optional: allow --seed or other flags to be passed through
EXTRA_FLAGS="${@:2}"

# ── Validation ───────────────────────────────────────────────────

if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found" >&2
    exit 1
fi

if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    echo "Error: Python script not found at $PYTHON_SCRIPT" >&2
    echo "Expected location: one directory up from launches/" >&2
    exit 1
fi

# Very basic check that it looks like an XURLs key
if [[ ! "$SOURCE_URL" =~ ^XURLs\.[A-Z_]+$ ]]; then
    echo "Warning: SOURCE_URL '$SOURCE_URL' does not look like 'XURLs.SOMETHING'" >&2
    echo "Continuing anyway — make sure it resolves correctly in the Python script." >&2
fi

# ── Run ──────────────────────────────────────────────────────────

echo "=============================================================="
echo " Generating classification prompts"
echo "=============================================================="
echo "Script       : $PYTHON_SCRIPT"
echo "Source URL   : $SOURCE_URL"
echo "Extra flags  : ${EXTRA_FLAGS:-(none)}"
echo ""

python3 "$PYTHON_SCRIPT" \
    --source_url "$SOURCE_URL" \
    $EXTRA_FLAGS

echo ""
echo "Done."
echo "Look for a file ending with _prompts.jsonl inside the corresponding *_sample/ folder."
echo ""