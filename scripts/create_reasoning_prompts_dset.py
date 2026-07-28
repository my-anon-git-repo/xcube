import argparse
import logging
import json
from pathlib import Path
import pandas as pd
from typing import Optional

# Import xcube libraries
from xcube.imports import *

def generate_prompts_dset(
    source_url: str,
    random_state: int = 42,
) -> Path:
    """
    Generate prompts for each example in the sampled dataset.
    
    Loads the sampled CSV and unique labels CSV from the sample_dir.
    For each row, creates a prompt with snippet (title + content) and unique label values.
    Saves as .jsonl with one JSON per line: {"uid": int, "title": str, "the_created_prompt": str, "labels": str}
    
    Returns the path to the generated .jsonl file.
    """

    # Resolve source path (same as sampling script)
    try:
        source_path = untar_xxx(eval(source_url))
        logging.info(f"Dataset at: {source_path}")
    except Exception as e:
        logging.error(f"Failed to resolve/extract dataset: {e}")
        raise

    # Derive the _sample sibling directory
    sample_dir = source_path
    # sample_dir = source_path.with_name(source_path.name + "_sample")
    # if not sample_dir.is_dir():
    #     raise FileNotFoundError(
    #         f"Sample directory not found: {sample_dir}\n"
    #         "Run the sampling script first with the same source_url."
    #     )

    logging.info(f"Starting prompt generation from sample directory: {sample_dir}")

    # Find sampled CSV
    sampled_csv_files = list(sample_dir.glob("*_sample.csv"))
    if not sampled_csv_files:
        raise FileNotFoundError("No sampled CSV found in directory")
    if len(sampled_csv_files) > 1:
        logging.warning(f"Multiple sampled CSVs found — using first: {sampled_csv_files[0].name}")
    sampled_csv = sampled_csv_files[0]
    logging.info(f"Loading sampled dataset: {sampled_csv.name}")

    # Load sampled data
    df_sample = pd.read_csv(sampled_csv)
    logging.info(f"Loaded {len(df_sample):,} examples from sampled CSV")

    # Find unique labels CSV
    unique_labels_files = list(sample_dir.glob("*_unique_labels.csv"))
    if not unique_labels_files:
        raise FileNotFoundError("No unique labels CSV found in directory")
    unique_labels_csv = unique_labels_files[0]
    logging.info(f"Loading unique labels: {unique_labels_csv.name}")

    # Load unique labels — assume columns 'label_key', 'label_values'
    df_labels = pd.read_csv(unique_labels_csv)
    # Use label_values, sorted alphabetically, one per line
    label_values = sorted(df_labels['label_values'].unique())
    labels_list_str = "\n".join(label_values)
    logging.info(f"Found {len(label_values):,} unique label values for prompts")

    # Prepare output .jsonl in same directory
    output_jsonl = sample_dir / f"{sampled_csv.stem}_prompts.jsonl"
    logging.info(f"Output JSONL: {output_jsonl}")

    # Template for the prompt
    PROMPT_TEMPLATE_1 = """You are an expert multi-label classifier for Wikipedia article snippets. Your task is to assign ALL relevant category labels to the given text snippet. This is a multi-label problem: one snippet can (and often should) match multiple categories.
Rules you MUST strictly follow:
1. ONLY use labels that are exactly in the provided list below. Do NOT invent, guess, add, or use any label not present in the list — even if it seems close.
2. Select as many labels as genuinely apply (0 is allowed if nothing fits well, but aim for up to around 10 at most for typical snippets).
3. Prioritize precision: only include a label if there is strong, direct evidence in the snippet.
4. Think step by step in detail:
   - First, read the snippet carefully and summarize the main topic, key entities, events, and themes in 2–3 sentences.
   - Then, scan the entire category list and identify candidate labels that seem potentially relevant.
   - For each candidate, explain briefly why it might match (or why it does NOT match) based on specific words, concepts, or context in the snippet.
   - Rank the strongest matches by relevance/confidence.
   - Finally, decide on the final set of labels (usually 2–10 for informative snippets).
5. Output ONLY in this exact structured format — nothing before <think> or after </answer>:
<think>
[Your full step-by-step reasoning, candidate evaluation, and justifications here]
</think>
<answer>
Relevant categories: label1, label2, label3, ... (comma-separated, exact spelling from list; empty if none apply)
Confidence order: most relevant first
</answer>
Category list (choose only from these — exact matches only):
{labels_list}

Wikipedia snippet to classify:
\"\"\"
{snippet}
\"\"\"
Now perform the multi-label classification following the rules above."""

    PROMPT_TEMPLATE_2 = """You are an expert multi-label classifier for Wikipedia article snippets. Your task is to assign ALL relevant category labels to the given text snippet. This is a multi-label problem: one snippet can (and often should) match multiple categories.

    **CRITICAL RULES YOU MUST FOLLOW UNDER NO CIRCUMSTANCES VIOLATE:**
    1. **ONLY USE LABELS EXACTLY AS THEY APPEAR IN THE PROVIDED LIST BELOW. DO NOT INVENT, GUESS, MODIFY, ADD, OR USE ANY LABEL NOT PRESENT IN THE LIST — EVEN IF IT SEEMS VERY CLOSE OR LOGICAL. NO EXCEPTIONS!** If a concept isn't exactly in the list, ignore it completely.
    2. Select as many labels as genuinely apply (0 is allowed if nothing fits well, but aim for up to around 10 at most for typical snippets).
    3. Prioritize precision: only include a label if there is strong, direct evidence in the snippet. **REMEMBER: ONLY FROM THE LIST!**
    4. Think step by step in detail:
    - First, read the snippet carefully and summarize the main topic, key entities, events, and themes in 2–3 sentences.
    - Second, scan the ENTIRE category list provided below and explicitly list out 15-20 potential candidate labels EXACTLY AS THEY APPEAR IN THE LIST (no more, no less; choose ones that seem even remotely related based on keywords). DO NOT ADD ANY LABELS NOT IN THE LIST HERE.
    - Third, for EACH of those exact candidates you listed, explain briefly why it might match (or why it does NOT match) based on specific words, concepts, or context in the snippet. **DO NOT EVALUATE ANY OTHER LABELS OUTSIDE YOUR 15-20 CANDIDATES.**
    - Fourth, rank the strongest matches from your candidates by relevance/confidence, eliminating any that don't have strong evidence.
    - Fifth, decide on the final set of labels (usually 2–10) from ONLY your evaluated candidates. **Final double-check: For each selected label, confirm it is verbatim (exact spelling, no changes) from the original list.**
    5. Output ONLY in this exact structured format — nothing before <think> or after </answer>. **VIOLATING FORMAT OR RULES WILL INVALIDATE YOUR RESPONSE.**

    <think>
    [Your full step-by-step reasoning, candidate extraction, evaluations, rankings, and final double-check here. STRICTLY FOLLOW THE RULES!]
    </think>

    <answer>
    Relevant categories: label1, label2, label3, ... (comma-separated, exact spelling from list; empty if none apply)
    Confidence order: most relevant first
    </answer>

    Category list (choose ONLY from these — EXACT MATCHES ONLY, NO ADDITIONS):
    {labels_list}

    Wikipedia snippet to classify:
    \"\"\"
    {snippet}
    \"\"\"

    Now perform the multi-label classification following the rules above. **REMEMBER: ONLY USE EXACT LABELS FROM THE LIST — NO EXCEPTIONS!**"""

    PROMPT_TEMPLATE_3 = """You are an expert multi-label classifier for Wikipedia article snippets. Your task is to assign ONLY categories from the provided list below.

**CRITICAL RULES - YOU MUST FOLLOW THESE EXACTLY:**
1. **ABSOLUTELY DO NOT CREATE NEW CATEGORIES** - Only use labels EXACTLY as they appear in the list below
2. **DO NOT USE ANY CATEGORY NOT IN THE LIST** - Even if it seems obvious or logical
3. **VERBATIM MATCHING ONLY** - Copy the labels exactly as written, including punctuation and spacing
4. If no categories from the list match, output an empty list

The ONLY valid categories are:
{labels_list}

Wikipedia snippet to classify:
\"\"\"
{snippet}
\"\"\"

First, think step by step about which categories from the list match the snippet.
Then, output your answer in this exact format:

<answer>
Relevant categories: category1, category2, category3 (must be EXACTLY from the list above, comma-separated)
</answer>

Remember: ONLY use categories from the list above. DO NOT invent new categories."""

    PROMPT_TEMPLATE_4 = """You are an expert multi-label classifier for Wikipedia article snippets. Assign categories to the snippet using only the ones from this list:
{labels_list}

Wikipedia snippet:
\"\"\"
{snippet}
\"\"\"
Output your reasoning followed by the final answer in this format:
<answer>
Relevant categories: category1, category2, category3 (comma-separated, only from the list)
</answer>
If none apply, use an empty list after 'Relevant categories:'."""

    PROMPT = PROMPT_TEMPLATE_4
    # Generate prompts and write to jsonl
    with output_jsonl.open("w", encoding="utf-8") as f:
        for _, row in df_sample.iterrows():
            # Snippet: title + content
            snippet = f"{row['title']}\n{row['content']}"

            # Fill template
            prompt = PROMPT.format(
                labels_list=labels_list_str,
                snippet=snippet
            )

            # JSON object
            example = {
                "uid": row['uid'],
                "title": row['title'],
                "the_created_prompt": prompt,
                "labels": row['labels']  # original labels string
            }

            # Write line
            f.write(json.dumps(example) + "\n")

    logging.info(f"Generated prompts for {len(df_sample):,} examples → {output_jsonl}")
    logging.info("Prompt generation completed successfully")

        # ── Create a small test file with one example prompt ────────────────────────────────
    logging.info("Creating sample prompt_test.py for quick model testing...")

    import random

    # Pick one example — can be random, or just the first one
    # Option A: random (recommended for variety)
    rng = random.Random(random_state)
    sample_idx = rng.randint(0, len(df_sample) - 1)
    sample_row = df_sample.iloc[sample_idx]

    # Option B: always first one (more deterministic)
    # sample_row = df_sample.iloc[0]
    # sample_idx = 0

    sample_snippet = f"{sample_row['title']}\n{sample_row['content']}".strip()

    sample_prompt = PROMPT.format(
        labels_list=labels_list_str,
        snippet=sample_snippet
    )

    test_file = sample_dir / "prompt_test.py"

    with test_file.open("w", encoding="utf-8") as f:
        f.write("""# prompt_test.py
# Auto-generated example prompt from the sampled LF-WikiSeeAlso dataset
# Generated on: {now}
# Original row index in sampled CSV: {idx}
# UID: {uid}
# Title: {title}

snippet = '''{snippet}'''

labels_list_str = '''{labels}'''

PROMPT = '''{prompt}'''

# You can copy-paste PROMPT into model playgrounds / eval scripts
# Original labels for reference (string form):
ORIGINAL_LABELS = {original_labels!r}

print("Sample prompt ready. Length:", len(PROMPT))
""".format(
            now=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            idx=sample_idx,
            uid=sample_row['uid'],
            title=sample_row['title'].replace('"', '\\"'),  # escape quotes
            prompt=sample_prompt,
            original_labels=sample_row['labels'],
            snippet=sample_snippet,
            labels=labels_list_str
        ))

    logging.info(f"Created quick-test file: {test_file}")

    return output_jsonl


def main():
    parser = argparse.ArgumentParser(
        description="Generate prompts for multi-label classification from sampled dataset"
    )
    parser.add_argument(
        "--source_url",
        type=str,
        required=True,
        help="Dataset identifier, e.g. 'XURLs.LF_WIKISEEALSO_320K'"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=678,
        help="Random seed (default: 876) — not used here but for consistency"
    )

    args = parser.parse_args()

    # Configure logging once here
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-7s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logging.info("=== Prompt Generation Script started ===")

    try:
        output_file = generate_prompts_dset(
            source_url=args.source_url,
            random_state=args.seed
        )

        print(f"\nDone!")
        print(f"  Source:   {args.source_url}")
        print(f"  Prompts:    {output_file}")

    except Exception as e:
        logging.error(f"Generation failed: {e}")
        raise


if __name__ == "__main__":
    main()