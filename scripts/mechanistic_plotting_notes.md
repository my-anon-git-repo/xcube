python plant_ablation_freq_binned_transfer.py --output-path ../tmp/publication_plots/

python  stage1_plant_pretraining_effect_singleplots.py  --data-path ./mimic_low_data/ --filename stage1_epoch_sweep2.csv --output-path ../tmp/publication_plots/ --verbose

python plot_plant_label_attention_histogram.py --output-path ../tmp/publication_plots/

Mechanistic Plotting:

python plot_phase1_attn_heads_kurtosis.py --output-path ../tmp/publication_plots/

python plot_attention_inset_medical_llama70b.py --output-path ../tmp/publication_plots/

# Mechanistic Plotting Notebook

Central documentation for all publication-quality plots used in the mechanistic interpretability paper.

**Style:** All plots use `publication_plot_style` (fallback included).  
**Output format:** High-quality PDF + PNG (400 dpi).  
**Location:** Plots are saved to `publication_plots/` by default.

---

## 1. Excess Kurtosis of Attention Heads (Phase 1)

**Purpose:** Rank attention heads by CoarseScore = κ(attention) × cos(Δʰ(q), c_anchor).
This directly supports Equation~\ref{eq:coarse_score} by showing heads that both attend sharply to hierarchical anchors and write semantically relevant updates into the residual stream.

**Script:** `plot_phase1_attn_kurtosis_llama70b.py`

**Command:**
```bash
python plot_phase1_attn_kurtosis_llama70b.py \
    --output-path ../tmp/publication_plots/ \
    --filename fig_phase1_attn_heads_kurtosis_llama70b \
    --figsize 11.8 7.3 \
    --dpi 300
```

## 2. Attention Inset: Early-Layer Head on Clinical Anchors (LLaMA-70B)

**Purpose:** Show that top early-layer heads (e.g. L3H22) produce sharply peaked (leptokurtic) attention on hierarchical clinical anchors during early CoT generation.

**Script:** `plot_attention_inset_medical_llama70b.py`

**Command:**
```bash
python plot_attention_inset_medical_llama70b.py \
    --output-path ../tmp/publication_plots/ \
    --filename inset_medical_early_head_attention_llama70b \
    --figsize 8.5 4.6 \
    --dpi 300
```

## 3. Phase 1 - Denoising Activation Patching (Cumulative Binning)

**Purpose:** Demonstrate the **sufficiency** of top CoarseScore-ranked early-layer heads for Phase 1 coarse hierarchical filtering by patching them in cumulative bins and measuring restoration of Reasoning Focus similarity.

**Script:** `plot_phase1_denoising_patching.py`

**Command:**
```bash
python plot_phase1_denoising_patching.py \
    --output-path ../tmp/publication_plots/ \
    --filename fig_phase1_denoising_patching \
    --figsize 12.8 7.6 \
    --dpi 300
```

## 4. Phase 1 - Noising Activation Patching (Cumulative Binning)

**Purpose:** Demonstrate the **necessity** of top CoarseScore-ranked early-layer heads for Phase 1 coarse hierarchical filtering by noising them in cumulative bins and measuring the disruption of Reasoning Focus similarity.

**Script:** `plot_phase1_noising_patching.py`

**Command:**
```bash
python plot_phase1_noising_patching.py \
    --output-path ../tmp/publication_plots/ \
    --filename fig_phase1_noising_patching \
    --figsize 12.8 7.6 \
    --dpi 300
```

## 5. Phase 2 - Refinement Head Ranking by RefineScore

**Purpose:** Rank mid-to-late layer attention heads by the combined RefineScore (QK-refinement + OV-helpfulness) as defined in Equation~\ref{eq:refine_score}. 
This plot identifies heads that iteratively re-attend to prior shortlist representations while suppressing near-miss ones, and whose residual updates help widen the logit margin between target and near-miss clinical subcategories.

**Script:** `plot_phase2_refine_score_ranking.py`

**Command:**
```bash
python plot_phase2_refine_score_ranking.py \
    --output-path ../tmp/publication_plots/ \
    --filename fig_phase2_refine_score_ranking \
    --figsize 18.0 11.1 \
    --dpi 300
```

## 6. Phase 2 - Temporal Dynamics of Top Refinement Heads

**Purpose:** Show how the top RefineScore-ranked heads evolve over the course of Chain-of-Thought generation. 
Specifically, how QK similarity to prior shortlist representations increases while similarity to near-miss representations decreases, and how OV-helpfulness (margin widening) grows stronger in later CoT segments.

**Script:** `plot_phase2_refine_temporal_dynamics.py`

**Command:**
```bash
python plot_phase2_refine_temporal_dynamics.py \
    --output-path ../tmp/publication_plots/ \
    --filename fig_phase2_refine_temporal_dynamics \
    --figsize 18.0 9.1 \
    --dpi 300
```

## 7. Phase 2 - Denoising Activation Patching (Cumulative Binning)

**Purpose:** Demonstrate the **sufficiency** of top RefineScore-ranked heads for Phase 2 iterative refinement by patching them in cumulative bins and measuring the downstream effect on Near-Miss Confusion (lower is better).

**Script:** `plot_phase2_denoising_patching.py`

**Command:**
```bash
python plot_phase2_denoising_patching.py \
    --output-path ../tmp/publication_plots/ \
    --filename fig_phase2_denoising_patching \
    --figsize 12.8 7.6 \
    --dpi 300
```

## 8. Phase 2 - Noising Activation Patching (Cumulative Binning)

**Purpose:** Demonstrate the **necessity** of the top RefineScore-ranked heads for Phase 2 iterative refinement by noising them in cumulative bins and measuring the increase in Near-Miss Confusion (lower is better).

**Script:** `plot_phase2_noising_patching.py`

**Command:**
```bash
python plot_phase2_noising_patching.py \
    --output-path ../tmp/publication_plots/ \
    --filename fig_phase2_noising_patching \
    --figsize 12.8 7.6 \
    --dpi 300
```


To rigorously evaluate causality, we conducted activation patching experiments on heads ranked by CoarseScore (Phase~1) and RefineScore (Phase~2). 
All experiments were performed in a controlled setting focused on the clinically important heart failure category (ICD-10 I50.x family and closely related codes, restricted to a closed set of approximately 50 labels). 
We used only discharge note snippets primarily concerning heart failure or related cardiac conditions and prompted the model to select diagnoses exclusively from this predefined set. 
This focused design enables precise measurement of refinement quality while remaining representative of real-world clinical complexity.
Importantly, both CoarseScore and RefineScore are category-agnostic: they depend only on the model's internal attention patterns and residual updates relative to shortlist and near-miss representations, not on any disease-specific features. 
Thus, the identification and validation scheme is readily generalizable to other clinical categories (e.g., respiratory diseases, renal failure, or infectious conditions) without requiring major modifications.

For both phases, we performed cumulative patching by simultaneously intervening on increasing bins of top-ranked heads (Top 3, Top 8, Top 16, Top 40, and Top 100). 
As negative controls, we applied identical patching to randomly selected heads and to low-ranked heads matched in number; these controls produced negligible effects on both Reasoning Focus (Phase~1) and Near-Miss Confusion (Phase~2).
In Phase~1, we measured \emph{Reasoning Focus}, defined as
\[
\mathsf{Focus} = \mathbb{E}_{q \in \mathcal{Q}_{\mathrm{early}}} \cos\!\big(e(r_q),\; e(x)\big).
\]
In Phase~2, we quantified refinement quality using \emph{Near-Miss Confusion} at refinement positions:
\[
\mathsf{Confusion}(q) = \mathbb{E}_{q \in \mathcal{R}_{\mathrm{refine}}} \cos\!\big(e(r_q),\; e(\mathcal{N}(x))\big),
\]
where lower values indicate better discrimination between the target diagnosis and semantically similar near-miss alternatives.

Denoising patching restored performance in corrupted runs, while noising patching disrupted it in clean runs. 
As negative controls, we applied the same patching procedure to randomly selected and low-ranked heads, which yielded negligible effects. 
This setup allows us to establish both the sufficiency and necessity of the identified heads for each phase of the reasoning process.

python plot_phase1and2_side_by_side_mimic.py     --output-path ../tmp/publication_plots/     --filename fig_reasoning_focus_confusion_mimic     --figsize 16 4.95


python plot_phase1and2_ablation.py     --figsize 22 7     --output-path publication_plots     --filename fig_mechanistic_distillation_ablations_golden
