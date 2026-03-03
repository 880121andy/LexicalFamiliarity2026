# Lexical Familiarity Predicts Processing Depth for Nonliteral Language in Large Language Models

> **Anonymity Notice:** This repository is fully anonymized for double-blind peer review. All author identities, affiliations, and specific institutional acknowledgments have been removed.

## Overview

This repository contains the datasets, code, and analysis results for the experiments from our paper, *"Lexical Familiarity Predicts Processing Depth for Nonliteral Language in Large Language Models."* Our research investigates how Large Language Models (LLMs) process language that deviates from literal, standard usage. By analyzing layer-wise representations using Gemma-Scope-2 Sparse Autoencoders (SAEs), we establish a "lexical familiarity gradient" across five categories of nonliteral language.

## Repository Structure

```
.
├── data/
│   ├── idioms.csv                          # Idioms dataset with position tags
│   ├── metaphors.csv                       # Metaphors dataset with position tags
│   ├── slang_neologisms.csv                # Neologisms with position tags
│   ├── slang_semantic_shift.csv            # Semantic shift pairs with position tags
│   ├── slang_constructional.csv            # Constructional slang with position tags
│   ├── literal_paraphrase.csv              # Baseline literal language data
│   └── identical_sentence_pair.csv         # Control pairs
├── scripts/
│   ├── run.py                              # Main SAE feature extraction and analysis
│   ├── lf_full.py                          # Lexical familiarity score computation (full, tokenizer-based)
│   ├── lf_simplified.py                    # Lexical familiarity score computation (simplified)
│   ├── normal_approx_ci.py                 # 95% confidence interval computation for results
│   ├── span_length.py                      # Divergence span length confound analysis
│   └── README.md                           # Script usage instructions
├── results/
│   ├── idioms.csv                          # Layer-wise SAE analysis results for idioms
│   ├── metaphors.csv                       # Layer-wise SAE analysis results for metaphors
│   ├── slang_neologisms.csv                # Layer-wise SAE analysis results for neologisms
│   ├── slang_semantic_shift.csv            # Layer-wise SAE analysis results for semantic shift
│   ├── slang_constructional.csv            # Layer-wise SAE analysis results for constructional slang
│   ├── literal_paraphrase.csv              # Layer-wise SAE analysis results for literal language
│   ├── identical_sentence_pair.csv         # Layer-wise SAE analysis results for control pairs
│   ├── ci/
│   │   ├── all_categories_with_ci.csv      # Cosine distances with 95% CIs across all categories
│   │   ├── overlap_analysis.csv            # CI overlap analysis across categories
│   │   ├── table2_with_ci.csv              # Table 2 data with confidence intervals
│   │   ├── table3_with_ci.csv              # Table 3 data with confidence intervals
│   │   ├── table2_latex.tex                # LaTeX source for Table 2
│   │   ├── table3_latex.tex                # LaTeX source for Table 3
│   │   ├── figure1_with_ci.png             # Figure 1 with confidence intervals
│   │   └── residual_analysis.png           # Residual diagnostic plots
│   └── lexical_familiarity/
│       ├── lf_scores_all_items.csv         # Per-item lexical familiarity scores
│       ├── lf_peak_layer_figure.png        # Category means with 95% CIs (Figure for paper)
│       ├── lf_analysis_plots.png           # Diagnostic panels (supplementary)
│       └── lf_table.tex                    # LaTeX table for LF analysis
├── requirements.txt                        # Python package dependencies
└── README.md
```

## Data Format

Each dataset CSV file contains sentence pairs with position annotations marking the divergence span between figurative and literal versions:

- **Figurative/nonliteral text**: The slang, idiom, metaphor, or figurative expression
- **Literal text**: The literal or paraphrased equivalent
- **Position tags**: Start and end word indices of the relevant words/phrases in both versions (as lists of `[start, end)` tuples)
- **Segment information**: The specific figurative words/phrases and their literal counterparts

Column names vary by category (e.g., `idiomatic`/`normal` for idioms, `metaphorical`/`normal` for metaphors, `gen_z`/`normal` for slang).

## Scripts

### `scripts/run.py` — Main SAE Analysis

This is the main analysis script that:

1. **Loads datasets** from the `data/` directory
2. **Extracts SAE features** from Gemma-Scope-2 Sparse Autoencoders for each sentence pair
3. **Computes layer-wise representations** across all 48 layers of the model
4. **Analyzes feature distributions** to identify the lexical familiarity gradient
5. **Generates results** and saves them to the `results/` directory

### `scripts/lf_full.py` — Lexical Familiarity (Full)

Computes the lexical familiarity (LF) score for each item using the actual Gemma tokenizer. The LF score combines:
- **Subword fragmentation index**: number of subword tokens / number of whitespace words (higher = less familiar)
- **Token frequency percentile**: average frequency rank of span tokens (higher = more frequent)
- **Combined LF score**: `LF = −z(fragmentation) + z(frequency)` (higher = more familiar)

Also runs regression analysis (`peak_layer ~ lf_score`) and generates publication-quality figures and LaTeX tables.

### `scripts/lf_simplified.py` — Lexical Familiarity (Simplified)

A streamlined version of `lf_full.py` using the same tokenizer-based computation but with a simpler interface. Suitable for quick reproduction of the key regression results.

### `scripts/normal_approx_ci.py` — Confidence Intervals

Computes 95% confidence intervals for layer-wise cosine distances using the normal approximation formula (`CI = mean ± 1.96 × SE`). Generates CI tables and LaTeX output for the paper.

### `scripts/span_length.py` — Span Length Confound Analysis

Checks whether divergence span length (in words) is a confound for the observed peak cosine distances. Runs correlation and regression analyses to verify that the lexical familiarity gradient holds independently of span length.

## Requirements

- Python 3.8+
- PyTorch with CUDA support (recommended for GPU acceleration)
- Hugging Face Transformers library
- SAE-Lens library for working with Sparse Autoencoders
- NumPy, Pandas, SciPy, statsmodels for data processing and statistics
- Matplotlib & Seaborn for visualization
- A Hugging Face API token (required to access Gemma models)

### Setup Instructions

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure your Hugging Face token:**
   Create a `.env` file in the repository root:
   ```
   HF_TOKEN=your_huggingface_token_here
   ```
   A token can be obtained from https://huggingface.co/settings/tokens

3. **Run the main SAE analysis:**
   ```bash
   cd scripts
   python run.py
   ```

4. **Compute lexical familiarity scores:**
   ```bash
   # Full analysis (recommended for reproduction)
   python scripts/lf_full.py --data-dir ./data --output ./results/lexical_familiarity

   # Simplified analysis
   python scripts/lf_simplified.py --data-dir ./data --output ./results/lexical_familiarity
   ```

5. **Compute confidence intervals:**
   ```bash
   python scripts/normal_approx_ci.py
   ```

## Results

The `results/` directory contains pre-computed outputs:

- **Root-level CSVs** (`results/*.csv`): Layer-wise feature activations, cosine distances between figurative and literal representations, and figurative-to-literal feature ratios across all 48 model layers — one file per nonliteral category.
- **`results/ci/`**: Cosine distance curves with 95% confidence intervals, CI overlap analysis, and LaTeX-formatted tables (Table 2 and Table 3 from the paper).
- **`results/lexical_familiarity/`**: Per-item LF scores, regression figures, and the LaTeX table for the lexical familiarity analysis section.

### Key Findings (from `results/lexical_familiarity/README.md`)

| Category | LF Score | Peak Layer | N |
|---|---|---|---|
| Idiom | +1.28 | L1 | 823 |
| Construction† | +0.44 | L7 | 37 |
| Metaphor | +0.45 | L8 | 625 |
| Semantic Shift | +0.02 | L9 | 1002 |
| Neologism | −1.37 | L41 | 1000 |

Lexical familiarity score significantly predicts peak divergence layer (β = −5.88, p < .001, R² = 0.342). At the category level, mean LF score correlates near-perfectly with peak layer (r = −0.95).
