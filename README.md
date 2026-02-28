# Lexical Familiarity Predicts Processing Depth for Nonliteral Language in Large Language Models

> **Anonymity Notice:** This repository is fully anonymized for double-blind peer review. All author identities, affiliations, and specific institutional acknowledgments have been removed.

## Overview

This repository contains the datasets, code, and analysis results for the experiments from our paper, *"Lexical Familiarity Predicts Processing Depth for Nonliteral Language in Large Language Models."* Our research investigates how Large Language Models (LLMs) process language that deviates from literal, standard usage. By analyzing layer-wise representations using Gemma-Scope-2 Sparse Autoencoders (SAEs), we establish a "lexical familiarity gradient" across five categories of nonliteral language.

## Repository Structure

```
.
├── data/
│   ├── idioms.csv                      # Idioms dataset with position tags
│   ├── metaphors.csv                   # Metaphor dataset with position tags
│   ├── slang_neologisms.csv            # Neologisms with position tags
│   ├── slang_semantic_shift.csv        # Semantic shift pairs with position tags
│   ├── slang_constructional.csv        # Constructional slang with position tags
│   ├── literal_paraphrase.csv          # Baseline literal language data
│   └── identical_sentence_pair.csv     # Control pairs
├── script/
│   └── run.py                          # Main SAE analysis script for SAE feature extraction and analysis
├── results/
│   ├── idioms.csv                      # Analysis results for idioms
│   ├── metaphors.csv                   # Analysis results for metaphors
│   ├── slang_neologisms.csv            # Analysis results for neologisms
│   ├── slang_semantic_shift.csv        # Analysis results for semantic shift
│   ├── slang_constructional.csv        # Analysis results for construction
│   ├── literal_paraphrase.csv          # Analysis results for literal language
│   └── identical_sentence_pair.csv     # Analysis results for control pairs
├── requirements.txt                    # Python package dependencies
└── README.md
```

## Data Format

Each dataset CSV file contains:
- **slang/figurative text**: The nonliteral or figurative expression
- **literal text**: The literal or paraphrased equivalent
- **Position tags**: Start and end positions of the relevant words/phrases in both versions
- **Segment information**: The specific slang words/phrases and their literal counterparts

## Main Script

### `script/run.py`

This is the main analysis script that:

1. **Loads datasets** from the `data/` directory
2. **Extracts SAE features** from Gemma-Scope-2 Sparse Autoencoders for each sentence pair
3. **Computes layer-wise representations** across all 48 layers of the model
4. **Analyzes feature distributions** to identify the lexical familiarity gradient
5. **Generates results** and saves them to the `results/` directory

## Requirements

The script requires:
- PyTorch with CUDA support (recommended for GPU)
- Hugging Face Transformers library
- SAE-Lens library for working with Sparse Autoencoders
- NumPy, Pandas, SciPy for data processing
- Matplotlib & Seaborn for visualization
- A Hugging Face API token (required to access Gemma models)

### Setup Instructions

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure your Hugging Face token:**
   Create a `.env` file in the repository root with:
   ```
   HF_TOKEN=your_huggingface_token_here
   ```
   You can obtain a token from https://huggingface.co/settings/tokens

3. **Run the analysis:**
   ```bash
   python script/run.py
   ```

## Results

The analysis generates output files in the `results/` directory containing:
- Layer-wise feature activations
- Cosine distances between figurative and literal representations
- Feature importance metrics
- Figurative-to-literal feature ratios across all 48 layers
