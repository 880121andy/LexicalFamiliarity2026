# Code and Data for: Lexical Familiarity Predicts Processing Depth for Nonliteral Language in Large Language Models

> **Anonymity Notice:** > This repository is fully anonymized for double-blind peer review. All author identities, affiliations, and specific institutional acknowledgments have been removed. 

## Overview

This repository contains the datasets and code required to reproduce the experiments from our paper, *"Lexical Familiarity Predicts Processing Depth for Nonliteral Language in Large Language Models."* Our research investigates how Large Language Models (LLMs) process language that deviates from literal, standard usage. By analyzing layer-wise representations using **Gemma-3-12B-IT** and **Gemma-Scope-2** Sparse Autoencoders (SAEs), we establish a "lexical familiarity gradient" across five categories of nonliteral language.

## Repository Structure

Below is an overview of the repository's organization:

```text
.
├── data/
│   ├── idioms/                 # Data sampled from PIE corpus
│   ├── metaphors/              # Data sampled from FLUTE dataset
│   ├── slang_neologisms/       # Neologisms derived from Urban Dictionary/LM-Lexicon
│   ├── slang_semantic_shift/   # Minimal pairs with identical lexical forms
│   ├── slang_constructional/   # Constructional slang datasets
│   └── literal_paraphrase/     # Baseline data from PAWS
├── scripts/
│   ├── extract_sae_features.py # Script to extract features using Gemma-Scope-2
│   ├── calculate_distance.py   # Computes layer-wise cosine distances
│   └── residual_analysis.py    # Calculates figurative-to-literal feature ratios
├── notebooks/
│   └── visualizations.ipynb    # Generates 48-layer profile figures (e.g., Fig 1 & Fig 2)
├── requirements.txt
└── README.md
