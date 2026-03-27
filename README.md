# PreTrain-Probes

A pipeline for studying how transformer representations evolve during pretraining. It extracts hidden-state activations from model checkpoints at different training steps, then trains binary classification probes (logistic regression and MLPs) to measure layer-wise representational capacity.

## Overview

The pipeline answers questions like: *When does a model learn to represent syntactic structure? How do intermediate layers differ from final layers across training?*

It works in two stages:

1. **Activation extraction** -- Load a model checkpoint, run probing datasets through it, and save per-layer residual stream activations.
2. **Probe training** -- Train logistic regression and MLP probes on the saved activations to measure what information each layer encodes at each training step.

## Installation

Requires Python >= 3.11.

```bash
pip install -e .
```

## Quick Start

### Run the full pipeline

```bash
bash pipeline.sh
```

This iterates over all configured model revisions (training checkpoints), extracts activations, then trains probes in a batched pass.

### Run stages independently

**Stage 1: Extract activations**

```bash
python generate_activations.py \
  --model_name pythia-70m \
  --revision step143000 \
  --devices cuda:0,cuda:1 \
  --max_seq_len 1024
```

**Stage 2: Train probes**

```bash
# Train both logistic regression and MLP probes
python run_probes.py --model_name pythia-70m --revision step143000

# Train only one type
python run_probes.py --model_name pythia-70m --revision step143000 --logreg_only
python run_probes.py --model_name pythia-70m --revision step143000 --mlp_only

# Batch multiple revisions (trains all revision x layer combinations together)
python run_probes.py --model_name pythia-70m --revision step0 step1 step128 step143000
```

## Supported Models

| Family | Models |
|--------|--------|
| Pythia (EleutherAI) | 70m, 160m, 410m, 1b, 1.4b, 2.8b, 6.9b |
| Gemma 2 (Google) | 2b, 9b |
| LLaMA 3.1 (Meta) | 8b |
| OLMo 1 (AI2) | 1b, 7b |

Model configurations (HuggingFace paths, hook layers, embedding inclusion) are defined in `utils_data.py:MODEL_CONFIG`.

## Data Layout

```
data/
  probing_datasets_MASTER.csv          # Registry of all probing datasets
  cleaned_data/*.csv                   # Binary classification datasets (prompt, target)
  model_activations_{model}_{rev}/     # Saved activation tensors (.pt)
  baseline_results_{model}_{rev}/      # Per-layer probe results

results/
  baseline_probes_{model}/
    normal_settings/                   # Coalesced best results per layer
```

Probing datasets are CSVs with `prompt` and `target` columns, where `target` is binary (0/1).

## Architecture

### Activation extraction (`generate_activations.py`)

- Uses [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) (`HookedTransformer`) to extract last-token residual stream activations at configured hook layers.
- Distributes datasets across GPUs using `multiprocessing.Pool`.
- Skips datasets whose activations already exist on disk (incremental/resumable).

### Probe training (`run_probes.py`)

- **Logistic regression**: Searches over 10 regularization strengths (C from 1e-5 to 1e5).
- **MLP probes**: Searches over 10 configurations of hidden sizes (32, 64, 32x2, 64x2), learning rates, and weight decay.
- **`ParallelMLP`**: Trains all layers' probes simultaneously in a single GPU batch using fused `einsum` operations -- no Python loop over layers.
- **Cross-validation strategy** adapts to dataset size:
  - LeavePOut (p=2) for <= 12 samples
  - StratifiedKFold (6 splits) for 12--128 samples
  - 80/20 train/test split for > 128 samples

### Utilities

- **`utils_data.py`** -- Model configs, dataset loading, balanced sampling, standardization.
- **`utils_probes.py`** -- `ParallelMLP`/`MultiLinear` for batched probe training, hyperparameter search, metrics (F1, accuracy, AUC).
- **`create_ovr_datasets.py`** -- Generates one-vs-rest binary datasets from multiclass sources.

## Key Design Decisions

- **Incremental computation**: Both stages check for existing outputs before running, making the pipeline safe to re-run and resume after interruption.
- **Batched training**: Revision batching reduces GPU invocations by up to 16x compared to training each (revision, layer) pair independently.
- **Model naming**: `run_probes.py` constructs `{model}_rev{revision}` names (e.g., `pythia-70m_revstep143000`); `utils_data.py` strips the revision suffix for config lookup.

## Dependencies

Core: `torch`, `transformer_lens`, `transformers`, `accelerate`, `scikit-learn`, `pandas`, `numpy`, `safetensors`, `tqdm`.

See `pyproject.toml` for the full list.
