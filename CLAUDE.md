# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Interpretability probe training pipeline that studies how transformer model representations evolve during pretraining. It extracts hidden-state activations from model checkpoints at different training steps, then trains binary classification probes (logistic regression and MLPs) to measure layer-wise representational capacity.

## Commands

### Full pipeline (both stages for all revisions)
```bash
bash pipeline.sh
```

### Stage 1: Generate activations
```bash
python generate_activations.py --model_name pythia-70m --devices cuda:0,cuda:1 --max_seq_len 1024 --revision step143000
```

### Stage 2: Train probes
```bash
python run_probes.py --model_name pythia-70m --revision step143000
python run_probes.py --model_name pythia-70m --revision step143000 --logreg_only
python run_probes.py --model_name pythia-70m --revision step143000 --mlp_only
```

### Install
```bash
pip install -e .
```

## Architecture

**Two-stage pipeline** orchestrated by `pipeline.sh`:

1. **`generate_activations.py`** — Loads models via TransformerLens (`HookedTransformer`), extracts last-token residual stream activations at configured hook layers. Distributes datasets across GPUs using `multiprocessing.Pool`. Outputs `.pt` files to `data/model_activations_{model}_{revision}/`.

2. **`run_probes.py`** — Trains logistic regression and MLP probes on the saved activations. All layers for a dataset are trained simultaneously on GPU using `ParallelMLP` (batched independent models via fused einsum). Hyperparameter selection uses cross-validation (LeavePOut for tiny datasets, StratifiedKFold for small, simple split for large). Results saved per-layer then coalesced.

**Utility modules:**
- **`utils_data.py`** — `MODEL_CONFIG` dict defines supported models (HF path, hook layers, embed inclusion). Dataset loading from `data/cleaned_data/*.csv` (requires `prompt` and `target` columns). Model names use `{base}_rev{revision}` convention (e.g., `pythia-70m_revstep143000`).
- **`utils_probes.py`** — `ParallelMLP`/`MultiLinear` train M independent models in one forward pass. `find_best_logreg` searches over C values (regularization); `find_best_mlp` searches over 10 `(hidden_sizes, lr, weight_decay)` configs grouped by shared `(hidden_sizes, lr)` for efficient batching.

## Key Design Patterns

- **Parallel model training**: `ParallelMLP` uses a `(n_models, batch, features)` tensor layout with `einsum` to train all layers' probes in a single batch, avoiding Python loops over layers.
- **Incremental computation**: Both stages skip work if output files already exist with correct shapes, enabling safe re-runs.
- **Model naming**: `run_probes.py` constructs `{model_name}_rev{revision}` as the full model name, which `utils_data.py` strips back to the base name for config lookup via `get_base_model_name()`.

## Data Layout

- `data/probing_datasets_MASTER.csv` — Registry of all probing datasets
- `data/cleaned_data/*.csv` — Binary classification datasets (columns: `prompt`, `target`)
- `data/model_activations_{model}_rev{revision}/` — Saved activation tensors
- `data/baseline_results_{model}_rev{revision}/normal/allruns/` — Per-layer probe results
- `results/baseline_probes_{model}/normal_settings/` — Coalesced results per layer
