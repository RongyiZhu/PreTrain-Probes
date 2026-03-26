"""
Train logreg + MLP probes for all layers of each dataset, then coalesce results.

Supports batching multiple revisions together: all (revision, layer) pairs for a
dataset are trained in a single GPU call, giving up to 16x fewer training invocations.

Usage:
    # Single revision (backward-compatible)
    python run_probes.py --model_name pythia-70m --revision step143000
    python run_probes.py --model_name pythia-70m --revision step143000 --logreg_only
    python run_probes.py --model_name pythia-70m --revision step143000 --mlp_only

    # Batched revisions (all trained together)
    python run_probes.py --model_name pythia-70m --revision step0 step1 step16 step128 step143000
"""

import os
import argparse

import pandas as pd
from tqdm import tqdm

from utils_data import get_xy_traintest, get_numbered_binary_tags, get_dataset_sizes, get_layers, get_datasets
from utils_probes import find_best_logreg, find_best_mlp

dataset_sizes = get_dataset_sizes()
all_dataset_tags = get_numbered_binary_tags()


# ─────────────────────── Raw results I/O ──────────────────────────────────

def _raw_results_path(full_name, layer, method_name):
    """Path to raw results CSV: all datasets × all configs for one (revision, layer, method)."""
    return f'data/baseline_results_{full_name}/normal/layer{layer}_{method_name}.csv'


def _load_completed_datasets(full_name, layer, method_name):
    """Return set of dataset tags already present in the raw results file."""
    path = _raw_results_path(full_name, layer, method_name)
    if not os.path.exists(path):
        return set()
    df = pd.read_csv(path)
    return set(df['dataset'].unique())


def _append_raw_results(full_name, layer, method_name, rows_df):
    """Append rows to the raw results CSV, creating it if needed."""
    path = _raw_results_path(full_name, layer, method_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        existing = pd.read_csv(path)
        combined = pd.concat([existing, rows_df], ignore_index=True)
    else:
        combined = rows_df
    combined.to_csv(path, index=False)


# ─────────────────────── Per-dataset runners ──────────────────────────────

def _run_probe_for_dataset(numbered_dataset, model_name, method_name, find_best_fn):
    """
    Train probes for ALL layers of a single dataset/revision simultaneously on GPU.
    Skips layers whose results already exist. Returns number of layers computed.
    """
    return _run_probe_for_dataset_batched(
        numbered_dataset, [model_name], method_name, find_best_fn
    )


def _run_probe_for_dataset_batched(numbered_dataset, revision_fullnames, method_name, find_best_fn):
    """
    Train probes for ALL (revision, layer) pairs of a single dataset simultaneously.
    Stacks all revisions × layers along the M dimension for a single GPU call.
    Skips pairs whose results already exist. Returns number of pairs computed.
    """
    # Collect (full_name, layer) pairs needing computation
    pairs_to_run = []
    for full_name in revision_fullnames:
        for layer in get_layers(full_name):
            done = _load_completed_datasets(full_name, layer, method_name)
            if numbered_dataset not in done:
                pairs_to_run.append((full_name, layer))

    if not pairs_to_run:
        return 0

    size = dataset_sizes[numbered_dataset]
    num_train = min(size - 100, 1024)

    X_trains, y_trains, X_tests, y_tests = [], [], [], []
    for full_name, layer in pairs_to_run:
        X_train, y_train, X_test, y_test = get_xy_traintest(
            num_train, numbered_dataset, layer, model_name=full_name
        )
        if hasattr(X_train, 'numpy'):
            X_train = X_train.numpy()
        if hasattr(X_test, 'numpy'):
            X_test = X_test.numpy()
        X_trains.append(X_train)
        y_trains.append(y_train)
        X_tests.append(X_test)
        y_tests.append(y_test)

    _, all_configs = find_best_fn(X_trains, y_trains, X_tests, y_tests)

    for i, (full_name, layer) in enumerate(pairs_to_run):
        # Build rows: all configs for this (dataset, layer) pair
        rows = []
        for cfg in all_configs[i]:
            row = {'dataset': numbered_dataset, 'method': method_name}
            row.update(cfg)
            rows.append(row)
        _append_raw_results(full_name, layer, method_name, pd.DataFrame(rows))

    return len(pairs_to_run)


def run_logreg_for_dataset(numbered_dataset, model_name):
    return _run_probe_for_dataset(numbered_dataset, model_name, 'logreg', find_best_logreg)


def run_mlp_for_dataset(numbered_dataset, model_name):
    return _run_probe_for_dataset(numbered_dataset, model_name, 'mlp', find_best_mlp)


# ─────────────────────── Coalesce ─────────────────────────────────────────

_METRIC_COLS = ['dataset', 'method', 'val_auc', 'val_loss',
                'test_f1', 'test_acc', 'test_loss', 'test_auc']


def coalesce_results(model_name):
    """Extract best config per dataset from raw results, keep only metrics, merge per layer."""
    methods = ['logreg', 'mlp']
    for layer in get_layers(model_name):
        best_rows = []
        for method in methods:
            path = _raw_results_path(model_name, layer, method)
            if not os.path.exists(path):
                print(f'Missing: {path}')
                continue
            df = pd.read_csv(path)
            best = df[df['is_best'] == 1].copy()
            # Keep only metric columns (drop config details)
            keep = [c for c in _METRIC_COLS if c in best.columns]
            best_rows.append(best[keep])

        if best_rows:
            out_dir = f'results/baseline_probes_{model_name}/normal_settings'
            os.makedirs(out_dir, exist_ok=True)
            pd.concat(best_rows, ignore_index=True).to_csv(
                f'{out_dir}/layer{layer}_results.csv', index=False
            )


# ─────────────────────── Main ────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--revision", type=str, nargs='+', required=True)
    parser.add_argument("--logreg_only", action="store_true")
    parser.add_argument("--mlp_only", action="store_true")
    args = parser.parse_args()

    revision_fullnames = [f'{args.model_name}_rev{r}' for r in args.revision]
    n_revs = len(revision_fullnames)

    # Gather datasets available per revision, then take union
    datasets_by_rev = {}
    for fn in revision_fullnames:
        datasets_by_rev[fn] = set(get_datasets(fn))
    all_datasets = sorted(set.union(*datasets_by_rev.values()))

    layers = get_layers(revision_fullnames[0])

    do_logreg = not args.mlp_only
    do_mlp = not args.logreg_only

    if do_logreg:
        print(f"[Logreg] {len(all_datasets)} datasets, {len(layers)} layers, {n_revs} revisions (M_max={len(layers)*n_revs})")
        for ds in tqdm(all_datasets, desc="Logreg batched"):
            revs = [fn for fn in revision_fullnames if ds in datasets_by_rev[fn]]
            _run_probe_for_dataset_batched(ds, revs, 'logreg', find_best_logreg)

    if do_mlp:
        print(f"[MLP] {len(all_datasets)} datasets, {len(layers)} layers, {n_revs} revisions (M_max={len(layers)*n_revs})")
        for ds in tqdm(all_datasets, desc="MLP batched"):
            revs = [fn for fn in revision_fullnames if ds in datasets_by_rev[fn]]
            _run_probe_for_dataset_batched(ds, revs, 'mlp', find_best_mlp)

    for fn in revision_fullnames:
        coalesce_results(fn)
    print("Done.")


if __name__ == "__main__":
    main()
