"""
Plot probe metrics across training steps, with separate curves per layer.
Averages results across all datasets for each (step, layer, method) combination.

Usage:
    python plot_results.py --model_name pythia-410m
    python plot_results.py --model_name pythia-6.9b
    python plot_results.py --model_name pythia-410m pythia-6.9b
"""

import os
import re
import argparse

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


RESULTS_DIR = 'results'
METRICS = ['test_f1', 'test_acc', 'test_auc', 'test_loss', 'val_auc', 'val_loss']
METRIC_LABELS = {
    'test_f1': 'Test F1',
    'test_acc': 'Test Accuracy',
    'test_auc': 'Test AUC',
    'test_loss': 'Test Loss',
    'val_auc': 'Validation AUC',
    'val_loss': 'Validation Loss',
}


def load_all_results(model_name):
    """Load all results for a model, returning a DataFrame with columns:
    [step, layer, method, test_f1, test_acc, test_auc, test_loss, val_auc, val_loss]
    where metric values are averaged across datasets.
    """
    rows = []
    pattern = re.compile(rf'baseline_probes_{re.escape(model_name)}_revstep(\d+)')

    for dirname in os.listdir(RESULTS_DIR):
        m = pattern.match(dirname)
        if not m:
            continue
        step = int(m.group(1))
        settings_dir = os.path.join(RESULTS_DIR, dirname, 'normal_settings')
        if not os.path.isdir(settings_dir):
            continue

        for fname in os.listdir(settings_dir):
            layer_match = re.match(r'layer(\d+)_results\.csv', fname)
            if not layer_match:
                continue
            layer = int(layer_match.group(1))
            df = pd.read_csv(os.path.join(settings_dir, fname))

            for method, group in df.groupby('method'):
                avg = group[METRICS].mean()
                rows.append({
                    'step': step, 'layer': layer, 'method': method,
                    **avg.to_dict()
                })

    return pd.DataFrame(rows).sort_values(['method', 'layer', 'step'])


def plot_model(model_name, df):
    """Generate a figure with subplots for each metric, curves per layer, averaged over methods."""
    methods = sorted(df['method'].unique())
    layers = sorted(df['layer'].unique())

    # Average across methods (logreg & mlp) for each (step, layer)
    avg_df = df.groupby(['step', 'layer'])[METRICS].mean().reset_index()
    avg_df = avg_df.sort_values(['layer', 'step'])

    n_metrics = len(METRICS)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows))
    axes = axes.flatten()

    cmap = plt.cm.viridis
    colors = [cmap(i / max(len(layers) - 1, 1)) for i in range(len(layers))]

    for idx, metric in enumerate(METRICS):
        ax = axes[idx]
        for i, layer in enumerate(layers):
            layer_data = avg_df[avg_df['layer'] == layer].sort_values('step')
            ax.plot(layer_data['step'], layer_data[metric],
                    marker='o', markersize=3, linewidth=1.5,
                    color=colors[i], label=f'Layer {layer}')

        ax.set_xlabel('Training Step')
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_title(METRIC_LABELS[metric])
        ax.set_xscale('symlog', linthresh=1)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f'{int(x):,}' if x >= 1 else f'{x:g}'
        ))
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',
               ncol=min(len(layers), 5), bbox_to_anchor=(0.5, -0.02),
               fontsize=9)

    fig.suptitle(f'{model_name} — Probe Metrics vs Training Step (avg over datasets & methods)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])

    out_dir = f'figs/{model_name}'
    os.makedirs(out_dir, exist_ok=True)
    out_path = f'{out_dir}/metrics_vs_step.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, nargs='+', required=True)
    args = parser.parse_args()

    for model_name in args.model_name:
        print(f"Loading results for {model_name}...")
        df = load_all_results(model_name)
        if df.empty:
            print(f"  No results found for {model_name}, skipping.")
            continue
        methods = df['method'].unique()
        print(f"  {len(df)} entries: methods={list(methods)}, "
              f"layers={sorted(df['layer'].unique())}, "
              f"steps={sorted(df['step'].unique())}")
        plot_model(model_name, df)


if __name__ == "__main__":
    main()
