import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
import os

BASEPATH = '.'

# ─────────────────────── Model configuration ─────────────────────────────

MODEL_CONFIG = {
    "gemma-2-9b": {
        "hf_model_path": "google/gemma-2-9b",
        "hook_layers": [9, 20, 31, 41],
        "include_embed": True
    },
    "llama-3.1-8b": {
        "hf_model_path": "meta-llama/Llama-3.1-8B",
        "hook_layers": [8, 16, 24, 31],
        "include_embed": True
    },
    "gemma-2-2b": {
        "hf_model_path": "google/gemma-2-2b",
        "hook_layers": [12],
        "include_embed": False
    },
    "pythia-70m": {
        "hf_model_path": "EleutherAI/pythia-70m",
        "hook_layers": [1, 2, 3, 4, 5],
        "include_embed": False
    },
    "pythia-160m": {
        "hf_model_path": "EleutherAI/pythia-160m",
        "hook_layers": [1, 4, 5, 7, 8, 10, 11],
        "include_embed": False
    },
    "pythia-410m": {
        "hf_model_path": "EleutherAI/pythia-410m",
        "hook_layers": [2, 10, 14, 17, 19, 22, 23],
        "include_embed": False
    },
    "pythia-1b": {
        "hf_model_path": "EleutherAI/pythia-1b",
        "hook_layers": [2, 3, 5, 6, 8, 10, 11, 13, 14, 15],
        "include_embed": False
    },
    "pythia-1.4b": {
        "hf_model_path": "EleutherAI/pythia-1.4b",
        "hook_layers": [2, 5, 7, 10, 12, 14, 17, 19, 22, 23],
        "include_embed": False
    },
    "pythia-2.8b": {
        "hf_model_path": "EleutherAI/pythia-2.8b",
        "hook_layers": [3, 10, 13, 19, 22, 29, 31],
        "include_embed": False
    },
    "pythia-6.9b": {
        "hf_model_path": "EleutherAI/pythia-6.9b",
        "hook_layers": [3, 6, 10, 13, 16, 19, 22, 26, 29, 31],
        "include_embed": False
    },
    "olmo-1-1b": {
        "hf_model_path": "allenai/OLMo-1B-0724-hf",
        "hook_layers": [4, 8, 12],
        "include_embed": False
    },
    "olmo-1-7b": {
        "hf_model_path": "allenai/OLMo-7B-0724-hf",
        "hook_layers": [8, 16, 24],
        "include_embed": False
    }
}


def get_base_model_name(model_name):
    """Extract base model name, removing revision suffix if present.
    E.g. 'pythia-70m_revstep143000' -> 'pythia-70m'
    """
    if '_rev' in model_name:
        return model_name.split('_rev')[0]
    return model_name


def get_model_config(model_name):
    """Get model configuration or raise error if model not supported."""
    base = get_base_model_name(model_name)
    if base not in MODEL_CONFIG:
        raise ValueError(f"Model {base} not supported. Available: {list(MODEL_CONFIG.keys())}")
    return MODEL_CONFIG[base]


def get_layers(model_name='gemma-2-9b'):
    """Get layer indices for a given model."""
    config = get_model_config(model_name)
    layers = config["hook_layers"].copy()
    if config["include_embed"]:
        layers = ['embed'] + layers
    return layers


# ─────────────────────── Dataset utilities ────────────────────────────────

def get_binary_df():
    df = pd.read_csv('data/probing_datasets_MASTER.csv')
    return df[df['Data type'] == 'Binary Classification']


def get_numbered_binary_tags():
    df = get_binary_df()
    return [name.split('/')[-1].split('.')[0] for name in df['Dataset save name']]


def read_numbered_dataset_df(numbered_dataset_tag):
    dataset_tag = '_'.join(numbered_dataset_tag.split('_')[1:])
    df = get_binary_df()
    dataset_save_name = df[df['Dataset Tag'] == dataset_tag]['Dataset save name'].iloc[0]
    return pd.read_csv(f'{BASEPATH}/data/{dataset_save_name}')


def get_dataset_sizes():
    """Returns dict of {dataset_tag: num_samples} for all binary datasets."""
    tags = get_numbered_binary_tags()
    sizes = {}
    for tag in tags:
        df = read_numbered_dataset_df(tag)
        sizes[tag] = len(df)
    return sizes


def get_datasets(model_name='llama-3.1-8b'):
    """List available datasets that have activations for the given model."""
    dataset_sizes = get_dataset_sizes()
    files = os.listdir(f'{BASEPATH}/data/model_activations_{model_name}')
    block_files = [f for f in files if 'blocks' in f]
    datasets = set()
    for f in block_files:
        dataset = f.split('_blocks')[0]
        if dataset in dataset_sizes:
            datasets.add(dataset)
    return sorted(list(datasets))


# ─────────────────────── Data loading ─────────────────────────────────────

def get_yvals(numbered_dataset_tag):
    df = read_numbered_dataset_df(numbered_dataset_tag)
    le = LabelEncoder()
    return le.fit_transform(df['target'].values)


def get_xvals(numbered_dataset_tag, layer, model_name='gemma-2-9b'):
    if layer == 'embed':
        fname = f'{BASEPATH}/data/model_activations_{model_name}/{numbered_dataset_tag}_hook_embed.pt'
    else:
        fname = f'{BASEPATH}/data/model_activations_{model_name}/{numbered_dataset_tag}_blocks.{layer}.hook_resid_post.pt'
    return torch.load(fname, weights_only=False)


def get_xyvals(numbered_dataset_tag, layer, model_name, MAX_AMT=1500):
    xvals = get_xvals(numbered_dataset_tag, layer, model_name)
    yvals = get_yvals(numbered_dataset_tag)
    return xvals[:MAX_AMT], yvals[:MAX_AMT]


def get_train_test_indices(y, num_train, num_test, pos_ratio=0.5, seed=42):
    np.random.seed(seed)
    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == 0)[0]

    pos_train_size = int(np.ceil(pos_ratio * num_train))
    neg_train_size = num_train - pos_train_size
    pos_test_size = int(np.ceil(pos_ratio * num_test))
    neg_test_size = num_test - pos_test_size

    train_pos = np.random.choice(pos_indices, size=pos_train_size, replace=False)
    train_neg = np.random.choice(neg_indices, size=neg_train_size, replace=False)
    remaining_pos = np.setdiff1d(pos_indices, train_pos)
    remaining_neg = np.setdiff1d(neg_indices, train_neg)
    test_pos = np.random.choice(remaining_pos, size=pos_test_size, replace=False)
    test_neg = np.random.choice(remaining_neg, size=neg_test_size, replace=False)

    train_indices = np.random.permutation(np.concatenate([train_pos, train_neg]))
    test_indices = np.random.permutation(np.concatenate([test_pos, test_neg]))
    return train_indices, test_indices


def get_xy_traintest(num_train, numbered_dataset_tag, layer, model_name, MAX_AMT=5000, seed=42):
    X, y = get_xyvals(numbered_dataset_tag, layer, model_name, MAX_AMT=MAX_AMT)
    num_test = X.shape[0] - num_train - 1
    if num_train + min(100, num_test) > X.shape[0]:
        raise ValueError(
            f"Requested {num_train + 100} samples but only {X.shape[0]} available in {numbered_dataset_tag}"
        )
    train_idx, test_idx = get_train_test_indices(y, num_train, num_test, pos_ratio=0.5, seed=seed)
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]
