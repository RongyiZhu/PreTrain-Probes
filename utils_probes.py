"""
GPU-accelerated probe training for binary classification.

Trains M independent probe models (one per layer) simultaneously using
fused einsum operations. Supports two probe types:
  - logreg: logistic regression (single linear layer), searches over C values
  - mlp: MLP with hidden layers, searches over (architecture, lr, weight_decay) configs

Both probes return (best_results, all_configs_results) for per-config analysis.
"""

import math
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeavePOut, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, log_loss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_device(device):
    global DEVICE
    DEVICE = device


# ─────────────────────── Model architecture ───────────────────────────────

class MultiLinear(nn.Module):
    """Linear layer that operates on M independent models in parallel."""
    def __init__(self, n_models: int, d_in: int, d_out: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_models, d_out, d_in))
        self.bias = nn.Parameter(torch.zeros(n_models, d_out))
        nn.init.normal_(self.weight, 0.0, 1 / math.sqrt(d_in))

    def forward(self, x: torch.Tensor):
        return torch.einsum("moi,mbi->mbo", self.weight, x) + self.bias[:, None, :]


class ParallelMLP(nn.Module):
    """M independent MLPs trained in parallel via batched linear layers.
    With hidden_sizes=(), this is logistic regression (single linear layer).
    """
    def __init__(self, n_models: int, d_in: int, hidden_sizes: tuple, d_out: int = 2):
        super().__init__()
        sizes = [d_in] + list(hidden_sizes) + [d_out]
        layers = []
        for i, (din, dout) in enumerate(zip(sizes, sizes[1:])):
            layers.append(MultiLinear(n_models, din, dout))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.net(x)


# ─────────────────────── Cross-validation ─────────────────────────────────

def get_cv(X_train):
    n_samples = X_train.shape[0]
    if n_samples <= 12:
        return LeavePOut(2)
    elif n_samples < 128:
        return StratifiedKFold(n_splits=6, shuffle=True, random_state=42)
    else:
        val_size = min(int(0.2 * n_samples), 100)
        train_size = n_samples - val_size
        return [(list(range(train_size)), list(range(train_size, n_samples)))]


def get_splits(cv, X_train, y_train):
    if hasattr(cv, 'split'):
        splits = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            if len(np.unique(y_train[val_idx])) == 2:
                splits.append((train_idx, val_idx))
        return splits
    return cv


# ─────────────────────── Data utilities ───────────────────────────────────

def _standardize_per_model(X_list):
    """Standardize each layer's data independently."""
    scalers = []
    X_scaled = []
    for X in X_list:
        scaler = StandardScaler()
        X_scaled.append(scaler.fit_transform(X))
        scalers.append(scaler)
    return X_scaled, scalers


# ─────────────────────── Training ─────────────────────────────────────────

def _train_model(model, X, y, weight_decays, lr, max_epochs, patience, batch_size):
    """
    Train a ParallelMLP with per-model L2 regularization via manual penalty.

    Args:
        model: ParallelMLP instance.
        X: (N_models, N_samples, d_in) tensor.
        y: (N_models, N_samples) long tensor.
        weight_decays: (N_models,) tensor of per-model L2 penalty coefficients.
    """
    N_models, N, _ = X.shape
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(max_epochs):
        perm = torch.randperm(N, device=X.device)
        X_shuffled = X[:, perm]
        y_shuffled = y[:, perm]

        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            X_batch = X_shuffled[:, start:end]
            y_batch = y_shuffled[:, start:end]

            logits = model(X_batch)
            ce_loss = F.cross_entropy(logits.reshape(-1, 2), y_batch.reshape(-1))

            l2_per_model = torch.zeros(N_models, device=X.device)
            for param in model.parameters():
                l2_per_model = l2_per_model + (param ** 2).reshape(N_models, -1).sum(dim=1)
            l2_penalty = (weight_decays * l2_per_model).sum() / (2.0 * N_models)

            loss = ce_loss + l2_penalty
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        if avg_loss < best_loss - 1e-5:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break


def _compute_metrics(y_true, probs, preds):
    """Compute standard metrics dict from predictions."""
    metrics = {
        'test_f1': f1_score(y_true, preds, average='weighted'),
        'test_acc': accuracy_score(y_true, preds),
        'test_loss': log_loss(y_true, probs),
    }
    try:
        metrics['test_auc'] = roc_auc_score(y_true, probs[:, 1])
    except ValueError:
        metrics['test_auc'] = 0.5
    return metrics


# ═══════════════════════════════════════════════════════════════════════════
#  LOGISTIC REGRESSION
# ═══════════════════════════════════════════════════════════════════════════

def find_best_logreg(X_trains, y_trains, X_tests, y_tests,
                     Cs=None, lr=1e-2, max_epochs=500,
                     patience=30, batch_size=256):
    """
    Train M independent logistic regression probes simultaneously on GPU,
    searching over C values with per-model selection.

    All n_Cs * M models are trained in a single batch (no loop over C).

    Returns:
        (best_results, all_configs_results)
    """
    if Cs is None:
        Cs = np.logspace(5, -5, 10)

    n_Cs = len(Cs)
    M = len(X_trains)
    N_expanded = n_Cs * M
    d_in = X_trains[0].shape[1]

    X_trains_scaled, scalers = _standardize_per_model(X_trains)
    X_tests_scaled = [scalers[i].transform(X_tests[i]) for i in range(M)]

    cv = get_cv(X_trains[0])
    splits = get_splits(cv, X_trains_scaled[0], y_trains[0])

    if len(splits) == 0 or X_trains[0].shape[0] <= 3:
        return _logreg_no_cv(X_trains_scaled, y_trains, X_tests_scaled, y_tests,
                             M, d_in, lr, max_epochs, patience, batch_size)

    weight_decays = torch.tensor(
        [1.0 / C for C in Cs for _ in range(M)],
        dtype=torch.float32, device=DEVICE
    )

    # ── Cross-validation ──
    fold_val_aucs = [[] for _ in range(N_expanded)]
    fold_val_losses = [[] for _ in range(N_expanded)]

    for train_idx, val_idx in splits:
        X_fold_base = np.stack([X_trains_scaled[m][train_idx] for m in range(M)])
        X_val_base = np.stack([X_trains_scaled[m][val_idx] for m in range(M)])
        y_fold_train = y_trains[0][train_idx]
        y_fold_val = y_trains[0][val_idx]

        X_ft = torch.tensor(np.tile(X_fold_base, (n_Cs, 1, 1)), dtype=torch.float32, device=DEVICE)
        y_ft = torch.tensor(np.tile(y_fold_train, (N_expanded, 1)), dtype=torch.long, device=DEVICE)
        X_fv = torch.tensor(np.tile(X_val_base, (n_Cs, 1, 1)), dtype=torch.float32, device=DEVICE)

        model = ParallelMLP(N_expanded, d_in, hidden_sizes=(), d_out=2).to(DEVICE)
        _train_model(model, X_ft, y_ft, weight_decays, lr, max_epochs, patience, batch_size)

        with torch.no_grad():
            probs = F.softmax(model(X_fv), dim=-1).cpu().numpy()

        for idx in range(N_expanded):
            try:
                fold_val_aucs[idx].append(roc_auc_score(y_fold_val, probs[idx, :, 1]))
                fold_val_losses[idx].append(log_loss(y_fold_val, probs[idx]))
            except ValueError:
                fold_val_aucs[idx].append(0.5)
                fold_val_losses[idx].append(1.0)

    mean_val_aucs = np.array([np.mean(a) for a in fold_val_aucs]).reshape(n_Cs, M)
    mean_val_losses = np.array([np.mean(l) for l in fold_val_losses]).reshape(n_Cs, M)
    best_c_idx = np.argmax(mean_val_aucs, axis=0)

    # ── Final retrain: all n_Cs * M models ──
    X_train_t = torch.tensor(np.stack(X_trains_scaled), dtype=torch.float32, device=DEVICE)
    X_test_t = torch.tensor(np.stack(X_tests_scaled), dtype=torch.float32, device=DEVICE)

    X_train_all = X_train_t.repeat(n_Cs, 1, 1)
    y_train_all = torch.tensor(np.tile(y_trains[0], (N_expanded, 1)), dtype=torch.long, device=DEVICE)
    X_test_all = X_test_t.repeat(n_Cs, 1, 1)

    final_model = ParallelMLP(N_expanded, d_in, hidden_sizes=(), d_out=2).to(DEVICE)
    _train_model(final_model, X_train_all, y_train_all, weight_decays, lr, max_epochs, patience, batch_size)

    with torch.no_grad():
        test_logits = final_model(X_test_all)
        test_probs = F.softmax(test_logits, dim=-1).cpu().numpy().reshape(n_Cs, M, -1, 2)
        test_preds = test_logits.argmax(dim=-1).cpu().numpy().reshape(n_Cs, M, -1)

    # ── Assemble results ──
    best_results = []
    all_configs_results = []
    for m in range(M):
        ci = best_c_idx[m]
        metrics = {
            'val_auc': float(mean_val_aucs[ci, m]),
            'val_loss': float(mean_val_losses[ci, m]),
            'best_C': float(Cs[ci]),
        }
        metrics.update(_compute_metrics(y_tests[m], test_probs[ci, m], test_preds[ci, m]))
        best_results.append(metrics)

        per_model = []
        for c_idx in range(n_Cs):
            cfg = {
                'config_idx': c_idx,
                'C': float(Cs[c_idx]),
                'is_best': int(c_idx == ci),
                'val_auc': float(mean_val_aucs[c_idx, m]),
                'val_loss': float(mean_val_losses[c_idx, m]),
            }
            cfg.update(_compute_metrics(y_tests[m], test_probs[c_idx, m], test_preds[c_idx, m]))
            per_model.append(cfg)
        all_configs_results.append(per_model)

    return best_results, all_configs_results


def _logreg_no_cv(X_trains_scaled, y_trains, X_tests_scaled, y_tests,
                  M, d_in, lr, max_epochs, patience, batch_size):
    """Fallback for <=3 samples. Uses default C=1.0."""
    wds = torch.full((M,), 1.0, device=DEVICE)

    X_train_t = torch.tensor(np.stack(X_trains_scaled), dtype=torch.float32, device=DEVICE)
    y_train_t = torch.tensor(np.tile(y_trains[0], (M, 1)), dtype=torch.long, device=DEVICE)
    X_test_t = torch.tensor(np.stack(X_tests_scaled), dtype=torch.float32, device=DEVICE)

    model = ParallelMLP(M, d_in, hidden_sizes=(), d_out=2).to(DEVICE)
    _train_model(model, X_train_t, y_train_t, wds, lr, max_epochs, patience, batch_size)

    with torch.no_grad():
        train_probs = F.softmax(model(X_train_t), dim=-1).cpu().numpy()
        test_logits = model(X_test_t)
        test_probs = F.softmax(test_logits, dim=-1).cpu().numpy()
        test_preds = test_logits.argmax(dim=-1).cpu().numpy()

    best_results = []
    all_configs_results = []
    for m in range(M):
        metrics = {'best_C': 1.0}
        try:
            metrics['val_auc'] = roc_auc_score(y_trains[m], train_probs[m, :, 1])
        except ValueError:
            metrics['val_auc'] = 0.5
        metrics['val_loss'] = log_loss(y_trains[m], train_probs[m])
        metrics.update(_compute_metrics(y_tests[m], test_probs[m], test_preds[m]))
        best_results.append(metrics)

        cfg = dict(metrics)
        cfg.update({'config_idx': 0, 'C': 1.0, 'is_best': 1})
        all_configs_results.append([cfg])

    return best_results, all_configs_results


# ═══════════════════════════════════════════════════════════════════════════
#  MLP
# ═══════════════════════════════════════════════════════════════════════════

MLP_CONFIGS = [
    # (hidden_sizes, lr, weight_decay)
    ((32,),    1e-3, 1e-4),
    ((32,),    1e-3, 1e-2),
    ((64,),    1e-3, 1e-5),
    ((64,),    1e-3, 1e-3),
    ((64,),    1e-2, 1e-4),
    ((32, 32), 1e-3, 1e-4),
    ((32, 32), 1e-3, 1e-2),
    ((64, 64), 1e-3, 1e-4),
    ((64, 64), 1e-3, 1e-2),
    ((64, 64), 1e-2, 1e-4),
]


def find_best_mlp(X_trains, y_trains, X_tests, y_tests,
                  configs=None, max_epochs=1000, patience=50, batch_size=256):
    """
    Train M independent MLPs simultaneously on GPU, searching over configs.
    Configs sharing (hidden_sizes, lr) are batched together.

    Returns:
        (best_results, all_configs_results)
    """
    if configs is None:
        configs = MLP_CONFIGS

    n_configs = len(configs)
    M = len(X_trains)
    d_in = X_trains[0].shape[1]

    X_trains_scaled, scalers = _standardize_per_model(X_trains)
    X_tests_scaled = [scalers[i].transform(X_tests[i]) for i in range(M)]

    cv = get_cv(X_trains[0])
    splits = get_splits(cv, X_trains_scaled[0], y_trains[0])

    if len(splits) == 0 or X_trains[0].shape[0] <= 3:
        hs, lr, wd = configs[0]
        return _mlp_no_cv(X_trains_scaled, y_trains, X_tests_scaled, y_tests,
                          M, d_in, hs, lr, wd, max_epochs, batch_size)

    # Group configs by (hidden_sizes, lr)
    groups = defaultdict(list)
    for cfg_idx, (hs, lr, wd) in enumerate(configs):
        groups[(hs, lr)].append((cfg_idx, wd))

    # ── Cross-validation ──
    val_aucs_per_cfg = [[[] for _ in range(M)] for _ in range(n_configs)]
    val_losses_per_cfg = [[[] for _ in range(M)] for _ in range(n_configs)]

    for train_idx, val_idx in splits:
        X_fold_base = np.stack([X_trains_scaled[m][train_idx] for m in range(M)])
        X_val_base = np.stack([X_trains_scaled[m][val_idx] for m in range(M)])
        y_fold_train = y_trains[0][train_idx]
        y_fold_val = y_trains[0][val_idx]

        for (hidden_sizes, lr), group_cfgs in groups.items():
            n_in_group = len(group_cfgs)
            N_group = n_in_group * M

            X_ft = torch.tensor(np.tile(X_fold_base, (n_in_group, 1, 1)), dtype=torch.float32, device=DEVICE)
            y_ft = torch.tensor(np.tile(y_fold_train, (N_group, 1)), dtype=torch.long, device=DEVICE)
            X_fv = torch.tensor(np.tile(X_val_base, (n_in_group, 1, 1)), dtype=torch.float32, device=DEVICE)
            wds = torch.tensor([wd for _, wd in group_cfgs for _ in range(M)], dtype=torch.float32, device=DEVICE)

            model = ParallelMLP(N_group, d_in, hidden_sizes, d_out=2).to(DEVICE)
            _train_model(model, X_ft, y_ft, wds, lr, max_epochs, patience, batch_size)

            with torch.no_grad():
                probs = F.softmax(model(X_fv), dim=-1).cpu().numpy()

            for g_idx, (cfg_idx, _) in enumerate(group_cfgs):
                for m in range(M):
                    flat_idx = g_idx * M + m
                    try:
                        val_aucs_per_cfg[cfg_idx][m].append(roc_auc_score(y_fold_val, probs[flat_idx, :, 1]))
                        val_losses_per_cfg[cfg_idx][m].append(log_loss(y_fold_val, probs[flat_idx]))
                    except ValueError:
                        val_aucs_per_cfg[cfg_idx][m].append(0.5)
                        val_losses_per_cfg[cfg_idx][m].append(1.0)

    mean_val_aucs = np.array([[np.mean(val_aucs_per_cfg[c][m]) for m in range(M)] for c in range(n_configs)])
    mean_val_losses = np.array([[np.mean(val_losses_per_cfg[c][m]) for m in range(M)] for c in range(n_configs)])
    best_cfg_idx = np.argmax(mean_val_aucs, axis=0)

    # ── Final retrain: ALL configs ──
    X_train_t = torch.tensor(np.stack(X_trains_scaled), dtype=torch.float32, device=DEVICE)
    X_test_t = torch.tensor(np.stack(X_tests_scaled), dtype=torch.float32, device=DEVICE)
    y_full = y_trains[0]

    test_probs_all = [[None] * M for _ in range(n_configs)]
    test_preds_all = [[None] * M for _ in range(n_configs)]

    for (hidden_sizes, lr), group_cfgs in groups.items():
        n_in_group = len(group_cfgs)
        N_group = n_in_group * M

        X_train_tiled = X_train_t.repeat(n_in_group, 1, 1)
        y_train_tiled = torch.tensor(np.tile(y_full, (N_group, 1)), dtype=torch.long, device=DEVICE)
        X_test_tiled = X_test_t.repeat(n_in_group, 1, 1)
        wds = torch.tensor([wd for _, wd in group_cfgs for _ in range(M)], dtype=torch.float32, device=DEVICE)

        final_model = ParallelMLP(N_group, d_in, hidden_sizes, d_out=2).to(DEVICE)
        _train_model(final_model, X_train_tiled, y_train_tiled, wds, lr, max_epochs, patience, batch_size)

        with torch.no_grad():
            logits = final_model(X_test_tiled)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            preds = logits.argmax(dim=-1).cpu().numpy()

        for g_idx, (cfg_idx, _) in enumerate(group_cfgs):
            for m in range(M):
                flat_idx = g_idx * M + m
                test_probs_all[cfg_idx][m] = probs[flat_idx]
                test_preds_all[cfg_idx][m] = preds[flat_idx]

    # ── Assemble results ──
    best_results = []
    all_configs_results = []
    for m in range(M):
        ci = best_cfg_idx[m]
        hs, lr, wd = configs[ci]
        metrics = {
            'val_auc': float(mean_val_aucs[ci, m]),
            'val_loss': float(mean_val_losses[ci, m]),
            'best_config_idx': int(ci),
            'best_hidden_sizes': str(hs),
            'best_lr': lr,
            'best_weight_decay': wd,
        }
        metrics.update(_compute_metrics(y_tests[m], test_probs_all[ci][m], test_preds_all[ci][m]))
        best_results.append(metrics)

        per_model = []
        for c in range(n_configs):
            hs, lr, wd = configs[c]
            cfg = {
                'config_idx': c,
                'hidden_sizes': str(hs),
                'lr': lr,
                'weight_decay': wd,
                'is_best': int(c == ci),
                'val_auc': float(mean_val_aucs[c, m]),
                'val_loss': float(mean_val_losses[c, m]),
            }
            cfg.update(_compute_metrics(y_tests[m], test_probs_all[c][m], test_preds_all[c][m]))
            per_model.append(cfg)
        all_configs_results.append(per_model)

    return best_results, all_configs_results


def _mlp_no_cv(X_trains_scaled, y_trains, X_tests_scaled, y_tests,
               M, d_in, hidden_sizes, lr, weight_decay, max_epochs, batch_size):
    """Fallback for <=3 samples."""
    wds = torch.full((M,), weight_decay, device=DEVICE)

    X_train_t = torch.tensor(np.stack(X_trains_scaled), dtype=torch.float32, device=DEVICE)
    y_train_t = torch.tensor(np.tile(y_trains[0], (M, 1)), dtype=torch.long, device=DEVICE)
    X_test_t = torch.tensor(np.stack(X_tests_scaled), dtype=torch.float32, device=DEVICE)

    model = ParallelMLP(M, d_in, hidden_sizes, d_out=2).to(DEVICE)
    _train_model(model, X_train_t, y_train_t, wds, lr, max_epochs, 50, batch_size)

    with torch.no_grad():
        train_probs = F.softmax(model(X_train_t), dim=-1).cpu().numpy()
        test_logits = model(X_test_t)
        test_probs = F.softmax(test_logits, dim=-1).cpu().numpy()
        test_preds = test_logits.argmax(dim=-1).cpu().numpy()

    best_results = []
    all_configs_results = []
    for m in range(M):
        metrics = {
            'best_config_idx': 0,
            'best_hidden_sizes': str(hidden_sizes),
            'best_lr': lr,
            'best_weight_decay': weight_decay,
        }
        try:
            metrics['val_auc'] = roc_auc_score(y_trains[m], train_probs[m, :, 1])
        except ValueError:
            metrics['val_auc'] = 0.5
        metrics['val_loss'] = log_loss(y_trains[m], train_probs[m])
        metrics.update(_compute_metrics(y_tests[m], test_probs[m], test_preds[m]))
        best_results.append(metrics)

        cfg = dict(metrics)
        cfg.update({'config_idx': 0, 'hidden_sizes': str(hidden_sizes),
                    'lr': lr, 'weight_decay': weight_decay, 'is_best': 1})
        all_configs_results.append([cfg])

    return best_results, all_configs_results
