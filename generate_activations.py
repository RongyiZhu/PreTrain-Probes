"""
Generate model activations for binary classification datasets.
Extracts last-token hidden states at configured hook layers using TransformerLens.
Supports multi-GPU parallelism and model revisions (e.g. Pythia checkpoints).
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import glob
import random
import argparse
from multiprocessing import Pool

import pandas as pd
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

from utils_data import MODEL_CONFIG, get_model_config

torch.set_grad_enabled(False)


def get_hook_names(model_name):
    config = get_model_config(model_name)
    hooks = [f"blocks.{layer}.hook_resid_post" for layer in config["hook_layers"]]
    if config["include_embed"]:
        hooks = ["hook_embed"] + hooks
    return hooks


def load_model(model_name, device, revision=None):
    config = get_model_config(model_name)
    kwargs = {"revision": revision} if revision else {}
    return HookedTransformer.from_pretrained(config["hf_model_path"], device=device, **kwargs)


def process_single_dataset(dataset_name, model, tokenizer, model_name, device, max_seq_len, hook_names, revision=None, batch_size=32):
    try:
        dataset = pd.read_csv(dataset_name)
        if "prompt" not in dataset.columns:
            return f"Skipping {dataset_name} (no 'prompt' column)"

        dataset_short_name = dataset_name.split("/")[-1].split(".")[0]
        revision_suffix = f"_rev{revision}" if revision else ""
        file_names = [
            f"data/model_activations_{model_name}{revision_suffix}/{dataset_short_name}_{hook}.pt"
            for hook in hook_names
        ]

        # Check if activations already exist with correct length
        text = dataset["prompt"].tolist()
        if all(os.path.exists(f) for f in file_names):
            lengths = [torch.load(f, weights_only=True).shape[0] for f in file_names]
            if all(l == len(text) for l in lengths):
                return f"Skipping {dataset_short_name} (activations exist)"

        # Tokenize once upfront, reuse for batching
        tokenized = tokenizer(text, padding=False, truncation=True, max_length=max_seq_len)
        text_lengths = [len(ids) for ids in tokenized['input_ids']]

        print(f"[GPU {device}] Generating activations for {dataset_short_name}")

        all_activations = {hook: [] for hook in hook_names}
        bar = tqdm(range(0, len(text), batch_size), desc=f"[GPU {device}] {dataset_short_name}")
        for i in bar:
            batch_text = text[i:i+batch_size]
            batch_lengths = text_lengths[i:i+batch_size]
            batch = tokenizer(batch_text, padding=True, truncation=True,
                              max_length=max_seq_len, return_tensors="pt").to(device)
            _, cache = model.run_with_cache(batch["input_ids"], names_filter=hook_names)
            for j, length in enumerate(batch_lengths):
                pos = min(length - 1, max_seq_len - 1)
                for hook in hook_names:
                    all_activations[hook].append(cache[hook][j, pos].unsqueeze(0).cpu())

        for hook, fname in zip(hook_names, file_names):
            torch.save(torch.cat(all_activations[hook]), fname)

        return f"[GPU {device}] Completed {dataset_short_name}"
    except Exception as e:
        return f"[GPU {device}] Error processing {dataset_name}: {e}"


def worker_process(datasets, model_name, device, max_seq_len, revision=None, batch_size=32):
    revision_suffix = f"_rev{revision}" if revision else ""
    os.makedirs(f"data/model_activations_{model_name}{revision_suffix}", exist_ok=True)
    hook_names = get_hook_names(model_name)

    # Load model once for all datasets on this GPU
    model = load_model(model_name, device, revision=revision)
    tokenizer = model.tokenizer
    tokenizer.truncation_side = 'left'
    tokenizer.padding_side = 'right'

    results = []
    for dataset_name in datasets:
        result = process_single_dataset(dataset_name, model, tokenizer, model_name, device, max_seq_len, hook_names, revision, batch_size)
        results.append(result)
        print(result)
    return results


def generate_activations(model_name, devices, max_seq_len=1024, revision=None, batch_size=32):
    dataset_names = glob.glob("data/cleaned_data/*.csv")
    random.shuffle(dataset_names)

    valid_datasets = []
    for name in dataset_names:
        try:
            df = pd.read_csv(name)
            if "prompt" in df.columns:
                valid_datasets.append(name)
        except Exception:
            continue

    if not valid_datasets:
        print("No valid datasets found")
        return

    num_gpus = len(devices)
    datasets_per_gpu = [[] for _ in range(num_gpus)]
    for i, name in enumerate(valid_datasets):
        datasets_per_gpu[i % num_gpus].append(name)

    print(f"Distributing {len(valid_datasets)} datasets across {num_gpus} GPUs")
    for device, ds in zip(devices, datasets_per_gpu):
        print(f"  GPU {device}: {len(ds)} datasets")

    worker_args = [(ds, model_name, dev, max_seq_len, revision, batch_size)
                   for ds, dev in zip(datasets_per_gpu, devices)]

    with Pool(processes=num_gpus) as pool:
        results = pool.starmap(worker_process, worker_args)

    print("\n=== Summary ===")
    for device, result_list in zip(devices, results):
        for r in result_list:
            print(f"  {r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--devices", type=str, nargs="*", default=["cuda:0"],
                        help="GPU devices (space or comma separated)")
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for activation extraction (default: 32)")
    args = parser.parse_args()

    if len(args.devices) == 1 and ',' in args.devices[0]:
        devices = [d.strip() for d in args.devices[0].split(',')]
    else:
        devices = args.devices

    generate_activations(args.model_name, devices, args.max_seq_len, args.revision, args.batch_size)
