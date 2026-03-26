"""
Create missing One-vs-Rest (OvR) binary datasets from multiclass parent datasets.

For each multiclass dataset, creates balanced binary CSVs where:
- target=1: all rows of the target class
- target=0: random sample of equal size from all other classes
- Total = 2N rows, shuffled

Naming: {number}_{parent_tag}_{class_name}.csv
MASTER.csv entries: Source=MULTICLASS, Data type=Binary Classification
"""

import pandas as pd
import numpy as np
import os

SEED = 42
DATA_DIR = "data/cleaned_data"
MASTER_PATH = "data/probing_datasets_MASTER.csv"


def safe_class_name(cls):
    """Convert class value to a safe string for filenames."""
    return str(cls).replace(" ", "_")


def main():
    rng = np.random.RandomState(SEED)

    # Read MASTER.csv
    master = pd.read_csv(MASTER_PATH)

    # Find multiclass datasets
    multiclass_mask = master["Data type"] == "Multiclass Classification"
    multiclass_rows = master[multiclass_mask]

    # Build set of existing OvR dataset tags (normalize spaces to underscores for matching)
    existing_tags = set()
    for tag in master["Dataset Tag"].dropna().values:
        existing_tags.add(str(tag))
        existing_tags.add(str(tag).replace(" ", "_"))

    # Find next available number by scanning existing save names
    existing_numbers = []
    for save_name in master["Dataset save name"].dropna():
        basename = os.path.basename(save_name)
        try:
            num = int(basename.split("_")[0])
            existing_numbers.append(num)
        except (ValueError, IndexError):
            pass
    next_number = max(existing_numbers) + 1 if existing_numbers else 164

    created = []
    new_master_rows = []

    for _, row in multiclass_rows.iterrows():
        parent_tag = row["Dataset Tag"]
        parent_save = row["Dataset save name"]

        if pd.isna(parent_tag) or pd.isna(parent_save):
            continue

        parent_path = os.path.join("data", parent_save) if not parent_save.startswith("data/") else parent_save
        if not os.path.exists(parent_path):
            print(f"WARNING: Parent file not found: {parent_path}, skipping {parent_tag}")
            continue

        # Load parent dataset
        parent_df = pd.read_csv(parent_path)
        if "target" not in parent_df.columns or "prompt" not in parent_df.columns:
            print(f"WARNING: {parent_tag} missing required columns, skipping")
            continue

        classes = sorted(parent_df["target"].unique(), key=str)

        for cls in classes:
            ovr_tag = f"{parent_tag}_{safe_class_name(cls)}"

            # Check if OvR already exists
            if ovr_tag in existing_tags:
                continue

            # Create balanced binary dataset
            pos_mask = parent_df["target"] == cls
            pos_df = parent_df[pos_mask].copy()
            neg_df = parent_df[~pos_mask].copy()

            n_pos = len(pos_df)
            if n_pos == 0:
                print(f"WARNING: No positive samples for {ovr_tag}, skipping")
                continue

            # Sample negative examples (with replacement if needed)
            if len(neg_df) >= n_pos:
                neg_sample = neg_df.sample(n=n_pos, random_state=rng, replace=False)
            else:
                neg_sample = neg_df.sample(n=n_pos, random_state=rng, replace=True)

            pos_df = pos_df.copy()
            neg_sample = neg_sample.copy()
            pos_df["target"] = 1
            neg_sample["target"] = 0

            # Keep only prompt, prompt_len (if exists), target
            cols = ["prompt", "target"]
            if "prompt_len" in parent_df.columns:
                cols = ["prompt", "prompt_len", "target"]

            combined = pd.concat([pos_df[cols], neg_sample[cols]], ignore_index=True)
            combined = combined.sample(frac=1, random_state=rng).reset_index(drop=True)

            # Save
            filename = f"{next_number}_{ovr_tag}.csv"
            filepath = os.path.join(DATA_DIR, filename)
            combined.to_csv(filepath, index=False)

            # Prepare MASTER row
            new_row = {col: "" for col in master.columns}
            new_row["Source"] = "MULTICLASS"
            new_row["Dataset name"] = parent_tag
            new_row["Dataset Tag"] = ovr_tag
            new_row["Dataset save name"] = f"cleaned_data/{filename}"
            new_row["Data type"] = "Binary Classification"
            new_master_rows.append(new_row)

            created.append((next_number, ovr_tag, len(combined), n_pos))
            next_number += 1

    # Append to MASTER.csv
    if new_master_rows:
        new_df = pd.DataFrame(new_master_rows, columns=master.columns)
        updated_master = pd.concat([master, new_df], ignore_index=True)
        updated_master.to_csv(MASTER_PATH, index=False)

    # Summary
    print(f"\nCreated {len(created)} new OvR datasets:")
    for num, tag, total, n_pos in created:
        print(f"  {num}_{tag}.csv  ({total} rows, {n_pos} pos + {n_pos} neg)")

    if not created:
        print("No new datasets needed — all OvR conversions already exist.")


if __name__ == "__main__":
    main()
