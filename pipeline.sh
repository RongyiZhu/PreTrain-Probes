#!/bin/bash

set -euo pipefail
export OMP_NUM_THREADS=1

revision_list=(
  step0
  step1
  step16
  step128
  step256
  step512
  step1000
  step2000
  step5000
  step10000
  step25000
  step50000
  step75000
  step100000
  step125000
  step143000
)

#model_name=("pythia-70m")
model_name=(
  #"pythia-70m"
  #"pythia-160m"
  "pythia-410m"
  #"pythia-1b"
  #"pythia-2.8b"
  #"pythia-6.9b"
  #"pythia-12b"
)

for model in "${model_name[@]}"; do
  echo "=============================="
  echo " Processing model: $model"
  echo "=============================="

  # Stage 1: Generate activations for all revisions
  for revision in "${revision_list[@]}"; do
    echo "=============================="
    echo " Generating activations: $revision  [$(date)]"
    echo "=============================="
    python generate_activations.py \
      --model_name "$model" \
      --revision "$revision" \
      --devices cuda:4,cuda:5 \
      --max_seq_len 1024 \
      --batch_size 32
  done
done

for model in "${model_name[@]}"; do
  # Stage 2: Train probes for all revisions at once (batched)
  echo "=============================="
  echo " Training probes: all revisions batched  [$(date)]"
  echo "=============================="
  python run_probes.py \
    --model_name "$model" \
    --revision "${revision_list[@]}" \
    --devices cuda:4,cuda:5 \
    --revisions_per_batch 4

  echo "[SUCCESS] $model done."
done

echo "All revisions processed successfully."
