#!/bin/bash
set -euo pipefail
GRADFORMER_WORKDIR="$PWD"
variant="$1"
seed="$2"
mkdir -p results
trap 'tar -cf "$GRADFORMER_WORKDIR/results.tar" -C "$GRADFORMER_WORKDIR" results' EXIT
export NETRC="$PWD/.netrc"
export WANDB_DIR="$PWD/results"
export WANDB_PROJECT=nanochat WANDB_ENTITY=yibozhong657-none
export OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1
export NANOCHAT_ATTENTION_BACKEND=sdpa NANOCHAT_DTYPE=bfloat16
cp /opt/gradformer-environment.txt results/environment.txt
cp /opt/gradformer-locks.txt results/environment-locks.txt
tar -xzf code.tar.gz
tar -xf dataset.tar
cd nanochat
case "$variant" in
    baseline) options=(--update baseline) ;;
    adam-no-aux) options=(--update adam --aux-weight 0) ;;
    adam-mse) options=(--update adam --beta 0) ;;
    adam-mse-cos) options=(--update adam) ;;
    sgd-mse-cos) options=(--update sgd) ;;
    *) echo "Unknown variant: $variant" >&2; exit 2 ;;
esac
torchrun --standalone --nproc_per_node=8 -m gradformer.train --backend nanochat "${options[@]}" --variant "$variant" --seed "$seed" --data ../dataset --output ../results/train --depth 8 --width 512 --heads 4 --sequence 256 --batch 16 --batch-tokens 65536 --tokens 200000000 --eval-every 250 --eval-batches 32 --save-every 1000 --backward reference
