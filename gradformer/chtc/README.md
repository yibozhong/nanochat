# CHTC environment

From the repository root, use `condor_submit -i gradformer/chtc/build.sub`. On the allocated build node, run:

```bash
mkdir -p tmp cache
export APPTAINER_TMPDIR="$PWD/tmp" APPTAINER_CACHEDIR="$PWD/cache"
apptainer build nanochat-uv.sif uv.def
sha256sum nanochat-uv.sif > build-manifest.txt
```

Move the image to your staging directory before exiting. The definition runs `uv sync --frozen --extra gpu --group gradformer --no-dev --no-install-project`, verifies the compiler, and records installed versions and lock hashes inside the image. The SIF digest identifies the exact runtime. The Docker base tag and Debian packages are not immutable: preserve the measured SIF and its digest as well as the definition when reproducing a run.

Train only on allocated GPUs. Set `NANOCHAT_ATTENTION_BACKEND=sdpa` for a pinned attention implementation and `NANOCHAT_DTYPE=bfloat16` for training. Set `NANOCHAT_DTYPE=float32` for correctness checks. Transfer W&B credentials privately and explicitly encrypt them; never include them in output archives. Give each job a distinct result destination. Archive output directories because W&B creates symbolic links that HTCondor directory transfer may reject.

The experiment harness is `python -m gradformer.train`. Use eight processes on a single node for nanochat and one GPU for llama2.c. Keep microbatch, global batch, sequence length and token budget fixed within each comparison. Run `python -m gradformer.check` and `python -m gradformer.check_compile` before enabling compiled experiments.

## Training submission

Create a committed source bundle with `python -m gradformer.package --prefix nanochat --output /absolute/code.tar.gz`. The data archive must be named `dataset.tar` and contain `dataset/train.bin`, `dataset/val.bin` and `dataset/manifest.json`. Override the three macros in the `train.sub` example. This template runs independent variants; the initial measured llama2.c experiment instead runs all variants sequentially on the same allocated GPU to control hardware variability.

The sample uses eager forward execution; the native nanochat parameter optimizer still compiles its own kernels. Add `--compile` only after the compiled gradient check passes on the chosen image. A checkpoint stores model, optimizer and data/RNG state; `--resume` requires transferring the existing result directory back into the job and keeping world size and training configuration unchanged. This template uses normal short jobs and does not claim automatic restart after eviction.
