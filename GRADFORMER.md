# Gradformer

Gradformer keeps the native model and trains each block to predict the final next-token loss gradient with respect to that block's input. The final NTP loss and the auxiliary prediction loss both update the model. Targets are detached, with no intermediate LM heads or second derivatives.

## Mathematical contract

For each block, define its original residual R(h), predicted positive gradient G(h) = -R(h), and update h_next = h - eta*G(h). With hidden SGD, eta=1 and auxiliary weight=0 reproduces the baseline up to floating-point addition order. A positive learnable eta is parameterized with softplus. The optional hidden Adam operator carries per-token moments across depth, resets them for each new forward, and uses predicted gradients during training and inference. It never reads labels in the forward operator.

Targets differentiate the unscaled mean CE of each microbatch, before gradient accumulation scaling. This convention is fixed across variants; changing microbatch size changes raw target magnitudes. MSE averages over hidden channels, valid tokens and layers. Cosine uses hidden-channel vectors and excludes zero/near-zero targets. Default alpha=1, beta=0.1, lambda=0.1. Targets are not normalized. All auxiliary losses retain their normal paths into earlier blocks; this is joint training, not isolated layer-local parameter updates.

The reference backward first obtains targets with autograd.grad and then differentiates the total loss. The reuse implementation differentiates NTP once, captures its pure hidden cotangents with temporary hooks, removes the hooks, then accumulates auxiliary parameter gradients. Both parameter-gradient results are checked numerically, including gradient accumulation.

Hidden Adam uses beta1=0.9, beta2=0.999, eps=0.01, and initial eta=0.01. The initial derivative eta/eps=1 avoids amplifying nanochat's zero-initialized residual projections. The zero-variance sqrt branch uses a finite zero derivative, preserving the derivative of the composite first-step update at g=0; finite-difference checks cover it. These are architectural hyperparameters, not the parameter optimizer's Adam settings.

## Environment

Use Python 3.11 and the committed uv.lock. nanochat: `uv sync --frozen --extra gpu --group gradformer`; llama2.c: `uv sync --frozen --extra gpu --group dev`. For CPU correctness, replace gpu with cpu. Parameter optimizers follow their respective native baselines: nanochat MuonAdamW and llama2.c AdamW. Baseline and Gradformer share optimizer settings, initialization seeds, batches, tokenization, data order and training schedules within each backend.

Authentication is supplied externally through W&B's netrc or environment. Never put credentials in source, submission files or logs. All runs use project nanochat and entity yibozhong657-none.

## Commands

Run these inside an allocated HTCondor job/container. Set NANOCHAT_DTYPE=float32 for correctness checks and bfloat16 for CUDA training.

- `python -m gradformer.check --backend nanochat` (or llama2)
- `torchrun --standalone --nproc_per_node=8 -m gradformer.benchmark --backend nanochat --output benchmark.json`
- `python -m gradformer.benchmark --backend llama2 --depth 6 --width 288 --heads 6`
- `torchrun --standalone --nproc_per_node=8 -m gradformer.train --backend nanochat --update baseline --data dataset --output outputs/baseline`
- Repeat the identical training command with `--update adam --output outputs/gradformer-adam` (or sgd).

Data preparation uses TinyStories V2 GPT-4's official train and validation files, a pinned dataset revision and the original llama2.c 32k tokenizer. The fixed token streams can be shared between the backends. This is a controlled small-model pretraining setting, not a reproduction of nanochat's upstream large-data speedrun. Data and tokenizer SHA256 values are saved in manifest.json.

Every training run saves configuration, JSONL metrics, W&B identity, best weights, and two alternating complete checkpoint generations. Each rank saves its optimizer and RNG/data state; the latest pointer changes only after all rank shards finish. `--resume` restores the same run and data stream. Do not mark a cluster job resumable until its output-transfer and interruption/restart behavior have also been checked.

Compilation preserves layer input boundaries for gradient targets: Gradformer compiles blocks, not the complete loss graph. Profiling reports actual step time and peak allocated/reserved GPU memory; do not infer Gradformer FLOPs from the unchanged baseline estimator. Performance and scientific conclusions belong in measured experiment reports.
