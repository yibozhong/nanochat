# Gradformer

Gradformer keeps the native model and trains each block to predict the final next-token loss gradient with respect to that block's input. The final NTP loss and the auxiliary prediction loss both update the model. Targets are detached, with no intermediate LM heads or second derivatives.

## Mathematical contract

For each block, define its original residual R(h), predicted positive gradient G(h) = -R(h), and update h_next = h - eta*G(h). With hidden SGD, eta=1 and auxiliary weight=0 reproduces the baseline up to floating-point addition order. A positive learnable eta is parameterized with softplus. The optional hidden Adam operator carries per-token moments across depth, resets them for each new forward, and uses predicted gradients during training and inference. It never reads labels in the forward operator.

Targets differentiate the unscaled mean CE of each microbatch, before gradient accumulation scaling. This convention is fixed across variants; changing microbatch size changes raw target magnitudes. MSE averages over hidden channels, valid tokens and layers. Cosine uses hidden-channel vectors and excludes zero/near-zero targets. Default alpha=1, beta=0.1, lambda=0.1. Targets are not normalized. All auxiliary losses retain their normal paths into earlier blocks; this is joint training, not isolated layer-local parameter updates.

The reference backward first obtains targets with autograd.grad and then differentiates the total loss. The reuse implementation differentiates NTP once, captures its pure hidden cotangents with temporary hooks, removes the hooks, then accumulates auxiliary parameter gradients. Both parameter-gradient results are checked numerically, including gradient accumulation.

Hidden Adam uses beta1=0.9, beta2=0.999, eps=0.01, and initial eta=0.01. The initial derivative eta/eps=1 avoids amplifying nanochat's zero-initialized residual projections. The zero-variance sqrt branch uses a finite zero derivative, preserving the derivative of the composite first-step update at g=0; finite-difference checks cover it. These are architectural hyperparameters, not the parameter optimizer's Adam settings.

Each layer has one scalar rho, shared across samples, tokens and hidden channels, with eta=softplus(rho). Eta has no direct training-step decay schedule; the parameter optimizer's learning rate for rho does. In the initial experiments, llama2.c puts rho in its no-weight-decay AdamW group (peak lr=5e-4), while nanochat puts it in the native smear/scalar AdamW group (peak lr=0.2, no weight decay). These are different cross-backend choices, not a tuned common step-size learning rate. The experiment runner initializes eta to 1 for hidden SGD and 0.01 for hidden Adam. Logged cosine compares raw G with its NTP target, before Adam moments, preconditioning or eta multiplication; it does not measure the actual hidden update direction.

## Environment

Use Python 3.11 and the committed uv.lock. nanochat: `uv sync --frozen --extra gpu --group gradformer`; llama2.c: `uv sync --frozen --extra gpu --group dev`. For CPU correctness, replace gpu with cpu. Parameter optimizers follow their respective native baselines: nanochat MuonAdamW and llama2.c AdamW. Baseline and Gradformer share optimizer settings, initialization seeds, batches, tokenization, data order and training schedules within each backend.

Authentication is supplied externally through W&B's netrc or environment. Never put credentials in source, submission files or logs. All runs use project nanochat and entity yibozhong657-none.

The first supervised runs used effective MSE/cosine coefficients 0.1/0.01. The requested stronger controls use `--aux-weight 1 --alpha 0.5 --beta 0.5` and `--aux-weight 1 --alpha 1 --beta 1`, so the actual coefficients in the total objective are 0.5/0.5 and 1/1; NTP retains coefficient 1. Keep other hyperparameters fixed and label these as separate experiments.

## Commands

Run these inside an allocated HTCondor job/container. Set NANOCHAT_DTYPE=float32 for correctness checks and bfloat16 for CUDA training.

- `python -m gradformer.check --backend nanochat` (or llama2)
- `torchrun --standalone --nproc_per_node=8 -m gradformer.benchmark --backend nanochat --output benchmark.json`
- `python -m gradformer.benchmark --backend llama2 --depth 6 --width 288 --heads 6`
- `torchrun --standalone --nproc_per_node=8 -m gradformer.train --backend nanochat --update baseline --data dataset --output outputs/baseline`
- Repeat the identical training command with `--update adam --output outputs/gradformer-adam` (or sgd).

Data preparation uses TinyStories V2 GPT-4's official train and validation files, a pinned dataset revision and the original llama2.c 32k tokenizer. The fixed token streams can be shared between the backends. This is a controlled small-model pretraining setting, not a reproduction of nanochat's upstream large-data speedrun. Data and tokenizer SHA256 values are saved in manifest.json.

Every training run saves configuration, JSONL metrics, W&B identity, best weights, and two alternating complete checkpoint generations. Each rank saves its optimizer and RNG/data state; the latest pointer changes only after all rank shards finish. `--resume` restores the same run and data stream. Do not mark a cluster job resumable until its output-transfer and interruption/restart behavior have also been checked.

Compilation preserves layer input boundaries for gradient targets: Gradformer compiles blocks, the hidden update, the auxiliary loss, and the final vocabulary projection/CE as separate functions, retaining explicit layer-input nodes. Profiling reports actual step time and peak allocated/reserved GPU memory; do not infer Gradformer FLOPs from the unchanged baseline estimator. Performance and scientific conclusions belong in measured experiment reports.

The recorded `training_seconds` sums timed optimizer steps. New runs also record `wall_seconds`, including setup, initial evaluation, compilation, periodic evaluation and checkpoint work from entry into the training function; time waiting in the cluster queue is excluded. Benchmark warmup is excluded from steady-state timings.

For CHTC, see `gradformer/chtc/README.md`, the compiler-verified uv container definition, and submission templates. Source bundles can be created with `python -m gradformer.package`; only committed files are included. Data preparation defaults to the exact dataset commit used in the first experiments.

## Single-GPU gradient-norm control

`--gradient-balance global --balance-ratio 1 --backward reuse --aux-weight 1 --alpha 1 --beta 1` keeps the auxiliary and NTP parameter-gradient norms equal over the union of native block parameters. Gradients are accumulated separately over the entire optimizer batch before measuring the ratio. The detached dynamic scalar multiplies all auxiliary parameter gradients, including embeddings and hidden-step parameters, so this remains a scalar-weighted joint objective. The norm measurement excludes embeddings, the output head and hidden-step scalars.

This first implementation explicitly supports one GPU only. It uses no EMA, weight cap or additional reverse pass compared with reuse backward; separate gradient buffers add memory. Near-zero norm cases disable the auxiliary contribution and log `balance/active=0`; ordinary active steps assert the measured weighted norm ratio. The configured outer auxiliary weight must be 1 because the dynamic coefficient replaces it. Per-block ratios and gradient cosines are logged; equal aggregate norms do not imply equality in every block, or equal effects after Adam/Muon. A common global clipping factor is applied after gradient combination as in the baseline.

Run `python -m gradformer.check_balance --backend BACKEND --device cuda --output balance-check.json` on an allocated node. It compares two consecutive updates against independent autograd reference objectives, covers accumulation 1 and 3, ratios 0.5 and 1, both hidden optimizers, and the zero-auxiliary guard. `gradformer.benchmark --include-balance --accumulation 8` includes the measured balancing cost in full accumulated optimizer steps.
