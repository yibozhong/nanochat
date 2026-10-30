# Hyper-Connections and manifold-constrained Hyper-Connections

This is the new experiment authorized on 2026-10-30. EBT is closed and its code
is archived separately. `nanochat.gpt.GPT` and the native SSSL training path
remain unchanged. `nanochat.connection_model.ConnectionModel` implements the
current full-attention vanilla baseline and optional HC/mHC residual routing.

## References and fidelity

- **HC:** the authors' PyTorch pseudocode in Appendix J of
  [Hyper-Connections v3](https://arxiv.org/html/2409.19606v3#A10), including
  LayerNorm in the router, zero dynamic projections, 0.01 learnable scales,
  tanh, cyclic one-hot reads, identity residual mixing and unit writes. Each
  attention and MLP sublayer has its own connection, indexed 2l and 2l+1.
  Streams are repeated initially and summed before final normalization.
  No author-owned GitHub implementation was found through the author homepage,
  author account and ByteDance organizations. The
  [lucidrains implementation](https://github.com/lucidrains/hyper-connections)
  was cross-checked at e89e30357d1f79945b0d3558e6721451ff68789a; it is third-party
  and uses RMSNorm instead of the appendix's LayerNorm. We keep the appendix
  choice and do not import its optional variants or depend on that package.
- **mHC:** [DeepSeek's official PyTorch reference](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/tests/test_mega_mhc.py)
  plus its [official V4 model](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/292e68aa01e9d6089f015fdfd876e5178ebbacf2/inference/model.py)
  and kernel. We preserve projection-before-RMS-scaling, sigmoid+epsilon pre
  weights, 2*sigmoid post weights, source/destination matrix orientation, row
  softmax+epsilon initialization, alternating row/column normalization ending
  with column normalization, and 20 iterations. The official learned final
  stream head is included. These released-code numerical details are more
  specific than the original paper equations. The unmodified official PyTorch
  Sinkhorn function is kept as an independent test oracle under its MIT license.

All routing math uses FP32 for BF16 training, with branch activations returned
to BF16. FP64 is retained for numerical tests. Native attention, ReLU-squared
MLP, RoPE, QK norm, RMSNorm and full-vocabulary CCE are shared with the baseline.
Both attention and MLP still compute width 512; four residual streams do not
mean four attention/MLP executions. Mixing and saved activations do add cost.

The public DeepSeek inference code allocates parameters with `torch.empty`
and loads trained weights; the tested kernel reference uses synthetic inputs.
Neither supplies a complete from-scratch training initializer. Our explicit
initialization choice: projection Normal(0,1/sqrt(nD)), dynamic scales 0.01,
pre bias logit(1/n), post bias 0, residual-logit bias identity. The final stream
head uses the same projection/scale/pre-bias rule. These choices are not claimed
to reproduce DeepSeek's unreleased training recipe. The random projections
avoid an exactly symmetric multi-stream initialization. Native MuonAdamW trains
the backbone; all connection parameters use a separate AdamW group at the
native unembedding LR, betas(.8,.96), no weight decay. This is a nanochat
adaptation, not a reproduction of the paper's model scale, data or optimizer.

Sinkhorn at finite iterations/epsilon only approximately satisfies unit row
and column sums. It is not exact identity, does not constrain the entire model
Jacobian, and does not by itself guarantee stable training. No quality or speed
advantage is claimed from correctness tests or 50-step smokes.

## Usage

Install the existing locked GPU environment with CCE:

```bash
uv sync --frozen --extra gpu --group research --group dev
python -m scripts.connection_train --variant baseline --steps 50 --data-dir data --output-dir outputs/baseline
python -m scripts.connection_train --variant hc --steps 50 --data-dir data --output-dir outputs/hc
python -m scripts.connection_train --variant mhc --steps 50 --data-dir data --output-dir outputs/mhc
```

Use compute nodes for these commands. The data directory is the existing
FineWeb-Edu packed cache from job 7444406, with train/dev/test manifests. Default
microbatch 16, accumulation 1, no activation checkpoint, effective batch 32768;
all three use 8 layers/D512/4 heads/2K/32K vocabulary. HC/mHC default to 4 streams.
`--steps` is required to prevent accidental long training. The schedule horizon
defaults to 30518, including native 40-step warmup and 65% warmdown; 50-step checks
are only a prefix of that schedule. W&B is always `yibozhong657-none/nanochat`.
Each run saves exact model/optimizer/RNG/data position; resume with `--resume`
and the same settings. A four-hour segment returns 85 after checkpointing.

`baseline-runtime.json` and `baseline-resources.sub` record the user's measured
16/1 vanilla default. Their 7GiB request is **not** a validated HC/mHC memory
bound. The new variants passed the 16x2048 smoke below on Blackwell; a 24GB
GPU-specific resource profile has not yet been measured. The validation submit
file allows compatible full GPUs with at least 22,000MiB (including the 24GB tier).

Run `python -m scripts.connection_validate` in the validation job. It checks
independent HC/mHC forward and parameter/input gradients, Sinkhorn finite
differences, initialization, exact old-baseline compatibility, optimizer
coverage, real 2K future-token isolation, KV-cache decoding, CCE, model state
restore and native attention regression tests. Then it runs 50 compiled updates
for each variant on one physical GPU with the same packed rows. It does not
automatically submit full pretraining or compare 50-step losses as final results.

## Validation result (2026-10-30)

Job **7546699.0** exited 0 on one RTX PRO 6000 Blackwell Server Edition.
Tested source: `8320fc6f7fd1606805569b2cd17e928bb7f3a8af`. All ten new
connection/model checks and five native SDPA fallback checks passed. Ten
FA3-specific comparison tests were skipped because FA3 was unavailable on this
GPU. W&B results and transferred reports were independently checked; see
[validation.json](validation.json) and the
[correctness run](https://wandb.ai/yibozhong657-none/nanochat/runs/da3193bb).

Each smoke used the same first 1,638,400 packed FineWeb-Edu tokens, seed 42,
50 updates, microbatch 16/accumulation 1, block compile and no activation
checkpointing. Initial dev CE was 10.398013 for all three models.

| Variant | Parameters | Dev CE after 50 steps | Mean step, steps 11–50 | Peak PyTorch reserved |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 58,720,256 | 6.496223 | 0.08237 s | 5.39 GiB |
| HC | 58,786,208 | 6.541191 | 0.18942 s | 14.60 GiB |
| mHC | 59,515,317 | 6.498918 | 0.18373 s | 11.47 GiB |

Timing excludes compilation, evaluation, checkpointing and scheduling; it is a
short measurement of this portable PyTorch implementation on this card. Allocator
peaks include validation and checkpoint phases but exclude non-PyTorch device
allocations. They are not an A30 measurement or a complete device-memory bound.
All updates and gradients were finite and dev CE decreased. These results do not
establish convergence or a quality advantage, and no full 1B run was launched.
Model-state restore passed; a full interrupted optimizer/RNG continuation has not
been separately exercised for this new training entry.
