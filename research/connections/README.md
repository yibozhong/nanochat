# Hyper-Connections and manifold-constrained Hyper-Connections

This is the new experiment authorized on2026-10-30. EBT is closed and its code
is archived separately. `nanochat.gpt.GPT` and the native SSSL training path
remain unchanged. `nanochat.connection_model.ConnectionModel` implements the
current full-attention vanilla baseline and optional HC/mHC residual routing.

## References and fidelity

- **HC:** the authors' PyTorch pseudocode in Appendix J of
  [Hyper-Connections v3](https://arxiv.org/html/2409.19606v3#A10), including
  LayerNorm in the router, zero dynamic projections,0.01 learnable scales,
  tanh, cyclic one-hot reads, identity residual mixing and unit writes. Each
  attention and MLP sublayer has its own connection, indexed2l and2l+1.
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
  weights,2*sigmoid post weights, source/destination matrix orientation, row
  softmax+epsilon initialization, alternating row/column normalization ending
  with column normalization, and20 iterations. The official learned final
  stream head is included. These released-code numerical details are more
  specific than the original paper equations. The unmodified official PyTorch
  Sinkhorn function is kept as an independent test oracle under its MIT license.

All routing math uses FP32 for BF16 training, with branch activations returned
to BF16. FP64 is retained for numerical tests. Native attention, ReLU-squared
MLP, RoPE, QK norm, RMSNorm and full-vocabulary CCE are shared with the baseline.
Both attention and MLP still compute width512; four residual streams do not
mean four attention/MLP executions. Mixing and saved activations do add cost.

The public DeepSeek inference code allocates parameters with `torch.empty`
and loads trained weights; the tested kernel reference uses synthetic inputs.
Neither supplies a complete from-scratch training initializer. Our explicit
initialization choice: projection Normal(0,1/sqrt(nD)), dynamic scales0.01,
pre bias logit(1/n), post bias0, residual-logit bias identity. The final stream
head uses the same projection/scale/pre-bias rule. These choices are not claimed
to reproduce DeepSeek's unreleased training recipe. The random projections
avoid an exactly symmetric multi-stream initialization. Native MuonAdamW trains
the backbone; all connection parameters use a separate AdamW group at the
native unembedding LR, betas(.8,.96), no weight decay. This is a nanochat
adaptation, not a reproduction of the paper's model scale, data or optimizer.

Sinkhorn at finite iterations/epsilon only approximately satisfies unit row
and column sums. It is not exact identity, does not constrain the entire model
Jacobian, and does not by itself guarantee stable training. No quality or speed
advantage is claimed from correctness tests or50-step smokes.

## Usage

Install the existing locked GPU environment with CCE:

```bash
uv sync --frozen --extra gpu --group research --group dev
python -m scripts.connection_train --variant baseline --steps 50 --data-dir data --output-dir outputs/baseline
python -m scripts.connection_train --variant hc --steps 50 --data-dir data --output-dir outputs/hc
python -m scripts.connection_train --variant mhc --steps 50 --data-dir data --output-dir outputs/mhc
```

Use compute nodes for these commands. The data directory is the existing
FineWeb-Edu packed cache from job7444406, with train/dev/test manifests. Default
microbatch16, accumulation1, no activation checkpoint, effective batch32768;
all three use8 layers/D512/4 heads/2K/32K vocabulary. HC/mHC default to4 streams.
`--steps` is required to prevent accidental long training. The schedule horizon
defaults to30518, including native40-step warmup and65% warmdown;50-step checks
are only a prefix of that schedule. W&B is always `yibozhong657-none/nanochat`.
Each run saves exact model/optimizer/RNG/data position; resume with `--resume`
and the same settings. A four-hour segment returns85 after checkpointing.

`baseline-runtime.json` and `baseline-resources.sub` record the user's measured
16/1 vanilla default. Their7GiB request is **not** a validated HC/mHC memory
bound. The new variants must pass the actual16x2048 GPU smoke before use.

Run `python -m scripts.connection_validate` in the validation job. It checks
independent HC/mHC forward and parameter/input gradients, Sinkhorn finite
differences, initialization, exact old-baseline compatibility, optimizer
coverage, real2K future-token isolation, KV-cache decoding, CCE, model state
restore and native attention regression tests. Then it runs50 compiled updates
for each variant on one physical GPU with the same packed rows. It does not
automatically submit full pretraining or compare50-step losses as final results.
