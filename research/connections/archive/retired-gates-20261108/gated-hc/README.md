# SiLU Gated HC

Per token/channel, four streams x are RMS-normalized over the stream axis only:

    v = x / sqrt(mean(x*x) + float32_epsilon)
    g_i = SiLU(b_i + sum_k W_ik * v_k)
    output_j = sum_i (x_i * g_i) * H_ij + branch * Hpost_j

W has shape [channel,4,4], b [channel,4]. Each gate can observe all four streams
of its own token and channel. Its channel-specific effective mixing matrix is
diag(g) H, a constrained matrix with four channel-specific degrees of freedom,
not an arbitrary channel-specific 4x4 matrix. H, Hpre and Hpost retain the
NVIDIA no-Sinkhorn HC token-wide projection. Attention and MLP retain separate
connectors. Read and the branch are unchanged; gating is on the residual input
to the write's shared mixing matrix.

The ordinary SiLU function is neither shifted nor bounded. W starts at zero,
b at 1.2784645427610737 (SiLU(b)=1), preserving HC's initial gate and RNG stream.
This initialization is not a stability guarantee. Gate parameters use the
existing connector AdamW group, no weight decay; the backbone is unchanged.
The 8-layer D512 model has 59,679,157 parameters, 163,840 more than shared HC.

The new Triton write fuses local RMS, the four gates, shared mixing and branch
addition. BF16 input/output, FP32 arithmetic. Its analytic backward recomputes
gates, uses deterministic parameter partials, and writes routing gradients with
one owner per token. No saved gate or gated-state bank, no atomic reductions.
The parameter partial buffer is 10 MiB at B16/T2048/C512. Retained kernels are
not modified. Compare measured full training steps before claiming a speedup.

The gate runs 25 correctness checks (FP64 all VJPs including zero/tiny/negative
gates, locality, identical initialization, compile, 2K causality, KV cache, CCE,
recomputation, optimizer/RNG restoration and retained model coverage), four
same-card 60-update profiles (baseline/shared no-Sinkhorn/channel-local/gated),
production-shape fused/compiled-Torch timing, then real NTP 25-to-128 resume.
Common microbatch 16/8/4, global tokens 32,768; failed numerics stop submission.

This is the selected gated HC implementation again as of 2026-11-07. The
independent-stream dense sigmoid pre/post variants are retired. Production
nanochat/, scripts/ and tests/ are restored byte-for-byte to validated source
ebf51bf38f2d038af89679293b95c21d6129f41c. GPU acceptance evidence is reused,
not rerun: gate7915083.0 completed all25 checks, same-card profiles and128steps.
The source archive is reconstructed from Git and its SHA256 matches the exact
original GPU archive. See restoration-20261107.json and validation.json.

The authorized new run uses ../gated-hc-train/submit.py and a fresh root
/home/yzhong239/gated-hc-silu-10b-20261107. It starts at seed42 with no old or
validation weights. The former SiLU run and both sigmoid runs stay stopped.
Training is limited to full H100/H200, FA3, micro16 and32768tokens/step, with
305176steps=10000007168tokens. Checkpoint/latest every500steps, separate best,
and exit85 every1800seconds preserve this new run across interruptions.
The first-step watcher never submits jobs and stops after a finite formal step.
