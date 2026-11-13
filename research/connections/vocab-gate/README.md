# Full dense vocabulary gate

User authorization: retire and remove the final-FFN readout route, implement the
first candidate vocabulary gate without low-rank factors, and run one replacement
10B-token experiment after correctness and performance checks on the same GPU.

For each final normalized D512 hidden state:

    z = W h; g = G h
    logits = 15 * tanh(z * (1 + 0.5 * tanh(g)) / 15)

W and G are each 32768 x 512, packed into one 65536 x 512 weight. G starts at zero,
preserving baseline predictions and initialization RNG. The first update learns
G from the usual cross-entropy gradient. The gate stays within [0.5, 1.5].
No auxiliary objective, low-rank factor, teacher, or custom gradient estimator.
The extra dense matrix adds 16,777,216 parameters; total 75,497,472. Head matrix
arithmetic doubles; the wall-clock cost must be measured on the same GPU.

The implementation uses packed cuBLAS GEMMs and independently written Triton
kernels for fused gating, softcap, log-sum-exp and both analytic score derivatives.
Gradient computation in forward avoids projection recomputation. A reusable
2048-token buffer bounds the paired BF16 logits to 256 MiB. dW accumulates in
FP32 over chunks, then rounds at the original BF16 classifier-gradient boundary.
Both one-row and tiled CE implementations are checked and timed. Baseline uses
its original unmodified Apple CCE. Disabled config retains original behavior.

References for the design and implementation strategy:
- [Mixtape (2019)](https://papers.neurips.cc/paper/9723-mixtape-breaking-the-softmax-bottleneck-efficiently.pdf): vocabulary-dependent gates; this experiment uses a different single dense multiplicative gate.
- [Liger fused linear CE](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/fused_linear_cross_entropy.py): chunked GEMM and compute-gradients-in-forward strategy; no code/dependency copied.

Protocol: original FineWeb-Edu packed2K/tokenizer, vanilla 8-layer D512/H4,
micro16/global32768, seed42, FA2, default MuonAdamW/LR schedule,
305176 updates = 10,000,007,168 tokens. One H100/H200 allocation performs GPU
tests, same-card timing, 200 diagnostic updates, trained causal/cache checks,
and a real CLI step2/exit85/step3 recovery. Diagnostics have no W&B and their
weights are discarded. Source-matched gate approval is required before a fresh
formal process creates the single online W&B run. Failure waits for bounded
in-place debug, never automatically retries or launches formal training.

Formal recovery retains model/optimizer/RNG/step/data position and run ID.
Every 1800 seconds exit85 returns atomic latest/best archives; archive failure
returns 74. Failed jobs hold without resubmission. GPU validation is complete; the authorized formal training is starting.

## Validated results (H100 80GB, source ada2613)

- 48 GPU tests passed, including production-width gradients for all parameters,
  FP32/BF16 kernels, saturation, upstream scaling, compile/checkpoint, baseline
  initialization/RNG, causal cache, and exact optimizer/RNG/next-update recovery.
- 200 diagnostic updates finite; gate gradients and trained weights nonzero.
  Trained 2K cache relative/peak errors: 0.002667 / 0.006206; causal prefix exact.
- Real CLI step2/exit85/step3 recovery passed, same diagnostic ID, no W&B.
- Interleaved 10 warm + 40 measured updates per model on the same H100:
  original baseline 67.674 ms; dense gate 43.929 ms. This includes replacing the
  old CCE compute path, not an architecture-only speed comparison.
- Tiled CE beats row CE. Production retains 2048-token chunks (256 MiB scratch);
  8192-token chunks save about 1.2 ms for the head while requiring 1 GiB scratch.
- Candidate peak allocated: 6,475,694,592 bytes (~6.03 GiB).
- One formal job 8547392.0, W&B 45674cbf. Pure training estimate ~3.72 hours,
  excluding evaluation, environment setup, queueing and checkpoint transport.

The first validation attempt exposed a preexisting static-graph/cache-position
incompatibility. Cached calls now use the ordinary block entry point while
training retains compiled blocks. The same GPU allocation completed the fix,
revalidation and formal launch; no replacement job was submitted.
