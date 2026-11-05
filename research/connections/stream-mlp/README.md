# Shared stream MLP mixing

The optional vanilla `mhc_stream_mlp` variant replaces only residual mixing
with a bias-free4->16->ReLU->4 MLP. For every token and channel, the four
stream values are its input. Weights are shared across channels and tokens;
each attention/MLP connector owns an independent pair of matrices. The stored
layout is already `[B,T,d,4]`, so no physical transpose is needed.

Original NVIDIA H_pre/H_post, branch RMSNorm, FA3 attention, backbone ReLU²
MLP and final stream collapse remain unchanged. No Sinkhorn is invoked.
There is no added mixing norm, bias, gate, runtime alpha, identity bypass or
auxiliary loss. Fused output is `MLP(old_streams) + H_post * branch_output`.

Both matrices are initialized nonzero: Win=[I;-I;Q;-Q] and
Wout=0.5[I,-I,Q.T,-Q.T], where Q is an orthogonal signed Hadamard basis.
Thus the MLP is initially identity using ReLU(x)-ReLU(-x)=x. The two bases
avoid identical duplicated units; Q has no zero row sums so initially
replicated streams can exercise all16 units across positive/negative tokens.
The0.5 is only an initial parameter value, not a fixed runtime coefficient.
This initialization is not a guarantee of stability after training updates.

Triton fuses the MLP and branch write, with a manually derived backward.
The16-dimensional intermediate is recomputed and never saved as a bank.
Activations are BF16; tiny mixer parameters/arithmetic/reductions use FP32
like current routing. New parameters stay in the connector AdamW group;
the original optimizer hyperparameters and backbone Muon groups are unchanged.
The8-layer/D512 model has58,992,805 parameters, including128 mixer parameters
per connector. Each mixer costs65,536 forward MAC/token excluding the branch
write; fewer parameters alone do not establish a throughput advantage.

The18-test GPU gate covers independent FP64 forward/all-gradient references,
zero/tiny/ragged inputs, identity initialization and derivative away from
ReLU kinks, channel sharing, token/channel locality, absence of saved hidden
banks, matched backbone/routing/optimizer,2K causality, cache/CCE/compile/
recompute, exact optimizer/RNG resume, and retained-model regressions.
Same-device profiling compares mHC, current channel-local mixing and the
new model; fused/compiled-PyTorch microbenchmarks are recorded separately.
Real25->exit85->fresh-process128 training uses the full10B schedule and
records initial/trained2K activation and gradient scales. No diagnostic
W&B records, credentials or checkpoints are returned.

After validation and push, one formal10B run uses the existing2K FineWeb-Edu
cache, seed42, micro16/accum1 and the original FA3/optimizer schedule.
Latest checkpoints overwrite every500 steps; best is separate; exit85
persists archives every1800 seconds. Formal logging uses nanochat/group
stream-mlp-10b-20261105. Monitoring polls every300 seconds and ends at the
first finite formal step. Existing channel-local training is untouched.
