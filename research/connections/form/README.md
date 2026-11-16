# Form: gradient-guided FFN structural reallocation

`scripts.connection_train --form taylor` enables Form on the existing vanilla,
single-pass baseline. `--form none` is the default. `--form random` provides a
random-reallocation control with the same schedule, budget, minimum and growth
rounding. Only the Taylor run is authorized for the current 10B submission.

The FFN remains `down(relu(up(x))²)`. Differentiable virtual gates between the
activation and down projection remain one and are buffers, not optimizer
parameters. The custom backward reduces gate gradients in FP32. The existing
training backward supplies all statistics; there are no candidate channels,
extra scoring passes, reconstruction, compensation, or auxiliary losses.

Each effective batch contributes its mean NTP gate gradient once, after normal
microbatch accumulation. The window statistics are uncorrected EMAs with decay
0.99 by default: `importance = -EMA(g) + 0.5 * lambda * EMA(g²)`, lambda 1.0.
This is the second moment of the batch gradient, not a per-token Fisher estimate.
Both decay and lambda are CLI/config options. Every Form resets all layer EMAs.

After updates `floor(k*T/11)`, k=1..10, Form selects the globally least important
channels, subject to each layer retaining at least hidden_dim channels. From
the same pre-Form scores it selects the top-Q retained channels. Their counts
per layer determine growth, rounded down to multiples of 64 and then completed
by largest remainders (ties by layer). Pruning remains individual-channel;
resulting layer widths need not be multiples of 64. There is no per-layer upper
bound. The random control shuffles the same eligible existing-channel pools
independently for pruning and growth using its own saved RNG.

Q is fixed from the initial widths: round `(sum(widths)-L*hidden_dim)/10` to the
nearest multiple of 64, with ties upward. The authorized d8/D512 model starts
with 2048 channels/layer, has minimum 512, and reallocates exactly 1216 channels
at each operation. Total width stays 16384 and model parameters stay 58,720,256.

Surgery physically slices input rows and output columns. It keeps retained
weights and the existing optimizer slots, samples new input rows from the
original uniform initialization, and zeros new output columns and new optimizer
slots. A separate saved Torch generator supplies new weights. The existing
Muon/AdamW optimizer, learning-rate schedule, update count and data position are
preserved. Muon slots are unpacked and regrouped by actual matrix shape; down
projection variance remains indexed by intermediate channel even at the square
minimum-width boundary. AdamW embedding/head slots and step counters are unchanged.

Checkpoints contain the actual widths, EMA buffers, window count, completed
Forms, history and independent RNGs. Loading builds the checkpoint shapes
before loading tensors and optimizer state. Dev-best evaluation builds that
checkpoint's architecture independently. Existing checkpoints default to Form
disabled. The packed trainer remains single-GPU; no distributed Form surgery
is claimed.

Metrics include validation CE, width/prune/growth counts, per-layer score
summaries, parameters, analytic FLOPs, step throughput, structural-update time
and the first step after each Form (including lazy recompilation). Detailed
scores and selected indices are local JSON. Loss changes use adjacent normal
training batches and are labeled accordingly; no additional loss probes run.
Steady throughput excludes the first step after surgery; total training time
includes it. Same-card preflight measures gate/surgery overhead against vanilla.

The worker runs 38 unit/regression checks, 256 updates each of vanilla and Form
at production dimensions (two structural operations), and a real exit85/resume
across the first Form. Tiny-model checks exercise all ten windows, minimum-width
state migration, function preservation after growth, causal behavior, compiled
gradients and exact fixed-gradient next-update recovery. The gate is limited to
2400 seconds and writes local results only. On success a separate process starts
the authorized 305176-update / 10,000,007,168-token training from seed 42 in the
same allocation. Failure stops training and retains the allocation for 900 seconds.

The submission script audits a held job, uses the existing packed2K cache and
frozen runtime, and checkpoints durably with exit85 every 1800 seconds. The
watcher changes only the GPU eligibility after a continuous idle hour, collects
small results, and never resubmits a failed run. Original baseline reference:
`6a07f085` (its historical cuDNN timing is not used for the same-card comparison).
