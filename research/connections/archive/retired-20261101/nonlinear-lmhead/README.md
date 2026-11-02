# Nonlinear LM-head experiments

The user authorized replacing the two channel-partition/expanded-head runs with
two fresh 10B-token runs. Both use the ordinary full-attention vanilla backbone,
not mHC. The user explicitly confirmed that h is the original LM-head input,
after the final RMSNorm.

- `nonlinear_lmhead`: features = SiLU(W_g h) * (W_u h).
- `nonlinear_residual_lmhead`: features = h + SiLU(W_g norm(h)) * (W_u norm(h)).

Both then use the existing W_vocab D->V matrix, softcap15, and one global NTP
cross-entropy. Gate/up are bias-free D->D matrices. There is no alpha, additional
down projection, or normalization after the nonlinear transform/addition. The
residual branch uses nanochat's existing parameter-free RMSNorm, including its
dtype-dependent epsilon. It may still learn a small contribution; deleting alpha
does not mathematically prevent that.

The two models have 59,244,544 parameters each, versus 58,720,256 for the linear
reference. Shared parameters and RNG state match the old vanilla initialization
exactly. Both new gate/up pairs start identically, with variance 1/D; they are not
zero-initialized. This does not match the three variants' initial feature or logit
variance. New matrices use the existing Muon group; W_vocab retains the original
unembedding AdamW settings. No LR tuning or extra formal control is included.

Each formal run uses seed42, 305176 steps (10,000,007,168 model tokens), batch32768,
microbatch16/accum1, D512, 8 layers, 4 heads, context2048 and vocabulary32768. The
existing FineWeb-Edu packed cache, tokenizer, dev/test, data order, cuDNN attention,
BF16/FP32 master weights and full10B LR schedule are unchanged. The ordinary linear
vanilla run is the historical reference; the GPU gate also measures its throughput
on the same device as the two new heads. Native/mHC results are separate comparisons.

Before training, the GPU gate checks 15 cases: independent FP64 formula and all
input/parameter derivatives, finite differences, no-alpha/residual behavior,
token isolation, exact shared initialization and optimizer ownership, BF16 CCE
against precision-matched dense logits with independent FP32 gradient checks,
real2K causality, KV cache, compiled/eager and recomputed gradients, old linear
behavior and optimizer/RNG checkpoint restoration. Each new mode then runs
25 steps, exits85, and restores in a fresh process to50 using the full305176-step
LR horizon. This is a diagnostic, not a separate1B training run.

Pinned CCE rounds logits to the feature dtype before FP32 softcap/LSE. Its forward
oracle uses that precision contract; the gradient oracle retains FP32 projection
arithmetic. No tolerance is changed in response to a failed run. Reports and
profiles are local JSON, with no W&B run or credentials. Gate checkpoints stay in
worker scratch. Formal training alone logs to `yibozhong657-none/nanochat`, group
`nonlinear-lmhead-10b-20261101`, with the exact head name in the run name/config.

`submit.py WORKFLOW_DIR` snapshots immutable code and inputs, submits two held
formal jobs and one gate. `watch.py WORKFLOW_DIR` audits the gate, publishes the
reviewed implementation/report and releases the existing two jobs; failures keep
them held. STOPPED.json and process locks prevent accidental continuation or
duplicates. Public GPU slots have a soft preference; backfill remains allowed.
The initial hardware gate uses the established Hopper runtime and >=22GB memory
budget. The head math itself does not require Hopper-specific mHC kernels.

Latest checkpoint overwrites every500 steps; dev-best is separate. Every1800
seconds exit85 transfers the two atomic archives to scheduler storage. Resume
restores the same model, optimizer, RNG, data position and W&B ID. No numbered
resume checkpoints accumulate. Final/dev-best tests use 2,097,152 tokens.

The pre-softcap Jacobian is W_vocab times the feature-transform Jacobian: its
pointwise rank remains at most D and the vocabulary column space remains fixed.
These runs test whether input-dependent nonlinear gradient routing improves LM
training; they do not establish removal of the D-dimensional bottleneck.

Stopped runs: channel_partition7647605.0/4f20c89a and dense_expanded7647606.0/e704b48e,
both scheduler JobStatus3/exit143. Durable snapshots at steps16794 and9500 plus
metadata remain under the previous workflow's `stop-nonlinear-20261101/`; W&B
records were not deleted. Neither run nor its monitor should be resumed.
