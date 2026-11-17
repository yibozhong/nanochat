# Nonzero-initialized vanilla and nGPT residual comparisons

On 2026-11-17 the user stopped the zero-output-init run `8841886.0/c600d3b7`
and authorized three new runs, each 10B tokens:

| Preset | Residual mode | Velocity | Parameters |
|---|---|---|---:|
| vanilla | original additive residual | none | 58720256 |
| ngpt | scalar normalized LERP | none | 58720272 |
| ngpt_momentum | normalized LERP + ambient Heavy-Ball | independent token embedding | 75497520 |

All share depth8/width512/heads4/context2048/vocabulary32768, seed42,
32768 batch tokens, microbatch16, existing FineWeb-Edu sample10BT packed2K,
FA2/CCE, MuonAdamW, and the same schedule. 305176 steps = 10000007168 tokens
per run. The old five sphere/momentum runs and the zero-init nGPT run remain
stopped; they must not be resumed by this launcher.

## Initialization

`--residual-projection-init normal --residual-projection-init-std .02`
changes only the previously zero Attention output and FFN down projections:
normal(mean=0, std=.02/sqrt(2*depth)), hence std=.005 at depth8. All other
backbone weights retain their original random initialization. Default `zero`
remains for legacy configuration/checkpoint compatibility. The three models
have exactly equal backbone tensors under the same seed; the momentum embedding
is created only after the backbone, normal std=.02.

## Residual updates

Borrow the scalar normalized LERP from
[NVIDIA/ngpt](https://github.com/NVIDIA/ngpt/blob/main/model.py):

```
h0 = normalize_RMS(embedding[token])
z = F(h) / max(RMS(F(h)), 1e-6)
h_next = normalize_RMS(h + alpha*(z-h))
```

Every Attention/FFN site has one learned positive scalar alpha, softplus,
initialized .05. Only one FP32 hidden stream in pure nGPT. The 1e-6 branch
floor is numerical protection, replacing the retired run's substantive .1
floor. Initial branch norms and target unit RMS are checked on actual data.
Hidden normalization also uses a numerical RMS floor1e-6.

For `ngpt_heavy_ball`:

```
u = beta*v + gamma*(z-h)
h_next = normalize_RMS(h + eta*u)
v_next = u
```

Defaults beta/gamma/eta=.5/.5/.1, effective alpha=.05. Momentum uses independent
token velocity embeddings. Carry the unscaled ambient velocity without tangent
projection/transport/normalization/detachment. The retained scalar residual
formulas are unchanged; only initialization and the branch floor differ here.

BF16 original branches, pre-norm, Attention/FFN, final norm/head and NTP loss
remain. No weight normalization or other full-nGPT architecture changes.

## Validation and launch

`python research/connections/ngpt/submit_train.py PRESET` submits exactly one
of the three fixed authorized plans. Each has its own run ID/root/lock/state;
held submission checks actual Args, inputs including container URI, resources,
credential encryption and checkpoint transport before release.

The same allocation first runs 29 GPU checks: matched initialization, first-step
inner-branch gradients, initial exact normalized targets, scalar nGPT equivalence
for small nonzero forces, first/second derivatives, compiled/checkpointed
forward/backward, CCE/dense gradients, causal/batch independence/KV decoding,
32 layers and exact optimizer/RNG recovery. All use production nonzero init.

Profiling gives the target128 updates and the other two modes32 each; timing
uses steps17-32 for all, with per-model memory peaks and isolated state. The
actual three-step CLI exit85 recovery checks exact model/optimizer/RNG/data
transport and identical fixed-gradient next updates. Diagnostic compute is
reported separately, no diagnostic W&B or weights enter formal training.
A fresh seed42 process starts formal training only after every check passes.
Gate timeout2400 seconds, failure debugging900 seconds, no blind resubmission.

H100/H200 preferred, eligible A100 after3600 continuously idle seconds, same
source/family gate required. No Blackwell eligibility. Durable latest/best
transfers via exit85 every1800 seconds preserve run ID/optimizer/RNG/step/data
position. Archive failures cannot return85. Scheduler only edits eligible
hardware requirements and collects results; it never releases/retries failed
jobs. STOPPED/STOPPED.json ends the watcher.

The stopped zero-init run summary is
`/home/yzhong239/ngpt-zero-init-stopped-20261117.json`.
