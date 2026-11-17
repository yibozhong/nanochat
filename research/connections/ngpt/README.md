# Pure nGPT-inspired residual, 10B tokens

One run authorized on 2026-11-17, with no Heavy-Ball or velocity embedding.
Borrow the scalar normalized LERP from [NVIDIA/ngpt](https://github.com/NVIDIA/ngpt/blob/main/model.py):

```
h0 = normalize_RMS(embedding[token])
z = F(h) / max(RMS(F(h)), 0.1)
h_next = normalize_RMS(h + alpha*(z-h))
```

Each Attention/FFN site has one learned positive scalar alpha (softplus,
initialized 0.05). One FP32 hidden stream, no velocity state or momentum
coefficients. Same branch and hidden normalization floors as the validated
hybrid; the 0.1 branch floor keeps original zero output-projection init finite.
No tangent projection or transport. BF16 original branches, final norm and NTP
head, original pre-norm/Attention/FFN, MuonAdamW and schedule are retained.
No weight normalization or other full-nGPT changes. The mode is `--residual-dynamics ngpt`.

Recipe: 8 layers, width512, 4 heads, vocabulary32768, context2048, batch32768
tokens, microbatch16, seed42, existing FineWeb-Edu sample10BT packed2K cache.
305176 steps = 10000007168 tokens, 58720272 parameters (16 extra scalars).
Only this formal run creates W&B. The old five runs remain stopped.

The same GPU allocation first checks independent nGPT scalar equivalence,
first/second derivatives, seed-matched backbone, gradients/compile/checkpoint,
causality/batch independence/KV cache, 32 layers, and exact optimizer/RNG/data
resume. A 128-update pure-mode short test and 32 vanilla timing updates use
separate diagnostic processes/weights/logs. No diagnostic W&B or checkpoint
enters formal training. Validation failure stops before training and retains
900 seconds for debugging. A successful gate starts a fresh seed42 process.

H100/H200 preferred; after 3600 continuously idle seconds allow the compatible
A100 FA2/CCE backend and run the same gate there. No Blackwell eligibility.
Durable latest/best checkpoint transfers every1800 seconds via exit85 preserve
optimizer/RNG/step/data position and W&B ID. Archive failure cannot report85.
`watch_train.py` only schedules eligible hardware and collects results; never
submits/retries/releases failed jobs. STOPPED/STOPPED.json terminates it.

`submit_train.py` uses one fixed run root, file lock, queue key and submission
intent/state checks, submits held, audits inputs/resources/transport, then
releases the job. GPU validation results are produced at runtime, not claimed
by this source. Small validation reports, CE, per-site alpha, throughput,
memory and theoretical/partial profiler FLOPs are retained separately.
