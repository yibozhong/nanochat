# Vanilla SSSL with three SWA passes

Each of the six S layers runs its own attention three times with shared Q/K/V/O,
then runs its MLP once. The two L layers run independent full attention and MLP
once. Every pass normalizes the current residual state, recomputes QKV, and adds
its output. No new parameters, gates, detach, or changes to backbone initialization
or the optimizer. The8-layer model remains58,720,256parameters.

At2K context, S sees512 tokens to the left plus itself. The layer pattern is
SSSLSSSL:18SWA evaluations,2full evaluations,8MLP evaluations. This differs from
the retired2026-11-04 experiment, where every layer ran128/128/full with shared
local/global attention parameters and a single MLP.

The optional configuration is `swa_looping=True`, `swa_loops=3` (total passes),
`swa_window=512` (left context), `window_pattern='SSSL'`, `streams=1`, on vanilla
baseline only. `swa_loops=1` provides the matched non-looping SSSL control.
Pattern tiling always ends in a full-attention layer, like native nanochat.
Disabled defaults preserve all retained computation paths.

Uses the existing native-window FA3 operator and compiles each entire layer;
the three attention evaluations remain serial. Cached decoding is explicitly
rejected because each pass needs its own cache. Current validation is Hopper-only.

The bounded gate runs21checks, six60-step optimizer profiles (full/SSSL/looped
SSSL and reverse), and a25-to128-step NTP restart. No formal10B, W&B, or returned
weights. See `protocol.json`; GPU validation passed on2026-11-08; see `validation.json`.

## GPU result

Job8088582.0 completed on one H200 (FA3, micro16,32Ktokens/step). All21checks
passed. Six independent60-step profiles, averaging the two runs per mode:

| Model | Complete optimizer step | Peak allocated |
|---|---:|---:|
| Full vanilla |60.6725ms|5.375GiB|
| Vanilla SSSL, one pass |59.7622ms|5.375GiB|
| Vanilla SSSL, three SWA passes |72.4958ms|8.394GiB|

Looping is21.307% slower than one-pass SSSL and19.487% slower than full vanilla.
The looped repetitions measured72.6084/72.3831ms. Timings include data copy,
CCE, backward, finite checks, and MuonAdamW, excluding compilation and evaluation.

Real NTP25-to128-step checkpoint restart passed; all steps had finite loss and
gradients. Initial/final dev CE10.398013/5.718440. This is an engineering stability
check, not a matched long-run quality comparison. No formal10B was submitted.
Diagnostic weights were deleted in the worker and none were returned.
