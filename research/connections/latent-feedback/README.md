# Detached latent feedback

Port of [Latent Feedback Is a Compute and Data Amplifier](https://www.alphaxiv.org/abs/2611.latent-feedback-transformers-scaling-law),
using the [authors' implementation](https://github.com/ifm-ai/LoRM/blob/main/lorm/model/feedback.py).

Enable with `scripts.connection_train --variant baseline --latent-feedback`.
Without the flag, the original baseline remains unchanged. The port retains
nanochat's normalized input embeddings, ReLU-squared FFN, attention, softcapped
LM head, optimizer settings, and packed FineWeb-Edu data.

- Feedback: `input[t] = normalized_embedding[t] + W RMSNorm(top[t-1])`.
  The first token has no feedback. W is full width, bias free, initialized
  with standard deviation .02; feedback RMSNorm epsilon is 1e-5.
- One seed/step draw per optimizer update: 15% P1, 85% P2, shared across
  accumulation microbatches. Previous passes run without gradients and their
  hidden states are detached. Only the final pass evaluates the LM head/CE.
- W uses a separate conditional Muon group. P1 skips its momentum and decay.
- Default evaluation is P4; `dev/ce` and `dev/ce_p4` report this value, with
  `dev/ce_p1` reported separately. `--feedback-eval-passes` changes the default.
- `FeedbackKVCache` stores the previous raw top state alongside KV. Cache calls
  use exact sequential recurrence, including token-by-token prompt prefill.
  Parallel P4 is a finite-pass approximation. Efficient parallel prefill is
  outside this initial correctness port.
- Existing compiled blocks, FA2 and Apple CCE are reused. The feedback mix is
  compiled separately. The preliminary pass does not retain a backward graph.

The bounded debug job runs GPU correctness/gradient/cache tests, 96 updates
per model at 8 layers / D512 / T2048 / batch 32768 on one GPU, and an actual
CLI exit85/resume comparison. Checkpoint tensors, optimizer/RNG/data position,
and the next fixed-gradient optimizer update must match exactly. Independent
FA2/CCE backward runs have floating-point reduction nondeterminism; their
trajectory differences are recorded separately. Data are the existing 128-step packed prefix.
No W&B run or formal 10B training is created. Timing excludes 16 warmup updates;
baseline and feedback alternate on the same card. Both model/optimizer states
are resident, so the reported peak memory is combined.

References: LoRM `lorm/model/feedback.py`, `lorm/train/algo.py`,
`works/latent_feedback/configs/feedback.yaml`. Its mechanism reuses deep
features forward; gradients do not cross the preliminary pass.

The separately authorized 10B run uses `submit_train.py` and `train.sh`:
305,176 updates, batch 32,768, microbatch 16, seed 42, 15% P1 / 85% P2,
P4 evaluation plus P1 dev logging. It starts from fresh weights, uses online
W&B, and preserves its run ID/model/optimizer/RNG/data position across
1800-second exit85 checkpoint transfers. Archive failures never return85.
H100/H200 reuse the completed validation. After one continuous idle hour,
the scheduler may include A100 (existing FA2/CCE backend), requiring the
feedback checks in that allocation before independently initializing training.
Failures hold the job; the watcher collects results without resubmitting.
