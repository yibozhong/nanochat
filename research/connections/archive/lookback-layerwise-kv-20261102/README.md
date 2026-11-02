# Retired per-layer memory Lookback

The user replaced per-layer historical KV with one shared projection pair of
completed/refined final hidden, while retaining local self-attention. The old
final-hidden-input implementation is in Git38b0c20; gate7675394.0 is historical
validation only. Formal7675499.0 / W&B d766589a was cancelled. The most recent
durable checkpoint (step7039,230653952tokens) and results remain in the stopped
workflow. It is not a10B result or a cancellation-time checkpoint. W&B and logs
remain. The completed vanilla16K baseline7669762.0 is retained unchanged.
