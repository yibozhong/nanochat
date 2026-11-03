# Default 2K FineWeb-Edu cache

The default training context is 2048 tokens. The cache is being rebuilt from
the retained FineWeb-Edu sample-10BT Parquet files with the original RustBPE
32K tokenizer and unchanged native BOS best-fit loader. This restores the
original 2K packing, rather than reshaping the retired 16K examples.

The complete cache contains 305176 updates of 32768 tokens (10,000,007,168
training tokens). Train uses shards 000–012; dev/test use shard 013 with 64/1024
packed rows. All three binary split hashes must match the original 2K cache.
The budget crosses source epochs and is not ten billion unique source tokens.

Data preparation is CPU-only, with no W&B credentials or training jobs. The
workflow also creates a small exact 128-update prefix for future implementation
checks, retaining the same dev/test splits. Full archive hashes are computed
on the worker; archive sizes, members and manifests are checked after transfer,
and the small prefix is fully rehashed on receipt.

Workflow: `/home/yzhong239/nanochat-data-2k-20261102`.

The current readiness record and verified URI are written to
`/home/yzhong239/datasets/fineweb-edu-10bt/current-packed-cache.json`.
Require `ready: true` before using the cache. Existing historical experiment
inputs remain immutable; new jobs must use the current record's versioned URI.

`workflow.py ROOT submit` creates one preparation job. `workflow.py ROOT watch`
collects and validates its output without submitting training. The existing
16K data archive and its local token-cache copies were deleted on request;
raw Parquet files, tokenizer, experiment reports and checkpoints remain.
