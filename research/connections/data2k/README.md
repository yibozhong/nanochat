# Default 2K FineWeb-Edu cache

The default training context is 2048 tokens. The cache is being rebuilt from
the retained FineWeb-Edu sample-10BT Parquet files with the original RustBPE
32K tokenizer and native BOS best-fit packing rules. This restores the
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

## Verified offline packing acceleration

`fast_pack.py` uses an ordered length index with FIFO ties, preserves native
buffer refill and cropping decisions, reads the text column, and reuses one CPU
row buffer. Native training code is unchanged. `prepare_indexed.py` installs
this offline adapter before running the original preparation and validation
script; final train/dev/test hashes remain mandatory.

CPU job 7685301.0 on one EPYC 7763 host compared two reversed-order rounds,
each with 32 warmup and 512 timed updates. All modes used the same allocation
of 8 CPUs; the indicated thread count controls tokenization. The native loader
produced 1.081M tokens/s; indexed packing with 4 threads produced 1.529M (1.414x),
and 8 threads produced 1.815M (1.679x). Text-column reading alone showed no
speedup. These are short same-host measurements, not full-job completion times.

Each mode matched the original 17.826M-token training prefix, per-update source
states, and complete dev/test hashes. A 5000-operation independent selection
oracle covered equal-length ties, shortest-document cropping and refill contents.
Results and exact source hashes are in `benchmark-results.json`.

For the validated optimized path, use `workflow.py ROOT submit --packing indexed8
--profile-report REPORT`. Submission checks the tested adapter hash and measured
speedup, requests 8 CPUs, and softly prefers the benchmark host. It does not
change the tokenizer, corpus, model, sample order or training budget.
