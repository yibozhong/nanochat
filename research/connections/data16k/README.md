# Packed 16K FineWeb-Edu data

The 16K token archives were deleted on 2026-11-02 at the user's request. The
default is now 2K; see `../data2k/README.md`. The information below documents
the former cache and its reproducible packing tools, not an available archive.

These shared data tools support native nanochat, full-attention vanilla and
NVIDIA mHC. `prepare_data.py` and `prepare_data.sh` were moved here without
changing packing, tokenization or split selection.

The validated cache contains 305176 updates of 32768 tokens, with 16384-token
contexts. It uses FineWeb-Edu sample-10BT shards 000–012 for training and shard
013 for held-out data, with the existing RustBPE 32K tokenizer. Dev and test
contain 8 and 128 packed rows respectively. Training reads into source epoch 2;
the budget does not mean ten billion unique source tokens.

- Archive: `osdf:///chtc/staging/y/yzhong239/nanochat-16k-10b-data-7667953.tar`
- SHA-256: `d84741d2a06d99d1fd07e33c42ff7e307feba62d4a64166301246ac450eb311c`
- Packing verification: `data-validation.json`.

The retained trainer accepts `--sequence-length 16384`; use a global batch of
32768 tokens and the validated microbatch for the selected architecture.
Data preparation and GPU profiling belong in HTCondor jobs. Historical completed
experiment inputs remain immutable and are not rewritten to the new source path.
