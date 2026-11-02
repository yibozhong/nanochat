# Full-attention context-length performance

Bounded single-GPU comparison of 2K/4K/8K/16K vanilla and NVIDIA mHC.
The model is the retained 8-layer D512 full-attention architecture. Every update
uses 32768 tokens; the desired microbatches are16/8/4/2 with no accumulation.
No native SSSL or no-Sinkhorn claim is made by this measurement.

Each configuration first executes four capacity-probe updates in a fresh
process. If either model exceeds90% of the22000MiB minimum-GPU budget,
both use the next smaller common microbatch with accumulation. If no setting
fits, report the limitation without enabling checkpointing or changing width.
Two fresh-process repetitions use20 warmup and40 timed complete optimizer
updates, reversing length/model order on the second repetition. Timing includes
host data copies, CCE, backward, finite checks, and the existing MuonAdamW step.
Setup and compilation are excluded; all timing samples are retained.

Use the existing1B FineWeb cache read-only. Temporary re-segmentation changes
example boundaries and does not recreate original long documents. This is a
performance measurement, not an evaluation of loss or long-context quality.
Source/data hashes, same-device UUID, finite updates, adjacent targets and
compiled full-context future-token/batch isolation are checked. Only JSON/logs
return; no W&B credentials, runs or model checkpoints are produced.

The formal trainer and its2K default are unchanged. Results will be saved in
`results.json` after the worker finishes and the monitor audits the report.
