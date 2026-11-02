# Retired residual and output-head experiments

The user ended these routes on 2026-11-01. Reports/protocols are historical;
their implementation, CLI options, tests, submission scripts, temporary
workspaces and checkpoints have been removed. No archived entry here is an
instruction to restart an experiment. Git history retains the old implementation.

- DFWR: first r32 GPU gate passed 12/13 checks; compiled/eager gradient relative
  difference was 0.2708109 against a 0.03 limit. No formal training began.
  Formal job 7662009.0 and isolation job 7666445.0 were cancelled by the user.
  The compile discrepancy was not resolved before this route ended.
- Nonlinear heads: both completed 10B tokens. Test CE was 3.2025114 for
  `nonlinear_lmhead` and 3.1757791 for `nonlinear_residual_lmhead`, compared with
  the existing vanilla control's 3.1600229; neither improved that result.
- Vocab/channel partition, expanded head and final stream mixing: stopped
  before completion. Partial curves and diagnostics are not final 10B results.
- Rank/mixing diagnostics: retain their progress-matching limitations and
  distinctions between matrix rank, covariance effective rank and Jacobian rank.

Online W&B history was not deleted. Additional local reports, cancellation
records and the exact file cleanup manifest are retained in
`/home/yzhong239/research-cleanup-20261101`. Core native/vanilla/mHC/no-Sinkhorn
checkpoints and reports remain in their original completed workflow directories.
