# Commit notes (for the unified-sweep-runner work)

Notes to fold into commit messages when this work is committed. Recorded here
because git is being managed manually and these facts would otherwise be lost.

## Commit 1 — resource single-source-of-truth + preflight

- Introduces `experiments/sweep_resources.yaml` as the sole home for CPU/GPU
  allocation; `experiments/resources.py` (no fallback defaults — missing/incomplete
  config raises) and `experiments/sweep_preflight.py` (GPU-capacity/VRAM/disk
  asserts, unit-testable via injected GPU count/VRAM). Entry points rewired to
  source resources centrally; `tests/` guards against resource literals reappearing.

- **`models/ffd/run.py` and `models/bert_fraud/run.py` have no clean commit-1-only
  diff.** Their diffs also subsume the earlier, never-committed CUDA-fallback fix
  (the `if not torch.cuda.is_available(): num_gpus_per_client = 0.0` block), which
  commit 1 replaced with the central `resources.for_model(..., gpu_available=...)`
  resolution. A bisect that lands here should be aware both concerns are entangled
  in these two files.

- FedXGBllr XGBoost `device`/`tree_method` are now explicit in
  `conf/base.yaml` (`tree_method: hist`, `device: cpu` placeholder); the runner
  overrides `device` from the central config at launch (commit 3). This closes the
  dormant "XGBoost auto-detects device outside the runner" path that the
  resource-literal regression scan would not have caught (device is not in its
  keyword list).

## Commit 4b — FedXGBllr cache routing (two incidental bugs fixed at the source)

Both were found while routing FedXGBllr through the shared cache; each would have
affected the sweep silently.

- **`+XGBoost.device` Hydra append conflict.** Commit 1 added an explicit
  `device: cpu` placeholder to `fedxgbllr/conf/base.yaml`. The sweep runner passed
  `+XGBoost.device=<dev>` (Hydra *append*), which then crashed with "Could not
  append to config. An item is already at 'XGBoost.device'." Changed the runner to
  a plain override (`XGBoost.device=<dev>`). Would have broken EVERY FedXGBllr
  sweep run.

- **cwd-relative results path.** `evaluation/results_writer._logs_dir` used a
  cwd-relative `results/logs`. FedXGBllr runs with cwd=`models/fedxgbllr` (Hydra
  needs its conf/), so its results were written to `models/fedxgbllr/results/logs/`
  — invisible to the collector and verify_hashes, which walk the repo `results/logs/`.
  This is the SAME root cause as the earlier quarantined confounded ULB FedXGBllr
  file (`results/_quarantine/`). Now fixed at the source: `_logs_dir` anchors at the
  repo root (computed from the module location), so all six models write to one
  tree regardless of cwd — rather than handling scattered files case by case.
