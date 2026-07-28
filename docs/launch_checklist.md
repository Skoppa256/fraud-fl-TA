# Sweep launch checklist (GPU box)

## Pre-launch gates (Part 8)

- [x] **Gate 1 — full `--dry-run` (all 3 datasets):** 108 planned / 96 to-run / 12
  skipped. No-ops are exactly BAF centralized + BAF iid (zero noniid → α=0.5
  ablation intact). Sub-6 clients only in PaySim noniid α=0.5 (sub6=2; census
  minimum n_minority=2 is PaySim α=0.5). Disk estimate ~2.3 GiB. Under
  `--gpu-fraction 1.0`, GPU models = 1 concurrent actor (fully sequential).
- [x] **Gate 2 — all six centralized executed:** lr/svm/gbm/ffd/xgb verified —
  complete rows (no empty columns), loadable artifacts, threshold+policy,
  calibration (SVM=NA), baseline_auprc. FedXGBllr routes to `run_xgb`, labelled
  `n/a (centralized); XGBoost (centralized upper bound)`. BERT centralized path
  confirmed (cache-consumed, model built, training) but CPU-slow to finish; its
  persist/calibration/hash wiring is byte-identical to FL BERT (verified
  end-to-end). The GPU box closes this.
- [ ] **Timing probe (THIS BOX'S JOB).** See priority below.

## Timing-probe priority — BERT-on-PaySim (NOT a footnote)

BERT is the pacing risk: 12 BERT runs in the matrix, and PaySim is 6.36M rows.
If 285k ULB rows block ~30–60 min on CPU, PaySim decides whether the sweep fits
the overnight window. Probe **BERT-on-PaySim federated first, then centralized**:

```bash
# 1) BERT on PaySim, FEDERATED (Non-IID α=0.5) — the dominant per-run cost
python experiments/run_sweep.py --timing-probe \
  --datasets paysim --models bert_fraud --conditions noniid --smote-arms no-smote

# 2) BERT on PaySim, CENTRALIZED
python experiments/run_sweep.py --timing-probe \
  --datasets paysim --models bert_fraud --conditions centralized --smote-arms no-smote
```

The probe reports total + **per-round** wall time (so extrapolation survives a
round-count change) + peak GPU memory + rounds_completed, and exits without
writing results. Multiply the per-run cost across the 96 runs (BERT + FedXGBllr
dominate) to check the window; if it doesn't fit, options are subsampling PaySim,
reducing global rounds, or splitting the sweep across nights.

## Launch (after the probe confirms the window)

Start under a session that survives an SSH drop:

```bash
tmux new -s sweep 'python experiments/run_sweep.py \
  --datasets paysim,creditcard,baf --models all \
  --conditions centralized,iid,noniid --alpha 0.5 --seeds 42 \
  --smote-arms both --gpu-fraction 1.0 --offline 2>&1 | tee results/sweep.out'
```

`--offline` keeps wandb local (a connectivity loss won't kill a run); `wandb sync`
the offline dir afterward. Use `--resume` to restart where a crash stopped.
Post-sweep: `python -m experiments.verify_hashes` (expect zero UNVERIFIED) and
`python -m experiments.collect_results`.
