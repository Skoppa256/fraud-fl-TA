# RQ3 SHAP v2 — run record (`experiments/shap_rq3.py`)

**Status: complete and frozen.** This file was a forward-looking playbook while
the run was pending; it is now a record of what was actually executed and what
came out. `results/shap_v2/**` is a frozen artifact (see the freeze list in
`CLAUDE.md`) — the commands below are documented so the run can be *read*, not
so it can be repeated. Re-running any of them is an explicit decision by the
author, never a step inside another task.

Everything ran on the GPU box in tmux session `TA_Deo`, under **shap 0.49.1**
(now pinned exactly in `requirements.txt`). The runner is sequential (no Ray —
the FedXGBllr OOM history is why), reads models/cache read-only, and writes
**only under `results/shap_v2/`**. `results/shap/` (v1, the pre-fix record) was
deliberately never touched, so every number that moved can be shown next to its
old value.

## What is on disk

`results/shap_v2/shap_summary_v2.csv` — **124 cells, every one `status = ok`**:

| stage | cells | explainer tier |
|---|---|---|
| `--stage main` | 96 | 33 multi-client KernelSHAP (`bg=local`) + 53 Linear/Tree + 10 single-client kernel |
| `--stage twobg` | 12 | shared-background arm, PaySim |
| `--stage paysim-exact` | 16 | `kernel-exact-grouped` (12 of them multi-client) |
| `--stage budget` | — | `budget_probe.csv` only, no cells |

Logs: `rq3_main.log`, `rq3_twobg.log`, `rq3_exact.log`.

Three tiers, and they carry different claims:

- **Exact by construction** — LinearSHAP (LR, SVM) and interventional TreeSHAP
  (GBM; XGB centralized only). Zero estimator noise; this is the calibration
  anchor the sampled tier is read against.
- **Sampled** — KernelSHAP at `nsamples = 500`, `l1_reg=False`, for FFD, BERT
  and FedXGBllr. Carries a noise floor **measured per client per cell**, not
  broadcast from one number.
- **Exact-grouped** — the same three kernel models on PaySim with the five
  `type_*` one-hots collapsed into one Shapley player: `M_eff = 9`, so
  `nsamples = 2^9 - 2 = 510` enumerates 100% of the kernel weight. Exact, and
  verified so: `seed_delta_max = 0.0` on all 16. **This tier is never an
  anchor** — it is the audited models re-run without sampling, so comparing the
  sampled tier against it could not fail.

## Stage 1 — main (`--stage main`)

```bash
python experiments/shap_rq3.py --stage main 2>&1 | tee results/shap_v2/main_run.log
```

Outcome: 96 cells, no errors, no `undefined`. The headline result is that
**29 of the 33 multi-client kernel cells are distinguishable from the
explainer's own floor** (exact exchangeability test over all 945 perfect
matchings at K = 5, BH-adjusted). Per model, weighted-between medians against
their own floors: FFD 0.963 (floor 0.9997, 11/11) · BERT 0.952 (0.9993, 10/11)
· FedXGBllr 0.958 (0.9988, 8/11).

For the side-by-side with v1: the old floors — one BAF-measured number
broadcast to every cell, under the `l1_reg` default — were FedXGBllr 0.9730,
BERT 0.9972, FFD 0.9966. The re-measured floors are all ≥ 0.9726 weighted and
now come as 5 per cell.

## Stage 2 — two-background arm (`--stage twobg`)

```bash
python experiments/shap_rq3.py --stage twobg --conditions dirichlet,iid \
    2>&1 | tee -a results/shap_v2/twobg.log
```

The **first** twobg implementation was invalid and its output was deleted before
the re-run: it pooled the backgrounds while keeping the *shared* explained rows,
so every client received byte-identical inputs, the client axis collapsed, and
`between` was 1.0 by construction. Fixed by giving each client its own explained
rows; a permanent guard now marks any collapsed cell `undefined`. No cell in the
committed tree is `undefined`.

**What the two arms measure.** They are not a one-factor contrast; each holds
one input fixed and varies the other, isolating one of two confounded mechanisms:

| arm | background | explained rows | isolates |
|---|---|---|---|
| `bg=local` (main) | per-client | shared central-test subset | each client's **background distribution** |
| `bg=shared` (twobg) | pooled over all clients | each client's **own partition** | the **data region** each client occupies |

Both arms show IID more stable than Dirichlet under a one-sided Wilcoxon
signed-rank test on the per-cell `weighted_between`: channel 1 (per-client
background, shared rows), paired within (dataset, model, arm), n = 15 → 11/15,
p = 0.0042, mean paired difference +0.054; channel 2 (pooled background,
per-client rows), paired within (model, arm), PaySim only, n = 6 → 5/6,
p = 0.046875, mean +0.073.

**Magnitudes are not comparable between the channels.** The arms differ in
background source, instance source, and train-vs-test rows: client partitions
cover `x_train` only, so the shared arm explains training rows while the main
arm explains held-out test rows. SHAP explains the fitted function rather than
generalization, so this does not invalidate either channel, but it does forbid
reading one channel's effect size against the other's.

## Stage 3 — PaySim exact tier (`--stage paysim-exact`)

```bash
python experiments/shap_rq3.py --stage paysim-exact     # grouped, 9 players, ns=510
```

Outcome: 16 cells, `seed_delta_max = 0.0` on every one. On the 6 IID/Dirichlet
pairs, all 6 Dirichlet cells sit below their IID counterpart (p = 1/64 =
0.0156), and the raw `max |client₀ − client₁|` attribution gap is 2.6× – 19.5×
larger under Dirichlet.

This tier also bounds the sampled tier's bias directly: over the 12 PaySim cells
that exist in both tiers, mean |exact − sampled| = 0.028, max 0.099 (7 down,
4 up, 1 tie). It produced exactly **one significance flip** — `bert/iid/none`,
p = 0.001 sampled → 0.111 exact: a false positive in the sampled tier that only
the exact tier could catch.

### `--include-ungrouped` was deliberately not run

The ungrouped exact variant (13 players, `nsamples = 2^13 - 2 = 8190`,
≈ 16.4× the per-instance cost — FedXGBllr ≈ 17 h/cell) is gated behind
`--include-ungrouped` and **was never executed**. The grouped tier already
confirmed the ranking it was meant to check, so the additional GPU time would
have bought no claim the thesis makes. This is a decision, not an omission: do
not run it to "complete" the grid.

## Stage 4 — budget probe (`--stage budget`)

```bash
python experiments/shap_rq3.py --stage budget    # cell baf/ffd/dirichlet/none
```

~15 min; wrote `results/shap_v2/budget_probe.csv` (two-seed floor at fixed eval
budget across nsamples/N_explain splits). It informed where additional GPU time
would go; no thesis claim rests on it.

## The exactness guard and its acceptance criterion

`process_cell()` in `experiments/shap_rq3.py` carries two guards that abort the
cell rather than record a suspect number:

1. **`l1_reg` regression guard** — for any kernel cell with M > 10 features and
   `groups is None`, a `max_nonzero_per_row` in `[0, 10]` raises. That is the
   LARS keep-10 signature; if it ever reappears the run stops instead of
   producing v1's numbers again.
2. **Exactness guard** — any cell claimed deterministic must satisfy
   `max |Δφ|` across the two coalition seeds **exactly 0.0**. Not "small", not
   "within tolerance": exactly zero, or the cell aborts. This is the acceptance
   criterion for the Linear/Tree tier and for the exact-grouped tier, and it is
   enforced in code rather than checked by hand.

There is also a width check on the grouped path: the one-hot grouping API
(`DenseData` groups) is verified on shap 0.51.0, so the runner width-checks the
output on 0.49.1 and aborts loudly if the grouping is not honoured.

Independent post-hoc audit: `python analysis/verify_kernel_tier.py` — six legs,
each printing PASS/FAIL on its own, so a reader can reject any single leg and
see what survives. All six PASS on the committed tree.

## What must never be re-run without an explicit decision

- **`--stage main`** — the 96 cells §4.3 reports.
- **`--stage twobg`** — the 12 shared-background cells; note that the `bg=local`
  rows must not be re-run by a twobg pass either.
- **`--stage paysim-exact`** — the 16 exact-grouped cells, including the
  false-positive catch.

`--stage budget` carries no thesis claim and would be cheap to redo, but it
still writes under a frozen path.

## Resume / skip-existing behaviour (unchanged, still documented)

Every stage resumes. Re-running the same command **skips any cell that already
has a `stability.json`** (printing `skip (exists) <path>`), and the summary CSV
is checkpointed after every cell, so a run is safe to kill and relaunch.
`--no-skip-existing` forces recomputation — which on a frozen tree means
overwriting results of record, so it needs the same explicit decision as a full
re-run.

The summary merge is keyed on dataset/model/condition/arm/bg/explainer, so a
re-run overwrites its own rows in place rather than appending duplicates; the
CSV never needs manual editing. `p_adj` and `verdict` are recomputed on every
summary write, so a partial run shows visible but non-final verdicts — the BH
pass needs all 33 multi-client kernel cells before they are final.

## Outputs schema (`results/shap_v2/shap_summary_v2.csv`)

One row per (dataset, model, condition, arm, bg, explainer):
`floor_mean`/`floor_min` (per-client two-seed floors), `between_mean`/
`between_sd` (20 CRN same-seed cross-client ρ), `delta`, `p_value` (exact
exchangeability test, 945 matchings), `p_adj` (BH per bg×explainer family),
`disattenuated` (secondary, flagged approximate), `weighted_between`/
`weighted_within` (magnitude-weighted rank correlation — this is the statistic
§4.3 reports), `max_nonzero_per_row` (l1 regression guard), `seed_delta_max`
(deterministic bit-identity), `status` ∈ {ok, undefined}, `verdict` ∈
{clients_differ, noise_limited, exact_reference, single_client, undefined}.

Per cell, under `results/shap_v2/<dataset>/<model>/<condition>/<arm>/`:
`stability.json`, `importance_per_client_seed.csv`, `profile.csv`. The
stability-profile figures and the §4.3 prose are produced laptop-side from these.
