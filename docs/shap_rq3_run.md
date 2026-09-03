# RQ3 SHAP v2 — box run playbook (`experiments/shap_rq3.py`)

Everything here runs on the GPU box in tmux session `TA_Deo`. The runner is
sequential (no Ray — the FedXGBllr OOM history is why), reads models/cache
read-only, and writes **only under `results/shap_v2/`**. `results/shap/` (v1,
the pre-fix record) is deliberately never touched, so every number that moves
can be shown next to its old value. Do not add the new CSVs to git while a run
is in progress (that has blocked `git pull` twice).

## Step 0 — attestation (one-liners, before anything else)

```bash
# 1. Installed shap version (expect 0.49.1; the pin in requirements.txt now says so)
conda run -n fraud-fl-TA python -c "import shap; print(shap.__version__)"

# 2. Was the version constant across the unlogged bert/gbm repair run?
ls -ld /home/justin/miniconda3/envs/fraud-fl-TA/lib/python3.10/site-packages/shap-*.dist-info

# 3. Recover the repair run's log if it exists (laptop production.log is the
#    pre-fix run only — the current artifacts have no synced provenance)
ls -lat ~/fraud-fl-TA/results/shap/*.log ~/fraud-fl-TA/*.log 2>/dev/null | head
history | grep shap_analysis | tail -20
```

If (1) prints anything other than `0.49.1`, stop and report — the version
audit's premises change.

## Step 1 — main stage (the decision run: Phase 1.1 + 2.1–2.3)

```bash
cd ~/fraud-fl-TA && git pull
tmux attach -t TA_Deo
mkdir -p results/shap_v2          # tee opens the log before Python creates the dir
python experiments/shap_rq3.py --stage main 2>&1 | tee results/shap_v2/main_run.log
```

- **Resume:** rerunning the same command skips every cell that already has a
  `stability.json`; the summary CSV is checkpointed after every cell. Safe to
  kill and relaunch.
- **Wall-clock estimate** (from the v1 noise-floor timings at nsamples=500,
  BAF): FedXGBllr ≈ 0.76 s/instance, BERT ≈ 0.57, FFD ≈ 0.14. Per multi-client
  kernel cell (5 clients × 2 seeds × 500 instances): FedXGBllr ≈ 63 min,
  BERT ≈ 47 min, FFD ≈ 12 min → ≈ 23–25 h for all 43 kernel cells, plus ~1–2 h
  for the deterministic tier. `l1_reg=False` drops the per-call LARS solve, so
  the real rate may be slightly better; the first cells print per-client-seed
  timings — extrapolate from those and use `--max-hours` if needed.
- The default order is v1's (cheap models first, then kernel BAF→ULB→PaySim).
  If you want the decision cells first: `--datasets paysim` then a second launch
  without the filter (resume skips what's done). The BH pass needs all 33
  multi-client kernel cells before verdicts are final — `p_adj`/`verdict` are
  recomputed on every summary write, so partial runs are visible but not final.

**Acceptance checks after the run:**

```bash
grep -c ERROR results/shap_v2/main_run.log            # expect 0
grep "l1_reg signature" results/shap_v2/main_run.log  # expect no hits (guard silent)
awk -F, 'NR>1 && $28=="undefined"' results/shap_v2/shap_summary_v2.csv | wc -l
python - <<'EOF'
import csv
rows = list(csv.DictReader(open("results/shap_v2/shap_summary_v2.csv")))
ker = [r for r in rows if r["explainer"] == "kernel" and r["bg"] == "local"
       and int(r["n_clients"]) > 1]
print(f"{len(ker)} multi-client kernel cells (expect 33)")
for m in ("ffd", "bert_fraud", "fedxgbllr"):
    f = [float(r["floor_mean"]) for r in ker if r["model"] == m]
    print(m, "floor_mean range:", round(min(f), 4), "-", round(max(f), 4),
          "| clients_differ:",
          sum(r["verdict"] == "clients_differ" for r in ker if r["model"] == m), "/",
          len(f))
EOF
```

Old floors for the side-by-side (measured under the l1 default, BAF-only,
seeds 11/22, N=250): FedXGBllr 0.9730, BERT 0.9972, FFD 0.9966. The
deterministic tier must show `seed_delta_max = 0.0` on every cell.

## Step 2 — two-background arm (Phase 2.5, PaySim Non-IID first)

**The first twobg run (before 2026-09-03) was invalid — delete it before re-running.**
It shared the background *and* the explanation set, so every client received
byte-identical inputs and the client axis collapsed: all agreement was 1.0 by
construction. Fixed by giving each client its own explained rows; a permanent
guard now marks any collapsed cell `undefined`.

```bash
rm -rf results/shap_v2/*/*/*/bg_shared          # ONLY bg_shared; leave the local arm alone
find results/shap_v2 -name bg_shared | wc -l    # expect 0
python experiments/shap_rq3.py --stage twobg --conditions dirichlet,iid \
    2>&1 | tee -a results/shap_v2/twobg.log
# default scope: paysim × kernel models; widen with --datasets/--conditions
```

The stale `bg=shared` rows in `shap_summary_v2.csv` are overwritten in place
(the merge is keyed on dataset/model/condition/arm/bg/explainer), so the CSV
needs no manual editing — but the `bg=local` rows must not be re-run.

**What the two arms measure.** They are not a one-factor contrast; each holds one
input fixed and varies the other, isolating one of the two confounded mechanisms:

| arm | background | explained rows | isolates |
|---|---|---|---|
| `bg=local` (main) | per-client | shared central-test subset | differences in each client's **background distribution** |
| `bg=shared` (twobg) | pooled over all clients | each client's **own partition** | differences in the **data region** each client occupies |

Read them together: high between-client agreement in the shared arm means clients'
data regions elicit the same feature ranking from the global model, so any
divergence seen in the main arm is background-driven — and vice versa.

Acceptance: no cell may come back with `status=undefined` and reason "collapsed
client axis" (that is the old bug); `n_explained_per_client` in each
`stability.json` should show real row counts (500 where the client is large
enough), and the per-client explained sets must differ.

Caveat to carry into the write-up: client partitions cover `x_train` only, so the
shared arm explains training rows while the main arm explains held-out test rows.
SHAP explains the fitted function rather than generalization, so this does not
invalidate the comparison, but it is a real difference between the arms.

## Step 3 — PaySim exact tier (Phase 3.1)

```bash
python experiments/shap_rq3.py --stage paysim-exact          # grouped, 9 players, ns=510
# then, only after the grouped tier confirms the ranking:
python experiments/shap_rq3.py --stage paysim-exact --include-ungrouped   # 13 players, ns=8190
```

Grouped costs ≈ what a normal PaySim pass costs (510 vs 500 evals/instance,
~6–7 h over the 11 PaySim kernel cells). Ungrouped is ≈ 16.4× per instance
(FedXGBllr ≈ 17 h/cell) — watch the first client's printed timing before
letting it run, and use `--models ffd` first if in doubt. Acceptance: every
exact cell reports `floor_mean = 1.0` (identical across seeds — enumeration is
complete, zero coalition sampling). Caveat: the one-hot grouping API
(`DenseData` groups) is verified on shap 0.51.0; the runner width-checks the
output on 0.49.1 and aborts loudly if the grouping is not honoured — if it
aborts, report it, don't work around it.

## Step 4 — budget probe (Phase 3.4)

```bash
python experiments/shap_rq3.py --stage budget    # default cell baf/ffd/dirichlet/none
```

~15 min; writes `results/shap_v2/budget_probe.csv` (two-seed floor at fixed
eval budget across nsamples/N_explain splits). Decides where any additional
GPU time goes.

## What to sync back to the laptop

`results/shap_v2/**` (summary CSV, per-cell `stability.json` /
`importance_per_client_seed.csv` / `profile.csv`, logs) — the stability-profile
figures and the thesis prose are produced laptop-side from these.

## Outputs schema (results/shap_v2/shap_summary_v2.csv)

One row per (dataset, model, condition, arm, bg, explainer):
`floor_mean`/`floor_min` (per-client two-seed floors), `between_mean`/
`between_sd` (20 CRN same-seed cross-client ρ), `delta`, `p_value` (exact
exchangeability test, 945 matchings), `p_adj` (BH per bg×explainer family),
`disattenuated` (secondary, flagged approximate), `weighted_between`/
`weighted_within` (magnitude-weighted rank correlation), `max_nonzero_per_row`
(l1 regression guard), `seed_delta_max` (deterministic bit-identity),
`status` ∈ {ok, undefined}, `verdict` ∈ {clients_differ, noise_limited,
exact_reference, single_client, undefined}.
