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

Recovered from `results/shap_v2/rq3_main.log` (2026-09-10). The stage ran as
**two invocations**, not one — the second is a resume pass:

```bash
# pass 1 — 96 cells in scope, ran them
python experiments/shap_rq3.py --stage main
# pass 2 — resume: same 96 in scope, 93 skipped as existing, last 3 completed
python experiments/shap_rq3.py --stage main
```

Both passes are recorded in `rq3_main.log` (945 lines; headers at lines 1 and
799, both `stage=main l1_reg=False seeds=(11, 22) nsamples=500 bg=100->kmeans10
explain=500`). The log's shell redirection is not recorded in the log itself —
see the transcript note at the end of this file for what is verified and what is
not.

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

Recovered from `results/shap_v2/rq3_twobg.log` (2026-09-10). The stage ran as
**three invocations**, and the log preserves the defect-and-repair sequence
described below:

```bash
# pass 1 — INVALID implementation: 6 cells (paysim x dirichlet)
python experiments/shap_rq3.py --stage twobg
# pass 2 — INVALID implementation: 6 cells (paysim x iid)
python experiments/shap_rq3.py --stage twobg --conditions iid
# pass 3 — corrected re-run after deleting pass 1-2 output: 12 cells
python experiments/shap_rq3.py --stage twobg --conditions dirichlet,iid
```

The conditions are read directly off each header's scope line ("two-background
arm: N kernel cells (paysim x ...)"); `--conditions` defaults to `dirichlet`, so
pass 1 may have passed it explicitly or relied on the default — indistinguishable
from the log.

**The log independently confirms the collapse defect.** Every one of the 12 cells
in passes 1-2 reports `between=1.0` — the arithmetic identity that a collapsed
client axis produces. Pass 3 reports varied values (0.8995 – 0.9863) against real
per-cell floors. Pass 3 also shows **zero** `skip (exists)` lines, which is the
log-level evidence that the invalid output was deleted before the re-run rather
than overwritten in place.

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

Recovered from `results/shap_v2/rq3_exact.log` (2026-09-10). The stage ran
**three times**, and all three produced identical output:

```bash
# run 3x — grouped, 9 players, ns=510; identical results each time
python experiments/shap_rq3.py --stage paysim-exact
```

Headers at lines 15, 209 and 403, each followed by "PaySim exact — grouped: 9
players, nsamples=510 (full enumeration)". All three report the same 16 cells
with byte-identical `between`, `delta` and `p` values — e.g.
`paysim/fedxgbllr/dirichlet_smote` at `between=0.6867 delta=0.3133 p=0.0011` in
every run. That triple repetition is independent corroboration of the tier's
`seed_delta_max = 0.0`: the exact tier reproduces itself exactly across separate
process invocations, not merely across the two coalition seeds within one.

All three passes show **zero** `skip (exists)` lines, so each recomputed rather
than resumed. Whether that came from `--no-skip-existing` or from the cell
directories being cleared between runs is **not recoverable** from the log.

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

## Close-out status (verified 2026-09-09)

Whether each RQ3 close-out item actually landed in `buku/content.typ` could not
previously be told from the repo. Each row below was confirmed by reading the
cited line, not by trusting a checklist. Line numbers are as of this date and
will drift if anything is inserted above them — the label and the quoted text
are the durable identifiers.

| # | close-out item | status | evidence |
|---|---|---|---|
| 1 | sampled-vs-exact bias reported | **done** | `content.typ:3450–3457` |
| 2 | `bert/iid/none` significance flip reported | **done** | `content.typ:3459–3466` |
| 3 | exact-tier table present | **done** | `content.typ:3407–3421`, label `<tab-4-shap-exact>` |
| 4 | stability grid figure present | **done** | `content.typ:3287–3290`, label `<fig-4-rq3-stability>` |
| 5 | two-seed measurement-scheme figure present | **done** | `content.typ:1782–1785`, label `<fig-3-two-seeds-gap>` |
| 6 | `--include-ungrouped` decision recorded | **done** | this file, §"`--include-ungrouped` was deliberately not run", plus the section below |

What each line actually says:

1. **Sampled-vs-exact bias** (`content.typ:3450`). "Pada kedua belas sel PaySim
   yang sama, selisih rerata absolut antara nilai tersampel dan nilai eksak
   sebesar 0,028 dengan selisih maksimum 0,099 yang terjadi pada sel FFD IID
   tanpa SMOTE, yang turun dari 0,973 menjadi 0,874. Tujuh dari dua belas sel
   bergerak turun, satu sel berimbang tepat, dan empat sel bergerak naik…" — the
   direction of bias is stated as conservative with respect to the reported
   conclusions.

2. **The `bert/iid/none` flip** (`content.typ:3459`). "Verdict statistik kedua
   tingkat sepakat pada sebelas dari dua belas sel. Pengecualiannya adalah sel
   BERT IID tanpa SMOTE, yang menghasilkan p = 0,001 pada tingkat tersampel
   namun p = 0,111 pada tingkat eksak. Sel tersebut merupakan positif palsu pada
   tingkat tersampel, dan pelaporannya bersifat wajib…" — reported as mandatory,
   not buried.

3. **Exact-tier table** (`content.typ:3407`, `<tab-4-shap-exact>`). Caption:
   "Stabilitas antar client pada tingkat eksak PaySim (Spearman berbobot
   magnitudo, atribusi bebas derau estimator)". Six model×arm rows, IID and
   Dirichlet columns; all six Dirichlet values below their IID counterpart.

4. **Stability grid** (`content.typ:3287`, `<fig-4-rq3-stability>`). The caption
   carries the null-not-threshold framing explicitly: "Sel berarsir tidak dapat
   dibedakan dari lantai derau explainer-nya menurut uji exchangeability eksak
   dan karenanya **tidak membawa klaim stabilitas**."

5. **Two-seed scheme** (`content.typ:1782`, `<fig-3-two-seeds-gap>`, §3.4.5). The
   caption states the correction directly: "Di bawah hipotesis nol kedua sebaran
   berimpit, sehingga **floor tidak dapat dipakai sebagai ambang deteksi** dan
   digantikan oleh uji exchangeability eksak atas seluruh 945 perfect matching."

### Standing decision — `--include-ungrouped` was not run, and must not be

Recorded here so it is visible in the repo and nobody runs it to "complete" the
grid. The ungrouped exact variant on PaySim uses all 13 features as separate
Shapley players: `nsamples = 2^13 - 2 = 8190`, ≈ 16.4× the per-instance cost of
the grouped tier, ≈ 17 h/cell for FedXGBllr alone. It is gated behind
`--include-ungrouped` in `experiments/shap_rq3.py` and **was never executed**.

The grouped exact tier (`M_eff = 9`, `nsamples = 510`, 100% of the kernel weight
enumerated, `seed_delta_max = 0.0` on all 16 cells) already confirmed the ranking
the ungrouped variant was meant to check: all six Dirichlet cells fall below
their IID counterparts, and the raw attribution gap widens 2.6×–19.5×. The extra
GPU time would buy no claim the thesis makes.

This is a **decision, not an omission**. Running it later would also write under
`results/shap_v2/`, which is frozen — so it carries the same cost as any other
regeneration of a result of record. `content.typ` does not mention the ungrouped
variant anywhere (`grep -n "ungrouped" buku/content.typ` returns nothing), so no
thesis claim depends on it either way.

## Kernel-weight enumeration per dataset (sourced 2026-09-09)

How much of the KernelSHAP kernel weight is enumerated **deterministically** at
`nsamples = 500`, before any coalition is drawn at random. §4.3 cites these to
justify measuring the noise floor per cell rather than broadcasting one number.

| tier | features M | nsamples | full subset sizes | kernel weight enumerated |
|---|---|---|---|---|
| PaySim, sampled | 13 | 500 | 1 of 6 | **34.9%** |
| ULB, sampled | 30 | 500 | 1 of 15 | **26.1%** |
| BAF, sampled | 55 | 500 | 1 of 27 | **22.3%** |
| PaySim, exact-grouped | 9 (`M_eff`) | 510 | 4 of 4 | **100.0%** |

**Derivation — re-derivable without rerunning anything.** In shap 0.49.1,
`KernelExplainer.explain()` (`shap/explainers/_kernel.py`) builds
`weight_vector[i] = (M-1) / (i*(M-i))` for `i = 1 … ceil((M-1)/2)`, doubles the
entries for paired sizes (a subset of size `i` and its complement `M-i` share a
weight), then normalises to sum 1. It walks subset sizes outward from `i = 1`,
fully enumerating size `i` while
`num_samples_left * remaining_weight_vector[i-1] / nsubsets >= 1 - 1e-8`, where
`nsubsets = binom(M, i)` (doubled when paired); after each accepted size it
subtracts `nsubsets` from the budget and renormalises the remaining weights by
`1 / (1 - remaining_weight_vector[i-1])`. The count it settles on is
`num_full_subsets`, and everything beyond it is sampled.

The enumerated fraction is therefore `1 - weight_left`, where `weight_left` is
the quantity shap itself logs at INFO. Confirmed against a live explainer rather
than by reading the source: `num_full_subsets = 1` with
`weight_left = 0.6509 / 0.7389 / 0.7774` for M = 13 / 30 / 55, and
`num_full_subsets = 4` (all sizes) for `M_eff = 9` at `nsamples = 510 = 2^9 - 2`,
which is the cap `2^M - 2` — hence exactly 100% and `seed_delta_max = 0.0`.

**Second, independent route — count the coalitions, do not read a summary field.**
After `shap_values()`, `explainer.maskMatrix` holds the coalitions actually used.
Counting rows whose subset size is *completely* enumerated (all `binom(M,s)`
distinct masks present) gives, at `nsamples = 500`: **26** coalitions for M = 13
(sizes 1 and its complement 12), **60** for M = 30, **110** for M = 55, and all
**510** for `M_eff = 9`. Recomputing the weight share from those counts alone
reproduces 34.9% / 26.1% / 22.3% / 100.0% — the two routes agree exactly. The
superseded 54.0% would require **182** coalitions at M = 13 (26 + 156, sizes 1
and 2 both complete); the explainer uses 26.

**Why 54.0% was wrong — a wrong model, not a wrong reading.** The error was
*omitting shap's acceptance criterion altogether* and substituting a pure
budget-capacity test ("does `binom(M,s)` still fit in the remaining samples?").
Under that test M = 13 accepts size 2 (156 ≤ 474) and reports 54.0%. shap does
not ask whether a size fits; it asks whether the size's share of the remaining
kernel weight justifies its cost, `num_samples_left * remaining[s-1] / nsubsets
>= 1`, which for size 2 at M = 13 gives **0.889 < 1** — rejected.

The `1/(1-w)` renormalisation is **not** the cause, and this was checked rather
than assumed: running the loop with the weight test but *without* renormalisation
still yields `num_full_subsets = 1` (criterion 0.5786 instead of 0.8889 — both
reject). Renormalisation changes the residual-weight bookkeeping, never the
accepted subset sizes at these M. M = 30 and M = 55 are unaffected by the whole
question because size 2 there costs 870 and 2970 coalitions against budgets of
440 and 390 — rejected by *either* model, so their agreement is structural, not
coincidental. M = 13 is the only case that separates the two models.

**Supersedes:** the `54.0%` figure in `docs/predictions.md:86`. Note that
`buku/content.typ` §4.3 also states "PaySim mengenumerasi sekitar 54 persen bobot
kernel" — flagged to the author, not edited. The ULB (~26%) and BAF (~22%)
figures in both places are correct. No measured result depends on this: the
per-cell floors are measured, not derived from these percentages, and P2's
"floors are approximately M-independent" is an empirical finding either way.

## Provenance of the `0,775` FedXGBllr figure (resolved 2026-09-09)

`buku/content.typ` §4.3 cites "0,775" as the FedXGBllr stability figure "reported
in the previous measurement", and argues it was an `l1_reg` artifact. The figure
is real and now has a primary source in the repo. The argument attached to it is
**wrong about which defect produced it**, and that is flagged below.

### Three states, not two

There were three RQ3 measurement states, not the two the v1/v2 naming suggests:

| state | source in repo | FedXGBllr mean ρ | condition |
|---|---|---|---|
| pre-repair (≤ 2026-08-06) | `results/shap/production.log` | **0.7751** | clipped-logit saturation → two all-zero PaySim cells; BERT absent (`KeyError`) |
| archived v1 | `results/shap/shap_summary.csv` | 0.9182 | logit repaired; `l1_reg` still `num_features(10)` |
| v2 | `results/shap_v2/shap_summary_v2.csv` | 0.9289 | `l1_reg=False`, per-cell floors |

Both pre-repair and archived-v1 states live under `results/shap/` — the log is
the pre-repair record, the CSV is post-repair. They are not the same run.

### Primary source, found and verified

`results/shap/production.log` (tracked) is the pre-repair run. It carries the
two all-zero cells directly:

```
[paysim/fedxgbllr/dirichlet_smote] clients=5 spearman=0.0000 kuncheva=1.0000 jac5=1.00
[paysim/fedxgbllr/dirichlet_none]  clients=5 spearman=0.0000 kuncheva=1.0000 jac5=1.00
```

`kuncheva=1.0000` alongside `spearman=0.0000` is the degenerate signature defect
#2's guard now catches: an all-zero vector scores as perfect rank agreement.
Every `bert_fraud` cell in the log is an `[ERROR] … KeyError: 'bert_fraud'`, so
BERT is absent from that run entirely.

The mean of the 11 FedXGBllr multi-client cells in the log is **0.7751** — the
disputed literal, exactly. The recipe is therefore not a reconstruction: it is
what the log records.

**Cross-check that fixes the ordering.** Of the 11 FedXGBllr cells, exactly the
two saturated ones differ between the log and the archived CSV; the other nine
are identical to four decimals. FFD sits at 0.9767 in both, unaffected by the
logit repair. That is the signature of a repair that touched only the saturated
cells, and it dates the log before the repair rather than after it.

### Dating, verified against the repo

Commit `1d661ac` "fix: shap on bert", **Fri 7 Aug 2026**, in
`experiments/shap_analysis.py`, removes `LOGIT_EPS = 1e-6` and the `_logit()`
helper and sets `cnn.final_layer = torch.nn.Identity()`, returning the
pre-Sigmoid activation unclipped. Its own added comment names the mechanism:
"on PaySim the probabilities compress to ~1e-9, which underflows the logit clip
floor so every prediction saturates to a constant -> KernelSHAP returns all-zero
attributions." The same commit fixes the BERT `KeyError`. The log predates it
(BERT absent, PaySim cells zero); the archived CSV postdates it (BERT at 0.9603,
PaySim cells repaired).

### The decomposition

| transition | cause | Δ FedXGBllr mean ρ |
|---|---|---|
| 0.7751 → 0.9182 | **clipped-logit repair** (defect #2, commit `1d661ac`) | **+0.143** |
| 0.9182 → 0.9289 | **`l1_reg=False`** (defect #1, the v2 re-run) | **+0.011** |

The logit repair accounts for roughly thirteen times the movement the `l1_reg`
fix does, and it moved only two cells rather than all eleven.

### Flagged for the author — `buku/content.typ` ~3248

The sentence attributes the 0.775 → ~0.95 movement to the `l1_reg` default. The
archive does not support that attribution: `l1_reg` moved the FedXGBllr mean by
**+0.011**, while the clipped-logit repair moved it by **+0.143**. On the
archived post-repair numbers FedXGBllr was still the lowest of the three sampled
models (0.9182 vs BERT 0.9603, FFD 0.9767), and it did not stop being lowest by
moving — it stopped being an outlier because the two saturated cells were no
longer zero.

The companion claim in the same passage — that the three sampled models converge
to roughly 0.95 with none standing out — is unaffected and holds: v2
`weighted_between` medians are FFD 0.963, BERT 0.952, FedXGBllr 0.958.

**Flag only. No thesis edit has been made and no replacement wording is
proposed.**

## `budget_probe.csv` — artifact absent, provenance undetermined (checked 2026-09-10)

Stage 4 above (line ~131) records that the budget probe "wrote
`results/shap_v2/budget_probe.csv`". **That file is not on the laptop**, and
whether it was ever produced could not be established. Recorded here rather than
corrected in place, because the claim may still be true of the GPU box.

**What was checked:**

| check | result |
|---|---|
| file on disk | **absent** — `results/shap_v2/` holds the summary CSV, four cell directories, and three logs; no `budget_probe.csv` |
| code path exists | **yes** — `stage_budget()` at `experiments/shap_rq3.py:628`, dispatched from the `--stage` map at line 706, writes `OUT / "budget_probe.csv"` at line 659 |
| reachable / gated | reachable; `--stage budget` is a normal choice, not behind an extra flag. `--budget-cell` defaults to `baf/ffd/dirichlet/none` |
| ever tracked by git | **no** — `git log --all --diff-filter=A -- '*budget_probe*'` and `--diff-filter=D` both return zero commits; git has never seen the path |
| would it be trackable | **yes** — `.gitignore:53 !results/shap_v2/**` un-ignores it, so it is not a `.gitignore` casualty |
| probe output in any log | **none** — no "budget probe", "budget_probe" or "n_explain=" line appears in any log under `results/`. The run record's own log list (line 28) names only `rq3_main.log`, `rq3_twobg.log`, `rq3_exact.log`; there is no budget log |
| invocation recorded anywhere | **no** — `--stage budget` appears nowhere outside this file |

**Verdict: absent, and undetermined between produced-and-unsynced and
never-produced.** No log, no git object and no invocation record is consistent
with the probe never having run; but the stage-4 text states a specific runtime
("~15 min") and a specific measurement, which reads as a report of something
observed, and the laptop only ever held a sync of the box. Both readings survive
the evidence, so neither is asserted here. Settling it requires looking on the
GPU box — a separate decision, not done here.

One caution against over-reading the stage-4 prose: **the command blocks in this
file are reconstructions, not transcripts.** Stage 1 documents
`… | tee results/shap_v2/main_run.log` and stage 2
`… | tee -a results/shap_v2/twobg.log`, but no `main_run.log` or `twobg.log`
exists anywhere in the repo — the actual logs are `rq3_main.log` (945 lines),
`rq3_twobg.log` (343) and `rq3_exact.log` (582). The stage content is real and
its logs survive under different names; only the invocation lines were written
from memory. So "wrote `budget_probe.csv`" carries the authority of a
recollection rather than of a transcript, which weakens — without settling — the
produced-and-unsynced reading.

**Nothing depends on it — verified, not assumed.** The `nsamples = 500` adoption
is argued from the inter-seed agreement measurement, *not* from this probe:

- `buku/content.typ:1728–1734` (§3.4.5): "vektor importance diukur pada
  nsamples ∈ {100, 500, 1000}. Nilai terkecil yang mencapai Spearman > 0,95 pada
  ketiga model yang dijelaskan KernelSHAP adalah nsamples = 500, yang karena itu
  digunakan."
- `buku/content.typ:2425–2426` (§3.4.5): "jumlah evaluasi fungsi ditetapkan
  nsamples = 500 dari pengukuran agreement antar-seed — bukan nilai bawaan
  pustaka".

Both point to `results/shap/noise_floor.txt`, which is present and tracked and
records exactly that sweep over nsamples ∈ {100, 500, 1000} with its `<= adopt`
markers. `content.typ` does not mention the budget probe anywhere — every
"budget"/"anggaran" occurrence in the thesis refers to the 5% false-alarm budget
or to the boosting-iteration budget, neither related.

So this is a **bookkeeping defect in the run record, not a hole in the evidence
chain**. Stage 4's own "no thesis claim rests on it" is now verified rather than
asserted. No thesis edit is proposed and no other line of this file is corrected.

## Transcript status of the command blocks (recovered 2026-09-10)

The stage-1/2/3 command blocks above were originally written from memory. They
have now been **recovered from the run logs** and rewritten. This section records
what is verified against artifacts, what is not, and what the recovery changed.

### What the logs record

Each invocation writes one header line:

```
shap 0.49.1 | stage=<name> l1_reg=False seeds=(11, 22) nsamples=500 bg=100->kmeans10 explain=500
```

followed by a scope line naming the cell count and, for `twobg`, the conditions.
So **`--stage` is recorded verbatim, and `--conditions` is recoverable for twobg
from the scope line.** The logs do **not** record `sys.argv`, a Hydra config
dump, or a wandb config — there is no full invocation record.

### Documented vs recovered, per block

| block | documented | recovered | discrepancy |
|---|---|---|---|
| stage 1 | one invocation, `… 2>&1 \| tee results/shap_v2/main_run.log` | **two** invocations of `--stage main`; second is a resume with 93 `skip (exists)` | invocation count wrong; `main_run.log` never existed (real log: `rq3_main.log`) |
| stage 2 | one invocation, `--conditions dirichlet,iid`, `… \| tee -a results/shap_v2/twobg.log` | **three** invocations: `dirichlet` (6 cells), `--conditions iid` (6), then `--conditions dirichlet,iid` (12) | invocation count wrong; the documented command is only the *third*; `twobg.log` never existed (real log: `rq3_twobg.log`) |
| stage 3 | one invocation | **three** invocations, all identical output | invocation count wrong; the command itself matches |

The `--stage` names and the stage-2 conditions are the substance, and they are
confirmed. What was wrong in every block was the **number of invocations** and
the `tee` target.

### Not recoverable, and not guessed

- the shell redirection (`tee` targets): not recorded in the logs at all. The
  rewritten blocks omit it rather than assert it.
- `--no-skip-existing`: stage 3's three passes show zero skips, which could come
  from the flag or from the cell directories being cleared between runs.
- `--datasets` / `--models`: not printed. The cell counts are consistent with the
  defaults (`baf,creditcard,paysim` for main; `paysim` for twobg), but that is
  inference, not a record.
- `--max-hours`: not printed.

### Independent checks — all pass

Checkable without the logs:

| check | result |
|---|---|
| every documented `--stage` name exists in the dispatch map | **yes** — `{"main", "twobg", "paysim-exact", "budget"}` at `experiments/shap_rq3.py:705–706`, matching the `--stage` `choices` list |
| every documented flag exists in the parser | **yes** — `--stage`, `--datasets`, `--models`, `--conditions`, `--max-hours`, `--no-skip-existing`, `--include-ungrouped`, `--budget-cell` |
| documented cell counts match `shap_summary_v2.csv` | **yes** — main 96, twobg 12, exact 16, total 124; all four exact |

### The logs are not in the archive

**Flagged, not fixed.** `rq3_main.log`, `rq3_twobg.log` and `rq3_exact.log` are
**untracked**: `.gitignore:56 *.log` matches them, and because it comes *after*
the `!results/shap_v2/**` negation at line 53, the later rule wins. The
`results/shap_v2/` negation therefore does not cover `.log` files.

These three logs are the sole evidence for everything in this section — the
invocation sequence, the collapse defect in twobg passes 1–2, and the exact
tier's cross-process reproducibility. They are also not regenerable. This is the
same class of exposure `production.log` had before it was covered, except that
`production.log` happened to be tracked and these are not. They are consequently
absent from `docs/frozen_manifest.sha256`, whose `results/shap_v2/**` block
covers 463 tracked files out of 466 on disk.

### Restated: what this means for `budget_probe.csv`

The verdict recorded above rested partly on stage 4 reading as an observation.
That reading is now weaker still: the blocks are confirmed reconstructions, and
in all three cases the reconstruction understated the invocation count and named
a log file that never existed. A section that mis-records its own commands three
times out of three is weak evidence that a fourth artifact it names was produced.

This does **not** flip the verdict to "never produced". The stage-1/2/3 errors
are all of one kind — collapsing a multi-pass reality into one tidy command line,
with the *substance* (stages, conditions, cell counts) correct every time. That
pattern is compression of things that did happen, not invention of things that
did not. So the evidence still does not distinguish produced-and-unsynced from
never-produced.

**Verdict unchanged: undetermined** — but the balance now leans further toward
never-produced than when first recorded, and the reason is documented here rather
than left as an impression.

## Available strengthening — cross-process reproducibility of the exact tier

**Flagged for the BAB 4 ledger, 2026-09-10. No thesis wording is proposed here.**

§4.3 currently rests the exact tier's exactness on `seed_delta_max = 0.0` — the
attribution vectors are bit-identical across the two coalition seeds *within* a
run. `results/shap_v2/rq3_exact.log` supports a strictly stronger statement at no
additional cost: the tier reproduces itself byte-for-byte across **three separate
process invocations**, on three separate occasions, each recomputing from scratch
(zero `skip (exists)` lines in all three).

Evidence — the same cell, three times, four decimals identical:

```
line  40:  [paysim/fedxgbllr/dirichlet_smote] K=5 floor=1.0 between=0.6867 delta=0.3133 p=0.0011 status=ok
line 234:  [paysim/fedxgbllr/dirichlet_smote] K=5 floor=1.0 between=0.6867 delta=0.3133 p=0.0011 status=ok
line 428:  [paysim/fedxgbllr/dirichlet_smote] K=5 floor=1.0 between=0.6867 delta=0.3133 p=0.0011 status=ok
```

All 16 cells behave this way in all three passes, headers at lines 15, 209 and
403.

**Why it is stronger.** `seed_delta_max = 0.0` rules out *coalition-sampling*
variance, since full enumeration leaves nothing to sample. It does not by itself
rule out anything else that could differ between runs — process-level RNG state,
dict or glob ordering, BLAS thread scheduling, or an accidental dependence on
which cells were already on disk. Three independent processes landing on
identical values closes those too, and it does so empirically rather than by
argument.

**Why it is available rather than required.** Nothing in §4.3 is wrong without
it; the exactness claim already holds on the enumeration argument plus the
bit-identity check. This is an additional, independently-verifiable support for a
claim the section already makes — and it now has an archived artifact behind it,
which the seed-delta check alone did not (`rq3_exact.log` was untracked until
2026-09-10).

Recorded so it is not lost. Whether §4.3 says anything about it is the author's
call.
