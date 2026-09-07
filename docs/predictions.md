# Pre-registered predictions — SMOTE / oversampling ablation

**Dated 2026-07-27, before any full-grid results exist.** Recorded so the
oversampling analysis in Bab 4 can be read against predictions made in advance,
not rationalised afterward. Contrary outcomes are legitimate results requiring
explanation, not failures.

Configuration under test: uniform `sampling_strategy = 0.01` across PaySim, ULB,
BAF; K = 5; α ∈ {0.5, 1.0, 5.0} plus IID; seeds {42, 123, 2024}. SMOTE local per
client, `k_neighbors = 5`, skipped below 6 minority or when the target is met.
The uncorrected (no-SMOTE) configuration is the baseline; SMOTE is the intervention.

## Predictions

- **Primary (discrimination).** SMOTE will not improve AUPRC on any dataset, and
  will degrade it at α = 0.5 where sparse clients dominate.

- **Secondary (threshold-dependent metrics).** F1 and Recall at the tuned
  threshold may *rise* under SMOTE while AUPRC does not, because oversampling
  inflates predicted probabilities (van den Goorbergh et al. 2022,
  `goorbergh2022harm`). Both patterns together indicate a calibration shift, not
  a genuine gain.

- **Calibration.** Under SMOTE the calibration intercept moves away from 0 in the
  overestimation direction and the calibration slope falls below 1, worsening as
  event fraction and client size fall. **Sign convention note:** this project
  uses the standard Van Calster / van den Goorbergh convention, where
  *overestimation → calibration intercept < 0* (predictions too high) and/or
  *slope < 1*. This is the opposite sign from the "intercept > 0 = overestimation"
  wording in the task prompt; the standard convention is used throughout the code
  (`evaluation/metrics.calibration_metrics`) and thesis. So concretely: expect
  **intercept < 0** under SMOTE.

- **Model family** (following Elor & Averbuch-Elor 2022, `elor2022smote` —
  preprint; 73 datasets incl. XGBoost/LightGBM/CatBoost): balancing helps weak
  classifiers but not strong ones. Expect GBM and FedXGBllr flat to slightly
  negative; LR and SVM most likely to benefit; FFD and BERT most exposed to
  memorising the wireframe segments.

- **Dataset.** ULB α = 0.5 the most degenerate cell (~344 training fraud total),
  ahead of BAF. (Census caveat: no dataset produced zero-minority clients at
  α = 0.5; ULB's exposure is chronic minority-poverty across all clients —
  median ≈ 53 — rather than zero-fraud clients.)

- **BAF IID.** SMOTE and no-SMOTE arms expected identical, since global prevalence
  (~1.10%) exceeds the 1:100 target — confirmed by the per-client census
  (`results/analysis/minority_census_summary.md`): 15/15 clients skip via
  `target_met`. The same holds for the BAF centralized arm.

## Notes

- Geometry evidence (measured, not predicted): at the worst-case BAF client
  (seed 42, α = 0.5, 21 real fraud), synthetic minority collapses onto 82
  one-dimensional segments regardless of target ratio; 16.0% of synthetic points
  land in majority territory. Volume scales with the ratio; dimensionality and
  contamination do not. See §3.4.3 and `results/visualizations/`.

---

# Pre-registered predictions — RQ3 SHAP re-run (`l1_reg=False`, per-cell floors)

**Dated 2026-08-28, before any cell of the v2 re-run exists.** Recorded so the
Phase-1.1/Phase-2 outcome is reportable either way, per the project standard.
Contrary outcomes are legitimate results requiring explanation, not failures.

Context: every v1 kernel cell and all three noise floors were measured under the
shap ≥ 0.47 default `l1_reg="num_features(10)"` — LARS keeps at most 10 features
per explained instance and sets every other attribution to exactly 0.0
(`git log --all -S "l1_reg"` is empty: the parameter never existed in this
repository). The v2 runner (`experiments/shap_rq3.py`) sets `l1_reg=False` on
every KernelSHAP call and replaces the single BAF-measured broadcast floor with
per-cell, per-client two-seed floors plus an exact exchangeability test
(`evaluation/shap_inference.py`, 945 matchings at K = 5, Benjamini–Hochberg
across the 33 multi-client kernel cells).

## Predictions

- **P1.** With `l1_reg=False`, the re-measured noise floors rise for all three
  kernel models (FedXGBllr, BERT, FFD): most of the measured floor is an
  artifact of the discontinuous LARS keep-10 selection, not of coalition
  sampling.

- **P2.** The floors become approximately M-independent: the spread across the
  PaySim-, ULB-, and BAF-specific floors shrinks relative to the current single
  BAF number, which is biased low for PaySim (at nsamples = 500, PaySim
  enumerates 54.0% of the kernel weight deterministically vs BAF's 22.3%, and
  the keep-10 default depresses floors more as M grows).

- **P3 — the falsifiable one.** Consequently a **majority** of the 33
  multi-client kernel cells shows between-client disagreement exceeding
  estimator noise (exchangeability test, BH-adjusted p ≤ 0.05), and the
  "at or below the floor" framing does not survive.

The between-client Spearman also moves when `l1_reg` changes, and its direction
is **not** predicted — both sides of the comparison shift together. If the
floors and the between-client values rise in step and the gaps stay put, P3 is
refuted and the "RQ3 answerable for the deterministic tier only" conclusion is
reinstated on much better evidence. Either way it is reported.

## Notes

- No run-to-run sd is carried over from synthetic stand-in models into the
  thesis. The sd that reaches Bab 4 is the measured one: the spread of the 20
  between-client ρ values per cell that Phase 2 produces.
- The deterministic tier (LR, SVM, GBM interventional) carries zero estimator
  noise by construction (verified per run via the bit-identity check); its
  between-client spread is the calibration anchor the kernel tier is read
  against.

## Outcomes (recorded 2026-09-07, after the v2 grid was complete and frozen)

The v2 re-run is finished: 124 cells in `results/shap_v2/`, all `status = ok`,
audited by `analysis/verify_kernel_tier.py` (six legs, all PASS). All three
predictions above stand.

- **P1 — confirmed.** The floors rose for all three kernel models once
  `l1_reg=False` removed the LARS keep-10 selection. On the pilot cell the
  FedXGBllr floor went from 0.9730 to 0.9984; across the full grid every one of
  the 33 multi-client kernel cells now carries a within-client floor ≥ 0.9726
  weighted, against v1's single broadcast BAF numbers of 0.9730 / 0.9972 /
  0.9966. Most of the noise v1 measured was the discontinuous feature selection,
  not coalition sampling.

- **P2 — confirmed.** The floors are approximately M-independent. Mean
  within-client floor (magnitude-weighted) per model across PaySim (M = 13),
  ULB (M = 30) and BAF (M = 55): FFD 0.9997 / 0.9996 / 0.9997, BERT 0.9992 /
  0.9982 / 0.9996, FedXGBllr 0.9951 / 0.9861 / 0.9993. The spread across M is
  at most 0.013 and is not monotone in M — BAF (M = 55) carries the *highest*
  floor for two of the three models, which is the opposite of the
  keep-10-driven M-dependence v1's single BAF number implied. The
  low-bias-for-PaySim artifact is gone.

- **P3 — confirmed.** The falsifiable one holds, and not narrowly: **29 of the
  33** multi-client kernel cells show between-client disagreement exceeding
  estimator noise under the exact exchangeability test (945 matchings at K = 5,
  BH-adjusted), against the "majority" the prediction asked for. Per model:
  FFD 11/11, BERT 10/11, FedXGBllr 8/11. The "at or below the floor" framing
  from v1 did not survive, and the "RQ3 answerable for the deterministic tier
  only" fallback is not reinstated.

The direction of the between-client Spearman was explicitly not predicted, and
it did not need to be: the floors and the between-client values did **not** rise
in step, which is what opened the gaps.

Two framing corrections came out of the review of this outcome and are recorded
in `docs/rq3_method_notes.md` rather than here, because they are about how the
result is read rather than about whether the prediction held: the floor is a
**null**, not a detection threshold, and the exact-grouped tier is not an anchor.
