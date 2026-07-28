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
