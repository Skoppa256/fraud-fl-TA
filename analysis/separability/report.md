# Class-separability analysis — hypothesis test

**Question.** BAF has 8.5× ULB's fraud prevalence yet ~1/30th the lift over baseline. Is separability (not imbalance) the driver?

**Verdict.** HOLDS. BAF minority is 93.1% rare+outlier vs ULB 18.6%, and survives the dimensionality-matched control (BAF 93.9%). Separability, not imbalance, drives the difficulty ordering.

## Minority-type typology (Napierala & Stefanowski 2016) — headline

| dataset | dim | safe | borderline | rare | outlier | rare+outlier |
|---|---|---|---|---|---|---|
| ULB | 30 | 71.51% | 9.88% | 2.33% | 16.28% | **18.6%** |
| BAF | 55 | 0.26% | 6.65% | 19.3% | 73.8% | **93.1%** |
| PaySim | 13 | 57.56% | 10.69% | 7.79% | 23.97% | **31.76%** |

## Complexity, univariate, overlap, ceiling

| dataset | N1 | N2 | N3 | F1 | LDA overlap | max AUC | XGBoost lift |
|---|---|---|---|---|---|---|---|
| ULB | 0.1279 | 0.2739 | 0.0727 | 0.2741 | 0.1691 | 0.9559 | ~484× |
| BAF | 0.4222 | 0.9006 | 0.2953 | 0.7036 | 0.4218 | 0.7054 | ~14× |
| PaySim | 0.0758 | 0.0817 | 0.0458 | 0.6587 | 0.3561 | 0.8892 | ~763× |

## Dimensionality-matched control (top-13 features by AUC)

| dataset | rare+outlier full → matched | N3 full → matched |
|---|---|---|
| ULB | 18.6% → 17.73% | 0.0727 → 0.0683 |
| BAF | 93.1% → 93.9% | 0.2953 → 0.3127 |
| PaySim | 31.76% → 31.76% | 0.0458 → 0.048 |

## Top-5 discriminative features

- **ULB**: V14(0.956); V4(0.950); V12(0.939); V11(0.924); V10(0.923)
- **BAF**: housing_status_BA(0.705); credit_risk_score(0.665); device_os_windows(0.663); customer_age(0.655); proposed_credit_limit(0.635)
- **PaySim**: errorBalanceOrig(0.889); oldbalanceOrg(0.813); amount(0.793); type_TRANSFER(0.708); newbalanceOrig(0.703)

## Method notes

- Typology on full train reference for ULB/BAF; PaySim on a uniform prevalence-preserving subsample (n=500000, seed 42).
- N1/N2/N3/F1 on a class-BALANCED subsample (all minority capped at 3000 + equal majority, seed 42) to isolate overlap from imbalance.
- F1 uses Lorena's inverse convention: higher = harder.
- Distance measures are dimensionality-dependent; the matched-dim control recomputes typology + N3 on the top-13 features so the ULB↔BAF ordering is checked at equal dimensionality. BAF's 26 one-hot columns inflate Euclidean distance between category-differing records.
- Read-only: nothing under results/ was modified.
