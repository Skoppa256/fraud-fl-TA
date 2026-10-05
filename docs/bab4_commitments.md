# The nine commitments — data built, wording proposed

Phase E1, 2026-09-29. **No thesis text was changed.** Everything here is for the
author to read, approve, edit or reject; E2 applies whatever he settles on.

`buku/content.typ`, `template.typ` and `bibliography.bib` are byte-identical to
their Phase C state.

---

## 0. Something the author should see before approving item 9

Building the calibration columns turned up a result that is **not flattering to
FedXGBllr**, and the decision to *report* calibration means reporting it.

**FedXGBllr's Brier score equals the dataset prevalence, on all eleven of its
cells, to within 0,15%:**

| dataset | prevalence | FedXGBllr Brier (all cells) | ratio |
|---|---|---|---|
| PaySim | 0,00129 | 0,001290–0,001291 | 1,0004–1,0007 |
| ULB | 0,00173 | 0,001728–0,001732 | 0,9986–1,0011 |
| BAF | 0,01103 | 0,010904–0,011026 | 0,9886–0,9996 |

That is the Brier score of a predictor that outputs the base rate for every row.
Its calibration slopes say the same thing — **9,19 to 35,08**, where the ideal is
1 and every other model sits between 0,28 and 1,86:

| model | n | min slope | median | max |
|---|---|---|---|---|
| LR | 16 | 0,58 | 0,97 | 1,86 |
| GBM | 16 | 0,45 | 0,97 | 1,82 |
| FFD | 16 | 0,28 | 0,96 | 1,42 |
| BERT | 16 | 0,29 | 0,86 | 1,29 |
| XGBoost | 5 | 1,12 | 1,23 | 1,47 |
| **FedXGBllr** | **9** | **0,92** | **12,34** | **35,08** |
| SVM | — | no calibration (hinge margin) | | |

**This does not touch any conclusion the thesis draws.** AUPRC and Recall\@5%FPR
are rank-based and unaffected by monotone miscalibration; FedXGBllr's ranking
performance — 0,996 on PaySim — stands. What it means is that FedXGBllr's
*probabilities* are not usable as probabilities, and reporting calibration makes
that visible in a chapter where FedXGBllr is otherwise the headline result.

Twelve of the 78 calibrated rows fall outside a slope of [0,5 – 2,0]: eight are
FedXGBllr, two are BAF Non-IID SMOTE cells (FFD 0,28, BERT 0,29), two are ULB GBM
without SMOTE (0,45–0,48).

**This is the author's call, and it may change the decision on item 9** — from
"report" to "report with a sentence of interpretation", or to "narrow". I have
built the data either way. I have not written the interpretation.

---

## 1. The new §4.1.1 table — data built

**Sources**, all `results/clean_summary.csv`:

| column | populated |
|---|---|
| `test_f1`, `test_precision`, `test_recall` | **96/96** |
| `test_actual_fpr` | **96/96** |
| `test_brier` | **80/96** |
| `test_cal_slope`, `test_cal_intercept` | **78/96** |

### 1.1 The threshold check — passes

§3.3.5 says F1/Precision/Recall use a threshold tuned for maximum F1 on the
validation set, distinct from the ROC threshold used for Recall\@5%FPR.

- `threshold_policy` is **`val_f1_tuned` on all 96 rows** — no other value occurs.
- `threshold` and `test_threshold_at_fpr` differ on **96/96 rows**.

**BAB 4 would be reporting exactly the numbers the methodology says it produces.**

### 1.2 Grouping — three candidates, costs stated

I followed the shape of Tabel 4.1–4.3: **one table per dataset, models as rows**.
That is candidate A.

| candidate | shape | rows | cost |
|---|---|---|---|
| **A. One table per dataset** | 3 tables, rows = model × condition × arm, 7 metric columns | 36 + 24 + 36 | Complete and matches the existing shape. ~96 rows over 3 tables, roughly 2 pages. Heaviest, but it is the only one that lets a reader check any cell. |
| **B. One table, best-per-model-per-dataset** | 1 table, 18 rows | 18 | Compact, one page. Loses the condition/arm breakdown — a reader cannot see that FFD's BAF calibration collapses only under Non-IID SMOTE. |
| **C. One table per dataset, no-SMOTE only** | 3 tables, 18 rows total | 18 | Halves A. But the SMOTE arm is where the interesting calibration failures are, so this hides the finding in §0 above. |

**Recommendation is the author's.** If §4.1.1 should stay readable, B; if the
table is meant to be checkable, A. I would not choose C — it drops exactly the
cells that carry information.

### 1.3 Precision of presentation

BAB 4 is already internally inconsistent here: **Tabel 4.2 renders BAF to three
decimals, Tabel 4.7 to four.** For the new table I propose:

- **F1, Precision, Recall, FPR aktual — three decimals**, matching Tabel 4.1–4.6,
  which is what §4.1.1 already contains.
- **Brier — five decimals.** Three would print 0,001 for every PaySim and ULB row
  and destroy the column; the values span 0,00001 to 0,01263.
- **Calibration slope and CITL — two decimals.** They range from 0,28 to 35,08;
  more precision adds nothing.

### 1.4 The NA cells — two kinds, and they should look different

| cells | why | proposed mark |
|---|---|---|
| **16 SVM rows**, all three calibration columns | `SGDClassifier(loss="hinge")` emits a decision margin, not a probability. §3.3.5 already states this. | `—` (em dash), with a caption note: *"SVM tidak menghasilkan probabilitas sehingga metrik kalibrasi tidak dihitung."* |
| **2 rows**, `paysim/fedxgbllr/dirichlet/{none,smote}` — Brier present, slope and intercept absent | Predicted probabilities saturate; the logistic recalibration has no spread to fit. The degenerate-output guard working correctly. | **a different mark**, e.g. `n/d` (tidak terdefinisi), with a caption note distinguishing it from the SVM case |

Using one mark for both would say the same thing about two different situations.

### 1.5 The data — candidate A, in full

Format below: `F1 · Prec · Rec · FPR akt · Brier · Slope · CITL`.

#### ULB (36 rows)

| Model | Kondisi | F1 | Prec | Rec | FPR akt | Brier | Slope | CITL |
|---|---|---|---|---|---|---|---|---|
| LR | Centralized none | 0,763 | 0,815 | 0,716 | 0,044 | 0,00067 | 0,95 | 0,18 |
| LR | Centralized SMOTE | 0,773 | 0,879 | 0,689 | 0,044 | 0,00065 | 0,83 | −2,08 |
| LR | Dirichlet none | 0,770 | 0,770 | 0,770 | 0,041 | 0,00066 | 0,92 | −0,09 |
| LR | Dirichlet SMOTE | 0,786 | 0,833 | 0,743 | 0,030 | 0,00060 | 0,82 | −1,63 |
| LR | IID none | 0,775 | 0,809 | 0,743 | 0,040 | 0,00066 | 0,93 | −0,09 |
| LR | IID SMOTE | 0,768 | 0,828 | 0,716 | 0,030 | 0,00062 | 0,78 | −1,83 |
| SVM | Centralized none | 0,818 | 0,889 | 0,757 | 0,049 | — | — | — |
| SVM | Centralized SMOTE | 0,806 | 0,945 | 0,703 | 0,031 | — | — | — |
| SVM | Dirichlet none | 0,770 | 0,770 | 0,770 | 0,039 | — | — | — |
| SVM | Dirichlet SMOTE | 0,778 | 0,800 | 0,757 | 0,048 | — | — | — |
| SVM | IID none | 0,770 | 0,770 | 0,770 | 0,049 | — | — | — |
| SVM | IID SMOTE | 0,763 | 0,815 | 0,716 | 0,049 | — | — | — |
| GBM | Centralized none | 0,806 | 0,945 | 0,703 | 0,018 | 0,00087 | 0,48 | −6,13 |
| GBM | Centralized SMOTE | 0,845 | 0,882 | 0,811 | 0,050 | 0,00046 | 0,72 | −0,41 |
| GBM | Dirichlet none | 0,788 | 0,897 | 0,703 | 0,037 | 0,00111 | 0,45 | −7,81 |
| GBM | Dirichlet SMOTE | 0,837 | 0,881 | 0,797 | 0,047 | 0,00048 | 0,70 | 0,18 |
| GBM | IID none | 0,797 | 0,898 | 0,716 | 0,007 | 0,00117 | 0,45 | −9,37 |
| GBM | IID SMOTE | 0,794 | 0,912 | 0,703 | 0,050 | 0,00052 | 0,56 | 1,33 |
| FFD | Centralized none | 0,851 | 0,896 | 0,811 | 0,039 | 0,00051 | 0,93 | −0,09 |
| FFD | Centralized SMOTE | 0,844 | 0,934 | 0,770 | 0,036 | 0,00058 | 0,92 | −1,65 |
| FFD | Dirichlet none | 0,836 | 0,933 | 0,757 | 0,026 | 0,00049 | 0,90 | −0,05 |
| FFD | Dirichlet SMOTE | 0,849 | 0,908 | 0,797 | 0,036 | 0,00051 | 0,88 | −1,33 |
| FFD | IID none | 0,847 | 0,921 | 0,784 | 0,023 | 0,00050 | 0,93 | −0,05 |
| FFD | IID SMOTE | 0,844 | 0,934 | 0,770 | 0,041 | 0,00054 | 0,93 | −1,50 |
| BERT | Centralized none | 0,812 | 0,875 | 0,757 | 0,034 | 0,00055 | 0,78 | −0,34 |
| BERT | Centralized SMOTE | 0,814 | 0,864 | 0,770 | 0,046 | 0,00093 | 0,66 | −1,40 |
| BERT | Dirichlet none | 0,812 | 0,875 | 0,757 | 0,045 | 0,00056 | 0,65 | −0,01 |
| BERT | Dirichlet SMOTE | 0,848 | 0,966 | 0,757 | 0,047 | 0,00057 | 0,53 | −1,01 |
| BERT | IID none | 0,806 | 0,900 | 0,730 | 0,046 | 0,00052 | 0,81 | −0,07 |
| BERT | IID SMOTE | 0,841 | 0,859 | 0,824 | 0,046 | 0,00048 | 0,61 | −0,05 |
| FedXGBllr | Dirichlet none | 0,806 | 0,900 | 0,730 | 0,045 | 0,00173 | 0,93 | 0,36 |
| FedXGBllr | Dirichlet SMOTE | 0,849 | 0,908 | 0,797 | 0,050 | 0,00173 | **14,06** | 2,28 |
| FedXGBllr | IID none | 0,814 | 0,864 | 0,770 | 0,050 | 0,00173 | 0,92 | 0,47 |
| FedXGBllr | IID SMOTE | 0,820 | 0,877 | 0,770 | 0,050 | 0,00173 | **12,34** | 4,03 |
| XGBoost | Centralized none | 0,830 | 0,918 | 0,757 | 0,050 | 0,00044 | 1,17 | −0,15 |
| XGBoost | Centralized SMOTE | 0,861 | 0,937 | 0,797 | 0,050 | 0,00044 | 1,12 | −1,16 |

#### BAF (24 rows)

| Model | Kondisi | F1 | Prec | Rec | FPR akt | Brier | Slope | CITL |
|---|---|---|---|---|---|---|---|---|
| LR | Centralized none | 0,218 | 0,181 | 0,273 | 0,050 | 0,01007 | 1,04 | 0,13 |
| LR | Dirichlet none | 0,208 | 0,159 | 0,299 | 0,050 | 0,01096 | 1,86 | 8,19 |
| LR | Dirichlet SMOTE | 0,174 | 0,134 | 0,249 | 0,050 | 0,01048 | 0,58 | −1,19 |
| LR | IID none | 0,217 | 0,182 | 0,270 | 0,050 | 0,01007 | 1,04 | 0,14 |
| SVM | Centralized none | 0,214 | 0,187 | 0,250 | 0,050 | — | — | — |
| SVM | Dirichlet none | 0,189 | 0,153 | 0,247 | 0,050 | — | — | — |
| SVM | Dirichlet SMOTE | 0,170 | 0,140 | 0,215 | 0,049 | — | — | — |
| SVM | IID none | 0,161 | 0,129 | 0,213 | 0,050 | — | — | — |
| GBM | Centralized none | 0,228 | 0,180 | 0,313 | 0,050 | 0,01000 | 1,02 | 0,02 |
| GBM | Dirichlet none | 0,245 | 0,200 | 0,317 | 0,050 | 0,01263 | 1,03 | −1,38 |
| GBM | Dirichlet SMOTE | 0,245 | 0,200 | 0,317 | 0,050 | 0,01263 | 1,03 | −1,38 |
| GBM | IID none | 0,216 | 0,175 | 0,280 | 0,050 | 0,01029 | 0,94 | 0,03 |
| FFD | Centralized none | 0,229 | 0,188 | 0,294 | 0,050 | 0,01001 | 1,02 | −0,18 |
| FFD | Dirichlet none | 0,236 | 0,198 | 0,292 | 0,050 | 0,01064 | 0,96 | −0,84 |
| FFD | Dirichlet SMOTE | 0,094 | 0,067 | 0,152 | 0,050 | 0,01102 | **0,28** | −1,00 |
| FFD | IID none | 0,226 | 0,177 | 0,313 | 0,050 | 0,00997 | 1,01 | −0,04 |
| BERT | Centralized none | 0,237 | 0,216 | 0,262 | 0,050 | 0,01005 | 1,29 | 1,33 |
| BERT | Dirichlet none | 0,239 | 0,207 | 0,284 | 0,050 | 0,01010 | 0,90 | −0,40 |
| BERT | Dirichlet SMOTE | 0,100 | 0,076 | 0,146 | 0,050 | 0,01111 | **0,29** | 8,45 |
| BERT | IID none | 0,239 | 0,200 | 0,297 | 0,050 | 0,00991 | 1,00 | 0,13 |
| FedXGBllr | Dirichlet none | 0,224 | 0,183 | 0,287 | 0,050 | 0,01103 | **9,19** | 6,40 |
| FedXGBllr | Dirichlet SMOTE | 0,209 | 0,160 | 0,301 | 0,050 | 0,01103 | **12,21** | 6,21 |
| FedXGBllr | IID none | 0,213 | 0,199 | 0,230 | 0,050 | 0,01090 | **24,68** | 0,42 |
| XGBoost | Centralized none | 0,236 | 0,195 | 0,299 | 0,050 | 0,01001 | 1,23 | 0,75 |

#### PaySim (36 rows)

| Model | Kondisi | F1 | Prec | Rec | FPR akt | Brier | Slope | CITL |
|---|---|---|---|---|---|---|---|---|
| LR | Centralized none | 0,599 | 0,793 | 0,481 | 0,048 | 0,00072 | 1,00 | −0,09 |
| LR | Centralized SMOTE | 0,614 | 0,784 | 0,504 | 0,050 | 0,00102 | 0,97 | −2,11 |
| LR | Dirichlet none | 0,608 | 0,780 | 0,498 | 0,050 | 0,00083 | 1,36 | 4,48 |
| LR | Dirichlet SMOTE | 0,607 | 0,787 | 0,494 | 0,050 | 0,00086 | 1,39 | 5,95 |
| LR | IID none | 0,601 | 0,807 | 0,479 | 0,050 | 0,00071 | 1,00 | −0,09 |
| LR | IID SMOTE | 0,642 | 0,740 | 0,567 | 0,048 | 0,00093 | 0,97 | −2,10 |
| SVM | Centralized none | 0,602 | 0,711 | 0,522 | 0,049 | — | — | — |
| SVM | Centralized SMOTE | 0,591 | 0,707 | 0,507 | 0,050 | — | — | — |
| SVM | Dirichlet none | 0,587 | 0,702 | 0,504 | 0,049 | — | — | — |
| SVM | Dirichlet SMOTE | 0,574 | 0,728 | 0,473 | 0,050 | — | — | — |
| SVM | IID none | 0,386 | 0,699 | 0,266 | 0,050 | — | — | — |
| SVM | IID SMOTE | 0,623 | 0,794 | 0,512 | 0,048 | — | — | — |
| GBM | Centralized none | 0,998 | 0,999 | 0,996 | 0,049 | 0,00005 | 0,91 | −10,20 |
| GBM | Centralized SMOTE | 0,998 | 1,000 | 0,996 | 0,040 | 0,00001 | 1,78 | −3,43 |
| GBM | Dirichlet none | 0,998 | 1,000 | 0,995 | 0,050 | 0,00001 | 1,00 | −0,27 |
| GBM | Dirichlet SMOTE | 0,998 | 1,000 | 0,996 | 0,050 | 0,00001 | 1,46 | −3,13 |
| GBM | IID none | 0,996 | 0,999 | 0,993 | 0,049 | 0,00001 | 1,05 | −0,98 |
| GBM | IID SMOTE | 0,998 | 1,000 | 0,996 | 0,040 | 0,00002 | 1,82 | −3,44 |
| FFD | Centralized none | 0,742 | 0,930 | 0,618 | 0,050 | 0,00051 | 0,95 | −0,05 |
| FFD | Centralized SMOTE | 0,784 | 0,892 | 0,699 | 0,050 | 0,00056 | 0,97 | −1,61 |
| FFD | Dirichlet none | 0,677 | 0,908 | 0,539 | 0,050 | 0,00082 | 1,42 | 11,31 |
| FFD | Dirichlet SMOTE | 0,706 | 0,936 | 0,567 | 0,050 | 0,00081 | 1,32 | 10,03 |
| FFD | IID none | 0,752 | 0,917 | 0,637 | 0,050 | 0,00049 | 1,03 | 0,85 |
| FFD | IID SMOTE | 0,796 | 0,871 | 0,732 | 0,050 | 0,00045 | 1,03 | −1,13 |
| BERT | Centralized none | 0,791 | 0,904 | 0,703 | 0,050 | 0,00043 | 0,92 | 2,15 |
| BERT | Centralized SMOTE | 0,870 | 0,927 | 0,820 | 0,050 | 0,00029 | 1,28 | 1,90 |
| BERT | Dirichlet none | 0,703 | 0,943 | 0,561 | 0,050 | 0,00065 | 0,79 | 3,13 |
| BERT | Dirichlet SMOTE | 0,669 | 0,874 | 0,542 | 0,050 | 0,00105 | 1,06 | 6,92 |
| BERT | IID none | 0,831 | 0,963 | 0,731 | 0,050 | 0,00043 | 1,22 | 2,27 |
| BERT | IID SMOTE | 0,861 | 0,958 | 0,782 | 0,050 | 0,00037 | 0,96 | −1,66 |
| FedXGBllr | Dirichlet none | 0,994 | 0,995 | 0,994 | 0,037 | 0,00129 | **n/d** | **n/d** |
| FedXGBllr | Dirichlet SMOTE | 0,992 | 0,993 | 0,991 | 0,008 | 0,00129 | **n/d** | **n/d** |
| FedXGBllr | IID none | 0,997 | 1,000 | 0,994 | 0,037 | 0,00129 | **35,08** | 4,42 |
| FedXGBllr | IID SMOTE | 0,997 | 1,000 | 0,994 | 0,048 | 0,00129 | **23,76** | 3,39 |
| XGBoost | Centralized none | 0,938 | 0,952 | 0,925 | 0,013 | 0,00012 | 1,47 | −0,18 |
| XGBoost | Centralized SMOTE | 0,939 | 0,948 | 0,929 | 0,050 | 0,00019 | 1,44 | −2,52 |

---

## 2. The Jaccard\@5 column for Tabel 4.13 — **the saturation premise does not hold**

Mean `jaccard_at5` over the same cells as the existing Kuncheva column
(`results/shap_v2/shap_summary_v2.csv`, **124/124 populated**):

| Dataset | | LR | SVM | GBM |
|---|---|---|---|---|
| **BAF** | Kuncheva (existing) | 0,956 | 0,927 | 0,853 |
| | **Jaccard\@5 (new)** | **0,933** | **0,889** | **0,778** |
| **ULB** | Kuncheva (existing) | 1,000 | 0,964 | 0,742 |
| | **Jaccard\@5 (new)** | **1,000** | **0,950** | **0,677** |
| **PaySim** | Kuncheva (existing) | 0,789 | 0,935 | 0,878 |
| | **Jaccard\@5 (new)** | **0,783** | **0,933** | **0,875** |

**The premise in the brief needs re-reading.** Jaccard\@5 does **not** saturate at
1,00 in the production values. Across the nine cells it ranges **0,677 – 1,000**;
across all 33 deterministic cells, **0,470 – 1,000**, mean 0,869, with 13 of 33
at exactly 1,000. So there is saturation in about 40% of cells, not across the
board. BAB 3's record of Jaccard saturating at 1,00 was from the **noise-floor
probe**, not from production.

**The column still demonstrates the chapter's argument**, by a different route:
Jaccard\@5 is **lower than Kuncheva in all nine cells**, and the gap widens where
agreement is weakest (ULB GBM: 0,742 → 0,677). That is what an uncorrected
overlap measure does. The argument for Kuncheva survives; the evidence for it is
"systematically different", not "degenerate".

---

## 3. The unweighted Spearman column for Tabel 4.12 — medians agree, minima do not

From the same 66 main federated cells, `spearman` column, **124/124 populated**.

| Model | n | weighted median (existing) | **unweighted median** | unweighted min | unweighted max |
|---|---|---|---|---|---|
| LR | 11 | 0,9845 | **0,9865** | 0,9286 | 0,9982 |
| SVM | 11 | 0,9895 | **0,9897** | 0,9082 | 0,9993 |
| GBM | 11 | 0,9774 | **0,9870** | 0,9287 | 0,9957 |
| FFD | 11 | 0,9634 | **0,9723** | 0,9374 | 0,9945 |
| BERT | 11 | 0,9517 | **0,9555** | 0,9247 | 0,9929 |
| FedXGBllr | 11 | 0,9576 | **0,9560** | 0,7676 | 0,9767 |

**The medians agree closely** — the largest gap is GBM at 0,0096. One sentence can
say the two measures tell the same story at the centre, and no conclusion changes.

**The minima do not, and that matters.** §4.3.1's prose uses the *minimum* to argue
FedXGBllr has wide variation. Under the weighted measure the minima are 0,8014
(GBM) and 0,6650 (FedXGBllr); unweighted they are 0,9287 and 0,7676. The five
largest divergences:

| dataset | model | condition | arm | weighted | unweighted | gap |
|---|---|---|---|---|---|---|
| ULB | GBM | IID | none | 0,8014 | 0,9870 | **0,1856** |
| PaySim | FFD | Dirichlet | none | 0,7979 | 0,9374 | 0,1395 |
| ULB | GBM | Dirichlet | none | 0,8253 | 0,9287 | 0,1034 |
| PaySim | FedXGBllr | Dirichlet | SMOTE | 0,6650 | 0,7676 | 0,1026 |
| BAF | BERT | Dirichlet | none | 0,8682 | 0,9552 | 0,0870 |

**This is a finding in its own right.** The low weighted values come from cells
where the *high-magnitude* features disagree while the long tail agrees — exactly
what the weighting is for. If the column is added, the minimum column would
benefit from a sentence saying which measure the "Minimum" refers to.

---

## 4. Proposed wording — drafts for the author to replace

**Every one of these is a draft, not prose I am proposing to keep.** They are as
short as the fix allows so that editing them is cheap.

### 4.1 §3.3.5 — mean feature importance, from reported to derived

**Current** (`content.typ` L1727–1729):
> "Terdapat empat ringkasan utama yang digunakan dalam analisis. **Pertama,
> rata-rata feature importance antar client menunjukkan fitur yang secara umum
> memberikan kontribusi terbesar terhadap prediksi.** Kedua, …"

**Proposed** — change only the first two sentences:
> "Terdapat tiga ringkasan utama yang digunakan dalam analisis, yang seluruhnya
> dihitung dari rata-rata feature importance antar client. Rata-rata tersebut
> digunakan untuk menurunkan himpunan lima fitur teratas setiap client dan tidak
> dilaporkan tersendiri. Pertama, …"

Then renumber *Kedua/Ketiga/Keempat* to *Pertama/Kedua/Ketiga*. Four words of
arithmetic, one new sentence.

### 4.2 §3.3.5 — the Kuncheva profile, from summary to robustness check

**Current** (L1738–1740):
> "Analisis tambahan dilakukan dengan menghitung indeks Kuncheva pada berbagai
> ukuran himpunan fitur teratas. **Profil tersebut digunakan untuk mengamati
> perubahan kesepakatan ketika semakin banyak fitur disertakan.**"

**Proposed** — replace the second sentence:
> "Profil tersebut diperiksa sebagai uji ketahanan terhadap pemilihan k dan tidak
> dilaporkan tersendiri; ringkasan yang dilaporkan adalah nilai pada k = 5."

### 4.3 §3.3.5 — the selected fraud count, drop the recording claim

**Current** (L1678–1680):
> "**Jumlah fraud yang benar-benar terpilih turut dicatat** karena prevalensi yang
> rendah dapat menghasilkan sangat sedikit sampel fraud dalam explanation data."

**Proposed:**
> "Prevalensi yang rendah dapat menghasilkan sangat sedikit sampel fraud dalam
> explanation data sehingga pemilihannya menerapkan batas minimum satu sampel
> fraud."

**Saran item 4's premise survives on §3.4.5 alone**, which already states the floor
and which this phase does not touch (L2086–2087):
> "Adapun pemilihan explanation data mengikuti proporsi kelas **dengan ketentuan
> sedikitnya satu sampel fraud**."

The actual counts are 1 / 1 / 6 for PaySim / ULB / BAF — stating them is a
separate decision the author has not made.

### 4.4 The new table's introducing sentence, and the two new columns

These three are the ones most likely to be rewritten; treat them as placeholders.

**§4.1.1, introducing the new table** (to go after the Recall\@5%FPR tables):
> "Sebagai pelengkap kedua metrik utama, @tab-4-supp menyajikan F1-score,
> Precision, Recall, FPR aktual, serta metrik kalibrasi pada seluruh konfigurasi."

**Status 2026-10-05 — superseded, on the author's instruction.** The placeholder
above shipped, then was cut back. The three `<tab-4-supp-*>` tables (ULB / BAF /
PaySim) were replaced by a single `<tab-4-cal-slope>` carrying **calibration slope
only**, because after the author rewrote the closing two paragraphs of §4.1.1 in
his own words, slope was the only supplementary metric any explanation cited by
value — F1, Precision, Recall, FPR aktual, Brier and CITL appeared nowhere but in
the introducing sentence itself. Those six metrics are now reported nowhere in the
thesis. The sentence in force is:
> "Sebagai pelengkap kedua metrik utama, @tab-4-cal-slope menyajikan calibration
> slope pada seluruh konfigurasi yang dievaluasi."

**§4.3.1, the Spearman column:**
> "Korelasi peringkat tanpa bobot disajikan berdampingan sebagai pembanding."

**§4.3.1 or §4.3.2, the Jaccard column:**
> "Jaccard\@5 disajikan berdampingan dengan indeks Kuncheva untuk menunjukkan
> pengaruh koreksi peluang."

### 4.5 `separabilitas` — one clause in BAB 2 §2.2.7

§4.1.3 is frozen, so BAB 2 is the only place this can land.

**Current** (L842–844):
> "Proporsi kelas minoritas sendiri belum sepenuhnya menggambarkan kesulitan
> klasifikasi, karena dua dataset dengan rasio fraud yang serupa dapat memiliki
> **tingkat tumpang tindih kelas yang berbeda**."

**Proposed** — add four words:
> "…dapat memiliki tingkat tumpang tindih kelas yang berbeda, yaitu tingkat
> **separabilitas kelas** yang berbeda."

### 4.6 The tree-ensemble naming — **free, if the author unifies on the most frequent**

Three prose wordings, 13 prose sites:

| form | count | sites |
|---|---|---|
| **`agregasi tree ensemble`** | **7** | BAB 1 L93, L137, L199 · BAB 2 L662 · **BAB 4 L2329, L2358, L2546** |
| `tree ensemble aggregation` | 5 | BAB 1 L208, L303 · BAB 2 L478, L739 · BAB 3 L1191 |
| `agregasi ensemble pohon` | 1 | BAB 3 L1563 |

**Unifying on `agregasi tree ensemble` (the most frequent) costs nothing in frozen
text: all three BAB 4 sites already use it.** Six sites would change — BAB 1 ×2,
BAB 2 ×2, BAB 3 ×2 — all in editable chapters.

Unifying the other way (`tree ensemble aggregation`) would require changing
**three sites inside frozen BAB 4** and a second unfreeze.

**Separately:** BAB 3's Tabel 3.5 cell reads **`Tree Ensemble Aggregation`**
(title case) where the equivalent cells in Tabel 2.1, 4.10 and 4.11 read
`Tree ensemble aggregation` (sentence case). One cell, editable chapter.

---

## 5. What the insertion will cost — renumbering forecast

Inserting one table in §4.1.1, after `<tab-4-rfpr-paysim>`:

| now | label | becomes |
|---|---|---|
| Tabel 4.1–4.6 | `tab-4-auprc-*`, `tab-4-rfpr-*` | **unchanged** |
| — | **the new table** | **Tabel 4.7** |
| Tabel 4.7 | `tab-4-baf-auprc-bench` | Tabel 4.8 |
| Tabel 4.8 | `tab-typology` | Tabel 4.9 |
| Tabel 4.9 | `tab-4-iid-smote` | Tabel 4.10 |
| Tabel 4.10 | `tab-4-baf-smote` | Tabel 4.11 |
| Tabel 4.11 | `tab-4-baf-smote-rfpr` | Tabel 4.12 |
| Tabel 4.12 | `tab-4-rq3-corr` | Tabel 4.13 |
| Tabel 4.13 | `tab-4-rq3-kuncheva` | Tabel 4.14 |
| Tabel 4.14 | `tab-4-rq3-paysim` | Tabel 4.15 |
| Tabel 4.15 | `tab-4-rq3-smote` | Tabel 4.16 |
| Tabel 4.16 | `tab-4-rq3-seeds` | Tabel 4.17 |

**The renumber is safe.** Verified:

- **17 `@label` references** to BAB 4 objects exist, **all inside BAB 4**. They
  resolve by referent, so every one follows the renumber automatically.
- **Zero literal "Tabel N.N" appears in prose** anywhere in `content.typ`. The only
  occurrence is inside BAB 5's commented-out `REFERENSI SINTAKS` block, which
  renders nothing.
- **BAB 5 names no table.** Confirmed — its single regex hit is that comment block.
- Three §4.3 tables (`tab-4-rq3-kuncheva`, `-smote`, `-seeds`) have **zero**
  references, matching the author's style of introducing tables by position.
- **DAFTAR TABEL goes from 25 to 26**; BAB 4 from 16 tables to 17. Figures
  unchanged at 10.

---

## 6. `CLAUDE.md` — seven fixes applied

The label map was **re-derived wholesale** from `content.typ` and cross-checked
against the rendered PDF: **25/25 tables and 10/10 figures match exactly.**

| # | before | after |
|---|---|---|
| 1 | `<tab-3-5>` → Tabel 3.6; `<tab-3-6>` → Tabel 3.9; rows for `<tab-smote-geometry>`, `<tab-minority-census>`, `<fig-3-4-smote-geometry>`; no row for `<tab-3-explainer>` | `<tab-3-5>` → **3.7**; `<tab-3-6>` → **3.8**; three dead rows **removed**; `<tab-3-explainer>` → **3.6** added; Gambar 3.5 row removed |
| 2 | skew note citing 3.9 / 3.5 / 3.6 / Gambar 3.5 | rewritten to 3.5 / 3.6 / 3.7 / 3.8, plus the standing fact that `content.typ` now contains **zero** literal "Tabel N.N" in prose |
| 3 | "Chapters 4 and 5 are off-limits… Keep BAB 4/5 byte-identical" | replaced with a per-region table: BAB 1–3 the author's text; **preamble/§4.1/§4.2 frozen, §4.1.1 unfrozen for exactly one insertion then re-frozen**; §4.3 and BAB 5 the author's replaced text |
| 4 | "§4.3 … cites the frozen `results/shap_v2/**` tree" | "**It cites nothing** — no `@key`, no author-year" |
| 5 | "BAB 5 (Penutup)" | "BAB 5 (KESIMPULAN DAN SARAN)" |
| 6 | two conventions undocumented | added: `set heading(supplement: [Subbab])` with the level-1 hazard and the `grep` to run before labelling a heading; and the walk-back-from-label splice rule with the defect that forced it |
| 7 | Tabel 4.9's caption row read "Paysim" | corrected to "PaySim", matching the caption after the Phase B2 fix |

Also added: a note that `<sec-hasil-rq3>` now has zero references and the label
stays; and a pointer to `docs/bab4_open.md`, `bab4_audit.md` and this file.

---

## 7. The freeze list and manifest

`results/analysis/**` added to the frozen table in `CLAUDE.md` and to
`docs/frozen_manifest.sha256`, using `git ls-files` so no untracked placeholder is
picked up.

| | before | after |
|---|---|---|
| manifest lines | 1196 | **1198** |
| files added | — | `results/analysis/minority_census.csv`, `results/analysis/minority_census_summary.md` |
| `shasum -a 256 -c` | 0 non-OK | **0 non-OK** |

### ⚠️ A second unprotected path the brief did not name

`analysis/separability/` is a **different directory** from `results/analysis/` and
§6 authorized only the latter, so I did not freeze it. But
**`analysis/separability/separability_summary.csv` is the source of record for
Tabel 4.8 and the 93,1% that BAB 5 now restates** — a thesis claim resting on an
unprotected file, exactly the condition §6 was written to fix. Six tracked files
live there. **Freezing it is an open decision for the author.**

---

## 8. Verification

| gate | result |
|---|---|
| `git diff` scoped to `buku/**` | **byte-identical** — `content.typ` SHA `31984257ed5b1692`, unchanged from Phase C |
| `typst compile` | **0 errors, 0 warnings** |
| `shasum -c docs/frozen_manifest.sha256` | **0 non-OK of 1198** |
| re-derived label map vs PDF front matter | **25/25 tables, 10/10 figures** |
| every §2 number traced to file + column + populated count | yes, §1–§3 above |

Files changed this phase: `CLAUDE.md`, `docs/frozen_manifest.sha256`, and this
file. Nothing under `buku/`.
