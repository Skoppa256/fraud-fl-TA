# BAB 4 audit — the last unverified surface

Read-only audit, 2026-09-27. **No file in `buku/` was modified.**

Every phase of the BAB 1–5 replacement programme verified numbers that *crossed a
chapter boundary*. The numbers living inside BAB 4's own prose and its §4.1/§4.2
tables were authored before the programme began and frozen throughout. This
document checks them.

**Headline: every number in BAB 4 verifies.** 115 prose literals, 11 tables,
~280 numeric cells, all against `results/clean_summary.csv` and
`analysis/separability/separability_summary.csv`. Two presentational notes and
one cross-artifact discrepancy are recorded below; no value is wrong.

---

## 1. The four figures named in the brief

### 1.1 The three lower-bound multiples — verified, denominator established

The multiples are **best AUPRC over all conditions including centralized, divided
by the dataset's class prevalence**. This is confirmed two ways: the arithmetic
works only with that numerator, and §4.1.3 quotes the three "nilai terbaik"
(0,997 / 0,838 / 0,169), which are exactly the centralized bests.

| dataset | best AUPRC (source cell) | prevalence | quotient | thesis | verdict |
|---|---|---|---|---|---|
| PaySim | 0,997304 (GBM centralized SMOTE) | 0,00129 | **773,10** | 773 | ✓ |
| ULB | 0,838465 (XGBoost centralized none) | 0,00173 | **484,66** | 484 | ✓ (see note) |
| BAF | 0,169037 (BERT centralized none) | 0,01103 | **15,33** | 15 | ✓ |

The federated-only reading does **not** reproduce them — it gives 772,48 / 477,90
/ 15,14, and 477,90 would not be written as 484. The denominator is settled.

**Note on ULB.** 484,66 truncates to 484 and *rounds* to 485. The thesis writes
484. PaySim (773,10) and BAF (15,33) are the same under both rules, so ULB is the
only one where the choice shows. Not an error — a truncation — but if the author
wants consistent rounding, 484 becomes 485.

### 1.2 The 6,4× fraud-rate ratio — verified, and the 8,5× confusion was **not** inherited

> §4.1.3: "Nilai fraud rate dataset BAF adalah 1,1% dibandingkan nilai fraud rate
> dataset ULB sebesar 0,172% yang berarti nilai fraud rate BAF adalah **6,4 kali**
> lebih besar daripada ULB."

- From the displayed values: 1,1 / 0,172 = **6,3953** → 6,4 ✓
- From the frozen prevalences: 0,01103 / 0,00173 = **6,3757** → 6,4 ✓

Both routes give 6,4. **The thesis has not inherited the 8,5× error.** Searching
`content.typ` for `8,5` returns exactly one hit — the substring inside `−18,5%`
in a Tabel 4.11 cell. BAF-over-PaySim is 0,01103 / 0,00129 = **8,5504**, which
appears nowhere.

### 1.3 The "~70%" worst drop — imprecise, not wrong

> §4.2.3: "Penurunan terburuk **sekitar 70%** terjadi pada model dengan aturan
> agregasi accuracy-weighted FedAvg"

The two accuracy-weighted FedAvg drops are **FFD −71,0068%** and
**BERT −73,2461%**. The worst is BERT at −73,2%, which "sekitar 70%" describes
loosely. §4.2.2 four paragraphs earlier states both precisely ("71,0% dan 73,2%").

**The author's call.** "Sekitar 70%" is defensible for a 71–73% band; "sekitar
73%" or "71,0% hingga 73,2%" would be tighter. Nothing is misstated.

---

## 2. Every numeral in BAB 4 prose

115 numeric literals, tables masked. The §4.3 literals (L2632–2841) were verified
in Phase A2 and re-verified in Phase C against BAB 5; they are not repeated here.
The §4.1/§4.2 literals — never checked before — are below. **All agree.**

### 2.1 Preamble (L2125–2134)

| value | claim | derivation | verdict |
|---|---|---|---|
| 108 | nominal cells | 3 datasets × 6 models × 3 conditions × 2 arms = 108 | ✓ |
| 12 | not run | 108 − 96 | ✓ |
| 96 | executed | rows in `clean_summary.csv` = 96 | ✓ |
| — | the 12 are BAF centralized+IID with SMOTE | all 12 such combinations absent from the CSV | ✓ |
| 1,1% | BAF fraud rate | prevalence 0,01103 | ✓ |
| 1% | SMOTE target | `sampling_strategy = 0,01` | ✓ |
| 0,00129 / 0,00173 / 0,01103 | AUPRC lower bounds | prevalence per dataset | ✓ |

### 2.2 §4.1.1–§4.1.3 prose

| L | value | claim | found | verdict |
|---|---|---|---|---|
| 2316 | "sekitar 0,5" | mean BAF Recall\@5%FPR | **0,4996** (n=24) | ✓ |
| 2331 | 0,656 | PaySim LR best | 0,655966 | ✓ |
| 2332 | 0,632 | PaySim SVM best | 0,632412 | ✓ |
| 2333 | 0,996 | PaySim FedXGBllr all fed | 0,995946 / 0,995729 | ✓ |
| 2334 | 0,772 | ULB LR best | 0,771595 | ✓ |
| 2334 | 0,743 | ULB SVM best | 0,742855 | ✓ |
| 2335 | 0,806 | ULB FedXGBllr best | 0,805952 | ✓ |
| 2343 | 0,158 / 0,167 / 0,141 | BAF FFD / BERT / FedXGBllr, IID none | 0,158120 / 0,166851 / 0,140882 | ✓ |
| 2346–7 | 0,045 / 0,045 / 0,137 | BAF FFD / BERT / FedXGBllr, Dirichlet SMOTE | 0,045469 / 0,044673 / 0,137397 | ✓ |
| 2349–50 | 0,647 / 0,660 / 0,996 | PaySim FFD / BERT / FedXGBllr, Dirichlet none | 0,647366 / 0,660028 / 0,995729 | ✓ |
| 2370–1 | 0,169 / 0,997 / 0,838 | best AUPRC BAF / PaySim / ULB | 0,169037 / 0,997304 / 0,838465 | ✓ |
| 2404 | 0,169 / 0,167 | BAF BERT centralized / federated Non-IID | 0,169037 / 0,166976 | ✓ |
| 2406 | 0,1607 | FT-Transformer, `@dong2026fcorr` | **external paper — not verifiable from frozen data** | n/a |
| 2416–7 | 1,1% / 0,172% / 6,4× | see §1.2 | | ✓ |

### 2.3 §4.2 prose

| L | value | claim | found | verdict |
|---|---|---|---|---|
| 2468 | 0,311 → 0,577 | PaySim SVM IID → Non-IID, no SMOTE | 0,311246 → 0,577040 | ✓ |
| 2470 | 0,858 → 0,660 | PaySim BERT IID → Non-IID, no SMOTE | 0,857594 → 0,660028 | ✓ |
| 2481 | "sekitar 1,0%" | ULB SVM IID SMOTE delta | **−1,0149%** | ✓ |
| 2482 | 103,2% | PaySim SVM IID SMOTE delta | **+103,1873%** | ✓ |
| 2546 | 9,3% | BAF FedXGBllr Non-IID SMOTE delta | −9,2869% | ✓ |
| 2548 | 30,5% / 31,9% | BAF LR / SVM | −30,4710% / −31,9222% | ✓ |
| 2550 | 71,0% / 73,2% | BAF FFD / BERT | −71,0068% / −73,2461% | ✓ |
| 2595 | "sekitar 70%" | worst drop | see §1.3 | imprecise |

---

## 3. Every cell of the eleven §4.1/§4.2 tables

**Zero mismatches.** ~280 numeric cells verified against `clean_summary.csv` and
`analysis/separability/separability_summary.csv`.

| table | label | cells checked | mismatches |
|---|---|---|---|
| Tabel 4.1 | `<tab-4-auprc-ulb>` | 36 | **0** |
| Tabel 4.2 | `<tab-4-auprc-baf>` | 24 | **0** |
| Tabel 4.3 | `<tab-4-auprc-paysim>` | 36 | **0** |
| Tabel 4.4 | `<tab-4-rfpr-ulb>` | 36 | **0** |
| Tabel 4.5 | `<tab-4-rfpr-baf>` | 24 | **0** |
| Tabel 4.6 | `<tab-4-rfpr-paysim>` | 36 | **0** |
| Tabel 4.7 | `<tab-4-baf-auprc-bench>` | 7 "studi ini" rows | **0** (6 `@dong2026fcorr` rows external) |
| Tabel 4.8 | `<tab-typology>` | 18 | **0** |
| Tabel 4.9 | `<tab-4-iid-smote>` | 36 | **0** |
| Tabel 4.10 | `<tab-4-baf-smote>` | 18 | **0** |
| Tabel 4.11 | `<tab-4-baf-smote-rfpr>` | 18 | **0** |

§4.3's five tables were verified in Phase A2 against `shap_summary_v2.csv` and
re-verified in Phase C. Nothing has changed; that result stands.

### 3.1 The delta convention holds throughout

All three delta tables — 4.9, 4.10, 4.11 — compute **`(new − old)/old`** at full
precision. Every delta cell was recomputed both ways and only `/old` reproduces
the table. The `/new` formula would print −244,9% for FFD in Tabel 4.10 instead of
−71,0%. **The two-formula problem is gone.**

### 3.2 `<tab-typology>` against the separability analysis

Every cell matches `analysis/separability/separability_summary.csv`:

| dataset | dim | safe | borderline | rare | outlier | rare+outlier |
|---|---|---|---|---|---|---|
| ULB (source) | 30 | 71,51 | 9,88 | 2,33 | 16,28 | 18,6 |
| ULB (Tabel 4.8) | 30 | 71,5% | 9,9% | 2,3% | 16,3% | **18,6%** |
| BAF (source) | 55 | 0,26 | 6,65 | 19,3 | 73,8 | 93,1 |
| BAF (Tabel 4.8) | 55 | 0,3% | 6,6% | 19,3% | 73,8% | **93,1%** |
| PaySim (source) | 13 | 57,56 | 10,69 | 7,79 | 23,97 | 31,76 |
| PaySim (Tabel 4.8) | 13 | 57,6% | 10,7% | 7,8% | 24,0% | **31,8%** |

The 93,1% that BAB 5 now restates is correct. One round-half case: BAF borderline
6,65 is shown as 6,6 (round-half-even); 6,7 would also be defensible.

---

## 4. Two presentational notes and one cross-artifact discrepancy

**4.1 BAF precision differs between two frozen tables.** Tabel 4.2 shows BAF at
three decimals (0,167 / 0,162 / 0,158 / 0,151); **Tabel 4.7 shows the same values
at four** (0,1670 / 0,1620 / 0,1581 / 0,1515). Both are frozen and both are
correct. This is worth knowing because the Phase C ruling rounded BAB 5's BAF
figures from four decimals to three to match Tabel 4.2 — which means BAB 5 now
matches Tabel 4.2 and differs in precision from Tabel 4.7.

**4.2 The lift figures differ between the thesis and the separability analysis.**
`separability_summary.csv` carries its own `xgb_lift` column — **ULB 484,1 ·
BAF 14,2 · PaySim 762,8** — computed from **XGBoost**'s AUPRC over prevalence. The
thesis reports **484 / 15 / 773**, computed from the **best model overall**. Both
are internally consistent; they answer different questions. The ULB figures agree
by coincidence (XGBoost *is* ULB's best). **Two project artifacts report different
lift numbers for the same datasets** — a reader comparing them would need this
explained.

---

## 5. The unreported commitments — the list is **nine**, not eight

For each: where promised (with the verb), whether the data exists, and where it
would go. **§4.1 and §4.2 are frozen; §4.3 is not.**

The verb matters. `dicatat`/`mencatat` promises only that a value was **recorded**
and is discharged by the value existing in a results file. `dilaporkan` promises
it is **reported** and is not.

### 5.1 F1-score, Precision, Recall — the strongest promise

**Promised, five sites:**
- BAB 1 §1.3: *"F1-score, Precision, dan Recall juga **dicatat** sebagai metrik pelengkap"* — record
- BAB 2 §2.2.8: *"Precision, Recall, dan F1-score **digunakan** sebagai metrik pelengkap"*, plus Persamaan 2.3–2.5 defining all three
- BAB 3 §3.3.1: *"penentuan ambang klasifikasi untuk **pelaporan** F1-score, Precision, dan Recall"* — **report**
- BAB 3 §3.3.5: *"AUPRC digunakan sebagai metrik utama sedangkan Recall\@5%FPR, F1-score, Precision, dan Recall digunakan sebagai metrik pelengkap… hasil **dilaporkan** sebagai satu nilai per konfigurasi"* — **report**
- BAB 3 §3.4.5: *"F1-score, Precision, dan Recall **dihitung** menggunakan ambang yang menghasilkan F1 tertinggi pada validation set"*

**Data:** `results/clean_summary.csv` — `test_f1`, `test_precision`, `test_recall`,
`threshold`, `threshold_policy`. **All 96/96 populated, zero NA.** Read-only; no
recomputation.

**Would go in:** §4.1.1, alongside Tabel 4.1–4.6. **Frozen.**

**Smallest honest fix:** either add one table (or three columns) to §4.1.1, or
narrow the promise in §3.3.1 and §3.3.5 to "recorded but not reported". The
threshold-selection procedure described in §3.3.5 and §3.4.5 exists solely to
produce these three numbers, so narrowing costs more than it looks.

### 5.2 Jaccard\@5

**Promised:** BAB 2 §2.2.9 (Persamaan 2.10 defines it); BAB 3 §3.3.5 lists it as
the **third of four main summaries**, and the closing paragraph — *"Hasil evaluasi
explainability **dilaporkan** … Spearman, Jaccard, dan Kuncheva digunakan untuk
mengukur kesepakatan interpretasi"* — is a report promise.

**Data:** `results/shap_v2/shap_summary_v2.csv`, column `jaccard_at5`,
**124/124 populated.** Read-only.

**Would go in:** §4.3.1 or §4.3.2. **Not frozen.**

**Smallest honest fix:** one column in the author's Tabel 4.13, or drop Jaccard
from §3.3.5's list of four. Note BAB 3 itself says Kuncheva is used *because*
Jaccard\@5 is not comparable across datasets — so dropping it is coherent.

### 5.3 Unweighted Spearman rank correlation — **a ninth, not in the brief's list**

**Promised:** BAB 2 §2.2.9 defines it (Persamaan 2.9) and gives three reasons for
choosing it; BAB 3 §3.3.5 lists it as the **second of four main summaries** and
names it in the `dilaporkan` paragraph.

**Not met under other wording.** BAB 4 reports *"korelasi peringkat berbobot"* —
the **weighted** variant, which BAB 2 introduces separately as a complement to
unweighted Spearman, not a replacement. `Spearman` appears **zero times** in BAB 4.

**Data:** `shap_summary_v2.csv`, column `spearman`, **124/124 populated.**

**Would go in:** §4.3.1. **Not frozen.**

**Smallest honest fix:** one column beside the weighted correlation in Tabel 4.12,
or a sentence in §3.3.5 saying the weighted variant is the one reported.

### 5.4 Mean feature importance across clients

**Promised:** BAB 3 §3.3.5 lists it as the **first of four main summaries**, in the
`dilaporkan` paragraph: *"Rata-rata importance digunakan untuk menjelaskan fitur
yang dominan"*. Repeated in §3.4.5.

**Data:** `results/shap_v2/**/importance_per_client_seed.csv`, column `mean`,
**124 files.** Read-only per cell, but a presentation would need selecting which
cells to show.

**Would go in:** §4.3.1. **Not frozen.**

**Smallest honest fix:** the author's Tabel 4.15 already reports *which* features
dominate and how they shift — arguably the interesting half. A sentence in §3.3.5
saying the mean vector is used to derive the top-5 sets rather than reported
directly would close it without new results.

### 5.5 Kuncheva profile across k

**Promised:** BAB 3 §3.3.5 — *"Analisis tambahan dilakukan dengan menghitung indeks
Kuncheva pada berbagai ukuran himpunan fitur teratas. Profil tersebut **digunakan**
untuk mengamati perubahan kesepakatan"* — "digunakan", the weakest verb of the
nine.

**Data:** `results/shap_v2/**/profile.csv`, **90 files**, columns
`k, between_mean, within_mean, within_min, within_max`, 13 rows each (k = 1…13).
Read-only. 90 rather than 124 because only multi-client cells get a profile.

**Would go in:** §4.3.1 or an appendix. **Not frozen.**

**Smallest honest fix:** one figure, or reword §3.3.5 to "computed as a robustness
check; the k = 5 summary is what is reported".

### 5.6 FPR aktual

**Promised twice, and the second is a report promise:**
- BAB 3 §3.3.5: *"FPR aktual dan ambang yang bersesuaian turut **dicatat**"* — record
- BAB 3 §3.4.5: *"FPR aktual juga **dilaporkan** karena skor prediksi yang tersedia tidak selalu memungkinkan pencapaian FPR tepat sebesar 5%"* — **report**

**Data:** `clean_summary.csv`, `test_actual_fpr` and `test_threshold_at_fpr`,
**both 96/96 populated.**

**Would go in:** §4.1.1, beside the Recall\@5%FPR tables. **Frozen.**

**Smallest honest fix:** change `dilaporkan` to `dicatat` in §3.4.5 — one word, and
BAB 3 is editable. That is the cheapest fix of the nine.

### 5.7 Synthesis multiplier

**Promised:** BAB 3 §3.4.3 — *"Modul juga **mencatat** multiplier sintesis"* —
record only. **Discharged by the value existing.**

**Data:** `results/analysis/minority_census.csv`, column `synthesis_multiplier`.
⚠️ **That file is not on the frozen list** — it sits under `results/analysis/`,
which `CLAUDE.md`'s frozen table does not cover.

**Would go in:** nowhere. The census table that once displayed it
(`<tab-minority-census>`) was removed from BAB 3 in Phase B3 by the author's
decision.

**Smallest honest fix: none needed.** The promise is `mencatat` and the record
exists. Flagging only that its home in the thesis is gone.

### 5.8 Selected fraud count in the explanation set

**Promised:** BAB 3 §3.3.5 — *"Jumlah fraud yang benar-benar terpilih turut
**dicatat** karena prevalensi yang rendah dapat menghasilkan sangat sedikit sampel
fraud dalam explanation data"* — record only.

**Data:** ⚠️ **not recorded in any results file.** `stability.json` carries
`n_explained_per_client` (the 500) but no fraud count. The value is deterministic —
`max(1, round(500 × prevalence))` = **1 / 1 / 6** for PaySim / ULB / BAF — so it
can be derived in one line, but it is not *read* from anywhere.

**Smallest honest fix:** this is the one place where the thesis says a value was
recorded and it was not. Either record it, or soften §3.3.5 to say the floor
guarantees at least one fraud sample (which §3.4.5 already says).

### 5.9 Calibration — Brier, slope, calibration-in-the-large

**Promised:** BAB 2 §2.2.8 (definitions and interpretation); BAB 3 §3.3.5 —
*"Evaluasi keluaran probabilitas **dilengkapi dengan** Brier score, calibration
slope, dan calibration-in-the-large"*; BAB 3 §3.4.5, same wording. Both read as
statements that the evaluation *includes* these.

**Data:** `clean_summary.csv` — `test_brier` **80/96**, `test_cal_slope` and
`test_cal_intercept` **78/96**. The 16 NA rows are SVM (hinge margin, no
probability — stated in §3.3.5). Two further rows,
`paysim/fedxgbllr/dirichlet/{none,smote}`, have a Brier but no slope: their
predicted probabilities saturate near zero, so the recalibration has no spread to
fit. **78 rows carry a complete triple.**

**Would go in:** §4.1.1. **Frozen.**

**Smallest honest fix:** a short subsection in §4.3 (not frozen) reporting
calibration for the 78 rows, or narrow §3.3.5 and §3.4.5 to say calibration was
computed and is available but not discussed. BAB 5 Saran item 5 already recommends
future work report calibration alongside AUPRC, which reads oddly if the study
computed it and said nothing.

### 5.10 Two confirmed non-gaps — set aside

- **Noise floor.** BAB 3 §3.3.5 defines it; §4.3.3 reports the same quantity as
  Tabel 4.16's column *"Median kesepakatan dua seed dalam client"*. **Met under
  different wording.**
- **The 945 matchings and Benjamini–Hochberg correction.** BAB 3 §3.3.5 describes
  both; §4.3.3 discharges them at `content.typ:2831` — *"Berdasarkan prosedur
  pengujian dan **koreksi pengujian berganda** yang digunakan, 29 dari 33
  konfigurasi…"*. **Met under different wording.**

**No other item was removed from the list as a false gap.** Each of the nine was
checked for an alternative wording in BAB 4 and none was found.

### 5.11 The pattern underneath

Five of BAB 2's eleven equations define a measure BAB 4 never names:

| equation | measure | named in BAB 4 |
|---|---|---|
| Persamaan 2.3 | Precision | **0 times** |
| Persamaan 2.4 | Recall | **0 times** |
| Persamaan 2.5 | F1 | **0 times** |
| Persamaan 2.9 | Spearman | **0 times** |
| Persamaan 2.10 | Jaccard | **0 times** |

(Persamaan 2.7, Shapley, is a method definition rather than a reported metric and
is not a gap.) An examiner who reads BAB 2's equations and then looks for them in
the results will find five that never appear.

---

## 6. Three cross-cutting items

### 6.1 `separabilitas`

Used **once**, in frozen §4.1.3, carrying the central BAF argument:

> `content.typ:2420` — *"Analisis menunjukkan bahwa perbedaan ini disebabkan oleh
> **separabilitas kelas** dimana meskipun kelas minoritas BAF banyak namun
> mayoritas merupakan outlier sehingga tidak membentuk pola yang bermakna."*

The concept is prepared in BAB 2 §2.2.7 (the Napierała retention added in Phase B2)
without the word:

> `content.typ:843` — *"…karena dua dataset dengan rasio fraud yang serupa dapat
> memiliki **tingkat tumpang tindih kelas** yang berbeda."*

**Minimum fix: one clause in BAB 2**, which is editable — e.g. naming class
separability as the term for that overlap. §4.1.3 is frozen, so a gloss there is
not available without unfreezing it.

### 6.2 The tree-ensemble paradigm's surface forms

Three distinct wordings, 17 sites:

| form | prose sites | table cells |
|---|---|---|
| `tree ensemble aggregation` | BAB 1 ×2, BAB 2 ×2, BAB 3 ×1 — **5** | — |
| `agregasi tree ensemble` | BAB 1 ×3, BAB 2 ×1, BAB 4 ×3 — **7** | — |
| `agregasi ensemble pohon` | BAB 3 ×1 — **1** | — |
| `Tree ensemble aggregation` | — | BAB 2 ×1, BAB 4 ×2 |
| `Tree Ensemble Aggregation` | — | BAB 3 ×1 |

**Legitimate styling:** the four capitalised instances are all table cells (row
labels in Tabel 3.5, 4.10, 4.11) — except `Tree Ensemble Aggregation` in BAB 3's
Tabel 3.5, which is title-case where the others are sentence-case.

**Free variation:** `agregasi tree ensemble` (7) is the most frequent prose form;
`tree ensemble aggregation` (5) and `agregasi ensemble pohon` (1) are the others.
Unifying on the most frequent would touch **BAB 1, BAB 2 and BAB 3** — all
editable — and **BAB 4**, which is frozen and already uses it.

**The other three paradigms vary only by sentence case in BAB 4 prose:**
`Accuracy-weighted FedAvg` at §4.1.2 (L2339, L2361–2362) and `Best-model selection`
at §4.2.2 (L2545), against lowercase everywhere else. Both inside frozen text.

### 6.3 `noise floor`

Defined in BAB 3 §3.3.5 (three occurrences) and **never used again**. Confirmed:
zero occurrences in BAB 4 and BAB 5.

**BAB 4 does report the quantity**, as Tabel 4.16's column *"Median kesepakatan dua
seed dalam client"*. So this is a term introduced and then not carried, not a
missing result. Harmless; worth knowing if the author wants the term used once in
§4.3.3, which is editable.

---

## 7. `CLAUDE.md` — what the finished thesis contradicts

### 7.1 The label map is stale in six rows

| CLAUDE.md line | claim | what is now true |
|---|---|---|
| Tabel 3.6 row | `<tab-3-5>` renders **Tabel 3.6** | renders **Tabel 3.7** |
| Tabel 3.7 row | `<tab-smote-geometry>` renders Tabel 3.7 | **object deleted** (Phase B3) |
| Tabel 3.8 row | `<tab-minority-census>` renders Tabel 3.8 | **object deleted** (Phase B3) |
| Tabel 3.9 row | `<tab-3-6>` renders **Tabel 3.9** | renders **Tabel 3.8** |
| Gambar 3.5 row | `<fig-3-4-smote-geometry>` renders Gambar 3.5 | **object deleted** (Phase B3) |
| — | (no row) | `<tab-3-explainer>` renders **Tabel 3.6** — **missing entirely** |

BAB 4's rows were re-derived in Phase B1 and are correct.

### 7.2 The skew note is wrong

> *"Note the label/number skew this table exists to prevent: `<tab-3-6>` renders as
> **Tabel 3.9**, `<tab-3-4>` as **Tabel 3.5**, `<tab-3-5>` as **Tabel 3.6**, and
> `<fig-3-4-smote-geometry>` as **Gambar 3.5**."*

Now: `<tab-3-6>` → **Tabel 3.8**; `<tab-3-4>` → Tabel 3.5 (still right);
`<tab-3-5>` → **Tabel 3.7**; `<fig-3-4-smote-geometry>` **no longer exists**.

### 7.3 "Chapters 4 and 5 are off-limits" (lines 158–169)

> *"BAB 4 … and BAB 5 … are off-limits… Keep BAB 4/5 byte-identical unless told
> otherwise."*

**False.** §4.3 and all of BAB 5 were replaced with the author's text. The correct
current state:

| region | status |
|---|---|
| BAB 1, BAB 2, BAB 3 | the author's text, applied |
| BAB 4 preamble, §4.1, §4.2 | **frozen** — unchanged except 20 `Paysim` → `PaySim` |
| BAB 4 §4.3 | the author's text, applied; **cites nothing** |
| BAB 5 | the author's text, applied; **cites nothing** |

### 7.4 "§4.3 … is complete and cites the frozen `results/shap_v2/**` tree"

**False.** The author's §4.3 contains **zero citations**. Its numbers derive from
`shap_v2` but the text does not cite it.

### 7.5 The chapter title

`CLAUDE.md` refers to *"BAB 5 (Penutup)"*. The chapter is now **KESIMPULAN DAN
SARAN**.

### 7.6 Two Typst conventions are undocumented

- `template.typ` now sets `set heading(supplement: [Subbab])` (added Phase B2) so
  heading references render "Subbab 4.3" rather than Typst's default "Section 4.3".
  **Safe only while every heading label targets level 2 or deeper** — a level-1
  label would render "Subbab 4". Not recorded in `CLAUDE.md`.
- The object-splice method that Phase B2's defect forced — derive a `#figure`
  block by walking back from its label and assert on both the first and last line,
  never by line range. Not recorded.

### 7.7 `results/analysis/` is not on the frozen list

`results/analysis/minority_census.csv` holds the `synthesis_multiplier` and
per-client census that BAB 3 §3.4.3 says are recorded. The frozen table covers
`clean_summary.csv`, `sweep/`, `logs/`, `models/`, `shap_v2/`, `shap/` — not
`analysis/`. Whether it should be frozen is the author's call.

---

## 8. What this audit did not check

- The six `@dong2026fcorr` rows in Tabel 4.7 are from an external paper and cannot
  be verified against frozen data.
- `analysis/separability/` outputs were treated as the source of record for the
  typology; they were not re-derived from the datasets.
- Prose *claims* were checked only where they carry a number. Statements of
  interpretation — that a difference is "signifikan", that a pattern is
  "konsisten" — are the author's judgement and were not second-guessed.
