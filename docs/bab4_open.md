# BAB 4 — open items after the §4.3 replacement

Opened 2026-09-24, when §4.3 was replaced in full with the author's own text
(`naskah-bab-4-3.md`). Each entry below records something the replacement moved,
dropped or exposed. **Nothing here is a defect to fix silently — these are
decisions for the author.**

Scope note: §4.1, §4.2 and the BAB 4 preamble (`content.typ` L2840–3319) are
frozen and were not touched. BAB 1–3 and BAB 5 were not touched.

---

## 1. Two SHAP measurement tiers are no longer disclosed in the thesis

The replaced §4.3 reported two channels that the author's version does not
mention at all.

**The exact-grouped tier.** 16 PaySim cells (`explainer = kernel-exact-grouped`
in `results/shap_v2/shap_summary_v2.csv`), in which the five one-hot `type`
columns are treated as one Shapley player so the coalition space can be
enumerated in full rather than sampled. The old §4.3 carried two dedicated
tables for it — `<tab-4-shap-exact>` and `<tab-4-shap-rawgap>` — plus the
sampled-versus-exact comparison: **mean absolute difference in weighted rank
correlation 0,028, maximum 0,099**, with the statistical decision agreeing on 11
of 12 configurations.

The disagreeing cell matters: on **BERT, IID, without SMOTE the p-value moved
from 0,0011 (sampled) to 0,1111 (exact)** — that is, the exact tier *reversed*
one significance decision. The old text drew the conclusion explicitly: that
configuration could not be used as evidence of a between-client difference on
the sampled result alone.

**The shared-background channel.** 12 cells (`bg = shared`), in which every
client uses one pooled background and explains its own data region — the
converse of the main arm. The old §4.3 reported 11 of 12 PaySim configurations
significant, and agreement lower under Non-IID on **5 of 6 pairs, p = 0,0469**.
Note the main arm's PaySim figure is 6 of 6; the two "sixes" are different
channels and the old text distinguished them. The author's §4.3 keeps the main
arm's 6 of 6 and drops the shared-background 5 of 6.

**Both remain in the record.** `results/shap_v2/shap_summary_v2.csv` still holds
all 124 rows (96 main + 16 exact-grouped + 12 shared-background), and
`docs/shap_rq3_run.md` still documents the method, including the standing
decision that `--include-ungrouped` was deliberately never run.

### Methodology check — the gap is transient, not permanent

The brief asked whether BAB 3 still promises these measurements. Checked both
versions:

- **The current `content.typ` BAB 3 §3.3.5 does describe both.** The
  shared-background arm at L2155 (*"background disatukan menjadi satu background
  bersama"*), and the exact-grouped route at L2164–L2178, including the
  deliberate non-run of the ungrouped route. **So as of today the thesis
  promises two measurements its results chapter no longer reports** — the same
  shape as the calibration gap.
- **The author's new BAB 3 (`docs/naskah/naskah-bab-3.md`) describes neither.**
  Searching it for `eksak`, `terkelompok`, `grouped`, `background bersama`,
  `two-bg`, `shared`, `pooled` returns three hits, none of them these: line 142
  is LinearSHAP's exactness for linear models, line 164 is the exact enumeration
  of the 945 matchings, line 232 is the minority-typology grouping.

**Conclusion: the gap exists only until the next phase applies the author's new
BAB 3, which closes it from the methodology side.** No action needed here; noted
so it is not re-opened as a finding.

---

## 2. The measured local-accuracy values lost their home

The deleted `<tab-4-shap-explainer>` (old Tabel 4.12, "Pemetaan explainer per
model") carried a measured *Local accuracy* column:

| models | explainer | local accuracy |
|---|---|---|
| LR, SVM | LinearSHAP (SVM: margin) | $9{,}26 \times 10^{-8}$ |
| GBM, XGB | TreeSHAP `interventional` | $0{,}00$ |
| FFD, BERT, FedXGBllr | KernelSHAP (`nsamples = 500`, `l1_reg=False`) | — |

The explainer *mapping* survives: it moves to the author's new BAB 3 Tabel 3.6,
which the next phase will build. But that table has a "Pengaturan utama" column,
not a measured-accuracy column, so these two numbers — and the explicit
`nsamples = 500` and `l1_reg=False` parameters — have no slot anywhere.

They are **results**, and the author's §4.3 has no place for them. Flagged for
the author; not restored.

---

## 3. Calibration availability is 78 rows, not 80 — and that is correct behaviour

`results/clean_summary.csv` carries three calibration columns: `test_brier`,
`test_cal_intercept`, `test_cal_slope`.

- `test_brier`: **80 of 96** populated
- `test_cal_intercept` and `test_cal_slope`: **78 of 96** populated

The 16 rows with no calibration at all are the SVM rows. That is by design:
`SGDClassifier(loss="hinge")` emits a decision margin, not a probability.

*(Updated after the BAB 1–2 replacement: the old BAB 2 §2.2.7 stated this — "untuk
SVM metrik kalibrasi dilaporkan sebagai `NA`". The author's new §2.2.8 does not.
See entry 10 below.)*

The extra two rows are **`paysim/fedxgbllr/dirichlet/none`** and
**`paysim/fedxgbllr/dirichlet/smote`**. Both carry a Brier score but no slope or
intercept, and both carry the *identical* Brier value `0.0012908728094771`.

**This is the degenerate-output guard working correctly, not missing data.**
Supporting arithmetic: PaySim's test prevalence is 0,00129 (the AUPRC lower
bound stated in the BAB 4 preamble). The Brier score of a predictor whose output
saturates near zero for every row equals the prevalence, 0,00129 — which is what
is recorded, to five significant figures. With predicted probabilities pinned
near 1e-09 there is no spread for a logistic recalibration to fit, so the slope
is undefined and correctly left empty rather than reported as a fabricated
number.

Record the true figure as **78 rows with a complete calibration triple**.

---

## 4. `dong2026fcorr` is introduced only in §4.1, never in BAB 2

`dong2026fcorr` is cited seven times in §4.1 — at `content.typ` L3091, in six
rows of `<tab-4-baf-auprc-bench>` (L3101–L3106), and at L3122 — and is
introduced nowhere in BAB 2's literature review.

**Pre-existing, and not caused by this edit.** §4.1 is frozen, so the fix, if
the author wants one, belongs in BAB 2. Logged only.

---

## 5. One resource file is now unreferenced

`buku/resources/fig-4-rq3-stability.png` — the old Gambar 4.2 — is no longer
referenced, because the author's §4.3 has no figure and BAB 4 now ends with
Gambar 4.1.

**Not deleted.** Per `CLAUDE.md`, an unreferenced resource is flagged, never
removed. Five other files in `buku/resources/` were already unreferenced before
this edit and are unrelated to it:
`chapter-2-power-digital-finance.png`, `fig-4-1-wireframe-weights.png`,
`fig-4-univariate-auc.png`, `fig-shap-jaccard-vs-kuncheva.png`,
`its-thesis-validation.png`.

---

# BAB 5 and bibliography — open items after the BAB 1–2 replacement

Appended 2026-09-24, when BAB 1 and BAB 2 were replaced with the author's text
(`naskah-bab-1.md`, `naskah-bab-2.md`). Kept in this file rather than a new
`bab5_open.md` because entries 1 and 5 are continuous with the BAB 4 items above
— the same replacement programme, and the author reads them together.

**Nothing here is acted on.** BAB 5 was not touched.

---

## 6. Three BAB 5 claims lose their BAB 1–3 support

By the author's decision (Phase B2 §3.5), old §2.2.6.1 and §2.2.6.2 are gone.
Three BAB 5 sentences now rest on material no longer in the thesis.

**6.1 The "21 seed" figure** — `content.typ:3373`:

> "…diperkirakan tidak membantu pada kondisi sekitar 21 seed, karena mayoritas
> seed akan tergolong sebagai *danger* atau *noise* — hal ini merupakan
> konsekuensi dari definisi algoritma-algoritma tersebut, bukan hasil yang telah
> diuji secara empiris."

The count 21 comes from the SMOTE geometry diagnostic in BAB 3 §3.4.3.1, which
the author's new BAB 3 drops. It still stands today because BAB 3 has not been
replaced; it becomes unsupported when BAB 3 lands.

**6.2 The one-hot fractional-value premise** — `content.typ:3355-3358`:

> "+ *Migrasi ke SMOTE-NC* untuk data bertipe campuran, sebagaimana diperkenalkan
> #cite(<chawla2002smote>, form: \"prose\"), guna menghindari nilai pecahan pada
> kolom kategorikal hasil one-hot. Rekomendasi ini kini memiliki dukungan
> empiris, bukan sekadar sitasi: pada @sec-hasil-rq3 di bawah SMOTE atribusi BAF
> SVM terkonsentrasi pada empat kolom `housing_status_*` sekaligus, konsekuensi
> teramati dari interpolasi kontinu SMOTE standar melintasi blok one-hot yang
> saling eksklusif."

Old §2.2.6.2 was the only place establishing that standard SMOTE produces
fractional values on one-hot columns. **This one is unsupported as of now.** The
*result* it points at survives — the author's §4.3 Tabel 4.15 reports the four
`housing_status` indicators — but the mechanism does not.

**6.3 The SMOTE variant survey** — `content.typ:3371-3372`:

> "Varian SMOTE yang membatasi seed pada perbatasan (@han2005borderline,
> @bunkhumpornpat2009safelevel) serta hibrida pembersihan (@batista2004balancing)
> diperkirakan tidak membantu…"

Old §2.2.6.1's last paragraph introduced all three variants. BAB 2 no longer
does, so BAB 5 cites three works the literature review never presents.
**Unsupported as of now.**

---

## 7. Bibliography — orphans, and what BAB 2 no longer introduces

`buku/thesis.typ:310-315` sets:

```typst
#bibliography("bibliography.bib", title: none, style: "apa", full: false)
```

**`full: false` means only cited keys render.** An orphaned key is therefore
invisible in DAFTAR PUSTAKA, not a stray entry. Confirmed as the setting in
force. Every key stays in `bibliography.bib`; nothing was removed.

**Orphaned now (2 of the 11 predicted):** `elor2022smote`, `fernandez2018smote`.
Both were cited only in the two BAB 2 subsections just removed.

**Still cited, all from BAB 3 (9 of the 11):** `aas2021explaining`,
`ducange2026fedshap`, `elreedy2024smote`, `lundberg2020treeshap`,
`riley2019minimum`, `sundararajan2020many`, `vansmeden2016epv`,
`vansmeden2019samplesize`, `visani2022stability`. These become orphaned only
when the author's BAB 3 lands — the 11-key figure is a prediction about the end
state, not the present one.

**Cited by a surviving chapter, no longer introduced in BAB 2 (11 keys):**

| key | cited in | note |
|---|---|---|
| `dong2026fcorr` | §4.1 (frozen) | pre-existing; never introduced in BAB 2 |
| `batista2004balancing` | BAB 5 | §6.3 above |
| `bunkhumpornpat2009safelevel` | BAB 5 | §6.3 above |
| `han2005borderline` | BAB 5 | §6.3 above |
| `blagus2013smote` | BAB 3, BAB 5 | BAB 3 citation goes when BAB 3 lands |
| `goorbergh2022harm` | BAB 3, BAB 5 | same |
| `elreedy2019smote` | BAB 5 | |
| `duan2019astraea` | BAB 5 | |
| `wang2021fedimbalance` | BAB 5 | |
| `weiss2007costsensitive` | BAB 5 | |

`napierala2016types` is **not** on this list: Phase B2 §3.2 retained it in BAB 2
§2.2.7, so §4.1's citation is introduced again.

---

## 8. `$1/N$` versus `$1/N_c$` — a wording candidate, not an error

The author's §2.2.9 writes `$g_(c,j) = 1/N_c sum ...$` and it was applied as
written. His BAB 3 restatement of the same formula writes `$1/N$`.

Both evaluate identically on this data. `experiments/shap_rq3.py:279` hands every
client the *same* explanation matrix in the main measurement
(`Xs = [X] * len(bgs)`, recorded in provenance as `"shared central test subset"`),
and every `stability.json` carrying the field records
`n_explained_per_client = [500, 500, 500, 500, 500]`. So $N_c$ does not vary by
client, and neither form is wrong.

`$1/N$` states the design more precisely — holding the explanation set fixed
across clients is exactly what isolates the background effect, and the subscript
implies a per-client count that never occurs. **Candidate BAB 2 wording change
for the author, not a correction.** Nothing changed.

---

## 9. `CLAUDE.md:158-169` no longer describes reality

The "Chapters 4 and 5 are off-limits" section, and its claim that "§4.3 …  is
complete and cites the frozen `results/shap_v2/**` tree", are both false now.
**Not rewritten — the standing rule is the author's to set.** The correct current
state, for when he does:

| region | status |
|---|---|
| BAB 1, BAB 2 | replaced with the author's text (this phase) |
| BAB 3 | not yet replaced; next phase |
| BAB 4 preamble, §4.1, §4.2 | **frozen** — byte-identical except the 20 `Paysim` → `PaySim` fixes |
| BAB 4 §4.3 | replaced with the author's text; **cites nothing at all** |
| BAB 5 | open, not yet written; no longer frozen |

---

## 10. The SVM calibration-`NA` basis is currently stated nowhere

A second transient gap of the same shape as entry 1, found while applying BAB 2.

The old BAB 2 §2.2.7 explained why SVM has no calibration metrics
(`content.typ` L1107–1110 before this phase): *"SVM diimplementasikan sebagai …
margin dan bukan probabilitas; untuk SVM metrik kalibrasi dilaporkan sebagai
`NA`"*. The author's new §2.2.8 describes Brier score, calibration slope and
calibration-in-the-large but **does not mention SVM or the `NA` case at all**.

The current BAB 3 does not state it either. So as of now the 16 `NA` rows in
`results/clean_summary.csv` have no stated basis anywhere in the thesis.

**It returns when BAB 3 lands.** The author's `naskah-bab-3.md` line 124 says:
*"Evaluasi tersebut dilakukan pada model yang menyediakan probabilitas, sedangkan
SVM dengan keluaran margin tidak dihitung metrik kalibrasinya."* The fact simply
relocates from BAB 2 to BAB 3 in his design.

Noted so it is not re-opened as a finding. **Nothing changed.**

*(Forward-consistency check while here: the author's BAB 3 cross-references
"Subbab 2.2.8" for the calibration metrics, and the new BAB 2 does render
Metrik Evaluasi untuk Imbalanced Classification as §2.2.8. The reference will
resolve.)*

---

# Chapters 1–4 closed — open items after the BAB 3 replacement

Appended 2026-09-25, when BAB 3 was replaced with the author's text
(`naskah-bab-3.md`). **BAB 1 through BAB 4 are now entirely in the author's own
words.** BAB 5 was not touched.

---

## 11. The "21 seed" claim in BAB 5 is now unsupported

This completes the set of three from entry 6. `content.typ:2909`:

> "Varian SMOTE yang membatasi seed pada perbatasan (@han2005borderline,
> @bunkhumpornpat2009safelevel) serta hibrida pembersihan (@batista2004balancing)
> **diperkirakan tidak membantu pada kondisi sekitar 21 seed**, karena mayoritas
> seed akan tergolong sebagai *danger* atau *noise* — hal ini merupakan
> konsekuensi dari definisi algoritma-algoritma tersebut, bukan hasil yang telah
> diuji secara empiris."

The count 21 came from the SMOTE geometry diagnostic in the old §3.4.3.1, which
this phase removed by the author's decision. Nothing in the thesis now states it.

**All three BAB 5 claims from entry 6 are now unsupported:** the "21 seed"
figure, the one-hot fractional-value premise behind the SMOTE-NC recommendation,
and the SMOTE variant survey with its three citation keys. The author will
resolve these when he writes BAB 5.

---

## 12. The eleven orphaned bibliography keys are now actually orphaned

Entry 7 recorded that only 2 of the 11 predicted orphans were orphaned, the other
9 still being cited from the old BAB 3. With BAB 3 replaced, **all eleven are
orphaned**:

`aas2021explaining` · `ducange2026fedshap` · `elor2022smote` ·
`elreedy2024smote` · `fernandez2018smote` · `lundberg2020treeshap` ·
`riley2019minimum` · `sundararajan2020many` · `vansmeden2016epv` ·
`vansmeden2019samplesize` · `visani2022stability`

45 of the 65 keys in `bibliography.bib` are cited; 20 are not (the eleven above
plus nine that were already unused before this programme began).

**`buku/thesis.typ` sets `full: false`, so an uncited key does not appear in
DAFTAR PUSTAKA.** Confirmed again after this edit. Every key stays in
`bibliography.bib`; nothing was removed.

---

## 13. Redraw briefs for the three raster figures

`<fig-3-1>`, `<fig-3-2>` and `<fig-3-3>` are raster PNGs with no generating
source. The author's new text changes what each must say. **No image file was
touched.**

**Gambar 3.1 — "Arsitektur umum sistem penelitian"** (`fig-3-1-*.png`).
The author's §3.1 now describes a **six-stage** pipeline in a fixed order: data
acquisition and preprocessing (three datasets, 70:15:15) → client partitioning
(five clients, IID or Dirichlet α = 0,5) → local training and imbalance handling
(with and without SMOTE) → federated aggregation (four paradigms: FedAvg for
LR/SVM, best-model selection for GBM, accuracy-weighted FedAvg for FFD/BERT,
tree ensemble aggregation for FedXGBllr) → experiment scenarios (centralized,
federated IID, federated Non-IID) → evaluation and explainability analysis. The
figure must show six stages in that order and must name the four aggregation
paradigms, since the text no longer enumerates them anywhere else near the
figure.

**Gambar 3.2 — "Lapisan logis sistem penelitian"** (`fig-3-2-*.png`).
The author's §3.3 now names **four** logical layers: data and preprocessing;
partition and client orchestration; model training with the four FL aggregation
paradigms; and evaluation covering both performance measurement and
explainability analysis. The image must show four layers with those labels — in
particular the fourth layer must show evaluation and explainability as one
layer, not two.

**Gambar 3.3 — "Skema pembagian data dua tingkat"** (`fig-3-3-*.png`).
The author's §3.3.2 describes a two-level split: level one is the global
70:15:15 stratified split at seed 42; level two partitions **only the training
set** across five clients by IID or Dirichlet. The image must show validation and
test sets staying whole at the simulation server and never being partitioned, and
must show SMOTE applied after partitioning and before training, on $D_k$ only.
It must not show the α-census variants (α = 1,0 and 5,0), which the author's text
no longer treats as training conditions.

---

## 14. Entry 1 (two SHAP tiers) — CLOSED

Entry 1 recorded that the thesis promised two measurements — the exact-grouped
tier and the shared-background arm — that the author's §4.3 no longer reports,
and predicted the gap would close when BAB 3 landed.

**It has.** Searching the rendered BAB 3 for `eksak terkelompok`, `terkelompok`,
`background bersama`, `two-bg`, `include-ungrouped` and `bg_shared` returns
nothing. The methodology no longer promises either measurement, so the results
chapter is no longer silent about something it committed to.

Both tiers remain in `results/shap_v2/shap_summary_v2.csv` (124 rows) and in
`docs/shap_rq3_run.md`. The thesis simply no longer discloses that they exist —
which is the author's decision, recorded in entry 1.

---

## 15. Entry 10 (SVM calibration-`NA` basis) — CLOSED

Entry 10 recorded that the basis for the 16 `NA` calibration rows was stated
nowhere, the old BAB 2 §2.2.7 having explained it and the author's §2.2.8 not.

**It is now stated in BAB 3 §3.3.5**, as predicted:

> "Evaluasi tersebut dilakukan pada model yang menyediakan probabilitas,
> sedangkan SVM dengan keluaran margin tidak dihitung metrik kalibrasinya."

The fact relocated from BAB 2 to BAB 3 in the author's design.

**Calibration itself stays open — see entry 3, unchanged.** §2.2.8 and §3.3.5
both commit to Brier score, calibration slope and calibration-in-the-large;
BAB 4 reports none; and a calibration subsection would belong in frozen §4.1.

---

## 16. One more resource file is unreferenced

`buku/resources/fig-3-4-smote-geometry-baf.png` — the old Gambar 3.5 — is no
longer referenced, the SMOTE geometry diagnostic having been removed.

**Not deleted.** Seven files in `buku/resources/` are now unreferenced across
`content.typ`, `thesis.typ` and `template.typ`:

`chapter-2-power-digital-finance.png` · `fig-3-4-smote-geometry-baf.png` ·
`fig-4-1-wireframe-weights.png` · `fig-4-rq3-stability.png` ·
`fig-4-univariate-auc.png` · `fig-shap-jaccard-vs-kuncheva.png` ·
`its-thesis-validation.png`

Two of those seven were orphaned by this programme (`fig-4-rq3-stability.png` in
B1, `fig-3-4-smote-geometry-baf.png` here); the other five predate it.
