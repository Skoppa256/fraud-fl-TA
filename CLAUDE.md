# CLAUDE.md — working agreement for this repository

Guidance for Claude Code (and any AI agent) working in `fraud-fl-TA/`.

## `buku/` is a first-class deliverable

`buku/` holds the undergraduate thesis (Tugas Akhir) source, written in Typst
(`thesis.typ`, `template.typ`, `content.typ`, `bibliography.bib`, `resources/`).
It is **authored documentation, not generated output** — treat it with the same
care as the code. Do not regenerate it, overwrite it wholesale, or treat its
contents as disposable.

Build/verify from the repo root:

```bash
typst compile buku/thesis.typ buku/thesis.pdf     # one-shot
typst watch   buku/thesis.typ buku/thesis.pdf     # live
```

## Frozen artifacts — read-only, regeneration is a decision

These paths are results of record. Prose in `buku/` cites them, and the
provenance argument depends on their not moving. **Do not re-run, overwrite,
re-sort, or "refresh" them.** Regenerating any of them requires an explicit
decision by the author — it is never a step inside another task.

| path | what it is |
|---|---|
| `results/clean_summary.csv` | v1 sweep summary, final |
| `results/sweep/sweep_master.csv` | v1 sweep master, final |
| `results/logs/**` | per-run round logs backing the above |
| `results/models/**` | persisted model artifacts the sweep produced |
| `results/shap_v2/**` | **RQ3 v2, final.** Read-only. Regenerating requires an explicit decision, not a re-run. |
| `results/analysis/**` | **Derived analysis of record.** `minority_census.csv` holds the per-client minority census and the `synthesis_multiplier` that BAB 3 §3.4.3 says are recorded. Frozen 2026-09-29 because a thesis claim rests on it. |
| `analysis/separability/**` | **Derived analysis of record — a different tree from `results/analysis/`.** `separability_summary.csv` is the source for the minority-typology table and the 93,1% that BAB 5 restates. Frozen 2026-09-29 (6 tracked files) for the same reason: a thesis claim rests on it. |
| `results/shap/**` | **v1 SHAP tree — superseded, but load-bearing.** `production.log` is the only surviving record of the pre-repair run, in which two PaySim FedXGBllr cells scored `spearman=0.0000 kuncheva=1.0000`; it cannot be regenerated (commit `1d661ac`, 2026-08-07, fixed the clipped-logit saturation that produced it). No figure from this log is cited in `buku/` — the v1 `0,775` this row previously claimed for BAB 4 §4.3 appears nowhere in `content.typ`. The log is kept as the record of the pre-repair state, not as a source for prose. `experiments/shap_analysis.py` targets this directory; since 2026-11-13 its `main()` refuses to start (the write guard below). |

`docs/frozen_manifest.sha256` records SHA-256 for the frozen trees. For files
that are tracked in git it is redundant with git's own object hashes; keep it
only as a portable, git-independent check.

Coverage is **partial**: `results/models/**` is not in the manifest, because the
persisted artifacts exist only on the GPU box and the manifest was generated on
the laptop. A green `shasum -a 256 -c docs/frozen_manifest.sha256` therefore
attests to seven of the eight frozen rows and says nothing about the model
artifacts. Run `analysis/append_model_manifest.sh` on the box to close it.

Note on `.gitignore`: the results rule is now `results/*` (ignore the
directory's *contents*) rather than `results` (ignore the *directory*), because
a directory-level ignore cannot be undone by a negation on a path inside it.
That change is what allowed `results/shap_v2/**` to be re-included and
committed. Results that were already tracked are unaffected by the switch.

**Per-seed trees — `results/seeds/seed<N>/**` (2026-11-13), not frozen.** The
multi-seed re-run writes here, one tree per training seed, each with the frozen
layout (`sweep/`, `logs/`, `models/`, `shap_v2/`, `clean_summary.csv`).
`experiments/run_sweep.py` sends every run to its seed's tree, seed 42 included,
so a launch — or the smoke gate — can no longer land on the seed-42 record.
`experiments/shap_rq3.py` requires `--seed N` and reads and writes only inside
that tree; `verify_hashes`, `analysis/verify_kernel_tier.py` and
`analysis/rq3_wilcoxon.py` read one with `--seed N`, and
`experiments/build_seed_summary.py --seed N` builds its `clean_summary.csv` with
the recipe that reproduces the frozen one. `results/cache/` stays shared: its
entries are named by seed. The tree is ignored by the `results/*` rule, so
nothing in it reaches git until the author decides to un-ignore it.

`evaluation/results_paths.py` holds the paths and a write guard: the shared
writers (`results_writer`, `model_persistence`), the runner and `shap_rq3.py`
refuse any path inside a frozen folder under `results/`. An entry point run by
hand therefore fails loudly unless `SWEEP_RESULTS_ROOT` (absolute) names a tree.
Regenerating a frozen artifact means editing the guard's `FROZEN` tuple — an
explicit decision, deliberately not a flag. The guard does **not** cover the
one-off scripts that wrote the frozen trees with their own writers —
`experiments/backfill_recall_at_fpr.py`, `experiments/migrate_rounds_schema.py`,
`notebooks/minority_census.py`, `analysis/separability/separability_analysis.py`
— so the rule above still applies to them: do not re-run them.

## Re-run code path (2026-11-13): every draw from the seed, FedXGBllr's trees fixed

The seed-42 record in `results/` came from code that differs from today's in two
ways, so today's code does not reproduce it bit for bit. A multi-seed comparison
therefore re-runs seed 42 as well, into `results/seeds/seed42/`.

- **Every random draw comes from the run seed** (`experiments/determinism.py`).
  Strategies average client results in client-id order (`in_client_order`),
  entry points call `seed_everything`, and torch is asked for deterministic
  kernels in the driver and in every Ray actor. FedXGBllr's per-round,
  per-client draws use `derive_seed`. Verified bit-identical on CPU; on a GPU box,
  a repeat run of one cell is the check, because `torch_deterministic` warns
  rather than raises on a kernel that has no deterministic implementation.
- **FedXGBllr's trees are fixed after round 0** (Ma et al. 2023, §3.4,
  Algorithm 1), as BAB 3 already describes (content.typ §3.3.4). In round 0
  every client builds its ensemble once; the server collects all five in
  client-id order and starts the CNN from the seed. Rounds 1..R train only the
  CNN, on 50 batches of 32 rows per round drawn from (seed, round, client). The
  Flower `hfedxgboost` baseline this code came from re-fit every client's trees
  in every round, and fed round 1 one client's trees in all five blocks; the
  seed-42 record was produced that way. Every other hyperparameter is the
  record's (BAB 3 `<tab-3-6>`): 50 trees per client, depth 6, CNN lr 0.0005, 20
  rounds for PaySim and 50 for ULB and BAF, early stopping with patience 10. On
  ULB IID the mean change in test AUPRC per round was 0.0156 with the old code
  (seed 42) and is 0.0015 / 0.0013 / 0.0012 now (seeds 42 / 123 / 2024); the
  last round is within 0.0026 of the best-validation round. The round-1
  difference moved the final AUPRC by at most 0.0015 (seeds 42 and 123, before
  `base_score` was set).
- The paper's own settings (§4.1: 100 trees per client, depth 8, E = 100 epochs
  over all rows at batch 64, lr 0.001, R = 10) were tried on 2026-11-13 and
  dropped by the author: on ULB IID they swung 0.0181 per round, and a PaySim
  cell is ~70M CNN steps (~20 h on a laptop CPU).
- **FedXGBllr's client XGBoost sets `base_score: 0.5`** (the author, 2026-11-13;
  `tests/test_fedxgbllr_config.py`). xgboost 2.0.0 (requirements.txt) otherwise
  estimates each booster's intercept from its data (0.12 on ULB), so every
  single-tree margin the CNN reads is about −2.09 and varies by only ~0.02 (one
  standard deviation) across rows and trees. The paper predates 2.0; earlier
  versions default to 0.5, an offset of 0. With the estimate the CNN learns
  little beyond the offset: on ULB IID at seeds 42 / 123 / 2024 it scored 0.7385
  / 0.7217 / 0.6629 test AUPRC, below the plain sum of its own trees (0.7754 /
  0.7703 / 0.6846). With 0.5 it scores 0.7716 / 0.7333 / 0.6656, at or above its
  trees' sum (0.7691 / 0.7321 / 0.6597), moves 0.0012–0.0015 per round and runs
  21–50 rounds. The seed-42 record has the offset. The centralized XGBoost
  comparator keeps the estimated intercept: it has no CNN to mislead.

**Thesis text deliberately not updated — the author's decision (2026-11-13).** The
author decides after analysing the new runs whether they enter `buku/`. BAB 3
already describes this FedXGBllr (trees once in round 0, then CNN-only rounds;
content.typ §3.3.4) and lists its settings (`<tab-3-6>`), so adopting the new runs
changes results, not methods:

- Every FedXGBllr number in BAB 4, and the other federated models' numbers
  (within run-to-run noise).
- If more than one seed enters the thesis, the sentences that name seed 42
  (§3.3.4, §3.4.4) name every seed used. §3.4.4's statement that the seed governs
  partitioning, sampling and model initialisation holds for the re-run; it did not
  fully hold for the seed-42 record (unseeded starting weights for FFD, BERT and
  FedXGBllr; client results averaged in arrival order).
- `<tab-3-6>` gains a FedXGBllr `base_score` 0,5 row, with a sentence on why.

Separately, §3.3.4 and §3.4.4 say the final model is chosen by the best
validation AUPRC. No federated model's code does that — all report the last
round. That mismatch predates this change and has been flagged to the author.

## Documentation ships with the code — same change, not later

Any change to the **pipeline, models, aggregation strategies, hyperparameters,
dataset handling, or environment/dependencies** requires a matching update to
`buku/content.typ` (and `buku/bibliography.bib` when a new method or citation is
involved) **within the same change**. A code change that leaves the thesis
describing the old behaviour is an **incomplete change**.

If a code change genuinely has **no** documentation impact, say so explicitly in
your summary (e.g. "no `buku/` update needed: refactor only, no
behaviour/config/dependency change") rather than skipping the question silently.

## Typst conventions

- **Never break a source line before** `=`, `-`, `+`, `*`, `#`, `@`, `<`, or a
  `digit.` — at the start of a line Typst reads these as markup (heading, list
  item, function call, reference, label, enumeration) rather than as content.
  Re-wrap the preceding line instead.
- **`buku/template.typ` sets `figure(kind: image)` as the document default.** A
  table figure must therefore pass `kind: table` **explicitly**, or it is
  numbered and captioned as "Gambar" and lands in the figure outline instead of
  the table outline. This has bitten once. Every table in `content.typ` carries
  `kind: table` — copy an existing one rather than writing a `#figure` from
  scratch.
- **All figure and table design lives in `buku/template.typ`, never at the call
  site** (author's instruction, 2026-10-31). A table passes only `kind: table`,
  `columns` (all `auto`), its cells (`table.header`, `rowspan`, `colspan`), a
  caption and a label. An image passes only its path, a caption and a label. No
  `text(size:)`, `align`, `inset`, `stroke`, `fill`, `width`, `#set`,
  `table.hline`, `#block` wrapper or manual `#pagebreak()` inside a chapter. The
  template supplies 10pt text, a bold first header row, centred non-justified
  cells that still hyphenate, rows that never split, code identifiers that wrap
  at `_`, images at 80% width, and tables that stay whole with their caption
  unless taller than a page, in which case they split and repeat their header.
  If one table or figure renders badly, change the global rule or raise it with
  the author; do not style that element. Bold on a single data cell
  (`<tab-4-cal-slope>` group rows, `<tab-4-baf-auprc-bench>`, `<tab-typology>`)
  is content emphasis and stays. The same rule covers the front matter: ABSTRAK
  and ABSTRACT use the template's spacing, so a long abstract runs onto a second
  page rather than being squeezed.
- Figure and table counters reset per BAB (`template.typ` resets both at each
  numbered level-1 heading), so numbers are per-chapter and shift whenever a
  figure or table is inserted earlier in the same chapter.
- **`template.typ` sets `set heading(supplement: [Subbab])`** so a heading
  cross-reference renders "Subbab 4.3" rather than Typst's English default
  "Section 4.3". A supplement set on `heading` applies at **every level**, so this
  is safe only while every labelled heading is level 2 or deeper — a level-1 label
  would render "Subbab 4", which is wrong. Before adding a label to any heading,
  run:

  ```bash
  grep -nE '^(=+) .*<[A-Za-z0-9_-]+>$' buku/content.typ   # every labelled heading
  ```

  and confirm none is level 1.
- **Splice a `#figure` block by walking back from its label, never by line
  range.** Find the line matching `) <label>`, walk backwards to the nearest
  `#figure(`, and assert on **both** the first and the last line of the block
  before moving it. A line-range grab once carried a whole trailing paragraph of
  prose into a replacement; the build was clean and the PDF looked right, and only
  a token-level diff against the source caught it.

## Writing consistency — one term, one form

The thesis text must be consistent throughout (author's instruction, 2026-10-31):
a term is written **identically everywhere it renders**, in prose, tables,
captions and headings. Capitalization varies only where position requires it: the
first word of a sentence or table cell, and title-case headings.

| write | not | note |
|---|---|---|
| `PaySim` | `Paysim`, `PAYSIM` | Lowercase `paysim` is fine only in labels and citation keys, which never render. |
| `terpusat` (`Terpusat` at the start of a table cell) | `centralized`, `Centralized` | See the exception below. |
| `IID` / `Non-IID` for a data condition | `non-IID`, `Non IID`, `Dirichlet` as a label | `Dirichlet` stays only for the method or distribution itself (`partisi Dirichlet`, `distribusi Dirichlet`, the parameter row in `<tab-3-hp-umum>`). Where α matters: `Non-IID ($alpha$ = 0,5)`. |
| `SMOTE` / `Tanpa SMOTE` for an imbalance condition | `none`, `no-SMOTE`, `Non-SMOTE`, `tanpa oversampling`, `Dengan SMOTE` as a label | `tanpa` is lowercase mid-cell (`IID tanpa SMOTE`). Prose may say `dengan SMOTE` where the sentence needs it; table labels say `SMOTE`. |
| `Federated Learning` | `federated learning` | `federated` alone, as an adjective, stays lowercase. |
| `deep learning` | `Deep Learning`, `DL` | Mid-sentence. `DL` is never defined; `ML` is, in §1.1. |
| `financial fraud detection` | `Financial Fraud Detection` | Mid-sentence. |
| `best-model selection`, `accuracy-weighted FedAvg` | `Best-Model Selection`, `Accuracy-Weighted FedAvg` | Mid-sentence. A table cell may capitalize its first word only. |
| `antar client`, `dalam client`, `antar kelas` | `antar-client`, `dalam-client`, `antarkelas` | |
| `dua seed`, `per-client` | `dua-seed`, `per client` | |
| `1D CNN`, `F1-score`, `validation set`, `non-linear`, `gradient` | `1D-CNN`, `F1` alone, `Validation-set`, `nonlinier`, `gradien` | |
| `di mana`, `di antara` | `dimana`, `diantara` | KBBI spelling. |

**The `terpusat` exception is language.** `centralized` stays only where English is
needed: the ABSTRACT, the author's `centralized machine learning` in ABSTRAK, and
the two glosses that introduce the English term, `terpusat (centralized)` in §1.1
and `terpusat atau centralized` in §2.1.2.

**Never unify these, because they are names:** paper titles in `<tab-2-1>`, dataset
names (`Financial Fraud Detection Dataset`, `Credit Card Fraud Detection Dataset`),
FFD's expansion (`Financial Fraud Detection network`), package and class names
(`shap`, `LogisticRegression`, `hfedxgboost`), Kaggle slugs, and labels and citation
keys.

**Check after any change to thesis text.** The only expected hits are the two
abstracts and the two glosses above:

```bash
grep -nE 'Paysim|PAYSIM|[Cc]entralized|Non IID|non-IID|[Dd]imana|diantara|[Nn]o-SMOTE|[Nn]on-SMOTE|\bnone\]|Dirichlet (none|SMOTE|\$alpha\$ =)|[Dd]engan SMOTE\]|[Tt]anpa [Oo]versampling' \
  buku/content.typ buku/thesis.typ | grep -v '^[^:]*:[0-9]*: *//'
```

**Grep cannot see inside images.** Check any new drawing by eye. Gambar 3.2 and 3.3
say `PaySim` as of 2026-10-31; a term changed later will not reach the hand-drawn
figures without a redraw by the author.

**Pairs the author has not ruled on.** These are different words for one idea, so
choosing is the author's call. Do not unify them without asking, and move a pair
into the table above once decided:

- `paradigma agregasi` / `skema agregasi` / `aturan agregasi`
- `explainability` / `interpretabilitas`
- `stabilitas` / `kestabilan` / `konsistensi` for what RQ3 measures
- `feature importance` / `kepentingan fitur`
- `berbasis tree` / `tree-based` / `berbasis pohon`, and `tree ensemble` / `ensemble pohon`
- `sampel minoritas` / `contoh minoritas`
- `penelitian ini` / `studi ini`
- `putaran` / `Global rounds`
- `performa` / `kinerja`
- `sel` / `konfigurasi`
- `pembanding terpusat` / `baseline terpusat`
- `learnable learning rates` / `Learnable LR` (in `<tab-3-4>`, where `LR` also means Logistic Regression)
- Title case or sentence case for table captions
- Decimal places for one metric across tables (`0,167` in `<tab-4-auprc-baf>`, `0,1670` in `<tab-4-baf-auprc-bench>`)

## Where to update — code change → thesis location

Find tables and figures by their **Typst label and caption**, not by displayed
number: both are auto-numbered, so label ids do NOT match the rendered
"Tabel 3.x" / "Gambar 3.x". The mapping below was re-derived from `content.typ`
and cross-checked against the rendered `thesis.pdf`. **Re-derive it — do not
patch single rows — after adding or removing any figure or table.**

Tables:

| renders as | label | caption |
|---|---|---|
| Tabel 2.1 | `<tab-2-1>` | Rangkuman Hasil Penelitian Terdahulu |
| Tabel 3.1 | `<tab-3-1>` | Karakteristik Dataset PaySim |
| Tabel 3.2 | `<tab-3-2>` | Deskripsi Fitur Dataset PaySim |
| Tabel 3.3 | `<tab-3-3>` | Karakteristik Dataset ULB Credit Card |
| Tabel 3.4 | `<tab-baf-char>` | Karakteristik Dataset Bank Account Fraud (BAF) |
| Tabel 3.5 | `<tab-3-smote>` | Konfigurasi Parameter SMOTE |
| Tabel 3.6 | `<tab-3-4>` | Pemetaan Model dengan Skema Agregasi Federated Learning dan Model Pembanding Terpusat |
| Tabel 3.7 | `<tab-3-explainer>` | Pemetaan explainer per model |
| Tabel 3.8 | `<tab-3-5>` | Spesifikasi Lingkungan Pengembangan |
| Tabel 3.9 | `<tab-3-hp-umum>` | Konfigurasi Umum Eksperimen |
| Tabel 3.10 | `<tab-3-6>` | Konfigurasi Hyperparameter Eksperimen per Model |
| Tabel 4.1 | `<tab-4-fitur-paysim>` | Fitur dan tipe data PaySim setelah preprocessing |
| Tabel 4.2 | `<tab-4-fitur-ulb>` | Fitur dan tipe data ULB setelah preprocessing |
| Tabel 4.3 | `<tab-4-fitur-baf>` | Fitur dan tipe data BAF setelah preprocessing |
| Tabel 4.4 | `<tab-4-auprc-ulb>` | AUPRC test pada ULB tanpa SMOTE |
| Tabel 4.5 | `<tab-4-auprc-baf>` | AUPRC test pada BAF tanpa SMOTE |
| Tabel 4.6 | `<tab-4-auprc-paysim>` | AUPRC test pada PaySim tanpa SMOTE |
| Tabel 4.7 | `<tab-4-rfpr-ulb>` | Recall\@5%FPR test pada ULB tanpa SMOTE |
| Tabel 4.8 | `<tab-4-rfpr-baf>` | Recall\@5%FPR test pada BAF tanpa SMOTE |
| Tabel 4.9 | `<tab-4-rfpr-paysim>` | Recall\@5%FPR test pada PaySim tanpa SMOTE |
| Tabel 4.10 | `<tab-4-cal-slope>` | Calibration slope test per model dan kondisi pada ketiga dataset |
| Tabel 4.11 | `<tab-4-baf-auprc-bench>` | Perbandingan AUPRC test pada BAF Base |
| Tabel 4.12 | `<tab-typology>` | Tipologi contoh minoritas |
| Tabel 4.13 | `<tab-4-iid-noniid>` | Perbandingan AUPRC test kondisi IID dan Non-IID tanpa SMOTE |
| Tabel 4.14 | `<tab-4-iid-smote>` | Efek SMOTE terhadap AUPRC pada kondisi IID untuk dataset ULB dan PaySim |
| Tabel 4.15 | `<tab-4-baf-smote>` | Efek SMOTE terhadap AUPRC pada kondisi Non-IID untuk dataset BAF |
| Tabel 4.16 | `<tab-4-baf-smote-rfpr>` | Efek SMOTE terhadap Recall\@5%FPR pada kondisi Non-IID untuk dataset BAF |
| Tabel 4.17 | `<tab-4-rq3-kuncheva>` | Rata-rata indeks Kuncheva dan Jaccard\@5 lima fitur terpenting per dataset |
| Tabel 4.18 | `<tab-4-rq3-noniid>` | Korelasi peringkat berbobot antar client pada kondisi IID dan Non-IID tanpa SMOTE |
| Tabel 4.19 | `<tab-4-rq3-noniid-smote>` | Korelasi peringkat berbobot antar client pada kondisi IID dan Non-IID dengan SMOTE |
| Tabel 4.20 | `<tab-4-rq3-kuncheva-noniid>` | Indeks Kuncheva antar client pada kondisi IID dan Non-IID tanpa SMOTE |
| Tabel 4.21 | `<tab-4-rq3-kuncheva-noniid-smote>` | Indeks Kuncheva antar client pada kondisi IID dan Non-IID dengan SMOTE |
| Tabel 4.22 | `<tab-4-rq3-smote>` | Perubahan konsistensi dan fitur penting setelah SMOTE pada beberapa konfigurasi Non-IID |
| Tabel 4.23 | `<tab-4-rq3-seeds>` | Kesepakatan pengulangan dan kesepakatan antar client pada KernelSHAP tersampel |

Figures:

| renders as | label | caption |
|---|---|---|
| Gambar 2.1 | `<fig-2-1>` | Arsitektur umum Federated Learning |
| Gambar 2.2 | `<fig-2-2>` | Arsitektur FedXGBllr |
| Gambar 2.3 | `<fig-2-3>` | Visualisasi pengaruh parameter $alpha$ pada distribusi label antar client |
| Gambar 2.4 | `<fig-2-4>` | Ilustrasi mekanisme SMOTE |
| Gambar 2.5 | `<fig-2-5>` | Contoh visualisasi SHAP dalam bentuk summary plot |
| Gambar 2.6 | `<fig-2-6>` | Arsitektur Flower |
| Gambar 3.1 | `<fig-3-1>` | Alur pelaksanaan penelitian. |
| Gambar 3.2 | `<fig-3-2>` | Arsitektur Umum Sistem. |
| Gambar 3.3 | `<fig-3-3>` | Skema pembagian data dua tingkat. |
| Gambar 3.4 | `<fig-3-4>` | Skema pengukuran dua seed. |
| Gambar 4.1 | `<fig-4-auprc-boxplot>` | Distribusi AUPRC test seluruh konfigurasi per dataset |
| Gambar 4.2 | `<fig-typology>` | Distribusi tipe contoh minoritas (k = 5 tetangga terdekat) per dataset. Kelas minoritas BAF hampir seluruhnya *rare* dan *outlier*, sedangkan ULB didominasi *safe*. |
| Gambar 4.3 | `<fig-4-rq3-spearman>` | Korelasi peringkat tanpa bobot antar client |
| Gambar 4.4 | `<fig-4-rq3-weighted>` | Korelasi peringkat berbobot antar client |

Note the label/number skew this table exists to prevent: `<tab-3-4>` renders as
**Tabel 3.6**, `<tab-3-explainer>` as **Tabel 3.7**, `<tab-3-5>` as **Tabel 3.8**,
`<tab-3-hp-umum>` as **Tabel 3.9**, and `<tab-3-6>` as **Tabel 3.10**. The skew
widened by one on 2026-10-05 when `<tab-3-smote>` was inserted in §3.3.3. Cite with `@label`; never write a literal number
into prose — a `@label` reference follows a renumber, a literal does not. As of
2026-09-29 `content.typ` contains **zero** literal "Tabel N.N" in prose; keep it
that way.

| Code area changed | Thesis location to update |
|---|---|
| Hyperparameters / any `conf/*.yaml` / `registry.yaml` values | Per-model values: Tabel «Konfigurasi Hyperparameter Eksperimen per Model» `<tab-3-6>`. Values shared by every model (K, rounds, α, seed, SMOTE): Tabel «Konfigurasi Umum Eksperimen» `<tab-3-hp-umum>` |
| Dependencies / versions (`requirements.txt`, `pyproject.toml`) | Tabel «Spesifikasi Lingkungan Pengembangan» `<tab-3-5>` |
| Preprocessing (`preprocessing/paysim.py`, dropped cols, features, encoding, scaling, split) | **§3.3.1**, **§3.4.1**, and the feature tables `<tab-4-fitur-paysim>`, `<tab-4-fitur-ulb>`, `<tab-4-fitur-baf>` with their intro paragraph in the BAB 4 preamble |
| Client partitioning (`partitioning/dirichlet.py`, IID/Dirichlet, α set, K) | **§3.3.2** and **§3.4.2** |
| Oversampling / SMOTE / ADASYN (`preprocessing/smote.py`, `adasyn.py`, `oversampling.py`) | Tabel «Konfigurasi Parameter SMOTE» `<tab-3-smote>` (`sampling_strategy`, `k_neighbors`, `random_state`, `enabled`), **§3.3.3** and **§3.4.3** |
| Models & aggregation strategies (`models/*/`, any `strategy.py`) | Tabel «Pemetaan Model dengan Skema Agregasi» `<tab-3-4>`, **§3.3.4**, **§3.4.4** |
| Evaluation metrics & SHAP (`evaluation/`, `experiments/shap_*.py`) | **§3.3.5** and **§3.4.5** |
| Dataset facts (row/feature count, fraud ratio, columns) | `<tab-3-1>` / `<tab-3-2>` (PaySim), `<tab-3-3>` (ULB), `<tab-baf-char>` (BAF) |
| Pipeline restructuring (stage order/count) | **Gambar 3.1** — and **Gambar 3.2** if the logical-layer / model / scheme counts change. Both are hand-drawn: flag it for the author, there is no regeneration step |
| New method or paper referenced | **`bibliography.bib`** + the citing sentence |

BAB 3/4 figures split into two kinds, and `analysis/figures/make_all.py`
enforces that every one of them is accounted for (`--check` fails on a figure
that is in neither set):

- **Generated** (`GENERATORS`) — Gambar 4.1 (`fig_4_auprc_boxplot.py`, AUPRC
  boxplot of all 96 cells, added in §4.1.1 on the author's instruction
  2026-11-05), Gambar 4.2 (`fig_4_1_minority_typology.py`; the script name
  predates the renumber) and Gambar 4.3 and 4.4 (both from
  `fig_4_rq3_korelasi_boxplot.py`: between-client Spearman tanpa bobot and
  berbobot per model over the 66 federated main cells, no inset and each with
  its own Y scale (lowest cell floored to 0,05 up to 1,00: 0,75 tanpa bobot,
  0,65 berbobot) since 2026-11-12, so they are not meant to be compared by
  eye; together they replaced Tabel `<tab-4-rq3-corr>` on
  2026-11-05, and the script checks that the paragraph after
  `<fig-4-rq3-weighted>` quotes the weighted medians and the FedXGBllr range
  the data gives),
  built from the experiment CSVs. Do not hand-edit
  those PNGs: change the script, or the data, and rebuild.
- **Authored by hand** (`EXTERNAL`) — the method diagrams Gambar 3.1–3.4. As of
  the 2026-10-04 revision (3.1–3.3) and 2026-10-05 (3.4) the author draws these
  in an external tool and delivers them in `TATATA.docx` (latest: 3.3 and 3.4
  redrawn 2026-11-05, dropped into `resources/` and renamed onto the existing
  filenames); `fig_3_1_pipeline.py`,
  `fig_3_2_logical_layers.py`, `fig_3_3_two_level_split.py` and
  `fig_3_4_two_seeds_gap.py` are kept but no longer own anything the thesis
  renders. **Nothing rebuilds these**, and unlike the generated versions they
  superseded they carry no row counts, fraud shares, per-client counts, K or α —
  so a pipeline change that moves those numbers will not reach them, and cannot
  silently contradict them either. A method change means asking the author for a
  redrawn diagram.

  **Gambar 3.4 is the one that cost something.** It replaced a *generated*
  figure, not a hand-drawn draft, so the swap dropped a data-bearing panel: the
  old `fig-3-two-seeds-gap.png` plotted a real cell (BAF BERT Non-IID tanpa
  SMOTE) from the RQ3 CSVs, and its caption stated that finding. The replacement
  is structural only. The superseded PNG is still in `resources/` and is now
  unreferenced — flagged, not deleted, per the rule below.

BAB 2 figures are external/illustrative and are not generated.
Not every PNG in `resources/` is referenced: some are from dropped drafts. An
unreferenced file is not automatically dead — flag it, do not delete it.

## What is frozen in `buku/content.typ`

All five chapters are now the author's own words, applied over 2026-09-24…27.
What is frozen is no longer "BAB 4 and 5" but the following:

| region | status |
|---|---|
| BAB 1, BAB 2, BAB 3 | the author's text. Change only on his instruction. |
| BAB 4 preamble, §4.1, §4.2 | **frozen**, with §4.1.1 and §4.2.1 re-frozen as of 2026-10-05. **No longer byte-identical since 2026-10-31**, on the author's two instructions of that date: table design moved to `template.typ`, and wording unified under «Writing consistency». Both changed no number, cell value or claim. The preamble also gained an intro paragraph and three feature tables (`<tab-4-fitur-paysim>`, `<tab-4-fitur-ulb>`, `<tab-4-fitur-baf>`) on the author's instruction (2026-10-31), which moved every other BAB 4 table number up by three. The six AUPRC and Recall\@5%FPR tables (`<tab-4-auprc-*>`, `<tab-4-rfpr-*>`) were reduced to their tanpa SMOTE columns, also on the author's instruction (2026-10-31); the removed values remain in `results/clean_summary.csv`, and prose quoting them is flagged to the author, not rewritten. On 2026-11-05, on the author's instruction (from `TATATA_konsisten.docx`), §4.1.1's paragraph after the three AUPRC tables was replaced by two paragraphs plus `<fig-4-auprc-boxplot>` (Gambar 4.1), which moved the typology figure to Gambar 4.2; the dropped "773/484/15 kali batas bawah" sentence still stands in §4.1.3. §4.2.1 gained one sentence and `<tab-4-iid-noniid>` on the author's instruction; its three paragraphs are otherwise untouched. Its history: it was unfrozen for a supplementary-metrics insertion, then revised again on the author's instruction — the three `<tab-4-supp-*>` tables were replaced by the single `<tab-4-cal-slope>` (calibration slope only; F1/Precision/Recall/FPR aktual/Brier/CITL are no longer reported anywhere), and the closing two paragraphs are now the author's own text. |
| BAB 4 §4.3 (`<sec-hasil-rq3>`) | the author's replaced text. **It cites nothing** — no `@key`, no author-year. On 2026-11-05, on the author's instruction, Tabel `<tab-4-rq3-corr>` was replaced by two boxplots, `<fig-4-rq3-spearman>` (Gambar 4.3) and `<fig-4-rq3-weighted>` (Gambar 4.4); its one introducing sentence was reworded to describe distributions, and Tabel 4.18–4.21 became 4.17–4.20. Later the same day, also on the author's instruction, the IID vs Non-IID comparison in §4.3.2 was widened from the three KernelSHAP models to all six: `<tab-4-rq3-noniid>` (tanpa SMOTE, 18 rows) and `<tab-4-rq3-noniid-smote>` (dengan SMOTE, 12 rows) cover all 30 pairs, and the prose quotes 20/30 decreases, mean 0,033, Wilcoxon p = 0,0078, with the per-group split (KernelSHAP 11/15, 0,054, p = 0,0042; LR/SVM/GBM 9/15, 0,011, p = 0,21). Source of those numbers: `analysis/rq3_wilcoxon.py` (reads `shap_summary_v2.csv`, writes nothing). |
| BAB 5 (KESIMPULAN DAN SARAN) | the author's replaced text. **Cites nothing.** Item 3 was updated 2026-11-05 on the author's instruction to the all-model IID vs Non-IID figures of §4.3.2 (`analysis/rq3_wilcoxon.py`). |

`<sec-hasil-rq3>` currently has a definition and **zero references** — BAB 3's and
BAB 5's went with their replacements. That is harmless and the label stays.

**Do not write, alter, or "sync" results, findings, numbers, tables, or
conclusions** without an explicit instruction from the author. If you find
experiment outputs (`results/`, `outputs/`, `wandb/`, notebooks, checkpoints), do
**not** narrate them into the thesis; mention them to the author instead. If a
code or methodology change makes a statement stale (e.g. a re-run supersedes a
reported number), **flag the exact location** — never edit it yourself.

Open items against the finished document are tracked in `docs/bab4_open.md`,
`docs/bab4_audit.md` and `docs/bab4_commitments.md`. Read them before proposing a
change to any chapter.

Likewise, do not rewrite the author's research questions (§1.2), objectives
(§1.4), contributions/manfaat (§1.5), or literature review (BAB 2) as part of a
routine doc-sync. Touch BAB 1–2 only when the code factually contradicts a
statement (e.g. a batasan masalah in §1.3), and flag such cases.

## No silent successes

`docs/rq3_method_notes.md` closes with two standing rules earned in this
repository. The second one binds code you write here: **no code path may
substitute stand-in data, skip work, or swallow an error and still report
success — missing or empty input must fail loudly, naming the path it looked
in.** Read that section before adding a fallback, a default, or a broad `except`.

## Verify before considering the change complete

Run `typst compile buku/thesis.typ buku/thesis.pdf` and confirm **zero errors and
zero warnings** before you call a change done. A code change that breaks the
thesis build is an incomplete change.

For a change to thesis text, also run the grep under «Writing consistency».

For anything touching the RQ3 SHAP path, also run:

```bash
python analysis/verify_kernel_tier.py     # six independent legs, all must PASS
pytest tests/                             # includes the RQ3 exactness guards
```
