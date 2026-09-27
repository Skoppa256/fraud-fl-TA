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
| `results/shap/**` | **v1 SHAP tree — superseded, but load-bearing.** `production.log` is the only surviving record of the pre-repair run, in which two PaySim FedXGBllr cells scored `spearman=0.0000 kuncheva=1.0000`; it cannot be regenerated (commit `1d661ac`, 2026-08-07, fixed the clipped-logit saturation that produced it). No figure from this log is cited in `buku/` — the v1 `0,775` this row previously claimed for BAB 4 §4.3 appears nowhere in `content.typ`. The log is kept as the record of the pre-repair state, not as a source for prose. `experiments/shap_analysis.py` still writes this directory — do not re-run it. |

`docs/frozen_manifest.sha256` records SHA-256 for the frozen trees. For files
that are tracked in git it is redundant with git's own object hashes; keep it
only as a portable, git-independent check.

Coverage is **partial**: `results/models/**` is not in the manifest, because the
persisted artifacts exist only on the GPU box and the manifest was generated on
the laptop. A green `shasum -a 256 -c docs/frozen_manifest.sha256` therefore
attests to four of the five frozen rows and says nothing about the model
artifacts. Run `analysis/append_model_manifest.sh` on the box to close it.

Note on `.gitignore`: the results rule is now `results/*` (ignore the
directory's *contents*) rather than `results` (ignore the *directory*), because
a directory-level ignore cannot be undone by a negation on a path inside it.
That change is what allowed `results/shap_v2/**` to be re-included and
committed. Results that were already tracked are unaffected by the switch.

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
- Figure and table counters reset per BAB (`template.typ` resets both at each
  numbered level-1 heading), so numbers are per-chapter and shift whenever a
  figure or table is inserted earlier in the same chapter.

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
| Tabel 3.5 | `<tab-3-4>` | Pemetaan Model dengan Skema Agregasi Federated Learning |
| Tabel 3.6 | `<tab-3-5>` | Spesifikasi Lingkungan Pengembangan |
| Tabel 3.7 | `<tab-smote-geometry>` | Geometri sintesis SMOTE pada client terburuk BAF |
| Tabel 3.8 | `<tab-minority-census>` | Sensus minoritas per-client |
| Tabel 3.9 | `<tab-3-6>` | Konfigurasi Hyperparameter Eksperimen |
| Tabel 4.1 | `<tab-4-auprc-ulb>` | AUPRC test pada ULB |
| Tabel 4.2 | `<tab-4-auprc-baf>` | AUPRC test pada BAF |
| Tabel 4.3 | `<tab-4-auprc-paysim>` | AUPRC test pada PaySim |
| Tabel 4.4 | `<tab-4-rfpr-ulb>` | Recall@5%FPR test pada ULB |
| Tabel 4.5 | `<tab-4-rfpr-baf>` | Recall@5%FPR test pada BAF |
| Tabel 4.6 | `<tab-4-rfpr-paysim>` | Recall@5%FPR test pada PaySim |
| Tabel 4.7 | `<tab-4-baf-auprc-bench>` | Perbandingan AUPRC test pada BAF Base |
| Tabel 4.8 | `<tab-typology>` | Tipologi contoh minoritas |
| Tabel 4.9 | `<tab-4-iid-smote>` | Efek SMOTE terhadap AUPRC pada kondisi IID untuk dataset ULB dan Paysim |
| Tabel 4.10 | `<tab-4-baf-smote>` | Efek SMOTE terhadap AUPRC pada kondisi Non-IID untuk dataset BAF |
| Tabel 4.11 | `<tab-4-baf-smote-rfpr>` | Efek SMOTE terhadap Recall@5%FPR pada kondisi Non-IID untuk dataset BAF |
| Tabel 4.12 | `<tab-4-rq3-corr>` | Korelasi peringkat berbobot antar client pada pengukuran utama |
| Tabel 4.13 | `<tab-4-rq3-kuncheva>` | Rerata indeks Kuncheva lima fitur terpenting per dataset |
| Tabel 4.14 | `<tab-4-rq3-paysim>` | Korelasi peringkat berbobot antar client pada PaySim menggunakan KernelSHAP tersampel |
| Tabel 4.15 | `<tab-4-rq3-smote>` | Perubahan konsistensi dan fitur penting setelah SMOTE pada beberapa konfigurasi Non-IID |
| Tabel 4.16 | `<tab-4-rq3-seeds>` | Kesepakatan pengulangan dan kesepakatan antar client pada KernelSHAP tersampel |

Figures:

| renders as | label | caption |
|---|---|---|
| Gambar 2.1 | `<fig-2-1>` | Arsitektur umum Federated Learning |
| Gambar 2.2 | `<fig-2-2>` | Arsitektur FedXGBllr |
| Gambar 2.3 | `<fig-2-3>` | Visualisasi pengaruh parameter α pada distribusi label antar client |
| Gambar 2.4 | `<fig-2-4>` | Ilustrasi mekanisme SMOTE |
| Gambar 2.5 | `<fig-2-5>` | Contoh visualisasi SHAP dalam bentuk summary plot |
| Gambar 3.1 | `<fig-3-1>` | Arsitektur umum sistem penelitian |
| Gambar 3.2 | `<fig-3-2>` | Lapisan logis sistem penelitian |
| Gambar 3.3 | `<fig-3-3>` | Skema pembagian data dua tingkat |
| Gambar 3.4 | `<fig-3-two-seeds-gap>` | Skema pengukuran dua-seed (sel contoh: BAF BERT Non-IID tanpa SMOTE) |
| Gambar 3.5 | `<fig-3-4-smote-geometry>` | Geometri sintesis SMOTE pada client terburuk BAF (seed 42, Dirichlet α = 0,5) |
| Gambar 4.1 | `<fig-typology>` | Distribusi tipe contoh minoritas per dataset |

Note the label/number skew this table exists to prevent: `<tab-3-6>` renders as
**Tabel 3.9**, `<tab-3-4>` as **Tabel 3.5**, `<tab-3-5>` as **Tabel 3.6**, and
`<fig-3-4-smote-geometry>` as **Gambar 3.5**. Cite with `@label`; never write a
literal number into prose.

| Code area changed | Thesis location to update |
|---|---|
| Hyperparameters / any `conf/*.yaml` / `registry.yaml` values | Tabel «Konfigurasi Hyperparameter Eksperimen» `<tab-3-6>` |
| Dependencies / versions (`requirements.txt`, `pyproject.toml`) | Tabel «Spesifikasi Lingkungan Pengembangan» `<tab-3-5>` |
| Preprocessing (`preprocessing/paysim.py`, dropped cols, features, encoding, scaling, split) | **§3.3.1** and **§3.4.1** |
| Client partitioning (`partitioning/dirichlet.py`, IID/Dirichlet, α set, K) | **§3.3.2** and **§3.4.2** |
| Oversampling / SMOTE / ADASYN (`preprocessing/smote.py`, `adasyn.py`, `oversampling.py`) | **§3.3.3** and **§3.4.3** |
| Models & aggregation strategies (`models/*/`, any `strategy.py`) | Tabel «Pemetaan Model dengan Skema Agregasi» `<tab-3-4>`, **§3.3.4**, **§3.4.4** |
| Evaluation metrics & SHAP (`evaluation/`, `experiments/shap_*.py`) | **§3.3.5** and **§3.4.5** |
| Dataset facts (row/feature count, fraud ratio, columns) | `<tab-3-1>` / `<tab-3-2>` (PaySim), `<tab-3-3>` (ULB), `<tab-baf-char>` (BAF) |
| Pipeline restructuring (stage order/count) | **Gambar 3.1** (regenerate the figure) — and **Gambar 3.2** if the logical-layer / model / scheme counts change |
| New method or paper referenced | **`bibliography.bib`** + the citing sentence |

Figures in `buku/resources/` (e.g. `fig-3-1-*`, `fig-3-2-*`) are raster images
ported from the proposal. If a pipeline/architecture change makes one stale, flag
it for the author to redraw — do not leave a diagram that contradicts the code.
Not every PNG in `resources/` is referenced: some are from dropped drafts. An
unreferenced file is not automatically dead — flag it, do not delete it.

## Chapters 4 and 5 are off-limits

BAB 4 (Hasil dan Pembahasan) and BAB 5 (Penutup) contain **authored results
prose**; §4.3 (`<sec-hasil-rq3>`) in particular is complete and cites the frozen
`results/shap_v2/**` tree. They are off-limits: **do not write, alter, or "sync"
results, findings, numbers, tables, or conclusions in them** without an explicit
instruction from the author. If you find experiment outputs (`results/`,
`outputs/`, `wandb/`, notebooks, checkpoints), do **not** narrate them into the
thesis; mention them to the author instead. If a code or methodology change
makes a BAB 4/5 statement stale (e.g. a re-run supersedes a reported number),
**flag the exact location to the author** — never edit it yourself. Keep BAB 4/5
byte-identical unless told otherwise.

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

For anything touching the RQ3 SHAP path, also run:

```bash
python analysis/verify_kernel_tier.py     # six independent legs, all must PASS
pytest tests/                             # includes the RQ3 exactness guards
```
