# Housekeeping pass — September 2026

A record of what moved, and why, so the archive explains itself. Six rounds of
repo housekeeping between 2026-09-09 and 2026-09-10. No thesis prose was written
or edited: `buku/content.typ` is byte-identical throughout. Nothing under a
frozen path was modified.

## Deleted

| what | why |
|---|---|
| `buku/resources/fig-shap-by-model.png`, `fig-shap-heatmap-deterministic.png`, `fig-shap-vs-floor.png` | Zero references in `buku/`. `fig-shap-vs-floor` encoded the noise-floor-as-threshold framing that §4.3 explicitly rejects; leaving it invited a re-reference. Copies remain in the frozen `results/shap/figures/` and in git history. |
| `analysis/plot_shap_stability.py` | v1-era script: defaulted to the invalidated v1 summary, hardcoded the broadcast floor `{fedxgbllr: 0.9730, bert_fraud: 0.9972, ffd: 0.9966}`, and titled its plot "N/M at or below floor". Run today it reproduced a retracted conclusion as a well-formed artifact. Nothing imported it. `analysis/make_rq3_heatmap.py` produces the current figure. |
| `results/shap/figures/rq3_stability.html`, `two_seeds_gap.html` | Untracked v2-era strays inside a frozen tree. Deleted only after byte-identical regenerated copies were confirmed in `results/visualizations/`. `results/shap/` is now 323 files on disk and 323 tracked. |
| The embedded-snapshot literals in `analysis/make_rq3_heatmap.py` and `two_seeds_gap.py` | See "Standing rule 2" below. |

## Force-added

All were matched by `.gitignore` and therefore invisible to the archive.

| what | why |
|---|---|
| `analysis/separability/separability_summary.csv` | The machine-readable source for `<tab-typology>` in BAB 4. Hidden by `.gitignore:37 *.csv`. All 18 values verified against the table. |
| `results/analysis/minority_census.csv` | Cited by BAB 4 §4.3 for the per-client minority counts. Verified: PaySim 2, BAF 3, ULB 4, all at Dirichlet α=0.5. Hidden by `.gitignore:51 results/*`. |
| `results/analysis/minority_census_summary.md` | Cited by `docs/predictions.md:47`. Verified: "baf IID: 15/15 skip via target_met". |
| `results/shap_v2/rq3_main.log`, `rq3_twobg.log`, `rq3_exact.log` | Unreproducible run records, and the sole evidence for the invocation sequence, for defect #3's collapse and its deletion-before-re-run, and for the exact tier's cross-process reproducibility. Hidden by `.gitignore:56 *.log`, which overrode the `!results/shap_v2/**` negation at line 53 because it came after it. A negation at line 60 now keeps them. |

Precedent note: the 93 tracked `.log` files under `results/logs/` got there the
same way. They are matched by `.gitignore:57 logs/` (not `*.log`), both rules
predate them, and no negation covers them — they were force-added.

## The manifest

`docs/frozen_manifest.sha256`, **407 → 1196 entries**, append-only at every step;
each round verified that the previous entries stayed byte-identical.

| tree | entries |
|---|---|
| v1 sweep (`clean_summary.csv`, `sweep_master.csv`, `results/logs/**`) | 407 |
| v1 SHAP (`results/shap/**`) | 323 |
| v2 SHAP (`results/shap_v2/**`) | 466 |

All 1196 verify OK. **Coverage is partial**: `results/models/**` is absent,
because those artifacts exist only on the GPU box. A green `shasum -c` attests to
four of `CLAUDE.md`'s five frozen rows. `analysis/append_model_manifest.sh`
closes it when run there — see `docs/box_todo.md`.

Two files were deliberately excluded: `results/shap/figures/rq3_stability.html`
and `two_seeds_gap.html`, being regenerable v2-era outputs rather than v1 record
(and since deleted).

## The `0,775` provenance question — resolved

BAB 4 §4.3 cites `0,775` as the FedXGBllr figure from "the previous measurement".
It was initially reported as unsubstantiated: the archived v1 summary
(`results/shap/shap_summary.csv`) gives a mean of 0.9182, and the two PaySim
Dirichlet cells it would need to be zero are recorded there as 0.8181 and 0.7560
with `"degenerate": false`. The figure is in fact exact and sourced —
`results/shap/production.log`, tracked, records a **pre-repair** run in which
those two cells are `spearman=0.0000` and BERT is absent (`KeyError`); the mean of
its 11 FedXGBllr cells is **0.7751**. Three measurement states, not two.

Full account, including the commit evidence dating the log and the split between
the clipped-logit repair (+0.143) and the `l1_reg` fix (+0.011), is in
`docs/shap_rq3_run.md` § "Provenance of the `0,775` FedXGBllr figure".

## The enumeration figure — corrected

The kernel weight PaySim enumerates deterministically at `nsamples = 500` is
**34.9%**, not the 54.0% recorded in `docs/predictions.md:86` and in
`content.typ`. The old value came from modelling shap's coalition selection as a
budget-capacity test; `KernelExplainer.explain()` applies a weight criterion that
rejects subset size 2 at M = 13. Confirmed by two independent routes (shap's own
logged `weight_left`, and a direct count of enumerated coalitions: 26, not 182).
ULB 26.1% and BAF 22.3% were correct. A dated correction is appended to
`docs/predictions.md`; the derivation is in `docs/shap_rq3_run.md`.

## Standing rule 2

`docs/rq3_method_notes.md` gained a second standing rule: no code path may
substitute stand-in data, skip work, or swallow an error and still report
success. Four instances are cited, all of which exited 0. The two live ones were
fixed in this pass — `make_rq3_heatmap.py` and `two_seeds_gap.py` now raise
`SystemExit` naming the path searched, instead of falling back to a hardcoded
snapshot of an earlier run and printing a success line.

## Handover — `content.typ` locations flagged but NOT edited

For the BAB 4 pass. Line numbers as of 2026-09-10; all are in prose or comments,
none affects the build.

| line | what | status |
|---|---|---|
| **3250** | "Nilai FedXGBllr sebesar 0,775 … merupakan artefak dari nilai bawaan `l1_reg`" | The figure is now sourced, but the **attribution is wrong**: `l1_reg` moved the FedXGBllr mean by +0.011; the clipped-logit repair moved it by +0.143. The companion claim in the same passage (three sampled models converge near 0.95) is unaffected and holds. |
| **3176** | "PaySim mengenumerasi sekitar 54 persen bobot kernel" | Wrong; ≈35%. ULB (~26%) and BAF (~22%) in the same sentence are correct. Nothing measured depends on it. |
| **4**, **2483** | `// BAB 1–3 terisi dari proposal; BAB 4–5 masih stub.` and `// BAB 4 — HASIL DAN PEMBAHASAN (stub)` | Stale comments; BAB 4 is written. |
| **3705**, **3715** | `// BAB 5 — PENUTUP (stub)` and "Kesimpulan … BELUM ditulis karena Bab 4 belum ada" | The second is now factually false. BAB 5 Kesimpulan remains unwritten — that is the real outstanding thesis work. |

Two available strengthenings, also flagged and not written: the exact tier's
**cross-process reproducibility** (three separate invocations, byte-identical
output — stronger than the within-run `seed_delta_max = 0.0` §4.3 currently
cites), and the **separability analysis**, whose ceiling evidence and
dimensionality-matched control were computed but never reached BAB 4. Both are
described in `docs/shap_rq3_run.md`.
