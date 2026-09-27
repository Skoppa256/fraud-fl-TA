# GPU-box to-do — items that cannot be done from the laptop

Three items left open by the September 2026 housekeeping pass. Each needs the GPU
box, because each needs either the persisted model artifacts (`results/models/**`,
box-only) or the box's own copy of `results/shap_v2/`. One session clears all
three.

**Why not `docs/launch_checklist.md`:** that file is a pre-launch gate list for
the sweep — its checkboxes are decision gates that must pass *before* a launch,
and it ends at the launch command. These are post-hoc verification items with no
relation to launching a sweep. Folding them in would blur a document whose value
is that it covers exactly one decision.

None of these is urgent. Nothing in the thesis depends on any of them; they close
provenance gaps.

---

## 1. Close the `results/models/**` manifest gap

`docs/frozen_manifest.sha256` covers four of `CLAUDE.md`'s five frozen rows. It
cannot cover `results/models/**` from the laptop, where that tree is empty. So a
green `shasum -c` today attests to nothing about the model artifacts.

```bash
bash analysis/append_model_manifest.sh
shasum -a 256 -c docs/frozen_manifest.sha256
```

**Pass looks like:** the script prints `appended N entries for results/models/`,
and the verify prints `N OK` with zero failures.

**It may refuse, and that is informative.** Two guards:

- *"REFUSING: … already has N results/models/ entries"* — already done; nothing
  to do.
- *"REFUSING: no TRACKED files under results/models/"* — the artifacts exist but
  are untracked (`.gitignore` has a model-artifacts block). Force-add them
  (`git add -f results/models/`), confirm they are the artifacts the sweep
  produced, then re-run. Do not edit the manifest by hand.

**Update afterwards:** `CLAUDE.md`'s manifest-description paragraph — delete the
"Coverage is **partial**" sentence once it is no longer true.

---

## 2. Settle `budget_probe.csv`

`docs/shap_rq3_run.md` records stage 4 as having written
`results/shap_v2/budget_probe.csv`. It is not on the laptop, git has never seen
it, no log records the probe running, and the verdict is recorded as
**undetermined**. One `ls` settles it either way.

```bash
ls -la results/shap_v2/budget_probe.csv
```

**If present:** note its size and mtime, sync it to the laptop, and force-add it
(`results/shap_v2/**` is un-ignored, so a plain `git add` works). Then append its
hash to `docs/frozen_manifest.sha256` in the file's idiom:

```bash
git ls-files results/shap_v2/budget_probe.csv | xargs shasum -a 256 >> docs/frozen_manifest.sha256
shasum -a 256 -c docs/frozen_manifest.sha256
```

**If absent:** the record can be finalised as *never produced* — the stage-4 entry
described an intent, not an artifact.

**Update afterwards:** the "`budget_probe.csv` — artifact absent, provenance
undetermined" section of `docs/shap_rq3_run.md`. Append the outcome; do not
rewrite the existing section.

---

## 3. Verify `plot_shap_beeswarm.py`'s new output path

Its default output moved from `results/shap/figures/beeswarm/` (a frozen tree) to
`results/visualizations/beeswarm/`. That change could not be verified on the
laptop: the script needs the model artifacts and exits 2 with
`NO artifacts under results/models/ — run on the box.`

```bash
python analysis/plot_shap_beeswarm.py
ls results/visualizations/beeswarm/ | head
git status --short results/shap/
```

**Pass looks like:** 18 figures under `results/visualizations/beeswarm/`, exit 0,
and `git status --short results/shap/` **empty** — nothing written into the
frozen v1 tree.

**Fail looks like:** anything appearing under `results/shap/figures/beeswarm/`.
That would mean a write path was missed; report it rather than deleting the
output, since `results/shap/**` is frozen.

**Update afterwards:** `docs/housekeeping_2026-09.md`, which currently records
this verification as outstanding.
