# RQ3 v2 — defects found and fixed, and two framing errors

A repo-local record of what went wrong during the RQ3 SHAP work and how it was
caught. It is written down because **every one of these was a silent success**:
the code ran to completion, produced well-formed numbers, and wrote them to
disk. Nothing raised, nothing warned, nothing looked wrong in a log. That class
of defect is not caught by "did it crash?" — it is caught only by asking what a
number would look like if the code were wrong, and then going to check.

The four defects below are all fixed; the guards that now catch each recurrence
are named. The two framing errors were caught in review of the *finished*
numbers and changed no code.

---

## 1. `shap >= 0.47` silently changed the KernelSHAP `l1_reg` default

**What happened.** From shap 0.47.0, `KernelExplainer.shap_values` defaults
`l1_reg="num_features(10)"`. LARS keeps at most 10 features per explained
instance and sets **every other attribution to exactly 0.0** — on BAF, 45 of 55
features zeroed on every row. The estimator was not approximating the Shapley
values; it was returning a 10-sparse projection of them.

**Why it was invisible.** The parameter is not passed anywhere in this
repository, and never has been: `git log --all -S "l1_reg"` returns nothing. The
behaviour changed underneath code that was never edited. The output is a
well-formed attribution vector of the right shape, and the zeros are
indistinguishable from "this feature genuinely does not matter" unless you count
them.

**Blast radius.** All 43 v1 kernel cells and all three v1 noise floors ran under
it. Because LARS selection is a *discontinuous* function of the coalition draw,
it inflated the measured noise floor far more than the sampling it was
supposedly regularising — which is exactly what made v1's "everything sits at
the floor" reading look plausible.

**Fix and guard.** The v2 runner sets `l1_reg=False` on every KernelSHAP call.
`process_cell()` in `experiments/shap_rq3.py` raises if any kernel cell with
M > 10 and no grouping reports `max_nonzero_per_row` in `[0, 10]`. Leg 1 of
`analysis/verify_kernel_tier.py` re-checks it post-hoc across the committed
tree (currently: 0 cells showing the signature; minimum nonzeros per row 13).

**Generalisation.** A floating floor on a library whose defaults encode
statistical choices is not a pin. `requirements.txt` now pins `shap==0.49.1`
exactly, and `pytest` is now declared rather than assumed.

## 2. The two all-zero cells were clipped-logit saturation, not underflow

**What happened.** Two FedXGBllr cells returned an identically-zero attribution
vector. The first reading was probability-scale underflow.

**What it actually was.** The FedXGBllr wrapper explained the *probability* and
then applied `logit(clip(p, 1e-6, 1 - 1e-6))`. On PaySim the CNN head's
probabilities compress to ≈ 1e-9 — far below the clip bound — so the clip pins
**every** input to the same value and the explained function is **constant** over
the whole background and explanation set. A constant function has all-zero
Shapley values by the efficiency axiom. The zeros were correct output from a
saturated link, not a numerical failure.

**Why the distinction matters.** "Underflow" invites a fix in the arithmetic
(higher precision, a smaller eps) that would have made things worse: a smaller
eps widens the logit range without recovering any signal. The real problem is
the clip in the link, and the real fix is not to go through the probability
scale at all.

**Fix and guard.** `_fedxgbllr_fn` now replaces the CNN's `final_layer` Sigmoid
with `Identity()` and explains the **pre-Sigmoid activation** — the exact,
unclipped log-odds, matching how BERT and FFD are explained. Both
`experiments/shap_analysis.py` and `experiments/shap_noise_floor.py` carry the
change and a comment recording why. On BAF the clip never bound, so historical
BAF floors measured through the old wrapper remain scale-consistent. Separately,
the degenerate-cell guard marks any all-zero or constant attribution vector
`status = undefined` rather than letting it flow into an aggregate as a
stability value of 1.0 — no cell in the committed tree is `undefined`.

## 3. The first `twobg` arm collapsed the client axis by construction

**What happened.** The two-background arm is supposed to pool the background
across clients while each client explains **its own** rows. The first
implementation pooled the background and kept the **shared** explained rows from
the main arm.

**Why the output looked perfect.** With the background pooled and the rows
shared, every client received byte-identical inputs to an identical global
model. Every client therefore computed the identical attribution vector, and
between-client agreement came out at exactly 1.0 — not as a finding, but as an
arithmetic identity. A result of "perfect agreement across all clients" is
precisely the shape a broken contrast produces, and it reads as a strong
positive result.

**Fix and guard.** Each client now explains rows drawn from its own partition;
`process_cell()` marks any cell whose per-client explanation sets are identical
as a collapsed client axis and refuses to score it. The invalid output was
deleted before the re-run rather than being overwritten in place.

**Generalisation.** For any contrast, write down what varies between the units
being compared *before* reading the number. If nothing varies, the number is a
tautology however clean it looks.

## 4. `bert/iid/none` was a false positive in the sampled tier

**What happened.** In the sampled tier (`nsamples = 500`), `bert/iid/none` came
back at p = 0.001 — comfortably significant. The exact-grouped PaySim tier, the
same cell with the kernel weight fully enumerated, gives p = 0.111.

**What caught it.** Nothing in the sampled tier could have. The exchangeability
test is exact given its inputs, the floor was measured correctly, the guards all
passed — the p-value was a faithful summary of attributions that were themselves
slightly off. Only re-running the same models *without* sampling exposed it.

**What it costs and what it buys.** It is one flip out of 12 cells present in
both tiers, against a mean |exact − sampled| of 0.028 and a max of 0.099
(7 down, 4 up, 1 tie). So the sampled tier's bias is small and the headline
survives — but "small bias" and "no significance flips" are different claims,
and only the second one was ever in doubt. The exact tier is what licenses
reporting sampled-tier p-values at all.

---

## Two framing errors corrected during review

Neither changed a number. Both changed what the numbers are allowed to say.

### The noise floor is a null, not a detection threshold

The floor was initially read as a threshold: a cell "counts" if its
between-client agreement falls below it. That is backwards. The floor is the
**null** — it is what between-client agreement would look like if clients did
not differ and only estimator noise separated their vectors. A cell sitting
below its own floor is therefore **evidence against H₀**, not evidence for it,
and the strength of that evidence is what the exchangeability test quantifies.
Read as a threshold, the same arithmetic silently converts "we cannot
distinguish this from noise" into "this is stable" — which is an acceptance of
the null on the basis of low power.

The consequence for the write-up: cells that are *not* distinguishable from
their floor carry **no stability claim in either direction**. They are shaded in
Gambar 4.3 (`<fig-4-rq3-stability>`) for exactly this reason. Four of the 33
are in that state, and leg 5 of `verify_kernel_tier.py` exists to confirm the
test can still say no — a test that always fires proves nothing.

### The exact-grouped tier is not an anchor

An early version of the anchor comparison folded the exact-grouped PaySim cells
in with the LinearSHAP / TreeSHAP cells to form the reference range. They are
not the same kind of thing. The Linear/Tree anchor is exact **independently of
the models being audited** — different models, different explainers. The
exact-grouped tier is the *same three kernel models* re-run without sampling.
Comparing the sampled tier against its own exact re-run cannot fail: any
discrepancy is bounded by the sampling bias the comparison is supposed to be
testing.

The anchor is therefore LR / SVM / GBM only (weighted between-client range
0.8014 – 0.9989, minimum at `creditcard/gbm/iid/none`), and the exact-grouped
tier is reported as its own tier. This is also why the observation that two
sampled-kernel cells are more divergent than *any* anchor cell
(`paysim/fedxgbllr/dirichlet/smote` at 0.6650 sampled / 0.5965 exact, and
`paysim/ffd/dirichlet/none` at 0.7979 in both tiers) is stated as a **finding**
about which model families diverge under Non-IID — confounded with model family,
since the models that admit an exact explainer are also the structurally
simplest — and explicitly **not** as a validity check. The comment block in
`analysis/verify_kernel_tier.py` records the same distinction at the point where
the anchor is selected, so the code cannot drift back.

---

## The common shape

All four defects and both framing errors share one property: the failure mode
produces a *cleaner-looking* result than the truth. Ten-sparse attributions look
decisive. All-zero vectors look like a feature that does not matter. A collapsed
contrast looks like perfect agreement. A sampled false positive looks like a
significant finding. A floor read as a threshold turns low power into stability.
An anchor that includes the audited models cannot reject them.

The practical rule this leaves: **when a result is unusually clean, that is the
moment to derive what the number would be if the code were wrong.** If the
answer is "the same", the number is not evidence.
