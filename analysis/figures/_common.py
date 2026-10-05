"""Shared plumbing for the thesis figure scripts (BAB 3 and BAB 4).

Every ``fig_*.py`` in this directory builds exactly one image in
``buku/resources/`` from CSV files the experiments wrote. Nothing here trains,
explains or re-runs anything: the inputs are read-only, and the frozen trees
(``results/clean_summary.csv``, ``results/sweep/``, ``results/logs/``,
``results/shap_v2/``) are opened for reading only.

Rules every script follows (CLAUDE.md, "No silent successes"):

* A missing, empty or wrongly-shaped input is a hard error that names the path
  it looked in. There is no fallback data, no embedded snapshot, no default.
* Any value a figure prints is computed from a CSV at build time. Text that is
  not data (stage names, arrows, captions inside the image) is authored here,
  but every number, dataset, model, paradigm, alpha, K and metric name that the
  image claims is checked against the CSVs first.
* The PNG carries its provenance in its metadata: the repo-relative path and
  SHA-256 prefix of every input it read.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESOURCES = PROJECT_ROOT / "buku" / "resources"
RESULTS = PROJECT_ROOT / "results"

# Inputs, all written by the experiment / analysis pipeline.
CLEAN_SUMMARY = RESULTS / "clean_summary.csv"                 # frozen, 96 rows
SWEEP_MASTER = RESULTS / "sweep" / "sweep_master.csv"         # frozen
MINORITY_CENSUS = RESULTS / "analysis" / "minority_census.csv"
SHAP_SUMMARY = RESULTS / "shap_v2" / "shap_summary_v2.csv"    # frozen
SHAP_V2 = RESULTS / "shap_v2"                                 # frozen
SEPARABILITY = PROJECT_ROOT / "analysis" / "separability" / "separability_summary.csv"

SEED = 42  # the single training seed of record

# --------------------------------------------------------------------------- #
# vocabulary: CSV codes -> thesis labels. An unknown code is an error, so a new
# dataset/model/paradigm in the CSV cannot silently appear under a raw code name.
# --------------------------------------------------------------------------- #
DATASET_LABEL = {"paysim": "PaySim", "creditcard": "ULB", "baf": "BAF"}
DATASET_LONG = {"paysim": "PaySim", "creditcard": "ULB Credit Card",
                "baf": "Bank Account Fraud (BAF)"}
DATASET_ORDER = ["paysim", "creditcard", "baf"]

MODEL_LABEL = {"lr": "LR", "svm": "SVM", "gbm": "GBM", "ffd": "FFD",
               "bert_fraud": "BERT", "fedxgbllr": "FedXGBllr", "xgb": "XGBoost"}
MODEL_ORDER = ["lr", "svm", "gbm", "ffd", "bert_fraud", "fedxgbllr"]

# sweep_master.csv `aggregation` codes -> label used in the thesis prose (BAB 3).
PARADIGM_LABEL = {
    "FedAvg": "FedAvg",
    "best-model-selection": "Best-model selection",
    "accuracy-weighted-FedAvg": "Accuracy-weighted FedAvg",
    "tree-ensemble+CNN-FedAvg": "Tree ensemble aggregation + Learnable LR",
}
PARADIGM_ORDER = ["FedAvg", "best-model-selection", "accuracy-weighted-FedAvg",
                  "tree-ensemble+CNN-FedAvg"]


def fail(msg: str) -> "None":
    raise SystemExit(f"FATAL: {msg}")


def rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


def label(mapping: Dict[str, str], code: str, what: str) -> str:
    if code not in mapping:
        fail(f"unknown {what} code {code!r} in the input CSV; add it to the "
             f"label table in {rel(Path(__file__))} rather than printing a raw code.")
    return mapping[code]


# --------------------------------------------------------------------------- #
# loud input loading + provenance
# --------------------------------------------------------------------------- #
_SOURCES: Dict[str, str] = {}


def sha256_prefix(path: Path, n: int = 16) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def _register(path: Path) -> None:
    _SOURCES[rel(path)] = sha256_prefix(path)


def read_csv(path: Path, required: Sequence[str], **kw) -> pd.DataFrame:
    """Read a CSV or stop, naming the path. Required columns must be present
    and the file must have at least one row."""
    path = Path(path)
    if not path.is_file():
        fail(f"input CSV not found.\n  looked in: {path}")
    df = pd.read_csv(path, **kw)
    if df.empty:
        fail(f"input CSV has no rows.\n  path: {path}")
    missing = [c for c in required if c not in df.columns]
    if missing:
        fail(f"input CSV is missing column(s) {missing}.\n  path: {path}\n"
             f"  has: {list(df.columns)}")
    _register(path)
    return df


def read_json(path: Path) -> dict:
    path = Path(path)
    if not path.is_file():
        fail(f"input JSON not found.\n  looked in: {path}")
    _register(path)
    return json.loads(path.read_text())


def one(values: Iterable, what: str):
    """The single distinct value of a column, or a loud error. Used wherever a
    figure prints one number (K, alpha, ...) that must be uniform in the data."""
    vals = sorted({v for v in values if not (isinstance(v, float) and v != v)})
    if len(vals) != 1:
        fail(f"expected exactly one distinct {what} in the data, found {vals}")
    return vals[0]


# --------------------------------------------------------------------------- #
# number formatting: Indonesian convention, as in the thesis tables
# (decimal comma, thousands dot): 0,5 · 71,5% · 391.355
# --------------------------------------------------------------------------- #
def idn(x: float, nd: int = 0) -> str:
    s = f"{x:,.{nd}f}"
    return s.replace(",", "\0").replace(".", ",").replace("\0", ".")


def idn_formatter(nd: int):
    from matplotlib.ticker import FuncFormatter
    return FuncFormatter(lambda v, _pos: idn(v, nd))


# --------------------------------------------------------------------------- #
# style
# --------------------------------------------------------------------------- #
# Text width of the thesis page is ~15.9 cm; figures are drawn at their printed
# size so 8-9 pt here is 8-9 pt on paper (no rescaling by Typst `width: ...%`
# beyond a few percent).
TEXT_WIDTH_IN = 6.3

INK = "#1f1f1f"
INK2 = "#4d4d4d"
MUTED = "#8a8a8a"
RULE = "#c8c8c8"

# Layer / stage tints for the method diagrams (light fill, darker edge + title).
TINTS = [
    ("#eaf1fb", "#7aa3d6", "#1f4e8c"),   # blue
    ("#e8f5ef", "#7fbf9f", "#1e6b47"),   # green
    ("#efedfb", "#a39ae0", "#463a99"),   # violet
    ("#fbf2e4", "#d9b27a", "#7a4f12"),   # amber
    ("#fbeaea", "#d99494", "#8c2a2a"),   # red
    ("#eef0f2", "#9aa4ae", "#36414c"),   # slate
]

# Minority typology (Napierala & Stefanowski): safe -> outlier, green -> red,
# the same order and hues as the figure it replaces.
TYPOLOGY_COLORS = {"safe": "#2a9d5c", "borderline": "#e9b13c",
                   "rare": "#e8812f", "outlier": "#c0392b"}

FLOOR_COLOR = "#0d366b"
BETWEEN_COLOR = "#3987e5"


FONT_CANDIDATES = ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"]


def resolved_font() -> str:
    """First installed family of FONT_CANDIDATES (Liberation Sans is metric-
    compatible with Arial, so wrapping matches across machines that lack Arial).
    DejaVu Sans ships with matplotlib, so this always resolves."""
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    return next(f for f in FONT_CANDIDATES if f in have)


def apply_style() -> None:
    font = resolved_font()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [font],
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "text.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        # mathtext ($H_0$, ...) in the body font instead of DejaVu, so a
        # subscript never depends on the font having U+2080-style glyphs
        "mathtext.fontset": "custom",
        "mathtext.rm": font,
        "mathtext.it": f"{font}:italic",
        "mathtext.bf": f"{font}:bold",
        "mathtext.sf": font,
        "mathtext.cal": font,   # unused; set so matplotlib stops hunting for 'cursive'
        "mathtext.tt": "DejaVu Sans Mono",
    })


def save(fig, name: str) -> Path:
    """Write buku/resources/<name> with input provenance in the PNG metadata.

    No timestamp is embedded, so re-running on unchanged inputs yields a
    byte-identical file and `git diff` stays quiet.
    """
    if not _SOURCES:
        fail(f"{name}: refusing to save a figure that read no input data.")
    out = RESOURCES / name
    RESOURCES.mkdir(parents=True, exist_ok=True)
    meta = {
        "Software": None,
        "Title": name,
        "Source": "; ".join(f"{p} sha256:{h}" for p, h in sorted(_SOURCES.items())),
        "Generator": rel(Path(sys.argv[0])) if sys.argv and sys.argv[0] else "",
    }
    # A glyph the font lacks is drawn as an empty box ("tofu") and matplotlib
    # only warns. Which glyphs exist depends on the machine's font (Liberation
    # Sans has U+2080, Arial does not), so this is checked on every save and a
    # missing glyph is a hard error, never a degraded figure reported as written.
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.04,
                    facecolor="white", metadata=meta)
    missing = sorted({str(w.message) for w in caught
                      if "missing from font" in str(w.message)})
    plt.close(fig)
    if missing:
        out.unlink(missing_ok=True)
        fail(f"{name}: font {plt.rcParams['font.sans-serif'][0]!r} lacks glyph(s) "
             f"used in the figure, which would render as empty boxes:\n  "
             + "\n  ".join(missing)
             + f"\n  {out} was not written; replace the character (e.g. use "
               f"mathtext $H_0$) or install a font that has it.")
    for w in caught:   # anything else matplotlib warned about stays visible
        warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
    print(f"wrote {rel(out)}")
    for p, h in sorted(_SOURCES.items()):
        print(f"  from {p}  (sha256 {h})")
    _SOURCES.clear()
    return out


# --------------------------------------------------------------------------- #
# experiment design facts, derived from the CSVs (shared by the BAB 3 diagrams)
# --------------------------------------------------------------------------- #
def split_ratio_from_code() -> List[int]:
    """Train/val/test percentages, read from the `test_size=` literals in the
    three preprocessing modules (the split is not recorded in any CSV). The two
    calls are train_test_split(test_size=a) then (test_size=b) on the holdout:
    train = 1-a, val = a*(1-b), test = a*b. All three datasets must agree."""
    found = {}
    for ds in ("paysim", "creditcard", "baf"):
        p = PROJECT_ROOT / "preprocessing" / f"{ds}.py"
        if not p.is_file():
            fail(f"preprocessing module not found.\n  looked in: {p}")
        sizes = [float(x) for x in re.findall(r"test_size\s*=\s*([0-9.]+)", p.read_text())]
        if len(sizes) != 2:
            fail(f"expected two test_size= literals in {p}, found {sizes}")
        a, b = sizes
        found[ds] = [round(100 * (1 - a)), round(100 * a * (1 - b)), round(100 * a * b)]
        _register(p)
    if len({tuple(v) for v in found.values()}) != 1:
        fail(f"datasets use different splits: {found}")
    return next(iter(found.values()))


def design_facts() -> dict:
    """Everything the method diagrams state about the experimental grid."""
    cs = read_csv(CLEAN_SUMMARY, ["dataset", "model", "scheme", "alpha",
                                  "oversampling", "num_clients", "random_seed",
                                  "rounds_configured", "test_auprc",
                                  "test_recall_at_fpr", "test_f1", "test_precision",
                                  "test_recall", "test_brier", "test_cal_slope",
                                  "threshold_policy"])
    sm = read_csv(SWEEP_MASTER, ["model", "condition", "aggregation", "status"])
    cen = read_csv(MINORITY_CENSUS, ["dataset", "seed", "scheme", "alpha",
                                     "client_id", "n_features", "n_total",
                                     "n_minority"])
    shap = read_csv(SHAP_SUMMARY, ["spearman", "kuncheva", "p_value", "explainer"])

    seed = int(one(cs["random_seed"], "random_seed"))
    if seed != SEED:
        fail(f"clean_summary seed is {seed}, figures assume {SEED}")
    fed = cs[cs["scheme"] != "centralized"]
    K = int(one(fed["num_clients"], "num_clients (federated rows)"))
    alpha = float(one(cs.loc[cs["scheme"] == "dirichlet", "alpha"], "Dirichlet alpha"))
    datasets = [d for d in DATASET_ORDER if d in set(cs["dataset"])]
    if set(cs["dataset"]) - set(datasets):
        fail(f"unknown dataset code(s): {set(cs['dataset']) - set(datasets)}")
    for d in datasets:
        label(DATASET_LABEL, d, "dataset")

    # model -> paradigm, from the federated sweep rows that actually succeeded
    ok = sm[(sm["status"] == "success") & (sm["condition"] != "centralized")]
    para = {}
    for model, grp in ok.groupby("model"):
        para[model] = one(grp["aggregation"], f"aggregation for model {model}")
        label(PARADIGM_LABEL, para[model], "aggregation")
    fed_models = [m for m in MODEL_ORDER if m in para]
    if set(para) - set(fed_models):
        fail(f"unknown federated model(s): {set(para) - set(fed_models)}")
    paradigms = [(p, [m for m in fed_models if para[m] == p])
                 for p in PARADIGM_ORDER if p in para.values()]

    # training-set size / prevalence / features per dataset, from the seed-42
    # IID census (5 clients that together are exactly the training set)
    c42 = cen[(cen["seed"] == seed) & (cen["scheme"] == "iid")]
    data = {}
    for d in datasets:
        g = c42[c42["dataset"] == d]
        if len(g) != K:
            fail(f"census has {len(g)} seed-{seed} IID clients for {d}, expected {K}"
                 f"\n  path: {MINORITY_CENSUS}")
        n, nmin = int(g["n_total"].sum()), int(g["n_minority"].sum())
        data[d] = {"n_train": n, "n_minority": nmin, "prevalence": nmin / n,
                   "n_features": int(one(g["n_features"], f"n_features for {d}"))}

    # per-client minority counts under the Dirichlet alpha actually run
    noniid = cen[(cen["seed"] == seed) & (cen["scheme"] == "dirichlet")
                 & (cen["alpha"] == alpha)]
    clients = {}
    for d in datasets:
        g = noniid[noniid["dataset"] == d].sort_values("client_id")
        if list(g["client_id"]) != list(range(K)):
            fail(f"census lacks seed-{seed} Dirichlet a={alpha} clients 0..{K-1} for {d}")
        clients[d] = {"n_total": [int(v) for v in g["n_total"]],
                      "n_minority": [int(v) for v in g["n_minority"]]}

    rounds = {m: sorted({int(v) for v in fed.loc[fed["model"] == m, "rounds_configured"]})
              for m in fed_models}

    return {
        "seed": seed, "K": K, "alpha": alpha, "datasets": datasets,
        "data": data, "clients": clients, "fed_models": fed_models,
        "paradigms": paradigms, "rounds": rounds,
        "arms": [a for a in ("none", "smote") if a in set(cs["oversampling"])],
        "schemes": [s for s in ("centralized", "iid", "dirichlet") if s in set(cs["scheme"])],
        "n_runs": len(cs),
        "threshold_policy": one(cs["threshold_policy"], "threshold_policy"),
        "explainers": sorted(set(shap["explainer"])),
        "split": split_ratio_from_code(),
    }
