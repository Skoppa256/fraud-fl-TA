"""Where run outputs are written: one tree per training seed, never the frozen one.

``results/`` itself holds the seed-42 results of record and is frozen
(CLAUDE.md, "Frozen artifacts"). A re-run writes under ``results/seeds/seed<N>/``
instead — the same layout (``sweep/``, ``logs/``, ``models/``, ``shap_v2/``,
``clean_summary.csv``), one tree per seed — so no run can land on another seed's
files, and none can land on the frozen ones.

``experiments/run_sweep.py`` sends every run to its seed's tree, seed 42
included, and hands that tree to the child process through ``SWEEP_RESULTS_ROOT``
(the same way ``SWEEP_GPU_FRACTION`` crosses the subprocess boundary). A child
started any other way resolves to ``results/``, where :func:`assert_writable`
refuses to create anything inside a frozen folder — so a by-hand run fails loudly
instead of overwriting a result of record.

``results/cache/`` is not part of any tree: it is an input cache whose entries are
already named by seed, shared by every seed.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LEGACY_ROOT = PROJECT_ROOT / "results"
SEEDS_ROOT = LEGACY_ROOT / "seeds"
RESULTS_ROOT_ENV = "SWEEP_RESULTS_ROOT"

# The frozen rows of CLAUDE.md that live under results/. Nothing written through
# this module may land inside them. All of results/sweep/ is included, not only
# sweep_master.csv: the master's `log` column points at the run logs beside it.
# Regenerating a frozen artifact is the author's decision, taken by editing this
# tuple — deliberately not by a flag or an environment variable.
FROZEN = (
    LEGACY_ROOT / "clean_summary.csv",
    LEGACY_ROOT / "sweep",
    LEGACY_ROOT / "logs",
    LEGACY_ROOT / "models",
    LEGACY_ROOT / "shap_v2",
    LEGACY_ROOT / "shap",
    LEGACY_ROOT / "analysis",
)


def seed_root(seed: int) -> Path:
    """The output tree of one training seed: ``results/seeds/seed<N>/``."""
    return SEEDS_ROOT / f"seed{int(seed)}"


def results_root() -> Path:
    """The tree this process writes into: ``SWEEP_RESULTS_ROOT``, else ``results/``.

    The value must be absolute: FedXGBllr runs with cwd=models/fedxgbllr, so a
    relative one would resolve to a different directory there than in the runner.
    """
    value = os.environ.get(RESULTS_ROOT_ENV, "").strip()
    if not value:
        return LEGACY_ROOT
    root = Path(value)
    if not root.is_absolute():
        raise ValueError(f"{RESULTS_ROOT_ENV} must be an absolute path, got {value!r}")
    return root


def assert_writable(path: os.PathLike | str) -> None:
    """Raise unless ``path`` lies outside every frozen folder under ``results/``."""
    target = Path(path).resolve()
    for frozen in FROZEN:
        f = frozen.resolve()
        if target == f or f in target.parents:
            raise RuntimeError(
                f"refusing to write {target}: it is inside the frozen "
                f"{f.relative_to(PROJECT_ROOT)} (CLAUDE.md, 'Frozen artifacts'). "
                f"Launch through experiments/run_sweep.py, which writes under "
                f"{SEEDS_ROOT.relative_to(PROJECT_ROOT)}/seed<N>/, or point this "
                f"process at a tree first: "
                f"export {RESULTS_ROOT_ENV}=\"$PWD/results/seeds/seed<N>\""
            )
