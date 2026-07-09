"""Check which planned runs are complete vs pending.

Cross-references ``experiments/registry.yaml`` against the summary CSVs under
``results/logs/`` to report what's done, what's still missing, and (optionally)
print the exact command to re-run each pending item.

Usage
-----
    python -m experiments.status
    python -m experiments.status --pending-only
    python -m experiments.status --print-commands  # ready-to-paste re-run cmds

A "completed" run is any ``<model>/<run_name>.csv`` that exists under
``results/logs/``. The check is by-filename only — it does not validate the
CSV contents, so corrupted runs may still register as complete. Inspect the
file or re-run if in doubt.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from typing import Dict, List, Tuple

import yaml


REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "registry.yaml")
LOGS_ROOT = "results/logs"


def _load_registry(path: str = REGISTRY_PATH) -> Dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _alpha_token(alpha) -> str:
    return "-" if alpha is None else str(alpha)


def fl_run_name(
    model: str, scheme: str, alpha, oversampling: str, seed: int
) -> str:
    return (
        f"{model}_{scheme}_alpha{_alpha_token(alpha)}_"
        f"{oversampling}_seed{int(seed)}"
    )


def centralized_run_name(model: str, oversampling: str, seed: int) -> str:
    return f"centralized_{model}_{oversampling}_seed{int(seed)}"


def _datasets(reg: Dict) -> List[str]:
    """Datasets in the sweep. Defaults to ``[paysim]`` for older registries."""
    return list(reg.get("datasets", ["paysim"]))


def _enumerate_fl_runs(reg: Dict) -> List[Tuple[str, str, str, object, str, int]]:
    """Yield ``(dataset, model, scheme, alpha, oversampling, seed)`` per planned FL run."""
    runs = []
    for dataset, model, oversampling, seed in itertools.product(
        _datasets(reg), reg["models"], reg["oversamplings"], reg["seeds"]
    ):
        # IID: alpha is None.
        runs.append((dataset, model, "iid", None, oversampling, seed))
        # Dirichlet: every alpha in the registry.
        for alpha in reg["alphas"]:
            runs.append((dataset, model, "dirichlet", alpha, oversampling, seed))
    return runs


def _enumerate_centralized_runs(reg: Dict) -> List[Tuple[str, str, str, int]]:
    return list(
        itertools.product(
            _datasets(reg),
            reg.get("centralized_models", []),
            reg["oversamplings"],
            reg["seeds"],
        )
    )


def _csv_path_fl(dataset: str, model: str, run_name: str) -> str:
    return os.path.join(LOGS_ROOT, dataset, model, f"{run_name}.csv")


def _csv_path_centralized(dataset: str, run_name: str) -> str:
    return os.path.join(LOGS_ROOT, dataset, "centralized", f"{run_name}.csv")


def _fl_command(
    dataset: str,
    model: str,
    scheme: str,
    alpha,
    oversampling: str,
    seed: int,
    num_rounds: int,
) -> str:
    if model == "fedxgbllr":
        bits = [
            "python -m hfedxgboost.main",
            f"dataset={dataset}",
            f"clients={dataset}_5_clients",
            f"run_experiment.num_rounds={num_rounds}",
            f"dataset.oversampling.method={oversampling}",
            f"random_seed={seed}",
            "use_wandb=true",
        ]
        if scheme == "dirichlet":
            bits.extend(
                [
                    "dataset.non_iid.enabled=true",
                    f"dataset.non_iid.alpha={alpha}",
                ]
            )
        return " \\\n  ".join(bits)
    module = {
        "ffd": "models.ffd.run",
        "lr": "models.fedavg_lr.run",
        "svm": "models.fedavg_svm.run",
        "gbm": "models.gbm_bestmodel.run",
    }[model]
    bits = [
        f"python -m {module}",
        f"--dataset {dataset}",
        f"--scheme {scheme}",
        f"--num_rounds {num_rounds}",
        f"--oversampling {oversampling}",
        f"--random_seed {seed}",
        "--use_wandb true",
    ]
    if scheme == "dirichlet":
        bits.insert(3, f"--alpha {alpha}")
    return " \\\n  ".join(bits)


def _centralized_command(dataset: str, model: str, oversampling: str, seed: int) -> str:
    return (
        f"python -m experiments.centralized_baseline.run_{model} "
        f"--dataset {dataset} --oversampling {oversampling} "
        f"--random_seed {seed} --use_wandb true"
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--registry",
        default=REGISTRY_PATH,
        help=f"Path to registry.yaml (default: {REGISTRY_PATH}).",
    )
    p.add_argument(
        "--logs-root",
        default=LOGS_ROOT,
        help=f"Directory to scan for completed CSVs (default: {LOGS_ROOT}).",
    )
    p.add_argument(
        "--pending-only",
        action="store_true",
        help="Only print pending runs (skip completed list).",
    )
    p.add_argument(
        "--print-commands",
        action="store_true",
        help="Print the re-run command for each pending run.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    reg = _load_registry(args.registry)
    global LOGS_ROOT
    LOGS_ROOT = args.logs_root

    fl_runs = _enumerate_fl_runs(reg)
    centralized_runs = _enumerate_centralized_runs(reg)

    fl_done: List[Tuple] = []
    fl_pending: List[Tuple] = []
    for run in fl_runs:
        dataset, model, scheme, alpha, oversampling, seed = run
        name = fl_run_name(model, scheme, alpha, oversampling, seed)
        if os.path.isfile(_csv_path_fl(dataset, model, name)):
            fl_done.append(run)
        else:
            fl_pending.append(run)

    cen_done: List[Tuple] = []
    cen_pending: List[Tuple] = []
    for run in centralized_runs:
        dataset, model, oversampling, seed = run
        name = centralized_run_name(model, oversampling, seed)
        if os.path.isfile(_csv_path_centralized(dataset, name)):
            cen_done.append(run)
        else:
            cen_pending.append(run)

    total = len(fl_runs) + len(centralized_runs)
    done = len(fl_done) + len(cen_done)
    pending = len(fl_pending) + len(cen_pending)
    print(
        f"[status] planned={total}  done={done}  pending={pending}  "
        f"({done * 100 // max(total, 1)}% complete)"
    )
    print(f"  FL          : {len(fl_done)}/{len(fl_runs)} done")
    print(
        f"  Centralized : {len(cen_done)}/{len(centralized_runs)} done"
    )

    if not args.pending_only:
        if fl_done:
            print("\n[done — FL]")
            for dataset, model, scheme, alpha, oversampling, seed in fl_done:
                print(
                    f"  ✓ [{dataset}] "
                    + fl_run_name(model, scheme, alpha, oversampling, seed)
                )
        if cen_done:
            print("\n[done — centralized]")
            for dataset, model, oversampling, seed in cen_done:
                print(
                    f"  ✓ [{dataset}] "
                    + centralized_run_name(model, oversampling, seed)
                )

    if fl_pending or cen_pending:
        print("\n[pending]")
        for dataset, model, scheme, alpha, oversampling, seed in fl_pending:
            name = fl_run_name(model, scheme, alpha, oversampling, seed)
            print(f"  · [{dataset}] {name}")
            if args.print_commands:
                cmd = _fl_command(
                    dataset,
                    model,
                    scheme,
                    alpha,
                    oversampling,
                    seed,
                    int(reg["num_rounds"][model]),
                )
                print("      " + cmd.replace("\n", "\n      "))
        for dataset, model, oversampling, seed in cen_pending:
            name = centralized_run_name(model, oversampling, seed)
            print(f"  · [{dataset}] {name}")
            if args.print_commands:
                print(
                    "      "
                    + _centralized_command(dataset, model, oversampling, seed)
                )

    return 0 if pending == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
