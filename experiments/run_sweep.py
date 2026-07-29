"""Single unified experiment runner for the full sweep.

Executes dataset × SMOTE-arm × model × condition as isolated subprocesses, safe
to launch unattended overnight. Reuses the existing per-model entry points (it
does NOT reimplement training) and the central resource config + preflight +
data cache from earlier commits.

    python experiments/run_sweep.py \
      --datasets paysim,creditcard,baf --models all \
      --conditions centralized,iid,noniid --alpha 0.5 --seeds 42 \
      --smote-arms both --gpu-fraction 1.0 [--dry-run|--resume|--offline|--timing-probe]

Running with NO arguments prints the dry-run manifest (it never launches jobs by
accident).

IMPORTANT — launch under a session that survives SSH drop:
    tmux new -s sweep 'python experiments/run_sweep.py ... 2>&1 | tee sweep.out'
    # or: nohup python experiments/run_sweep.py ... &
A plain SSH session dropping overnight sends SIGHUP and kills the whole run.

Scope of this commit (commit 3): orchestration only. Each child still writes its
own metrics CSV via evaluation.results_writer; the runner records orchestration
metadata + cache-derived data/partition hashes + smote_inoperative + sub-6 count.
Wiring the child eval to record its OWN hashes (for the cross-run equality assert)
and calibration is the next commit.
"""

from __future__ import annotations

import argparse
import csv
import os
import signal
import subprocess
import sys
import time
import yaml
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments import data_cache, resources  # noqa: E402
from experiments import sweep_preflight  # noqa: E402
from evaluation import model_persistence as _mp  # noqa: E402

# --- Experiment constants (mirror the model configs; uniform across datasets). --
SAMPLING_STRATEGY = 0.01
K_NEIGHBORS = 5
NUM_CLIENTS = 5
ALL_MODELS = ("lr", "svm", "gbm", "ffd", "bert_fraud", "fedxgbllr")
ALL_CONDITIONS = ("centralized", "iid", "noniid")
ALL_DATASETS = ("paysim", "creditcard", "baf")
SMOTE_ARMS = ("smote", "no-smote")

# model -> FL entry-point module (iid/noniid). fedxgbllr is Hydra (special-cased).
_FL_MODULE = {
    "lr": "models.fedavg_lr.run",
    "svm": "models.fedavg_svm.run",
    "gbm": "models.gbm_bestmodel.run",
    "ffd": "models.ffd.run",
    "bert_fraud": "models.bert_fraud.run",
}
# model -> centralized entry-point script. NB: fedxgbllr centralized == run_xgb,
# the plain-XGBoost UPPER BOUND (no CNN head — architecture not held constant).
_CENTRAL_SCRIPT = {
    "lr": "run_lr", "svm": "run_svm", "gbm": "run_gbm",
    "ffd": "run_ffd", "bert_fraud": "run_bert_fraud", "fedxgbllr": "run_xgb",
}

SWEEP_DIR = PROJECT_ROOT / "results" / "sweep"
MASTER_CSV = SWEEP_DIR / "sweep_master.csv"
WANDB_OFFLINE_DIR = SWEEP_DIR / "wandb_offline"

PYEXE = sys.executable


@dataclass
class RunSpec:
    dataset: str
    model: str
    arm: str            # "smote" | "no-smote"
    condition: str      # "centralized" | "iid" | "noniid"
    seed: int
    alpha: Optional[float] = None   # only for noniid
    smote_inoperative: bool = False
    sub6_count: int = 0

    @property
    def oversampling(self) -> str:
        return "smote" if self.arm == "smote" else "none"

    @property
    def alpha_label(self) -> str:
        return "na" if self.alpha is None else f"{self.alpha:g}"

    @property
    def group(self) -> str:
        return f"{self.dataset}_{self.arm}_{self.condition}_{self.model}"

    @property
    def run_name(self) -> str:
        return f"{self.group}_a{self.alpha_label}_s{self.seed}"

    @property
    def run_dir(self) -> Path:
        return SWEEP_DIR / self.run_name


# --------------------------------------------------------------------------- #
# Matrix
# --------------------------------------------------------------------------- #
def build_matrix(datasets, models, conditions, alphas, seeds, arms) -> List[RunSpec]:
    """Cartesian matrix. alpha multiplies ONLY noniid (separate condition
    instances per alpha); centralized/iid are alpha-independent (one each). Seeds
    are always in the name so nothing collides when more are added later."""
    out: List[RunSpec] = []
    for s in seeds:
        for ds in datasets:
            for model in models:
                for arm in arms:
                    for cond in conditions:
                        alpha_list = alphas if cond == "noniid" else [None]
                        for a in alpha_list:
                            out.append(RunSpec(ds, model, arm, cond, seed=int(s), alpha=a))
    return out


# --------------------------------------------------------------------------- #
# smote_inoperative — computed from cache metadata, before any training
# --------------------------------------------------------------------------- #
def _client_skips(n_min: int, n_maj: int) -> bool:
    """A client is a SMOTE no-op iff it lacks enough minority OR already meets the
    target ratio (mirrors preprocessing.smote's skip guards)."""
    if n_min < K_NEIGHBORS + 1:
        return True
    if n_maj > 0 and (n_min / n_maj) >= SAMPLING_STRATEGY:
        return True
    return n_maj == 0


def smote_status(spec: RunSpec) -> Tuple[bool, int]:
    """(inoperative, sub6_count) for the SMOTE arm, from cached data/partitions.

    * centralized: no partition — evaluate the whole training set as one group.
    * iid/noniid: inoperative iff EVERY client skips. sub6_count = clients with
      n_minority < k_neighbors+1.
    """
    if spec.condition == "centralized":
        data, _ = data_cache.get_preprocessed(spec.dataset, spec.seed)
        y = data["y_train"]
        n_min = int((y == 1).sum())
        n_maj = int(len(y) - n_min)
        inop = _client_skips(n_min, n_maj)
        return inop, (1 if n_min < K_NEIGHBORS + 1 else 0)

    indices, _ = data_cache.get_partition_indices(
        spec.dataset, spec.seed, spec.alpha, spec.condition, NUM_CLIENTS
    )
    data, _ = data_cache.get_preprocessed(spec.dataset, spec.seed)
    y = data["y_train"]
    all_skip, sub6 = True, 0
    for ix in indices:
        yk = y[ix]
        n_min = int((yk == 1).sum())
        n_maj = int(len(yk) - n_min)
        if n_min < K_NEIGHBORS + 1:
            sub6 += 1
        if not _client_skips(n_min, n_maj):
            all_skip = False
    return all_skip, sub6


def cache_hashes(spec: RunSpec) -> Tuple[str, str]:
    """(data_hash, partition_hash) from the cache. partition_hash='' for centralized."""
    _data, data_hash = data_cache.get_preprocessed(spec.dataset, spec.seed)
    if spec.condition == "centralized":
        return data_hash, ""
    _idx, phash = data_cache.get_partition_indices(
        spec.dataset, spec.seed, spec.alpha, spec.condition, NUM_CLIENTS
    )
    return data_hash, phash


# --------------------------------------------------------------------------- #
# Command + environment
# --------------------------------------------------------------------------- #
def build_command(spec: RunSpec, gpu_available: bool, use_wandb: bool) -> Tuple[List[str], Path]:
    """Return (argv, cwd). Reuses existing entry points — no training logic here."""
    ov = spec.oversampling
    wb = "true" if use_wandb else "false"
    if spec.condition == "centralized":
        argv = [
            PYEXE, "-m", f"experiments.centralized_baseline.{_CENTRAL_SCRIPT[spec.model]}",
            "--dataset", spec.dataset, "--oversampling", ov,
            "--sampling_strategy", str(SAMPLING_STRATEGY),
            "--random_seed", str(spec.seed), "--use_wandb", wb,
        ]
        return argv, PROJECT_ROOT

    if spec.model == "fedxgbllr":
        xgb_device = resources.xgboost_params(gpu_available=gpu_available)["device"]
        argv = [
            PYEXE, "-m", "hfedxgboost.main",
            f"dataset={spec.dataset}", f"clients={spec.dataset}_5_clients",
            f"dataset.oversampling.method={ov}",
            f"dataset.oversampling.sampling_strategy={SAMPLING_STRATEGY}",
            f"XGBoost.device={xgb_device}",  # override (device now exists in base.yaml; not +append)
            f"random_seed={spec.seed}", f"use_wandb={wb}",
        ]
        if spec.condition == "noniid":
            argv += ["dataset.non_iid.enabled=true", f"dataset.non_iid.alpha={spec.alpha}"]
        else:
            argv += ["dataset.non_iid.enabled=false"]
        return argv, PROJECT_ROOT / "models" / "fedxgbllr"

    # argparse FL models
    scheme = "iid" if spec.condition == "iid" else "dirichlet"
    argv = [
        PYEXE, "-m", _FL_MODULE[spec.model],
        "--dataset", spec.dataset, "--scheme", scheme,
        "--oversampling", ov, "--sampling_strategy", str(SAMPLING_STRATEGY),
        "--num_clients", str(NUM_CLIENTS),
        "--random_seed", str(spec.seed), "--use_wandb", wb,
    ]
    if spec.condition == "noniid":
        argv += ["--alpha", str(spec.alpha)]
    return argv, PROJECT_ROOT


def build_env(spec: RunSpec, offline: bool, use_wandb: bool,
              data_hash: str = "", partition_hash: str = "") -> Dict[str, str]:
    """Child environment: propagate thread pins explicitly (they do NOT inherit
    automatically) and set wandb group/name/tags/mode/dir + config via env (wandb
    honours these), so no entry point needs editing. The extra config fields
    (smote_arm, condition, alpha, seed, smote_inoperative, both hashes) are merged
    into wandb.config via a per-run yaml pointed to by WANDB_CONFIG_PATHS."""
    env = dict(os.environ)
    n = str(resources.threads_per_actor())
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[var] = n
    # Ensure children (esp. fedxgbllr under a different cwd) import repo packages.
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    if use_wandb:
        env["WANDB_RUN_GROUP"] = spec.group
        env["WANDB_NAME"] = spec.run_name
        env["WANDB_TAGS"] = ",".join([spec.dataset, spec.model, spec.condition])
        env["WANDB_DIR"] = str(WANDB_OFFLINE_DIR)
        env["WANDB_MODE"] = "offline" if offline else "online"
        # Extra config fields merged into wandb.config (no entry-point edit).
        # wandb's dict_from_config_file expects the NESTED form
        # ``key: {value: ...}`` (plus optional ``wandb_version: 1``) and crashes at
        # ``v["value"]`` on a flat ``key: value`` file. Emit the nested form.
        spec.run_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = spec.run_dir / "wandb_config.yaml"
        cfg_fields = {
            "dataset": spec.dataset,
            "model": spec.model,
            "smote_arm": spec.arm,
            "condition": spec.condition,
            "alpha": spec.alpha_label,
            "seed": spec.seed,
            "smote_inoperative": bool(spec.smote_inoperative),
            "data_hash": data_hash,
            "partition_hash": partition_hash,
        }
        doc = {"wandb_version": 1}
        for k, v in cfg_fields.items():
            doc[k] = {"value": v}
        cfg_path.write_text(yaml.safe_dump(doc, sort_keys=False))
        env["WANDB_CONFIG_PATHS"] = str(cfg_path)
    return env


# --------------------------------------------------------------------------- #
# Resume markers (config-hash, success-only)
# --------------------------------------------------------------------------- #
def _marker_path(spec: RunSpec) -> Path:
    return spec.run_dir / "completion.json"


def _config_fingerprint(spec: RunSpec, data_hash: str, partition_hash: str) -> str:
    import hashlib
    import json
    payload = {
        "run_name": spec.run_name, "oversampling": spec.oversampling,
        "sampling_strategy": SAMPLING_STRATEGY, "k_neighbors": K_NEIGHBORS,
        "num_clients": NUM_CLIENTS, "alpha": spec.alpha,
        "data_hash": data_hash, "partition_hash": partition_hash,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def is_complete(spec: RunSpec, fingerprint: str) -> bool:
    """A run is resumable-complete only if its success marker exists AND its config
    fingerprint matches (so a config change re-runs rather than being skipped)."""
    import json
    m = _marker_path(spec)
    if not m.is_file():
        return False
    try:
        rec = json.loads(m.read_text())
    except Exception:
        return False
    return rec.get("status") == "success" and rec.get("fingerprint") == fingerprint


def _write_marker(spec: RunSpec, fingerprint: str, extra: dict) -> None:
    """Written ONLY on success."""
    import json
    spec.run_dir.mkdir(parents=True, exist_ok=True)
    rec = {"status": "success", "fingerprint": fingerprint, "run_name": spec.run_name, **extra}
    _marker_path(spec).write_text(json.dumps(rec, indent=2))


# --------------------------------------------------------------------------- #
# Ray orphan check + cleanup
# --------------------------------------------------------------------------- #
def ray_orphan_check_and_clean(context: str) -> None:
    """Detect + clean up stray Ray actors that could survive a killed subprocess
    and starve the next run. Best-effort; safe when Ray isn't running.

    NB: the `ray stop` CLI is broken in this env (click/Sentinel incompatibility:
    "ValueError: not a valid Sentinel"), so we use ``ray.shutdown()`` (no-op if
    Ray isn't up in this process). A child's own Ray actors live in the child's
    process group and are reaped by the process-group kill in execute_run.
    """
    try:
        r = subprocess.run(["pgrep", "-f", "ray::"], capture_output=True, text=True)
        stray = [p for p in r.stdout.split() if p.strip()]
        if stray:
            print(f"[ray] {context}: {len(stray)} stray Ray process(es) detected "
                  f"(pids {','.join(stray)}); relying on process-group kill.")
    except Exception:
        pass
    try:
        import ray
        if ray.is_initialized():
            ray.shutdown()
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #
# The currently-running child Popen, so a SIGTERM/SIGINT to the runner tears down
# the WHOLE child process group (Ray actors included) instead of orphaning it —
# `pkill -f run_sweep` previously left a models.bert_fraud.run child alive 25 min.
_CURRENT_CHILD: "Optional[subprocess.Popen]" = None


def _now_stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def _kill_child_group(sig=signal.SIGTERM) -> None:
    proc = _CURRENT_CHILD
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), sig)  # child started with start_new_session=True
    except ProcessLookupError:
        pass
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass


def _install_signal_handlers() -> None:
    def _handler(signum, _frame):
        print(f"\n[runner] received signal {signum} — killing child process group and exiting.")
        _kill_child_group(signal.SIGTERM)
        raise SystemExit(130)
    for _s in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(_s, _handler)
        except Exception:
            pass


def _child_run_name(spec: RunSpec) -> str:
    """The run_name the child entry point writes under (build_run_name convention).
    Centralized FedXGBllr is run_xgb → model 'xgb'."""
    ov = spec.oversampling
    if spec.condition == "centralized":
        model = "xgb" if spec.model == "fedxgbllr" else spec.model
        return f"centralized_{model}_{ov}_seed{spec.seed}"
    scheme = "iid" if spec.condition == "iid" else "dirichlet"
    a = "-" if spec.condition == "iid" else f"{spec.alpha:g}"
    return f"{spec.model}_{scheme}_alpha{a}_{ov}_seed{spec.seed}"


def _artifact_dir(spec: RunSpec) -> Path:
    return PROJECT_ROOT / "results" / "models" / spec.dataset / _child_run_name(spec)


def _rounds_completed(spec: RunSpec) -> object:
    """Rounds actually run (early stopping ends at different rounds). From the
    child's per-round CSV; 'n/a' for centralized."""
    if spec.condition == "centralized":
        return "n/a"
    subdir = "fedxgbllr" if spec.model == "fedxgbllr" else spec.model
    # child run_name convention: <model>_<scheme>_alpha<alpha|->_<ov>_seed<seed>
    scheme = "iid" if spec.condition == "iid" else "dirichlet"
    a = "-" if spec.condition == "iid" else f"{spec.alpha:g}"
    child = f"{subdir}_{scheme}_alpha{a}_{spec.oversampling}_seed{spec.seed}_rounds.csv"
    p = PROJECT_ROOT / "results" / "logs" / spec.dataset / subdir / child
    if p.is_file():
        try:
            return max(0, sum(1 for _ in open(p)) - 1)  # minus header
        except Exception:
            return ""
    return ""


def execute_run(spec: RunSpec, gpu_available: bool, offline: bool,
                use_wandb: bool, data_hash: str, partition_hash: str) -> dict:
    spec.run_dir.mkdir(parents=True, exist_ok=True)
    logf = spec.run_dir / "run.log"
    argv, cwd = build_command(spec, gpu_available, use_wandb)
    env = build_env(spec, offline, use_wandb, data_hash=data_hash, partition_hash=partition_hash)

    ray_orphan_check_and_clean(f"before {spec.run_name}")
    # Append (never truncate): a re-run of the same cell must not destroy the
    # previous attempt's log (that erased forensic evidence during a diagnosis).
    attempt_ts = _now_stamp()
    header = (
        f"\n=== ATTEMPT {attempt_ts} | {spec.run_name} ===\n"
        f"cwd: {cwd}\ncmd: {' '.join(argv)}\n"
        f"resolved resources: {resources.for_model(spec.model, gpu_available=gpu_available)}\n"
        f"threads/actor: {resources.threads_per_actor()}\n"
        f"data_hash: {data_hash}  partition_hash: {partition_hash}\n"
        f"{'-'*70}\n"
    )
    global _CURRENT_CHILD
    t0 = time.time()
    status, exit_code = "failed", None
    try:
        with open(logf, "a") as fh:   # append mode — accumulate attempts
            fh.write(header)
            fh.flush()
            # start_new_session=True → child is its own process-group leader, so a
            # signal to the runner can tear down the whole group (Ray actors too).
            proc = subprocess.Popen(argv, cwd=str(cwd), env=env, stdout=fh,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            _CURRENT_CHILD = proc
            exit_code = proc.wait()
        status = "success" if exit_code == 0 else "failed"
    except Exception as exc:  # noqa: BLE001
        with open(logf, "a") as fh:
            fh.write(f"\n[runner] subprocess raised: {exc}\n")
        status = "failed"
    finally:
        _CURRENT_CHILD = None
        ray_orphan_check_and_clean(f"after {spec.run_name}")  # cleanup in finally

    # Completeness assert (commit 5b): a run that trained successfully but
    # persisted no loadable artifact is a FAILURE, not a pass. Silent absence is
    # the failure mode to design against — this also catches any entry point not
    # yet wired for persistence.
    artifact_dir = _artifact_dir(spec)
    if status == "success" and not _mp.manifest_ok(artifact_dir):
        status = "failed_no_artifact"
        with open(logf, "a") as fh:
            fh.write(
                f"\n[runner] PERSISTENCE MISSING: trained OK but no valid manifest "
                f"at {artifact_dir} — recording as failure.\n"
            )

    wall = round(time.time() - t0, 1)
    rec = {
        "run_name": spec.run_name, "dataset": spec.dataset, "model": spec.model,
        "smote_arm": spec.arm, "condition": spec.condition,
        "alpha": "" if spec.alpha is None else spec.alpha, "seed": spec.seed,
        "status": status, "exit_code": exit_code, "wall_seconds": wall,
        "smote_inoperative": spec.smote_inoperative, "sub6_count": spec.sub6_count,
        "data_hash": data_hash, "partition_hash": partition_hash,
        "rounds_completed": _rounds_completed(spec),
        "aggregation": (
            "n/a (centralized); XGBoost (centralized upper bound)"
            if spec.condition == "centralized" and spec.model == "fedxgbllr"
            else "n/a (centralized)" if spec.condition == "centralized"
            else _AGG.get(spec.model, "")
        ),
        "log": str(logf.relative_to(PROJECT_ROOT)),
    }
    if status == "success":
        fp = _config_fingerprint(spec, data_hash, partition_hash)
        _write_marker(spec, fp, {"wall_seconds": wall, "rounds_completed": rec["rounds_completed"]})
    return rec


_AGG = {
    "lr": "FedAvg", "svm": "FedAvg", "gbm": "best-model-selection",
    "ffd": "accuracy-weighted-FedAvg", "bert_fraud": "accuracy-weighted-FedAvg",
    "fedxgbllr": "tree-ensemble+CNN-FedAvg",
}

_MASTER_COLUMNS = [
    "run_name", "dataset", "model", "smote_arm", "condition", "alpha", "seed",
    "status", "exit_code", "wall_seconds", "smote_inoperative", "sub6_count",
    "data_hash", "partition_hash", "rounds_completed", "aggregation", "log",
]


def _append_master(rec: dict) -> None:
    """Atomic-ish append: single sequential writer, flush+fsync per row so a
    crash mid-sweep leaves a complete prefix, never a corrupt line."""
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    new = not MASTER_CSV.is_file()
    with open(MASTER_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_MASTER_COLUMNS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(rec)
        f.flush()
        os.fsync(f.fileno())


# --------------------------------------------------------------------------- #
# Disk estimate + preflight
# --------------------------------------------------------------------------- #
def _dir_size(path: Path) -> int:
    total = 0
    if path.is_dir():
        for dp, _d, fs in os.walk(path):
            for fn in fs:
                try:
                    total += os.path.getsize(os.path.join(dp, fn))
                except OSError:
                    pass
    return total


def estimate_disk_bytes(n_runs: int) -> int:
    """Data cache (measured) + per-run (model artifact + logs) + wandb offline dir
    (grows across the sweep and needs `wandb sync` afterwards)."""
    cache = _dir_size(data_cache.CACHE_ROOT)
    # Measured (commit 5): model artifacts persist REFERENCES to the shared cache,
    # not data copies — ~0.5 MiB/run avg (FFD largest ~1.5 MiB), so ~46 MiB across
    # 96 runs. Budget 2 MiB/run for model+scaler+names+manifest+logs, generous.
    per_run_artifacts = 2 * 1024 ** 2
    wandb_per_run = 15 * 1024 ** 2       # offline run dir (dominant per-run cost)
    return cache + n_runs * (per_run_artifacts + wandb_per_run)


# --------------------------------------------------------------------------- #
# Dry-run manifest
# --------------------------------------------------------------------------- #
def annotate_noops(specs: List[RunSpec]) -> None:
    for s in specs:
        if s.arm == "smote":
            inop, sub6 = smote_status(s)
            s.smote_inoperative, s.sub6_count = inop, sub6
        else:
            _inop, sub6 = smote_status(RunSpec(s.dataset, s.model, "smote", s.condition, s.seed, s.alpha))
            s.sub6_count = sub6


def print_manifest(specs: List[RunSpec], gpu_available: bool) -> None:
    annotate_noops(specs)
    to_run = [s for s in specs if not (s.arm == "smote" and s.smote_inoperative)]
    skipped = [s for s in specs if s.arm == "smote" and s.smote_inoperative]
    print("=== DRY-RUN MANIFEST ===")
    print(f"planned cells: {len(specs)} | to run: {len(to_run)} | "
          f"skipped SMOTE no-ops: {len(skipped)}")
    print(f"\nper-model resource allocation (gpu_available={gpu_available}, "
          f"gpu_fraction default={resources.gpu_fraction_default()}):")
    for m in sorted({s.model for s in specs}):
        print(f"  {m:12} {resources.for_model(m, gpu_available=gpu_available)}")
    print("\nSMOTE no-op cells (smote arm == no-smote arm; skipped):")
    for s in skipped:
        print(f"  {s.run_name}  (sub6={s.sub6_count})")
    if not skipped:
        print("  (none)")
    print(f"\nestimated disk for {len(to_run)} runs: "
          f"{estimate_disk_bytes(len(to_run)) / 1024**3:.1f} GiB "
          f"(incl. data cache + model artifacts + logs + wandb offline)")
    print("\nall planned runs:")
    for s in specs:
        tag = "SKIP(inoperative)" if (s.arm == "smote" and s.smote_inoperative) else "run"
        print(f"  [{tag:17}] {s.run_name}  sub6={s.sub6_count}")
    print("\n[launch] start under tmux/screen/nohup — a dropped SSH sends SIGHUP "
          "and kills the whole sweep.")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _csv_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified sweep runner (safe defaults; no-arg → manifest).")
    p.add_argument("--datasets", type=_csv_list, default=list(ALL_DATASETS))
    p.add_argument("--models", default="all")
    p.add_argument("--conditions", type=_csv_list, default=list(ALL_CONDITIONS))
    p.add_argument("--alpha", type=lambda s: [float(x) for x in _csv_list(s)], default=[0.5])
    p.add_argument("--seeds", type=lambda s: [int(x) for x in _csv_list(s)], default=[42])
    p.add_argument("--smote-arms", choices=["both", "smote", "no-smote"], default="both")
    p.add_argument("--gpu-fraction", type=float, default=None,
                   help="override per-client GPU fraction (default: central config value).")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="pre-launch gate: run the cheapest cell (centralized LR on "
                        "creditcard, no-smote) end-to-end exactly as the runner "
                        "invokes it (subprocess + real wandb.init via "
                        "WANDB_CONFIG_PATHS), and FAIL loudly unless it yields a "
                        "complete results row AND a loadable artifact. Run with "
                        "--offline. Catches a sweep that goes green while producing "
                        "nothing.")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--offline", action="store_true", help="wandb offline mode.")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--timing-probe", action="store_true",
                   help="run ONE cell (the first of the filtered matrix), report "
                        "total+per-round wall time + peak GPU mem, exit. PRIORITY on "
                        "the GPU box: BERT-on-PaySim FEDERATED first, then CENTRALIZED "
                        "(6.36M rows × 12 BERT runs decide whether the sweep fits the "
                        "window). E.g.: --timing-probe --datasets paysim --models "
                        "bert_fraud --conditions noniid --smote-arms no-smote, then "
                        "--conditions centralized.")
    return p.parse_args(argv)


def _resolve(args) -> Tuple[List[str], List[str], List[str]]:
    models = list(ALL_MODELS) if args.models == "all" else _csv_list(args.models)
    return args.datasets, models, args.conditions


def main(argv=None) -> int:
    _install_signal_handlers()
    args = parse_args(argv)
    datasets, models, conditions = _resolve(args)
    arms = list(SMOTE_ARMS) if args.smote_arms == "both" else [args.smote_arms]
    specs = build_matrix(datasets, models, conditions, args.alpha, args.seeds, arms)
    gpu_count, _vram = sweep_preflight.detect_gpus()
    gpu_available = gpu_count > 0

    # Resolve --gpu-fraction and publish it via env so it crosses the subprocess
    # boundary to every child (build_env copies os.environ) AND so the runner's own
    # manifest/preflight (which call resources.for_model) report what will actually
    # run — not the config default. This is the fix for "--gpu-fraction had no
    # effect": children read SWEEP_GPU_FRACTION in resources.for_model.
    gpu_fraction = args.gpu_fraction if args.gpu_fraction is not None else resources.gpu_fraction_default()
    os.environ["SWEEP_GPU_FRACTION"] = str(gpu_fraction)

    # No args at all → manifest only (never launch 108 jobs by accident).
    if args.dry_run or len(sys.argv) == 1:
        print_manifest(specs, gpu_available)
        return 0

    if args.smoke:
        return _smoke_gate(args.offline)

    if args.timing_probe:
        return _timing_probe(specs[0], gpu_available, args)

    # Preflight (aborts before run 1 on any capacity/disk violation).
    to_run_est = sum(1 for s in specs if not (s.arm == "smote" and smote_status(s)[0]))
    sweep_preflight.run_preflight(
        models=models, gpu_fraction=gpu_fraction,
        n_runs=to_run_est,
        est_bytes_per_run=(2 + 15) * 1024 ** 2,  # model artifacts (measured ~0.5) + wandb offline
    )

    annotate_noops(specs)
    use_wandb = not args.no_wandb
    completed = failed = skipped = 0
    for spec in specs:
        if spec.arm == "smote" and spec.smote_inoperative:
            print(f"[skip] {spec.run_name} — SMOTE inoperative (== no-smote arm)")
            data_hash, partition_hash = cache_hashes(spec)
            _append_master({
                "run_name": spec.run_name, "dataset": spec.dataset, "model": spec.model,
                "smote_arm": spec.arm, "condition": spec.condition,
                "alpha": "" if spec.alpha is None else spec.alpha, "seed": spec.seed,
                "status": "skipped_inoperative", "exit_code": "", "wall_seconds": 0,
                "smote_inoperative": True, "sub6_count": spec.sub6_count,
                "data_hash": data_hash, "partition_hash": partition_hash,
                "rounds_completed": "n/a", "aggregation": "", "log": "",
            })
            skipped += 1
            continue

        data_hash, partition_hash = cache_hashes(spec)
        fp = _config_fingerprint(spec, data_hash, partition_hash)
        if args.resume and is_complete(spec, fp):
            print(f"[resume] {spec.run_name} — already complete, skipping")
            skipped += 1
            continue

        print(f"[run] {spec.run_name}")
        rec = execute_run(spec, gpu_available, args.offline, use_wandb, data_hash, partition_hash)
        _append_master(rec)
        if rec["status"] == "success":
            completed += 1
        else:
            failed += 1
            print(f"[FAIL] {spec.run_name} (exit {rec['exit_code']}) — see {rec['log']}; continuing")

    print(f"\n=== SWEEP DONE === completed={completed} failed={failed} skipped={skipped}")
    if use_wandb and args.offline:
        print(f"[wandb] offline runs in {WANDB_OFFLINE_DIR} — sync later with `wandb sync`")
    return 0 if failed == 0 else 1


# Metric/identity columns that MUST be populated for a centralized summary row to
# count as complete. The intentionally-blank FL-only columns (num_rounds,
# best_round, alpha) are excluded on purpose. "NA" counts as present — it is a
# recorded value (e.g. SVM calibration), not an empty cell.
_SMOKE_REQUIRED_COLS = (
    "test_auprc", "test_f1", "test_precision", "test_recall",
    "best_val_auprc", "best_val_f1", "threshold", "data_hash",
    "baseline_auprc", "timestamp", "duration_seconds", "run_name",
)


def _summary_csv_path(spec: RunSpec) -> Path:
    """Path to the single-row summary CSV the child writes for any cell
    (centralized or FL), following the results_writer subdir convention."""
    if spec.condition == "centralized":
        subdir = "centralized"
    else:
        subdir = "fedxgbllr" if spec.model == "fedxgbllr" else spec.model
    return (PROJECT_ROOT / "results" / "logs" / spec.dataset / subdir /
            f"{_child_run_name(spec)}.csv")


def _summary_row_complete(path: Path) -> Tuple[bool, List[str]]:
    """(complete?, missing_cols) for the last data row of a summary CSV."""
    if not path.is_file():
        return False, ["<summary CSV missing>"]
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return False, ["<no data row>"]
    row = rows[-1]
    missing = [c for c in _SMOKE_REQUIRED_COLS if str(row.get(c, "")).strip() == ""]
    return (not missing), missing


# The smoke gate exercises two structurally different paths end-to-end:
#   1. centralized LR on creditcard — cheapest cell; the argparse + wandb.init +
#      WANDB_CONFIG_PATHS + sklearn-persist path.
#   2. federated FedXGBllr on creditcard (iid, no-smote) — the Hydra two-stage
#      path (per-client boosters + aggregator CNN) that the LR cell never touches:
#      early-stopping capture, two-stage persistence, and CNN-probability
#      calibration. This one early-stops (~round 12) so it also guards the
#      last-executed-round persistence capture (~4 min).
_SMOKE_CELLS: Tuple[RunSpec, ...] = (
    RunSpec("creditcard", "lr", "no-smote", "centralized", seed=42, alpha=None),
    RunSpec("creditcard", "fedxgbllr", "no-smote", "iid", seed=42, alpha=None),
)


def _smoke_gate(offline: bool) -> int:
    """Execute representative cells end-to-end EXACTLY as the runner invokes them
    (subprocess, WANDB_CONFIG_PATHS, real ``wandb.init()``) and fail loudly unless
    EACH produces BOTH a complete results row (no empty metric columns) AND a
    loadable artifact.

    Pre-launch gate: a matrix that goes green while silently producing nothing —
    the ``wandb.init()`` crash that took down all 96 runs, an early-stop
    persistence miss on FedXGBllr, or any persistence regression — is caught here
    before an overnight window is burnt. Run with ``--offline`` (wandb still reads
    WANDB_CONFIG_PATHS in offline mode, so the exact format that broke is
    exercised)."""
    gpu_count, _vram = sweep_preflight.detect_gpus()
    gpu_available = gpu_count > 0
    os.environ.setdefault("SWEEP_GPU_FRACTION", str(resources.gpu_fraction_default()))

    results = []
    for spec in _SMOKE_CELLS:
        print(f"\n=== SMOKE CELL: {spec.run_name} (offline={offline}) ===")
        data_hash, partition_hash = cache_hashes(spec)
        # use_wandb=True so wandb.init() + the WANDB_CONFIG_PATHS yaml are exercised.
        rec = execute_run(spec, gpu_available, offline, use_wandb=True,
                          data_hash=data_hash, partition_hash=partition_hash)
        summary_csv = _summary_csv_path(spec)
        row_ok, missing = _summary_row_complete(summary_csv)
        artifact_ok = _mp.manifest_ok(_artifact_dir(spec))
        ok = rec["status"] == "success" and row_ok and artifact_ok
        results.append((spec, rec, summary_csv, row_ok, missing, artifact_ok, ok))

    print("\n=== SMOKE GATE RESULT ===")
    for spec, rec, summary_csv, row_ok, missing, artifact_ok, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {spec.run_name}")
        print(f"        status={rec['status']} (exit {rec['exit_code']}) | "
              f"row={'OK' if row_ok else 'INCOMPLETE'} | "
              f"artifact={'OK' if artifact_ok else 'MISSING'}")
        if not row_ok:
            print(f"        empty/missing columns: {missing}  ({summary_csv})")
        if not artifact_ok:
            print(f"        artifact dir: {_artifact_dir(spec)}")
        print(f"        log: {rec['log']}")
    passed = all(r[-1] for r in results)
    print(f"\n  SMOKE GATE: {'PASS' if passed else 'FAIL'}")
    if not passed:
        print("  Refusing to certify launch: a green sweep would produce nothing.")
    return 0 if passed else 1


def _timing_probe(spec: RunSpec, gpu_available: bool, args) -> int:
    """Run ONE cell end-to-end; report total + per-round wall time + peak GPU mem +
    rounds_completed; do NOT write master results."""
    print(f"=== TIMING PROBE: {spec.run_name} ===")
    data_hash, partition_hash = cache_hashes(spec)
    peak_mem = {"mib": 0}
    stop = {"flag": False}

    import threading

    def _sample():
        while not stop["flag"]:
            try:
                r = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5)
                vals = [int(x) for x in r.stdout.split() if x.strip().isdigit()]
                if vals:
                    peak_mem["mib"] = max(peak_mem["mib"], max(vals))
            except Exception:
                pass
            time.sleep(2)

    sampler = threading.Thread(target=_sample, daemon=True)
    if gpu_available:
        sampler.start()
    rec = execute_run(spec, gpu_available, args.offline, use_wandb=False,
                      data_hash=data_hash, partition_hash=partition_hash)
    stop["flag"] = True
    rounds = rec["rounds_completed"]
    total = rec["wall_seconds"]
    per_round = (total / rounds) if isinstance(rounds, int) and rounds > 0 else "n/a"
    print("\n=== TIMING PROBE RESULT ===")
    print(f"  cell           : {spec.run_name}")
    print(f"  status         : {rec['status']} (exit {rec['exit_code']})")
    print(f"  total wall (s) : {total}")
    print(f"  rounds_completed: {rounds}")
    print(f"  per-round (s)  : {per_round}   (use this to extrapolate if rounds change)")
    print(f"  peak GPU mem   : {peak_mem['mib'] if gpu_available else 'n/a (no GPU)'} MiB")
    return 0 if rec["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
