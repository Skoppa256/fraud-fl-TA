"""Tests for the sweep runner's pure logic (no training, no cache needed).

Matrix shape, SMOTE no-op predicate, command construction, and resume-fingerprint
semantics. Runnable via ``pytest`` or ``python tests/<file>.py``.
"""

import contextlib
import csv
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation import results_paths as P  # noqa: E402
from experiments import run_sweep as R  # noqa: E402
from experiments.run_sweep import RunSpec  # noqa: E402


@contextlib.contextmanager
def _tmp_seeds_root():
    """Redirect results/seeds/ to a throwaway dir, so tests leave nothing in the repo."""
    prev = P.SEEDS_ROOT
    with tempfile.TemporaryDirectory() as d:
        P.SEEDS_ROOT = Path(d)
        try:
            yield Path(d)
        finally:
            P.SEEDS_ROOT = prev


def test_matrix_counts_single_alpha_single_seed():
    specs = R.build_matrix(
        list(R.ALL_DATASETS), list(R.ALL_MODELS), list(R.ALL_CONDITIONS),
        alphas=[0.5], seeds=[42], arms=list(R.SMOTE_ARMS),
    )
    # 3 datasets × 6 models × 2 arms × 3 conditions = 108
    assert len(specs) == 108
    # noniid alpha attached; centralized/iid alpha None
    for s in specs:
        assert (s.alpha == 0.5) == (s.condition == "noniid")


def test_matrix_alpha_multiplies_only_noniid():
    specs = R.build_matrix(["baf"], ["lr"], ["centralized", "iid", "noniid"],
                           alphas=[0.5, 1.0, 5.0], seeds=[42], arms=["smote"])
    # 1 centralized + 1 iid + 3 noniid = 5
    assert len(specs) == 5
    assert sum(s.condition == "noniid" for s in specs) == 3
    assert sum(s.condition == "centralized" for s in specs) == 1


def test_run_name_and_group():
    s = RunSpec("baf", "ffd", "no-smote", "noniid", seed=42, alpha=0.5)
    assert s.group == "baf_no-smote_noniid_ffd"
    assert s.run_name == "baf_no-smote_noniid_ffd_a0.5_s42"
    c = RunSpec("paysim", "lr", "smote", "centralized", seed=42, alpha=None)
    assert c.run_name == "paysim_smote_centralized_lr_ana_s42"


def test_client_skip_predicate():
    # too few minority (< k+1=6) → skip
    assert R._client_skips(n_min=5, n_maj=1000) is True
    # already meets target ratio (>= 0.01) → skip
    assert R._client_skips(n_min=20, n_maj=1000) is True   # 0.02 >= 0.01
    # below target with enough minority → NOT a skip (SMOTE fires)
    assert R._client_skips(n_min=6, n_maj=100000) is False  # 6e-5 < 0.01
    # zero majority → skip
    assert R._client_skips(n_min=100, n_maj=0) is True


def test_build_command_centralized_maps_fedxgbllr_to_xgb():
    s = RunSpec("baf", "fedxgbllr", "smote", "centralized", seed=42)
    argv, cwd = R.build_command(s, gpu_available=False, use_wandb=False)
    assert "experiments.centralized_baseline.run_xgb" in argv
    assert "--dataset" in argv and "baf" in argv and "--oversampling" in argv and "smote" in argv


def test_build_command_fedxgbllr_noniid_hydra():
    s = RunSpec("paysim", "fedxgbllr", "no-smote", "noniid", seed=42, alpha=0.5)
    argv, cwd = R.build_command(s, gpu_available=False, use_wandb=False)
    assert "hfedxgboost.main" in argv
    assert "dataset=paysim" in argv and "clients=paysim_5_clients" in argv
    assert "dataset.oversampling.method=none" in argv
    assert "dataset.non_iid.enabled=true" in argv and "dataset.non_iid.alpha=0.5" in argv
    assert "XGBoost.device=cpu" in argv           # CPU-resolved on this box (override, not +append)
    assert str(cwd).endswith(os.path.join("models", "fedxgbllr"))


def test_build_command_argparse_iid():
    s = RunSpec("creditcard", "ffd", "smote", "iid", seed=123)
    argv, _ = R.build_command(s, gpu_available=False, use_wandb=True)
    assert "models.ffd.run" in argv
    assert "--scheme" in argv and "iid" in argv
    assert "--alpha" not in argv          # iid has no alpha
    assert "--random_seed" in argv and "123" in argv


def test_env_propagates_thread_pins_and_wandb():
    s = RunSpec("baf", "bert_fraud", "smote", "noniid", seed=42, alpha=0.5)
    with _tmp_seeds_root():
        env = R.build_env(s, offline=True, use_wandb=True,
                          data_hash="dh" * 32, partition_hash="ph" * 32)
        assert env["OMP_NUM_THREADS"] == env["MKL_NUM_THREADS"] == str(R.resources.threads_per_actor())
        assert env["WANDB_MODE"] == "offline"
        assert env["WANDB_RUN_GROUP"] == s.group and env["WANDB_NAME"] == s.run_name
        assert env["WANDB_TAGS"] == "baf,bert_fraud,noniid"
        # Extra config fields merged into wandb.config via WANDB_CONFIG_PATHS.
        # Validate against wandb's REAL parser (dict_from_config_file), not just
        # substring presence — a flat key:value file crashes it at v["value"],
        # which is exactly how all 96 runs failed. This is the regression guard.
        from wandb.sdk.lib.config_util import dict_from_config_file
        parsed = dict_from_config_file(env["WANDB_CONFIG_PATHS"])
        assert parsed["dataset"] == "baf"
        assert parsed["model"] == "bert_fraud"
        assert parsed["smote_arm"] == "smote"
        assert parsed["condition"] == "noniid"
        assert str(parsed["alpha"]) == "0.5"
        assert parsed["seed"] == 42
        assert parsed["smote_inoperative"] is False
        assert parsed["data_hash"] == "dh" * 32
        assert parsed["partition_hash"] == "ph" * 32


def test_build_env_carries_gpu_fraction_to_child():
    """SWEEP_GPU_FRACTION (set by main() from --gpu-fraction) must survive into
    the child env — build_env copies os.environ, so the child's for_model reads
    the requested fraction rather than the config default."""
    prev = os.environ.get("SWEEP_GPU_FRACTION")
    s = RunSpec("baf", "bert_fraud", "no-smote", "noniid", seed=42, alpha=0.5)
    try:
        with _tmp_seeds_root():
            os.environ["SWEEP_GPU_FRACTION"] = "0.2"
            env = R.build_env(s, offline=True, use_wandb=False)
            assert env["SWEEP_GPU_FRACTION"] == "0.2", env.get("SWEEP_GPU_FRACTION")
    finally:
        if prev is None:
            os.environ.pop("SWEEP_GPU_FRACTION", None)
        else:
            os.environ["SWEEP_GPU_FRACTION"] = prev


def test_every_run_writes_into_its_seed_tree():
    """Seed 42 included: the runner never targets the frozen seed-42 record."""
    for seed in (42, 123):
        tree = P.SEEDS_ROOT / f"seed{seed}"
        for s in (RunSpec("paysim", "fedxgbllr", "smote", "noniid", seed=seed, alpha=0.5),
                  RunSpec("baf", "lr", "no-smote", "centralized", seed=seed)):
            assert s.root == tree
            paths = (s.run_dir, R._artifact_dir(s), R._summary_csv_path(s),
                     R._master_csv(seed), R._wandb_dir(seed))
            for p in paths:
                assert tree in p.parents, p
                P.assert_writable(p)   # outside every frozen folder
        assert R._summary_csv_path(RunSpec("baf", "lr", "no-smote", "centralized", seed=seed)) \
            == tree / "logs" / "baf" / "centralized" / f"centralized_lr_none_seed{seed}.csv"


def test_build_env_points_child_at_seed_tree():
    """The child gets its tree as an ABSOLUTE path (fedxgbllr runs from another cwd)."""
    s = RunSpec("creditcard", "fedxgbllr", "no-smote", "iid", seed=123)
    with _tmp_seeds_root() as tmp:
        env = R.build_env(s, offline=True, use_wandb=True)
        assert env[P.RESULTS_ROOT_ENV] == str(tmp / "seed123")
        assert Path(env[P.RESULTS_ROOT_ENV]).is_absolute()
        # wandb falls back to the system temp dir when WANDB_DIR is missing.
        assert env["WANDB_DIR"] == str(tmp / "seed123" / "sweep" / "wandb_offline")
        assert Path(env["WANDB_DIR"]).is_dir()


def test_build_command_fedxgbllr_smote_seed_follows_run_seed():
    """Every other entry point seeds SMOTE from --random_seed; FedXGBllr's YAML pins 42."""
    for seed in (42, 123):
        s = RunSpec("baf", "fedxgbllr", "smote", "noniid", seed=seed, alpha=0.5)
        argv, _ = R.build_command(s, gpu_available=False, use_wandb=False)
        assert f"dataset.oversampling.base_seed={seed}" in argv
        assert f"random_seed={seed}" in argv


def test_build_env_pins_determinism_settings():
    """cuBLAS reads its workspace setting when CUDA starts, so it must arrive via env."""
    s = RunSpec("baf", "ffd", "no-smote", "iid", seed=123)
    with _tmp_seeds_root():
        env = R.build_env(s, offline=True, use_wandb=False)
    assert env["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert env["PYTHONHASHSEED"] == "123"


def test_append_master_writes_one_master_per_seed():
    with _tmp_seeds_root() as tmp:
        for seed in (123, 2024):
            R._append_master({"run_name": f"r_s{seed}", "seed": seed, "status": "success"})
        for seed in (123, 2024):
            rows = list(csv.DictReader(open(tmp / f"seed{seed}" / "sweep" / "sweep_master.csv")))
            assert [r["run_name"] for r in rows] == [f"r_s{seed}"]
            assert list(rows[0]) == R._MASTER_COLUMNS


def test_resume_fingerprint_changes_with_config():
    s = RunSpec("baf", "lr", "smote", "iid", seed=42)
    fp1 = R._config_fingerprint(s, "datahashA", "parthashA")
    assert R._config_fingerprint(s, "datahashA", "parthashA") == fp1  # stable
    assert R._config_fingerprint(s, "datahashB", "parthashA") != fp1  # data changed
    assert R._config_fingerprint(s, "datahashA", "parthashB") != fp1  # partition changed


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} run_sweep logic tests passed.")
