"""Tests for the sweep runner's pure logic (no training, no cache needed).

Matrix shape, SMOTE no-op predicate, command construction, and resume-fingerprint
semantics. Runnable via ``pytest`` or ``python tests/<file>.py``.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments import run_sweep as R  # noqa: E402
from experiments.run_sweep import RunSpec  # noqa: E402


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
    import shutil
    s = RunSpec("baf", "bert_fraud", "smote", "noniid", seed=42, alpha=0.5)
    try:
        env = R.build_env(s, offline=True, use_wandb=True,
                          data_hash="dh" * 32, partition_hash="ph" * 32)
        assert env["OMP_NUM_THREADS"] == env["MKL_NUM_THREADS"] == str(R.resources.threads_per_actor())
        assert env["WANDB_MODE"] == "offline"
        assert env["WANDB_RUN_GROUP"] == s.group and env["WANDB_NAME"] == s.run_name
        assert env["WANDB_TAGS"] == "baf,bert_fraud,noniid"
        # Extra config fields merged into wandb.config via WANDB_CONFIG_PATHS.
        cfg = open(env["WANDB_CONFIG_PATHS"]).read()
        for needed in ("smote_arm: smote", "condition: noniid", "alpha: 0.5",
                       "seed: 42", "smote_inoperative: false",
                       "data_hash: " + "dh" * 32, "partition_hash: " + "ph" * 32):
            assert needed in cfg, f"missing wandb config field: {needed!r}"
    finally:
        shutil.rmtree(s.run_dir, ignore_errors=True)


def test_build_env_carries_gpu_fraction_to_child():
    """SWEEP_GPU_FRACTION (set by main() from --gpu-fraction) must survive into
    the child env — build_env copies os.environ, so the child's for_model reads
    the requested fraction rather than the config default."""
    import shutil
    prev = os.environ.get("SWEEP_GPU_FRACTION")
    s = RunSpec("baf", "bert_fraud", "no-smote", "noniid", seed=42, alpha=0.5)
    try:
        os.environ["SWEEP_GPU_FRACTION"] = "0.2"
        env = R.build_env(s, offline=True, use_wandb=False)
        assert env["SWEEP_GPU_FRACTION"] == "0.2", env.get("SWEEP_GPU_FRACTION")
    finally:
        if prev is None:
            os.environ.pop("SWEEP_GPU_FRACTION", None)
        else:
            os.environ["SWEEP_GPU_FRACTION"] = prev
        shutil.rmtree(s.run_dir, ignore_errors=True)


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
