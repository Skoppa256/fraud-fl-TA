"""Regression guard: no CPU/GPU resource literal outside the central config.

Preflight only validates runner-controlled paths; it cannot catch a bare
per-model invocation that carries its own hardcoded allocation (exactly how a
wrong per-config value escaped before). This test statically scans the model
entry points, their Hydra/YAML configs, and the sweep drivers for resurrected
resource literals and fails if any reappear.

Flagged patterns (outside the allowlisted central files):
  * ``num_gpus_per_client`` / ``num_cpus_per_client`` keys anywhere
  * a ``client_resources:`` YAML key
  * ``num_gpus`` / ``num_cpus`` assigned a NUMERIC literal (dict or YAML),
    e.g. ``"num_gpus": 0`` or ``num_cpus: 10`` — but NOT ``"num_gpus": _res[...]``
    or ``detect_gpus()[0]`` (values sourced from the central config).

Runnable via ``pytest`` or ``python tests/<file>.py``.
"""

import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

# Directories scanned for stray literals.
_SCAN_DIRS = ["models", "experiments", "preprocessing", "evaluation"]

# Files/dirs that legitimately contain these tokens.
_ALLOW_SUBSTR = (
    os.path.join("experiments", "sweep_resources.yaml"),
    os.path.join("experiments", "resources.py"),
    os.path.join("experiments", "sweep_preflight.py"),
    os.sep + "tests" + os.sep,
    os.sep + "__pycache__" + os.sep,
    os.sep + "outputs" + os.sep,   # stale Hydra run artifacts
    os.sep + "wandb" + os.sep,     # stale wandb run artifacts
    os.sep + "results" + os.sep,
)

_PER_CLIENT = re.compile(r"num_(?:gpus|cpus)_per_client")
_CLIENT_RES_KEY = re.compile(r"^\s*client_resources\s*:")            # YAML key
_NUMERIC_ALLOC = re.compile(r"""num_(?:gpus|cpus)["']?\s*[:=]\s*["']?\s*[0-9]""")


def _iter_files():
    for d in _SCAN_DIRS:
        base = os.path.join(ROOT, d)
        for dirpath, _dirs, names in os.walk(base):
            for n in names:
                if not (n.endswith(".py") or n.endswith(".yaml") or n.endswith(".sh")):
                    continue
                full = os.path.join(dirpath, n)
                if any(sub in full for sub in _ALLOW_SUBSTR):
                    continue
                yield full


def find_resource_literals():
    offenders = []
    for full in _iter_files():
        with open(full, encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f, 1):
                stripped = line.strip()
                if stripped.startswith("#") or stripped.startswith("//"):
                    continue  # comment line — a mention, not a literal
                if (
                    _PER_CLIENT.search(line)
                    or _CLIENT_RES_KEY.search(line)
                    or _NUMERIC_ALLOC.search(line)
                ):
                    offenders.append(f"{os.path.relpath(full, ROOT)}:{i}: {stripped}")
    return offenders


def test_no_resource_literals_outside_central_config():
    offenders = find_resource_literals()
    assert not offenders, (
        "resource literals found outside experiments/sweep_resources.yaml — route "
        "them through experiments.resources:\n  " + "\n  ".join(offenders)
    )


if __name__ == "__main__":
    off = find_resource_literals()
    if off:
        print("FAIL — resource literals found:")
        for o in off:
            print("  " + o)
        sys.exit(1)
    print("PASS — no resource literals outside the central config.")
