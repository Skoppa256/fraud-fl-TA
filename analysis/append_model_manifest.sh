#!/usr/bin/env bash
# Append results/models/** hashes to docs/frozen_manifest.sha256 — RUN ON THE BOX.
#
# Why this exists: results/models/ is empty on the laptop (the persisted sweep
# artifacts live only on the GPU box), so the manifest generated there covers
# four of CLAUDE.md's five freeze rows. A green `shasum -a 256 -c` on the laptop
# therefore proves nothing about the model artifacts. This script closes that gap
# when run where the artifacts actually are.
#
#   bash analysis/append_model_manifest.sh
#
# Idempotent: refuses to append if the manifest already carries results/models/
# entries. It appends only; it never rewrites or reorders existing lines, and it
# hashes tracked files only, matching how the rest of the manifest was built.
#
# Afterwards, verify and commit the manifest:
#   shasum -a 256 -c docs/frozen_manifest.sha256
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MANIFEST="docs/frozen_manifest.sha256"
PREFIX="results/models/"

if [[ ! -f "$MANIFEST" ]]; then
    echo "FATAL: $MANIFEST not found (run from a full checkout)." >&2
    exit 1
fi

# --- idempotence guard: refuse rather than duplicate ------------------------- #
existing="$(grep -c "  ${PREFIX}" "$MANIFEST" || true)"
if [[ "$existing" -gt 0 ]]; then
    echo "REFUSING: $MANIFEST already has $existing ${PREFIX} entries."
    echo "Nothing appended. To re-hash, remove those lines deliberately first —"
    echo "that is a decision about a result of record, not a step inside a re-run."
    exit 0
fi

# --- collect tracked artifacts ---------------------------------------------- #
mapfile -t files < <(git ls-files "$PREFIX" | sort)
if [[ "${#files[@]}" -eq 0 ]]; then
    echo "REFUSING: no TRACKED files under ${PREFIX}."
    echo "On the box, the artifacts may exist but be untracked — .gitignore has a"
    echo "model-artifacts block. Force-add them first (git add -f), confirm they"
    echo "are the artifacts the sweep produced, then re-run this script."
    echo "Nothing appended."
    exit 1
fi

before="$(wc -l < "$MANIFEST")"
printf '%s\n' "${files[@]}" | xargs shasum -a 256 >> "$MANIFEST"
after="$(wc -l < "$MANIFEST")"

echo "appended $(( after - before )) entries for ${PREFIX} (${before} -> ${after})"
echo "now verify:  shasum -a 256 -c $MANIFEST"
