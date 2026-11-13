#!/usr/bin/env python3
"""Rebuild every BAB 3 / BAB 4 figure from the experiment CSVs.

    python analysis/figures/make_all.py           # build all, then check coverage
    python analysis/figures/make_all.py --check   # coverage check only, build nothing

Coverage guard: every `image("resources/...")` that BAB 3 or BAB 4 of
buku/content.typ actually renders must be produced by a script in GENERATORS,
or be listed in EXTERNAL (figures reproduced from outside sources). A figure the
thesis shows but no script owns is an error, so a hand-made PNG cannot slip back
into these chapters unnoticed.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONTENT = ROOT / "buku" / "content.typ"

# buku/resources/<file>  ->  script in this directory that writes it
#
# The four method diagrams (Gambar 3.1-3.4) are no longer here: as of
# 2026-10-04 (3.1-3.3) and 2026-10-05 (3.4) the author draws them by hand in an
# external tool and the thesis renders those files instead (see EXTERNAL). Their
# generators are kept — they still run and still read the CSVs — but nothing in
# BAB 3 renders their output, so they are no longer part of the rebuild.
#
# Gambar 3.4 is the newest of these. It replaced a GENERATED figure, so unlike
# 3.1-3.3 it also dropped a data-bearing panel: the old
# fig_3_4_two_seeds_gap.py plotted a real cell (BAF BERT Non-IID tanpa SMOTE)
# from the RQ3 CSVs. The hand-drawn version is structural only, so no experiment
# number reaches it any more.
GENERATORS = {
    "fig-4-2-minority-typology.png": "fig_4_1_minority_typology.py",
    "fig-4-auprc-boxplot.png": "fig_4_auprc_boxplot.py",
    "fig-4-rq3-spearman-boxplot.png": "fig_4_rq3_korelasi_boxplot.py",
    "fig-4-rq3-weighted-boxplot.png": "fig_4_rq3_korelasi_boxplot.py",
}

# Figures in BAB 3/4 that are NOT generated from experiment data.
#
# The method diagrams are authored by hand (2026-10-04 revision, delivered in
# TATATA.docx) and are structural rather than data-bearing: unlike the generated
# versions they superseded, they carry no row counts, fraud shares, per-client
# counts, K or alpha. Nothing rebuilds them, so a pipeline change that moves
# those numbers will NOT reach these diagrams — update them by hand.
EXTERNAL: set = {
    "fig-3-1-alur-penelitian.png",
    "fig-3-2-lapisan-sistem.png",
    "fig-3-3-pembagian-dua-tingkat.png",
    "fig-3-4-skema-dua-seed.png",
}

# BAB 3 and BAB 4 are found by POSITION, not by title: they are the 3rd and 4th
# level-1 headings (`= ...`) in content.typ, and BAB 4 ends at the 5th heading or
# at end of file. Matching on titles broke as soon as BAB 5 was renamed
# ("= PENUTUP" -> "= KESIMPULAN DAN SARAN"), and the guard then failed on every
# run without ever reaching the coverage check.
FIRST_BAB, LAST_BAB = 3, 4


def chapter_images() -> list:
    if not CONTENT.is_file():
        raise SystemExit(f"FATAL: thesis source not found.\n  looked in: {CONTENT}")
    lines = CONTENT.read_text(encoding="utf-8").splitlines()
    heads = [i for i, l in enumerate(lines) if re.match(r"= \S", l)]
    if len(heads) < LAST_BAB:
        raise SystemExit(f"FATAL: found {len(heads)} level-1 headings in {CONTENT}; "
                         f"need at least {LAST_BAB} to locate BAB {FIRST_BAB}-{LAST_BAB}.")
    a = heads[FIRST_BAB - 1]
    b = heads[LAST_BAB] if len(heads) > LAST_BAB else len(lines)
    end_name = lines[b].strip() if b < len(lines) else "end of file"
    print(f"scanning BAB {FIRST_BAB}-{LAST_BAB}: '{lines[a].strip()}' "
          f"(line {a + 1}) up to '{end_name}'")
    out = []
    for l in lines[a:b]:
        if l.lstrip().startswith("//"):
            continue
        out += re.findall(r'image\(\s*"resources/([^"]+)"', l)
    return out


def check() -> None:
    used = chapter_images()
    if not used:
        raise SystemExit("FATAL: found no image() calls in BAB 3/4 — the chapter "
                         "markers or the image syntax changed; fix the parser.")
    orphan = [f for f in used if f not in GENERATORS and f not in EXTERNAL]
    if orphan:
        raise SystemExit("FATAL: BAB 3/4 render figure(s) with no generator script:\n  "
                         + "\n  ".join(orphan)
                         + "\n  add a script to GENERATORS, or list the file in "
                           "EXTERNAL if it is reproduced from an outside source.")
    unused = sorted(set(GENERATORS) - set(used))
    ext = sorted(f for f in used if f in EXTERNAL)
    print(f"coverage OK: {len(used)} figure(s) in BAB 3/4, "
          f"{len(used) - len(ext)} generated from data, {len(ext)} authored by hand.")
    if ext:
        print("note: NOT rebuilt by this script — a data change will not reach "
              "them:\n  " + "\n  ".join(ext))
    if unused:
        print("note: generated but not referenced in BAB 3/4: " + ", ".join(unused))


def build() -> None:
    for name, script in GENERATORS.items():
        print(f"--- {script}")
        r = subprocess.run([sys.executable, str(HERE / script)], cwd=ROOT)
        if r.returncode != 0:
            raise SystemExit(f"FATAL: {script} failed (exit {r.returncode}); "
                             f"{name} was not rebuilt.")


if __name__ == "__main__":
    if "--check" not in sys.argv[1:]:
        build()
    check()
