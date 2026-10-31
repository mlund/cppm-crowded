#!/usr/bin/env bash
# Regenerate all analysis figures from data into figures.
# Static schematics and VMD renders live in figures/schematics and are not rebuilt here.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export SOURCE_DATE_EPOCH=0 # fixed PDF timestamps, so reruns give identical files
DATA=data
OUT=figures
py() { uv run --frozen python "$@"; }

# Main κ analysis; best-κ figures (ESI) use the minimum of the matching score S from kappa_invariants.py.
# Other criteria remain available via --fit-mode peak|rmsd|weighted|mayer|kb.
py analysis/plot_kappa_analysis.py "$DATA" "$OUT" --no-titles --fit-mode joint

# Explicit-ion pressure, ion counts and convergence
py analysis/plot_explicit_analysis.py "$DATA/explicit" "$OUT/explicit" --no-titles

# Signed Kirkwood–Buff and orientational invariants, joint χ², PCA check
py analysis/kappa_invariants.py "$DATA" "$OUT/invariants"

# Explicit-ion g(r) with orientational order at contact (main-text structure figure)
py analysis/plot_explicit_structure.py "$DATA" "$OUT"

# SDF renders (static, VMD) titled with the variant symbols
py analysis/plot_sdf_panel.py figures/schematics/sdfs.png "$OUT"

# CPPM variants: renders and Mollweide maps (Inkscape composite) with the table redrawn
py analysis/plot_cppm_table.py figures/schematics/cppm-table_MW.png "$OUT"

# SDF Mollweide maps (ESI); the voxel order of each .dx file is detected from output.yaml
mkdir -p "$OUT/sdf"
for v in isotropic dipolar quadrupolar; do
    py analysis/plot_sdf_sphere.py "$DATA/explicit/$v-exp" --projection mollweide --mode shell \
        --r-ref 25 --cmap RdBu_r --vcenter 1.0 -o "$OUT/sdf/$v-exp.png"
done
# Neutral surface groups carry no charge, so only the ions are mapped
py analysis/plot_sdf_sphere.py "$DATA/explicit/neutral-exp/sdf-cl.dx" "$DATA/explicit/neutral-exp/sdf-na.dx" \
    --projection mollweide --mode shell --r-ref 23 --cmap RdBu_r --vcenter 1.0 -o "$OUT/sdf/neutral-exp.png"
