"""Salt susceptibility: variance of the Cl⁻ count vs. the ideal Donnan value (ESI table).

For ideal partitioning, ⟨δN_s²⟩ = N₊N₋/(N₊ + N₋); the ratio tests how close the
macroion solution is to that limit.

Usage: python salt_susceptibility.py data/fluctuations
"""

import sys
from pathlib import Path

import numpy as np

from autocorr import read_gz


def main(root):
    for variant in ("bulk", "isotropic", "dipolar", "quadrupolar", "neutral"):
        var, ratio = [], []
        for d in sorted(root.glob(f"{variant}-exp*")):
            na = read_gz(d / "n_na.csv.gz")["value"].to_numpy()
            cl = read_gz(d / "n_cl.csv.gz")["value"].to_numpy()
            var.append(cl.var())
            ratio.append(cl.var() / (na.mean() * cl.mean() / (na.mean() + cl.mean())))
        var = np.array(var)
        sem = var.std(ddof=1) / np.sqrt(len(var))
        print(f"{variant:12s} ⟨δN_s²⟩ = {var.mean():.1f} ± {sem:.1f}  ratio = {np.mean(ratio):.2f}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
