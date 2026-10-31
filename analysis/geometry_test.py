"""Test whether the long isotropic λ* reflects the Yukawa geometry rather than ionic screening.

The implicit model screens between surface charges through the macroion interior. Scaling
the bead charges by A_ex/A_sh, the Debye–Hückel amplitude of a sphere that excludes ions
from radius a_ex over that of a charged shell in electrolyte, gives pair energies with the
ion-excluded amplitude. Runs at the free-salt λ (data/geometry) are compared with the
explicit-ion reference on the same terms as the λ scan.

Usage: python geometry_test.py data
"""

import re
import sys
from pathlib import Path

import numpy as np

import invariants as I

SIGMA = 40.0
V_SIGMA = np.pi * SIGMA**3 / 6


def groups(folder, pattern):
    """Replica folders keyed by name without replica suffix."""
    out = {}
    for f in sorted(folder.iterdir()):
        m = re.fullmatch(pattern, f.name)
        if m and f.is_dir():
            out.setdefault(m.group(1), []).append(f)
    return out


def main(root):
    exp = I.explicit_folders(root / "explicit", "isotropic")
    L = I.common_cutoff(exp)
    r_grid = np.arange(SIGMA, I.load_rdf(exp[0]).iloc[:, 0].max(), 0.25)
    _, g_exp, sem_exp = I.replica_mean_sem(exp, I.load_rdf, "g(r)", x=r_grid)
    G_exp = [I.replica_invariants(f, L)["KB"] for f in exp]

    imp = groups(
        root / "implicit" / "isotropic",
        r"(isotropic-imp_(?:0\.0699|0\.0571|0\.0373|0\.1040_exp))(?:_\d)?",
    )
    geo = groups(root / "geometry", r"(isotropic-geo\d+_[0-9.]+)(?:_\d)?")
    print(f"{'runs':28s} {'ΔG/vσ':>16s} {'D/N':>6s} {'g_max':>6s}")
    for name, fs in {**imp, **geo}.items():
        G = [I.replica_invariants(f, L)["KB"] for f in fs]
        dG = (np.mean(G) - np.mean(G_exp)) / V_SIGMA
        s = (
            np.hypot(
                np.std(G, ddof=1) / np.sqrt(len(G)), np.std(G_exp, ddof=1) / np.sqrt(len(G_exp))
            )
            / V_SIGMA
        )
        _, g, sem = I.replica_mean_sem(fs, I.load_rdf, "g(r)", x=r_grid)
        dn = I.rmsd_r2(r_grid, g - g_exp) / I.rmsd_r2(r_grid, np.hypot(sem, sem_exp))
        print(f"{name:28s} {dG:+8.3f} ± {s:5.3f} {dn:6.2f} {np.nanmax(g):6.2f}")
    print(f"{'explicit':28s} {'':16s} {'':6s} {np.nanmax(g_exp):6.2f}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
