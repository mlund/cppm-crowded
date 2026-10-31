"""Robustness of the isotropic Kirkwood–Buff match (ESI, finite size and weighting).

- G_exp(L): the KB integral does not plateau below half the box, so the cutoff matters.
- Root of ΔG and joint λ* under shorter cutoffs and the Ganguly–van der Vegt
  closed-system correction (J. Chem. Theory Comput. 9, 1347 (2013)).
- λ* when the structure term is reweighted, since the score has no unique weighting.
- ΔG(L) for representative λ: finite-size biases largely cancel in the difference,
  so what matters is whether its sign and distance from zero hold across L.

Usage: python finite_size_kb.py data
"""

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import invariants as I
import kappa_invariants as K

warnings.filterwarnings("ignore")
SIGMA, N_MACRO, BOX = 40.0, 30, 149.644
V_SIGMA = np.pi * SIGMA**3 / 6
_load, _cutoff = I.load_rdf, I.common_cutoff


def gvdv(folder):
    """Rescale g(r) for the particles a closed box withholds from the tail."""
    df = _load(folder)
    if df is None:
        return df
    r, g = df.iloc[:, 0].to_numpy(), df["g(r)"].to_numpy()
    V = BOX**3
    f = (g - 1) * r**2
    excess = (
        N_MACRO
        / V
        * 4
        * np.pi
        * np.concatenate([[0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(r))])
    )
    n_out = N_MACRO * (1 - 4 / 3 * np.pi * r**3 / V)
    df = df.copy()
    df["g(r)"] = g * n_out / (n_out - excess - 1)
    return df


def patched(loader, frac):
    I.load_rdf = loader
    I.common_cutoff = lambda fs: _cutoff(fs) * frac


def main(root):
    exp = I.explicit_folders(root / "explicit", "isotropic")
    L_max = _cutoff(exp)
    print(f"L_max = {L_max:.1f} Å")
    for L in (50, 55, 60, 65, 70, L_max):
        G = np.mean(
            [
                I.kb_integral(_load(f).iloc[:, 0].to_numpy(), _load(f)["g(r)"].to_numpy(), L)
                for f in exp
            ]
        )
        print(f"  G_exp/vσ(L={L:.1f}) = {G / V_SIGMA:.3f}")

    for name, loader in (("raw", _load), ("GvdV", gvdv)):
        for frac in (0.8, 0.9, 1.0):
            patched(loader, frac)
            res = K.analyse_system(root, "isotropic", SIGMA)
            print(
                f"  {name:4s} L×{frac}: KB root {res['stars']['KB']['lam']:.1f} Å, joint λ* {res['lam_joint']:.1f} Å"
            )
    patched(_load, 1.0)

    for sys_type in ("isotropic", "dipolar", "quadrupolar"):
        res = K.analyse_system(root, sys_type, SIGMA)
        t, grid = res["chi2_terms"], res["stars"]["KB"]["grid"]
        lams = [grid[np.argmin(t["KB"] + w * t["g(r)"])] for w in (0.25, 0.5, 1, 2, 4)]
        print(f"  {sys_type}: λ*(w = 0.25, 0.5, 1, 2, 4) =", " ".join(f"{x:.1f}" for x in lams))


def replica_G(folders, loader, Ls):
    """KB integral per replica (rows) and cutoff (columns), per macroion volume."""
    out = []
    for f in folders:
        df = loader(f)
        r, g = df.iloc[:, 0].to_numpy(), df["g(r)"].to_numpy()
        out.append([I.kb_integral(r, g, L) for L in Ls])
    return np.array(out) / V_SIGMA


def delta_G_vs_L(root, out, lams=(9.6, 14.3, 17.1, 22.6, 26.8, 37.9)):
    exp = I.explicit_folders(root / "explicit", "isotropic")
    groups = I.implicit_groups(root / "implicit", "isotropic")
    Ls = np.arange(40.0, _cutoff(exp) + 0.01, 0.5)
    kappas = sorted(groups)
    picks = [min(kappas, key=lambda k: abs(1 / k - lam)) for lam in lams]
    fig, axes = plt.subplots(1, 2, figsize=(7, 3), sharey=True, layout="constrained")
    cmap = plt.get_cmap("plasma_r")
    for ax, (name, loader) in zip(
        axes, (("Krüger weight", _load), ("+ Ganguly–van der Vegt", gvdv))
    ):
        ge = replica_G(exp, loader, Ls)
        for k in picks:
            gi = replica_G(groups[k], loader, Ls)
            d = gi.mean(0) - ge.mean(0)
            s = np.hypot(gi.std(0, ddof=1) / np.sqrt(len(gi)), ge.std(0, ddof=1) / np.sqrt(len(ge)))
            c = cmap((1 / k - 9) / 30)
            ax.plot(Ls, d, color=c, label=f"{1 / k:.1f} Å")
            ax.fill_between(Ls, d - s, d + s, color=c, alpha=0.25, lw=0)
            for L in (60, 65, 70, Ls[-1]):
                i = np.argmin(abs(Ls - L))
                print(
                    f"  {name[:6]} λ={1 / k:4.1f} L={Ls[i]:5.1f}: ΔG/vσ = {d[i]:+.3f} ± {s[i]:.3f} ({d[i] / s[i]:+.1f} SEM)"
                )
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel(r"Cutoff $L$ (Å)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(r"$\Delta G/v_\sigma$")
    axes[1].legend(title=r"$\ell_D$", fontsize=7, title_fontsize=8, frameon=False)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"delta_G_vs_L.{ext}", dpi=300)


if __name__ == "__main__":
    root = Path(sys.argv[1])
    if len(sys.argv) > 2:
        delta_G_vs_L(root, Path(sys.argv[2]))
    else:
        main(root)
