"""Explicit-ion pair structure: g(r) for all variants with orientational order beneath it.

Strips share the r/σ axis with g(r), so the reader sees that orientational order lives in the
first coordination shell. Isotropic and neutral variants have no patch orientation, so their
correlations are noise and are not drawn.

Usage: python analysis/plot_explicit_structure.py data figures
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from invariants import (
    explicit_folders,
    load_multipole,
    load_rdf,
    percus_yevick_gr,
    replica_mean_sem,
)

VARIANTS = {  # name → (label, colour); labels match the manuscript macros
    "isotropic": (r"Isotropic ($P^{0}_{0}$)", "C0"),
    "dipolar": (r"Dipolar ($P^{1}_{8}$)", "C3"),
    "quadrupolar": (r"Quadrupolar ($P^{2}_{8}$)", "C2"),
    "neutral": (r"Neutral ($P^\mathrm{n}$)", "0.45"),
}
ORIENTED = ("dipolar", "quadrupolar")
STRIPS = {  # multipole_dist.csv column → axis label
    "mucorr": r"$\langle\hat{\mu}_i\cdot\hat{\mu}_j\rangle$",
    "p2": r"$\langle P_2(\cos\theta_{ij})\rangle$",
    "quadcorr_norm": r"$\langle\hat{\mathbf{Q}}_i{:}\hat{\mathbf{Q}}_j\rangle$",
}
PHI = 0.3
X_MAX = 1.5  # r/σ; beyond this g(r) is near 1 and orientational order has decayed


def py_peak_fit(r, g, sigma):
    """PY hard spheres whose first peak matches g(r); φ scales with the effective diameter."""

    def phi(d):
        return PHI * (d / sigma) ** 3

    d = brentq(lambda d: percus_yevick_gr(r, d, phi(d)).max() - g.max(), sigma, sigma + 8)
    return d, phi(d)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("path", type=Path, help="Folder with explicit/")
    p.add_argument("plots", type=Path, help="Output folder")
    p.add_argument("--sigma", type=float, default=40.0, help="Macroion diameter σ [Å]")
    args = p.parse_args()
    sigma = args.sigma
    explicit = args.path / "explicit"

    fig, axes = plt.subplots(
        1 + len(STRIPS),
        1,
        figsize=(3.37, 4.3),  # one column of a two-column journal page
        sharex=True,
        gridspec_kw={"height_ratios": [2.2] + [1] * len(STRIPS), "hspace": 0},
    )
    for name, (label, colour) in VARIANTS.items():
        folders = explicit_folders(explicit, name)
        r, g, sem = replica_mean_sem(folders, load_rdf, "g(r)")
        m = (r >= sigma) & (r <= X_MAX * sigma)
        axes[0].plot(r[m] / sigma, g[m], color=colour, lw=1.1, label=label)
        axes[0].fill_between(
            r[m] / sigma, (g - sem)[m], (g + sem)[m], color=colour, alpha=0.2, lw=0
        )
        if name == "neutral":
            d, phi = py_peak_fit(r[m], g[m], sigma)
            rp = np.linspace(d, X_MAX * sigma, 18)
            axes[0].plot(
                rp / sigma,
                percus_yevick_gr(rp, d, phi),
                "o",
                color=colour,
                ms=3,
                mew=0.8,
                mfc="white",
                zorder=4,
                label="PY hard spheres",
            )
            print(f"PY peak fit: d_eff = {d:.1f} Å, φ_eff = {phi:.2f}")
        if name not in ORIENTED:
            continue
        for ax, col in zip(axes[1:], STRIPS):
            R, c, sc = replica_mean_sem(folders, load_multipole, col)
            k = (R >= 1.02 * sigma) & (R <= X_MAX * sigma)  # below contact the averages are empty
            ax.plot(R[k] / sigma, c[k], color=colour, lw=1.1)
            ax.fill_between(R[k] / sigma, (c - sc)[k], (c + sc)[k], color=colour, alpha=0.2, lw=0)
            i = np.argmax(np.abs(c[k]))
            print(f"{name:12s} {col:14s} contact extremum {c[k][i]:+.2f} at r = {R[k][i]:.1f} Å")
    axes[0].axhline(1, color="gray", lw=0.6, ls=":")
    axes[0].set_ylabel(r"$g(r)$", fontsize=9)
    axes[0].legend(fontsize=7, frameon=False, handlelength=1.5, labelspacing=0.3)
    for ax, lab in zip(axes[1:], STRIPS.values()):
        ax.axhline(0, color="gray", lw=0.6, ls=":")
        ax.set_ylabel(lab, fontsize=8)
        ax.yaxis.set_major_locator(plt.MaxNLocator(3, prune="upper"))
    axes[-1].set_xlim(0.99, X_MAX)
    axes[-1].set_xlabel(r"$r/\sigma$", fontsize=9)
    for ax in axes:
        ax.tick_params(labelsize=7.5)
    fig.align_ylabels(axes)
    args.plots.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(args.plots / f"explicit_structure.{ext}", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
