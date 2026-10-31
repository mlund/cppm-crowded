"""Find the implicit-ion Debye length that reproduces explicit-ion structure and thermodynamics.

For each CPPM variant and each invariant in `invariants.py`, ΔX(λ) = X_imp(λ) − X_exp is
modelled with a Gaussian process over the Debye length λ = 1/κ; λ* is its zero. The matching
score S combines the Kirkwood–Buff integral with the shell-weighted g(r) residual, each scaled
by its own statistical uncertainty. The orientational invariants are left out of S: P00 has none and P1/P2 only some, so they would weigh the variants unequally.
They are still reported, as a check of λ*. Non-overlapping λ* intervals mean
a single screened-Coulomb κ cannot represent the explicit-ion system.

Usage: python analysis/kappa_invariants.py data figures
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA

from invariants import (
    debye_length,
    delta_table,
    explicit_folders,
    gp_curve,
    implicit_groups,
    kappa_star,
    kb_integral,
    load_rdf,
    molarity,
    replica_mean_sem,
    rmsd_r2,
)

SYSTEMS = ("isotropic", "dipolar", "quadrupolar")
SYMBOLS = {"isotropic": r"$P^{0}_{0}$", "dipolar": r"$P^{1}_{8}$", "quadrupolar": r"$P^{2}_{8}$"}
# Orientational invariants only where the variant has the corresponding patch symmetry
INVARIANTS = {"isotropic": ["KB"], "dipolar": ["KB", "mu", "Q"], "quadrupolar": ["KB", "Q"]}
FIT_KEYS = ("KB",)  # invariants that enter the joint χ², alongside the g(r) residual
INV_LABELS = {
    "KB": r"$\Delta G/v_\sigma$",
    "mu": r"$\Delta X_\mu/v_\sigma$",
    "Q": r"$\Delta X_Q/v_\sigma$",
}
INV_TITLES = {"KB": "Kirkwood–Buff", "mu": "Dipolar correlation", "Q": "Quadrupolar correlation"}
CMAP = ListedColormap(plt.get_cmap("plasma")(np.linspace(0.9, 0.15, 256)))
DEBYE_LABEL = r"Debye length, $1/\kappa$ [Å]"
PHI = 0.3  # macroion volume fraction
LAMBDA_BULK = 1 / 0.104  # Debye–Hückel at the 100 mM reservoir; the κ of the *_exp runs
LAMBDA_CYTOSOL = 22.0  # E. coli cytosol estimate, Wennerström et al., PNAS 117, 10113 (2020)
TERMS = {  # χ² term → (legend label, colour)
    "g(r)": (r"Structure, $g(r)$", "C0"),
    "KB": ("Kirkwood–Buff", "C3"),
    "mu": (r"Dipolar, $X_\mu$", "C2"),
    "Q": (r"Quadrupolar, $X_Q$", "C1"),
}


def save(fig, out, stem, layout=True):
    if layout:
        fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def analyse_system(root, sys_type, sigma):
    groups = implicit_groups(root / "implicit", sys_type)
    exp = explicit_folders(root / "explicit", sys_type)
    keys = INVARIANTS[sys_type]
    v_sigma = np.pi * sigma**3 / 6

    df = delta_table(groups, exp, keys)
    for k in keys:  # dimensionless, per macroion volume
        df[[f"d{k}", f"s{k}", f"x{k}", f"e{k}"]] /= v_sigma
    x_exp = {k: tuple(np.array(v) / v_sigma) for k, v in df.attrs["exp"].items()}
    stars = {k: kappa_star(df["debye"], df[f"d{k}"], df[f"e{k}"], x_exp[k][1]) for k in keys}
    grid = stars["KB"]["grid"]

    r_grid = np.arange(sigma, load_rdf(exp[0]).iloc[:, 0].max(), 0.25)
    _, g_exp, sem_exp = replica_mean_sem(exp, load_rdf, "g(r)", x=r_grid)
    stats = [replica_mean_sem(fs, load_rdf, "g(r)", x=r_grid)[1:] for fs in groups.values()]
    ns = [len(fs) for fs in groups.values()]
    # Per-run SD pooled over replicated κ stands in for single-replica SEM (cf. delta_table)
    sd_g = np.sqrt(np.mean([sem**2 * n for (_, sem), n in zip(stats, ns) if n > 1], axis=0))
    g_imp, z_struct = [], []
    for (g, sem), n in zip(stats, ns):
        sem = sd_g if n == 1 else sem
        g_imp.append(g)
        # Structure residual in units of its own noise floor, so it enters χ² like the others
        z_struct.append(rmsd_r2(r_grid, g - g_exp) / rmsd_r2(r_grid, np.hypot(sem, sem_exp)))

    # Joint χ² on the GP grid; every term smoothed by a GP so none is tied to the κ nodes.
    # KB: latent mean² / (latent sd² + data SEM²). Structure: z² from its own GP.
    by_lam = df.sort_values("debye")  # np.interp needs increasing x; df is in κ order
    chi2_terms = {
        k: st["mu"] ** 2 / (st["sd"] ** 2 + np.interp(grid, by_lam["debye"], by_lam[f"s{k}"] ** 2))
        for k, st in stars.items()
        if k in FIT_KEYS
    }
    _, z_mu, _, _ = gp_curve(df["debye"], z_struct, np.ones(len(z_struct)), grid)
    chi2_terms["g(r)"] = z_mu**2
    chi2 = sum(chi2_terms.values())

    na, cl = (np.mean([molarity(f, a) for f in exp]) for a in ("Na", "Cl"))
    return {
        # Free salt per box volume vs. per volume left by the macroions bracket the free-salt λ
        "lam_free_salt": (debye_length(cl / (1 - PHI)), debye_length(cl)),
        # Linear response (Chung & Denton, PRE 88, 022306 (2013)): all small ions screen,
        # counted in the free volume; ionic strength (c_Na + c_Cl)/2
        "lam_linear": debye_length((na + cl) / 2 / (1 - PHI)),
        "df": df,
        "x_exp": x_exp,
        "stars": stars,
        "chi2_terms": chi2_terms,
        "chi2": chi2,
        # Unsmoothed χ² at the scanned κ, to show the GP curve follows the data
        "chi2_raw": sum(df[f"d{k}"] ** 2 / df[f"s{k}"] ** 2 for k in FIT_KEYS).to_numpy()
        + np.array(z_struct) ** 2,
        "lam_joint": grid[np.argmin(chi2)],
        "r_grid": r_grid,
        "g_exp": g_exp,
        "g_imp": np.array(g_imp),
        "z_struct": np.array(z_struct),
        "exp": exp,
        "groups": groups,
    }


def running_dG(res, sigma):
    """ΔG(L')/v_σ vs L' at the discrete κ closest to the KB λ*; should plateau at large L'."""
    df, lam = res["df"], res["stars"]["KB"]["lam"]
    i = int(np.argmin(np.abs(df["debye"] - lam)))
    imp = res["groups"][df["kappa"][i]]
    rdfs = {f: load_rdf(f) for f in imp + res["exp"]}
    L_max = min(d.iloc[:, 0].max() for d in rdfs.values())
    Ls = np.linspace(1.2 * sigma, L_max, 60)

    def G(folders, Lp):
        return np.mean(
            [kb_integral(rdfs[f].iloc[:, 0].values, rdfs[f]["g(r)"].values, Lp) for f in folders]
        )

    v_sigma = np.pi * sigma**3 / 6
    dG = [(G(imp, Lp) - G(res["exp"], Lp)) / v_sigma for Lp in Ls]
    return Ls / sigma, np.array(dG), df["debye"][i]


def plot_invariants(results, out):
    ncol = max(len(v) for v in INVARIANTS.values())
    fig, axes = plt.subplots(
        len(SYSTEMS), ncol, figsize=(3.3 * ncol, 2.6 * len(SYSTEMS)), squeeze=False
    )
    for i, s in enumerate(SYSTEMS):
        res = results[s]
        df = res["df"]
        for j in range(ncol):
            ax = axes[i][j]
            if j >= len(INVARIANTS[s]):
                ax.set_visible(False)
                continue
            k = INVARIANTS[s][j]
            st = res["stars"][k]
            ax.axhline(0, color="gray", lw=0.6, ls="--")
            ax.fill_between(
                st["grid"],
                st["mu"] - st["sd_ref"],
                st["mu"] + st["sd_ref"],
                color="C0",
                alpha=0.2,
                lw=0,
            )
            ax.plot(st["grid"], st["mu"], color="C0", lw=1.2, label="GP mean ± 1 sd")
            ax.errorbar(
                df["debye"],
                df[f"d{k}"],
                yerr=df[f"s{k}"],
                fmt="o",
                ms=3,
                color="black",
                capsize=2,
                lw=0.8,
                label="Replica mean ± SEM",
            )
            if st["bracketed"]:
                ax.axvspan(st["lo"], st["hi"], color="C3", alpha=0.15, lw=0)
                ax.axvline(st["lam"], color="C3", lw=1.2, label=r"$\lambda^*$ (68%)")
            ax.set_title(f"{s.capitalize()} ({SYMBOLS[s]}): {INV_TITLES[k]}", fontsize=9)
            ax.set_ylabel(INV_LABELS[k], fontsize=9)
            ax.tick_params(labelsize=8)
            if i == len(SYSTEMS) - 1 or j >= len(INVARIANTS[SYSTEMS[i + 1]]):
                ax.set_xlabel(DEBYE_LABEL, fontsize=9)
    axes[0][0].legend(fontsize=7, frameon=False)
    save(fig, out, "kappa_star_invariants")


def scan_max(res):
    return res["df"]["debye"].max()


def at_scan_edge(res):
    """Joint minimum on the last grid points: only a lower bound on λ*."""
    return res["lam_joint"] >= scan_max(res) - 0.02 * np.ptp(res["df"]["debye"])


def mark_references(ax, res):
    lo, hi = res["lam_free_salt"]
    ax.axvspan(lo, hi, color="0.85", lw=0, zorder=0, label="Free salt (counterions excluded)")
    ax.axvline(LAMBDA_BULK, color="0.4", lw=1, ls="--", zorder=0, label="Reservoir salt")
    ax.axvline(LAMBDA_CYTOSOL, color="C2", lw=1, ls="-.", zorder=0, label="Cytosol (ref.)")


def panel_label(ax, s, y=0.93):
    ax.text(
        0.95,
        y,
        f"{s.capitalize()} ({SYMBOLS[s]})",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
    )


def plot_chi2(results, out):
    fig, axes = plt.subplots(1, len(SYSTEMS), figsize=(10, 3.3), sharex=True, sharey=True)
    for ax, s in zip(axes, SYSTEMS):
        res = results[s]
        grid = res["stars"]["KB"]["grid"]
        mark_references(ax, res)
        # Beyond the scan the GP has no data; hatch it so the cut-off is not read as a minimum
        ax.axvspan(scan_max(res), 40, facecolor="none", edgecolor="0.75", hatch="///", lw=0)
        for k, term in res["chi2_terms"].items():
            label, colour = TERMS[k]
            ax.plot(grid, term, lw=1, color=colour, label=label)
        ax.plot(grid, res["chi2"], color="black", lw=1.8, label=r"Score $S$")
        ax.axhline(len(res["chi2_terms"]), color="black", lw=0.8, ls=":", label="Number of terms")
        i = np.argmin(res["chi2"])
        ax.plot(grid[i], res["chi2"][i], "o", color="black", ms=5, zorder=5)
        ax.set_yscale("log")
        ax.set_xlabel(DEBYE_LABEL, fontsize=11)
        ax.tick_params(labelsize=10)
        panel_label(ax, s)
    axes[0].set_xlim(8, 39)
    axes[0].set_ylim(1e-1, 1e3)
    axes[0].set_ylabel(r"$S$", fontsize=11)
    # One legend for all: the dipolar panel carries every term
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, -0.24),
    )
    save(fig, out, "kappa_joint_chi2")


def plot_gr_at_star(results, sigma, out):
    """Explicit g(r) vs implicit at the scan κ nearest the joint λ* and at the reservoir λ."""
    norm = lambda_norm(results)
    fig, axes = plt.subplots(
        2,
        len(SYSTEMS),
        figsize=(10, 4.4),
        sharex=True,
        sharey="row",
        gridspec_kw={"height_ratios": [2.2, 1]},
    )
    for j, s in enumerate(SYSTEMS):
        res = results[s]
        r, x = res["r_grid"], res["r_grid"] / sigma
        _, g_exp, sem_exp = replica_mean_sem(res["exp"], load_rdf, "g(r)", x=r)
        top, bot = axes[0][j], axes[1][j]
        kappas = np.array(list(res["groups"]))
        picks = [
            (kappas[np.argmin(np.abs(1 / kappas - res["lam_joint"]))], None),
            (kappas[np.argmin(np.abs(1 / kappas - LAMBDA_BULK))], "0.55"),
        ]
        top.fill_between(x, g_exp - sem_exp, g_exp + sem_exp, color="black", alpha=0.15, lw=0)
        top.plot(x, g_exp, color="black", lw=1.2, label="Explicit ions", zorder=3)
        for kappa, colour in picks:
            _, g, sem = replica_mean_sem(res["groups"][kappa], load_rdf, "g(r)", x=r)
            colour = colour or CMAP(norm(1 / kappa))
            label = f"$1/\\kappa$ = {1 / kappa:.1f} Å"
            top.plot(x, g, color=colour, lw=1.2, label=label)
            dg, sdg = g - g_exp, np.hypot(sem, sem_exp)
            bot.fill_between(x, dg - sdg, dg + sdg, color=colour, alpha=0.25, lw=0)
            bot.plot(x, dg, color=colour, lw=1)
        for ax in (top, bot):
            ax.axhline(1.0 if ax is top else 0.0, color="gray", lw=0.6, ls="--", zorder=0)
            ax.tick_params(labelsize=10)
        panel_label(top, s)
        top.legend(fontsize=8.5, frameon=False, loc="center right", bbox_to_anchor=(1, 0.6))
        bot.set_xlabel(r"$r/\sigma$", fontsize=11)
    axes[0][0].set_xlim(1.0 - 0.01, 1.5)
    axes[0][0].xaxis.set_major_locator(plt.MultipleLocator(0.1))
    axes[0][0].set_ylabel(r"$g(r)$", fontsize=11)
    axes[1][0].set_ylabel(r"$\Delta g(r)$", fontsize=11)
    save(fig, out, "gr_at_lambda_star")


def lambda_norm(results):
    """Colour scale shared with rdf_overlay_1x3: Debye length, from ≤10 Å to the longest run."""
    debyes = [d for res in results.values() for d in res["df"]["debye"]]
    return plt.Normalize(min(10.0, min(debyes)), max(debyes))


def plot_fit_observables(results, sigma, out):
    """What the joint χ² compares: g(r) (top), its Kirkwood–Buff integral vs λ, the g(r)
    distance D/N vs λ (≈1 when implicit and explicit agree within noise), and the joint χ²
    (GP-smoothed curve; raw values at the scanned κ as markers).

    Implicit runs share one λ colour scale; the explicit reference is black. The integral uses
    the same finite-size weight as the fit, so the points are exactly what χ² compares.
    """
    norm = lambda_norm(results)
    fig = plt.figure(figsize=(10, 6.8))
    outer = fig.add_gridspec(
        2, 1, height_ratios=[1.4, 1.5], hspace=0.22, left=0.09, right=0.87, top=0.98, bottom=0.08
    )
    top = outer[0].subgridspec(1, len(SYSTEMS), wspace=0.28)
    bot = outer[1].subgridspec(3, len(SYSTEMS), wspace=0.28, hspace=0)
    axes = np.array(
        [[fig.add_subplot(top[0, j]) for j in range(len(SYSTEMS))]]
        + [[fig.add_subplot(bot[i, j]) for j in range(len(SYSTEMS))] for i in range(3)]
    )
    z_max = 1.1 * max(res["z_struct"].max() for res in results.values())
    # One λ axis for all variants, so scan ranges and λ* compare by eye
    lam_lo = min(res["df"]["debye"].min() for res in results.values()) - 0.6
    lam_hi = max(res["df"]["debye"].max() for res in results.values()) + 0.6
    n_sub = 16  # colour steps between adjacent κ curves, so the band reads as continuous
    for j, s in enumerate(SYSTEMS):
        res, df = results[s], results[s]["df"]
        x = res["r_grid"] / sigma
        m = x <= 1.5  # beyond 1.5σ g(r) ≈ 1 and carries no contrast
        ax = axes[0][j]
        ax.sharey(axes[0][0])
        lams = df["debye"].to_numpy()
        for (la, ga), (lb, gb) in zip(zip(lams, res["g_imp"]), zip(lams[1:], res["g_imp"][1:])):
            for i in range(n_sub):
                s0, s1 = i / n_sub, (i + 1) / n_sub
                ka = 1 / la + (s0 + s1) / 2 * (1 / lb - 1 / la)
                ax.fill_between(
                    x[m],
                    (ga + s0 * (gb - ga))[m],
                    (ga + s1 * (gb - ga))[m],
                    color=CMAP(norm(1 / ka)),
                    lw=0,
                )
        ax.plot(x[m], res["g_exp"][m], color="black", lw=1.2, label="Explicit ions", zorder=3)
        ax.axhline(1.0, color="gray", lw=0.6, ls="--", zorder=0)
        ax.set_xlim(0.99, 1.5)
        ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
        ax.set_xlabel(r"$r/\sigma$", fontsize=10)
        ax.tick_params(labelsize=9)
        panel_label(ax, s)

        kb, dist, chi = axes[1][j], axes[2][j], axes[3][j]
        dist.sharex(kb)
        chi.sharex(kb)
        if j == 2:  # dipolar and quadrupolar G span the same few hundredths
            kb.sharey(axes[1][1])
        if j:
            dist.sharey(axes[2][0])
            chi.sharey(axes[3][0])
        me, se = res["x_exp"]["KB"]
        kb.axhspan(me - se, me + se, color="black", alpha=0.15, lw=0)
        kb.axhline(me, color="black", lw=1.2)
        kb.errorbar(lams, df["xKB"], yerr=df["eKB"], fmt="none", ecolor="0.5", lw=0.8)
        kb.scatter(
            lams, df["xKB"], c=CMAP(norm(lams)), s=22, edgecolor="black", linewidth=0.4, zorder=3
        )
        dist.axhline(1, color="black", lw=0.8, ls="--", zorder=0)
        dist.scatter(
            lams,
            res["z_struct"],
            c=CMAP(norm(lams)),
            marker="s",
            s=18,
            edgecolor="black",
            linewidth=0.4,
            zorder=3,
        )
        grid = res["stars"]["KB"]["grid"]
        g = (grid >= lams.min()) & (grid <= lams.max())  # no GP extrapolation beyond the scan
        chi.plot(grid[g], res["chi2"][g], color="black", lw=1.2, zorder=2)
        chi.scatter(
            lams,
            res["chi2_raw"],
            c=CMAP(norm(lams)),
            marker="D",
            s=14,
            edgecolor="black",
            linewidth=0.4,
            zorder=3,
        )
        chi.set_yscale("log")
        lam = res["lam_joint"]
        for ax in (kb, dist, chi):
            ax.tick_params(labelsize=9)
        kb.set_xlim(lam_lo, lam_hi)
        kb.tick_params(labelbottom=False)
        kb.yaxis.set_major_locator(plt.MaxNLocator(4, prune="lower"))
        for ax in (kb, dist, chi):
            ax.axvline(lam, color="black", lw=0.8, ls=":", zorder=0)
        # Label on the side with room: left of the line when it sits in the right half
        right = lam > np.mean(kb.get_xlim())
        kb.annotate(
            rf"$\ell_D^* = {lam:.1f}$ Å",
            xy=(lam, 1),
            xycoords=("data", "axes fraction"),
            xytext=(-4 if right else 4, -4),
            textcoords="offset points",
            ha="right" if right else "left",
            va="top",
            fontsize=9,
        )
        dist.set_ylim(0, z_max)
        dist.tick_params(labelbottom=False)
        dist.yaxis.set_major_locator(plt.MaxNLocator(4, prune="lower"))
        chi.set_xlabel(DEBYE_LABEL, fontsize=10)
    axes[0][0].set_ylabel(r"$g(r)$", fontsize=11)
    axes[1][0].set_ylabel(r"$G/v_\sigma$", fontsize=10)
    axes[2][0].set_ylabel(r"$D/N$", fontsize=10)
    axes[3][0].set_ylabel(r"$S$", fontsize=10)
    axes[0][0].legend(fontsize=9, frameon=False, loc="lower right")
    fig.align_ylabels(axes[:, 0])
    cax = fig.add_axes([0.89, 0.08, 0.015, 0.9])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=CMAP, norm=norm), cax=cax)
    cb.set_label(DEBYE_LABEL, fontsize=11)
    cb.ax.yaxis.set_major_locator(plt.MultipleLocator(5))
    save(fig, out, "fit_observables", layout=False)


def plot_summary(results, out):
    """λ* per variant against the reference screening lengths."""
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    for y, s in enumerate(SYSTEMS):
        res = results[s]
        lo, hi = res["lam_free_salt"]
        ax.fill_between(
            [lo, hi],
            y - 0.3,
            y + 0.3,
            color="0.85",
            lw=0,
            zorder=0,
            label="Free salt (counterions excluded)" if y == 0 else None,
        )
        ax.plot(
            [res["df"]["debye"].min(), scan_max(res)],
            [y - 0.22] * 2,
            color="0.6",
            lw=3,
            solid_capstyle="butt",
            label="Simulated range" if y == 0 else None,
        )
        kb = res["stars"]["KB"]
        if kb["bracketed"]:
            ax.errorbar(
                kb["lam"],
                y + 0.12,
                xerr=[[kb["lam"] - kb["lo"]], [kb["hi"] - kb["lam"]]],
                fmt="o",
                mfc="white",
                mec="C3",
                ecolor="C3",
                ms=6,
                capsize=3,
                label="Kirkwood–Buff only" if y == 0 else None,
            )
        lam = res["lam_joint"]
        ax.plot(
            lam,
            y,
            "o",
            color="black",
            ms=7,
            zorder=4,
            label=r"Score $S$ minimum" if y == 0 else None,
        )
        if at_scan_edge(res):  # minimum not reached: λ* is at least this
            ax.annotate(
                "",
                xy=(lam + 4, y),
                xytext=(lam + 0.6, y),
                arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.2},
            )
        ax.plot(
            res["lam_linear"],
            y,
            "|",
            color="C0",
            ms=14,
            mew=2,
            label="Linear response, all ions" if y == 0 else None,
        )
    ax.axvline(LAMBDA_BULK, color="0.4", lw=1, ls="--", label="Reservoir salt")
    ax.axvline(LAMBDA_CYTOSOL, color="C2", lw=1, ls="-.", label="Cytosol (ref.)")
    ax.set_yticks(range(len(SYSTEMS)), [f"{s.capitalize()}\n{SYMBOLS[s]}" for s in SYSTEMS])
    ax.set_ylim(-0.6, len(SYSTEMS) - 0.4)
    ax.invert_yaxis()
    ax.set_xlim(6, 39)
    ax.set_xlabel(DEBYE_LABEL, fontsize=11)
    ax.legend(fontsize=8, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    save(fig, out, "lambda_star_summary")


def plot_running(results, sigma, out):
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for n, s in enumerate(SYSTEMS):
        x, dG, lam = running_dG(results[s], sigma)
        ax.plot(x, dG, color=f"C{n}", label=f"{SYMBOLS[s]}, $1/\\kappa$ = {lam:.1f} Å")
    ax.axhline(0, color="gray", lw=0.6, ls="--")
    ax.set_xlabel(r"Truncation $L/\sigma$")
    ax.set_ylabel(r"$\Delta G(L)/v_\sigma$")
    ax.legend(fontsize=8, frameon=False)
    save(fig, out, "kb_running")


def pca_check(results, out):
    """Does explicit g(r) lie in the span of the implicit family? Residual in noise units."""
    fig, axes = plt.subplots(1, len(SYSTEMS), figsize=(10, 3.2), squeeze=False)
    summary = {}
    for ax, s in zip(axes[0], SYSTEMS):
        res = results[s]
        G = res["g_imp"]
        pca = PCA(n_components=2).fit(G)
        z_imp = pca.transform(G)
        z_exp = pca.transform(res["g_exp"][None, :])
        recon = pca.inverse_transform(z_exp)[0]
        r = res["r_grid"]
        imp_resid = np.median(
            [rmsd_r2(r, g - pca.inverse_transform(pca.transform(g[None, :]))[0]) for g in G]
        )
        summary[s] = {
            "pca_var_explained": pca.explained_variance_ratio_.sum(),
            "exp_residual_over_imp": rmsd_r2(r, res["g_exp"] - recon) / imp_resid,
        }
        sc = ax.scatter(z_imp[:, 0], z_imp[:, 1], c=res["df"]["debye"], cmap=CMAP, s=18)
        ax.scatter(z_exp[0, 0], z_exp[0, 1], marker="*", s=120, color="black", label="Explicit")
        ax.set_title(f"{s.capitalize()} ({SYMBOLS[s]})", fontsize=10)
        ax.set_xlabel("PC1")
        fig.colorbar(sc, ax=ax, label=r"$1/\kappa$ [Å]")
    axes[0][0].set_ylabel("PC2")
    axes[0][0].legend(fontsize=8, frameon=False)
    save(fig, out, "gr_pca")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("path", type=Path, help="Folder with implicit/ and explicit/")
    p.add_argument("plots", type=Path, help="Output folder")
    p.add_argument("--sigma", type=float, default=40.0, help="Macroion diameter σ [Å]")
    args = p.parse_args()
    args.plots.mkdir(parents=True, exist_ok=True)

    results = {s: analyse_system(args.path, s, args.sigma) for s in SYSTEMS}
    plot_invariants(results, args.plots)
    plot_chi2(results, args.plots)
    plot_gr_at_star(results, args.sigma, args.plots)
    plot_fit_observables(results, args.sigma, args.plots)
    plot_summary(results, args.plots)
    plot_running(results, args.sigma, args.plots)
    pca = pca_check(results, args.plots)

    rows = []
    for s, res in results.items():
        for k, st in res["stars"].items():
            ok = st["bracketed"]
            df = res["df"]
            lam, lo, hi = (st["lam"], st["lo"], st["hi"]) if ok else (np.nan,) * 3
            # Unbracketed: data consistent with zero (undetermined) vs. root outside scan
            if ok:
                note = ""
            elif (np.abs(df[f"d{k}"]) < 2 * df[f"s{k}"]).all():
                note = "undetermined: |dX| < 2 SEM at every kappa"
            else:
                note = f"outside scan: GP mean {'>' if st['mu'].mean() > 0 else '<'} 0 throughout"
            rows.append(
                {
                    "system": s,
                    "invariant": k,
                    "debye_star": lam,
                    "debye_lo": lo,
                    "debye_hi": hi,
                    "kappa_star": 1 / lam,
                    "frac_samples_bracketed": st["frac_bracketed"],
                    "scan_debye_min": df["debye"].min(),
                    "scan_debye_max": df["debye"].max(),
                    "note": note,
                }
            )
        rows.append(
            {
                "system": s,
                "invariant": "joint",
                "debye_star": res["lam_joint"],
                "kappa_star": 1 / res["lam_joint"],
                "chi2_min": res["chi2"].min(),
                "n_terms": len(res["chi2_terms"]),
                **pca[s],
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(args.plots / "kappa_star_invariants.csv", index=False, float_format="%.4g")
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(table.to_string(index=False, float_format=lambda v: f"{v:.3g}"))


if __name__ == "__main__":
    main()
