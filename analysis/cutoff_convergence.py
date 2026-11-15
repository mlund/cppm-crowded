"""Cutoff test: screened Fanourgakis vs Yukawa Ewald for the implicit-ion P18 runs.

Frames sampled with the production Hamiltonian H₀ = Fanourgakis(L/2) are reweighted to each
test Hamiltonian H₁ with w ∝ exp(−β[U₁ − U₀]), so the averages are what MC with H₁ would give.
Constant terms (self energy, k = 0 at fixed V) drop out of the weights.

`prepare` condenses a run tree from simulation/cutoff_test/ into data/cutoff/; `plot` reads
data/cutoff/ only.
"""

import argparse
import re
import shutil
from math import comb
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from invariants import implicit_groups, kb_integral, replica_mean_sem, rmsd_r2

KT = 8.314462618e-3 * 298.15  # kJ/mol; energy.csv is in kJ/mol
BJERRUM = 7.15  # Å
SIGMA = 40.0  # macroion diameter, Å
V_SIGMA = np.pi * SIGMA**3 / 6
DR = 1.0  # g(r) bin width; reweighted and reference g share frames, so noise mostly cancels
N_BLOCKS = 20  # block bootstrap over consecutive frames
MIN_N_EFF = 0.1  # below this effective-sample fraction, reweighted averages are not trusted
REFERENCE = "ewald_40_1e-06"
PRODUCTION = f"fan_{149.64407707272974 / 2:.1f}"
CHARGE = {"MP": 0.0, "NP": -1.0, "PP": 1.0, "UP": 0.0}


# ---------------------------------------------------------------- prepare
def read_xyz(path):
    """Per-frame macroion centre and dipole, from a Faunus xyz trajectory."""
    lines = Path(path).read_text().splitlines()
    n = int(lines[0])
    n_frames = len(lines) // (n + 2)
    names = [lines[2 + i].split()[0] for i in range(n)]
    xyz = np.array(
        [
            [list(map(float, lines[f * (n + 2) + 2 + i].split()[1:4])) for i in range(n)]
            for f in range(n_frames)
        ]
    )
    return names, xyz


def centres_and_dipoles(names, xyz, box, per_molecule):
    q = np.array([CHARGE[a] for a in names])
    n_mol = len(names) // per_molecule
    xyz = xyz.reshape(len(xyz), n_mol, per_molecule, 3)
    q = q.reshape(n_mol, per_molecule)
    mp = [i for i, a in enumerate(names[:per_molecule]) if a == "MP"][0]
    centre = xyz[:, :, mp, :]
    d = xyz - centre[:, :, None, :]
    d -= box * np.round(d / box)
    dipole = np.einsum("mi,fmij->fmj", q, d)
    return centre, dipole


def prepare(tree, out, box):
    for root in sorted(Path(tree).glob("P18_*")):
        dest = out / root.name
        dest.mkdir(parents=True, exist_ok=True)
        names, xyz = read_xyz(root / "traj/traj.xyz")
        per_mol = len(names) // 30
        centre, dipole = centres_and_dipoles(names, xyz, box, per_mol)
        np.savez_compressed(dest / "frames.npz", centre=centre, dipole=dipole, box=box)
        for sub in ("traj", *(f"rerun/{p.name}" for p in (root / "rerun").iterdir())):
            for f in ("input.yml", "output.yaml", "energy.csv", "pressure.csv"):
                src = root / sub / f
                if src.exists():
                    (dest / sub).mkdir(parents=True, exist_ok=True)
                    shutil.copy(src, dest / sub / f)
        speed = []
        for p in sorted((root / "speed").iterdir()):
            if not (p / "output.yaml").exists():
                print(f"  no timing for {p.name}")
                continue
            # Plain regex: output.yaml carries custom tags, and only two numbers are needed
            doc = (p / "output.yaml").read_text()
            log = (p / "run.log").read_text()
            kvec = re.findall(r"k-vectors=(\d+)", log)
            alpha = re.findall(r"α=([0-9.]+)", log)
            speed.append(
                {
                    "label": p.name,
                    "steps": int(re.search(r"^\s*steps:\s*(\d+)", doc, re.M).group(1)),
                    "seconds": float(re.search(r"elapsed_seconds:\s*([0-9.eE+-]+)", doc).group(1)),
                    "k_vectors": int(kvec[-1]) if kvec else 0,
                    "alpha": float(alpha[-1]) if alpha else np.nan,
                }
            )
        pd.DataFrame(speed).to_csv(dest / "speed.csv", index=False)
        print(f"{root.name}: {len(centre)} frames -> {dest}")


# ---------------------------------------------------------------- physics
def fanourgakis_s(r, rc, kappa, C=4, D=3):
    """Screened Poisson short-range function, as coulomb::Poisson<4, 3>::short_range_f0."""
    rk = rc * kappa
    qp = (1 - np.exp(2 * rk * r / rc)) / (1 - np.exp(2 * rk))
    s = sum(comb(D - 1 + c, c) * (C - c) / C * qp**c for c in range(C))
    return np.where(r < rc, (1 - qp) ** (D + 1) * s, 0.0)


def mean_field_pressure_offset(kappa, rc, n, z, volume):
    """βP_ex(Yukawa) − βP_ex(Fanourgakis) from the net-charge mean field, Q²∫u/(2V)."""
    r = np.linspace(1e-6, rc, 20001)
    missing = 4 * np.pi / kappa**2 - 4 * np.pi * np.trapezoid(
        r * np.exp(-kappa * r) * fanourgakis_s(r, rc, kappa), r
    )
    u = BJERRUM * (n * z) ** 2 * missing / (2 * volume)  # kT
    return u / volume  # −∂U/∂V with U ∝ 1/V


# ---------------------------------------------------------------- analysis
def energies(folder, label):
    return pd.read_csv(folder / f"rerun/{label}/energy.csv")["total"].to_numpy() / KT


def pressure_du(folder, label):
    return pd.read_csv(folder / f"rerun/{label}/pressure.csv")["dU/kT"].to_numpy()


def log_weights(folder, label):
    du = energies(folder, label) - energies(folder, PRODUCTION)
    return -(du - du.mean())


def pair_histograms(centre, dipole, box, edges):
    """Per-frame pair counts and Σ μ̂ᵢ·μ̂ⱼ per bin."""
    n = centre.shape[1]
    i, j = np.triu_indices(n, 1)
    d = centre[:, i] - centre[:, j]
    d -= box * np.round(d / box)
    r = np.linalg.norm(d, axis=-1)
    u = dipole / np.linalg.norm(dipole, axis=-1, keepdims=True)
    cos = np.einsum("fpk,fpk->fp", u[:, i], u[:, j])
    counts = np.array([np.histogram(rf, edges)[0] for rf in r])
    dip = np.array([np.histogram(rf, edges, weights=cf)[0] for rf, cf in zip(r, cos)])
    return counts, dip


def weighted_structure(counts, dip, logw, edges, box, n):
    w = np.exp(logw - logw.max())
    w /= w.sum()
    c = w @ counts
    shell = 4 * np.pi / 3 * (edges[1:] ** 3 - edges[:-1] ** 3)
    g = c / (n * (n - 1) / 2 * shell / box**3)
    mu = np.divide(w @ dip, c, out=np.full_like(c, np.nan), where=c > 0)
    return g, mu


def block_indices(n_frames, rng):
    blocks = np.array_split(np.arange(n_frames), N_BLOCKS)
    return np.concatenate([blocks[k] for k in rng.integers(0, N_BLOCKS, N_BLOCKS)])


def analyse_kappa(folder, implicit_dir, n_boot=200):
    kappa = float(folder.name.split("_")[1])
    frames = np.load(folder / "frames.npz")
    centre, dipole, box = frames["centre"], frames["dipole"], float(frames["box"])
    n = centre.shape[1]
    half = box / 2
    edges = np.arange(SIGMA * 0.75, half + DR, DR)
    r = 0.5 * (edges[1:] + edges[:-1])
    counts, dip = pair_histograms(centre, dipole, box, edges)
    labels = sorted(p.name for p in (folder / "rerun").iterdir())
    logw = {lab: log_weights(folder, lab) for lab in labels}
    n_frames = len(centre)
    assert all(len(v) == n_frames for v in logw.values()), "rerun and xyz frame counts differ"

    # Production noise at this κ: replica SEM of g(r) and of G, the scale the paper resolves
    reps = implicit_groups(implicit_dir, "dipolar")[kappa]
    _, _, sem_g = replica_mean_sem(reps, lambda f: pd.read_csv(f / "rdf.csv"), "g(r)", x=r)
    g_reps = [
        kb_integral(r, np.interp(r, *pd.read_csv(f / "rdf.csv").iloc[:, :2].T.values), half)
        for f in reps
    ]
    sem_G = np.std(g_reps, ddof=1) / np.sqrt(len(g_reps)) / V_SIGMA

    def observables(idx):
        out = {}
        for lab in labels:
            g, mu = weighted_structure(counts[idx], dip[idx], logw[lab][idx], edges, box, n)
            du_v = pressure_du(folder, lab)[idx]
            lw = logw[lab][idx]
            dV = 0.1
            p = np.log(np.sum(np.exp(lw - lw.max() - du_v)) / np.sum(np.exp(lw - lw.max()))) / dV
            out[lab] = {"g": g, "mu": mu, "G": kb_integral(r, g, half) / V_SIGMA, "P": p}
        return out

    full = observables(np.arange(n_frames))
    rng = np.random.default_rng(0)
    boot = [observables(block_indices(n_frames, rng)) for _ in range(n_boot)]
    ref = full[REFERENCE]
    rows = []
    for lab in labels:
        lw = logw[lab]
        w = np.exp(lw - lw.max())
        dG_boot = [b[lab]["G"] - b[REFERENCE]["G"] for b in boot]
        dP_boot = [b[lab]["P"] - b[REFERENCE]["P"] for b in boot]
        dg = full[lab]["g"] - ref["g"]
        # Noise of Δg within these frames, to tell a real difference from reweighting noise
        dg_noise = np.std([b[lab]["g"] - b[REFERENCE]["g"] for b in boot], axis=0)
        n_eff = w.sum() ** 2 / (w**2).sum() / n_frames
        rows.append(
            {
                "kappa": kappa,
                "label": lab,
                "scheme": lab.split("_")[0],
                "rc": float(lab.split("_")[1]),
                "sigma_beta_dU": np.std(-lw),  # vs the sampling Hamiltonian H₀
                "sigma_beta_dU_ref": np.std(lw - logw[REFERENCE]),  # vs Ewald
                "n_eff_frac": n_eff,
                "reliable": n_eff >= MIN_N_EFF,
                "dG": full[lab]["G"] - ref["G"],
                "dG_err": np.std(dG_boot),
                "dG_over_sem": (full[lab]["G"] - ref["G"]) / sem_G,
                "sem_G": sem_G,
                "D_over_N": rmsd_r2(r, dg) / rmsd_r2(r, sem_g),
                "D_over_N_self": (
                    rmsd_r2(r, dg) / rmsd_r2(r, dg_noise) if lab != REFERENCE else 0.0
                ),
                # Smallest D/N these frames can resolve: reweighting noise in production SEM
                "resolution": rmsd_r2(r, dg_noise) / rmsd_r2(r, sem_g),
                "max_abs_dg": np.nanmax(np.abs(dg)),
                "max_abs_dmu": np.nanmax(np.abs(full[lab]["mu"] - ref["mu"])[r < half - 5]),
                "P": full[lab]["P"],
                "dP": full[lab]["P"] - ref["P"],
                "dP_err": np.std(dP_boot),
                "dP_meanfield": (
                    -mean_field_pressure_offset(kappa, float(lab.split("_")[1]), n, 8, box**3)
                    if lab.startswith("fan")
                    else 0.0
                ),
            }
        )
    return pd.DataFrame(rows), r, full


def _style(ax, d, x, y, color, label=None, yerr=None):
    """Line through all points; hollow markers where reweighting is unreliable."""
    ax.plot(x, y, "-", color=color, lw=1, label=label)
    if yerr is not None:
        ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor=color, elinewidth=0.8, capsize=2)
    ok = d.reliable.to_numpy()
    ax.plot(x[ok], y[ok], "o", color=color, ms=4)
    ax.plot(x[~ok], y[~ok], "o", mfc="white", mec=color, ms=4)


def plot(table, speed, out):
    out.mkdir(parents=True, exist_ok=True)
    colors = dict(zip(sorted(table.kappa.unique()), ["#1f77b4", "#ff7f0e", "#2ca02c"]))
    fan = table[table.scheme == "fan"].sort_values("rc")

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), layout="constrained")
    for kappa, d in fan.groupby("kappa"):
        c, rc = colors[kappa], d.rc.to_numpy()
        _style(axes[0], d, rc, d.D_over_N.to_numpy(), c, rf"$\lambda_D$ = {1 / kappa:.1f} Å")
        axes[0].plot(rc, d.resolution, ":", color=c, lw=1)
        _style(axes[1], d, rc, d.dG_over_sem.to_numpy(), c, yerr=(d.dG_err / d.sem_G).to_numpy())
        _style(axes[2], d, rc, d.sigma_beta_dU_ref.to_numpy(), c)
    axes[0].axhline(1, ls="--", c="k", lw=0.8)
    axes[1].axhspan(-1, 1, color="0.92", zorder=0)
    axes[0].set(ylabel="D/N vs Ewald (dotted: resolution)", yscale="log")
    axes[1].set(ylabel=r"$\Delta G$ / SEM$_\mathrm{prod}$", ylim=(-35, 5))
    axes[2].set(ylabel=r"$\sigma(\beta U_\mathrm{F} - \beta U_\mathrm{Ewald})$", yscale="log")
    for ax, tag in zip(axes, "abc"):
        ax.set_xlabel(r"Fanourgakis cutoff $r_c$ [Å]")
        ax.set_title(f"({tag})", loc="left", fontsize=10)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.savefig(out / "cutoff_convergence.png", dpi=200)
    fig.savefig(out / "cutoff_convergence.pdf")

    ew = table[table.scheme.isin(["ewaldopt", "ewaldfit"])]
    order = sorted(ew.label.unique())
    fig, ax = plt.subplots(figsize=(5, 3.2), layout="constrained")
    for i, (kappa, d) in enumerate(ew.groupby("kappa")):
        d = d.set_index("label").loc[order]
        x = np.arange(len(order)) + (i - 1) * 0.2
        ok, y, c = d.reliable.to_numpy(), d.D_over_N.to_numpy(), colors[kappa]
        ax.plot(x[ok], y[ok], "o", color=c, ms=4, label=rf"{1 / kappa:.1f} Å")
        ax.plot(x[~ok], y[~ok], "o", mfc="white", mec=c, ms=4)
    ax.axhline(1, ls="--", c="k", lw=0.8)
    ax.set_xticks(range(len(order)), order, rotation=45, ha="right", fontsize=8)
    ax.set(ylabel="D/N vs Ewald reference", yscale="log")
    ax.legend(frameon=False, fontsize=8, title=r"$\lambda_D$")
    fig.savefig(out / "cutoff_ewald.png", dpi=200)
    fig.savefig(out / "cutoff_ewald.pdf")

    fig, ax = plt.subplots(figsize=(4, 3.2), layout="constrained")
    for kappa, d in fan.groupby("kappa"):
        c = colors[kappa]
        ax.errorbar(
            d.rc,
            d.dP * 1e4,
            yerr=d.dP_err * 1e4,
            fmt="o",
            color=c,
            ms=4,
            label=rf"{1 / kappa:.1f} Å",
        )
        ax.plot(d.rc, d.dP_meanfield * 1e4, "--", color=c, lw=0.8)
    ax.set(
        xlabel=r"Fanourgakis cutoff $r_c$ [Å]",
        ylabel=r"$\beta P_\mathrm{ex}^\mathrm{F} - \beta P_\mathrm{ex}^\mathrm{Ewald}$ [$10^{-4}$ Å$^{-3}$]",
    )
    ax.legend(frameon=False, fontsize=8, title=r"$\lambda_D$ (dashed: mean field)")
    fig.savefig(out / "cutoff_pressure.png", dpi=200)
    fig.savefig(out / "cutoff_pressure.pdf")

    if speed.empty or PRODUCTION not in set(speed.label):
        return
    s = speed.assign(per_step=speed.seconds / speed.steps)
    base = s[s.label == PRODUCTION].set_index("kappa").per_step
    order = sorted(s.label.unique())
    fig, ax = plt.subplots(figsize=(5, 3.2), layout="constrained")
    for i, (kappa, d) in enumerate(s.groupby("kappa")):
        d = d.set_index("label").reindex(order)
        x = np.arange(len(order)) + (i - 1) * 0.2
        ax.plot(
            x, d.per_step / base[kappa], "o", color=colors[kappa], ms=4, label=rf"{1 / kappa:.1f} Å"
        )
    ax.axhline(1, ls="--", c="k", lw=0.8)
    ax.set_xticks(range(len(order)), order, rotation=60, ha="right", fontsize=7)
    ax.set(ylabel=r"MC step time / Fanourgakis($L/2$)", yscale="log")
    ax.legend(frameon=False, fontsize=8, title=r"$\lambda_D$")
    fig.savefig(out / "cutoff_speed.png", dpi=200)
    fig.savefig(out / "cutoff_speed.pdf")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("prepare")
    pp.add_argument("tree", type=Path)
    pp.add_argument("data", type=Path)
    pp.add_argument("--box", type=float, default=149.64407707272974)
    pl = sub.add_parser("plot")
    pl.add_argument("data", type=Path, help="data/ (holds cutoff/ and implicit/)")
    pl.add_argument("out", type=Path)
    args = p.parse_args()
    if args.cmd == "prepare":
        prepare(args.tree, args.data, args.box)
        return
    tables, speeds = [], []
    for folder in sorted((args.data / "cutoff").glob("P18_*")):
        t, _, _ = analyse_kappa(folder, args.data / "implicit")
        tables.append(t)
        speed = (
            pd.read_csv(folder / "speed.csv") if (folder / "speed.csv").stat().st_size > 1 else None
        )
        if speed is not None and not speed.empty:
            speeds.append(speed.assign(kappa=t.kappa.iloc[0]))
    table = pd.concat(tables)
    speed = pd.concat(speeds) if speeds else pd.DataFrame(columns=["label", "steps", "seconds"])
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "cutoff_convergence.csv", index=False, float_format="%.4g")
    speed.to_csv(args.out / "cutoff_speed.csv", index=False, float_format="%.4g")
    with pd.option_context("display.width", 200):
        print(table.drop(columns=["scheme"]).to_string(index=False, float_format="%.3g"))
        print(speed.to_string(index=False, float_format="%.3g"))
    plot(table, speed, args.out)


if __name__ == "__main__":
    main()
