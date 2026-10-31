#!/usr/bin/env python3
import argparse
import gzip
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

parser = argparse.ArgumentParser(description="Compare multiple single simulation folders.")
parser.add_argument(
    "--sim", required=True, help="Comma-separated list of simulation folders to compare"
)
parser.add_argument("--plots", required=True, help="Output folder for plots")
args = parser.parse_args()

sim_dirs = [Path(p.strip()).resolve() for p in args.sim.split(",")]
plot_dir = Path(args.plots).resolve()
plot_dir.mkdir(parents=True, exist_ok=True)

labels = [d.name for d in sim_dirs]
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


# ── helpers ───────────────────────────────────────────────────────────────────


def yaml_load_faunus(path):
    loader = yaml.SafeLoader
    loader.add_multi_constructor(
        "",
        lambda l, tag, node: (
            l.construct_yaml_map(node)
            if isinstance(node, yaml.MappingNode)
            else (
                l.construct_sequence(node)
                if isinstance(node, yaml.SequenceNode)
                else l.construct_scalar(node)
            )
        ),
    )
    with open(path) as f:
        return yaml.load(f, Loader=loader)


def find_pex(node):
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "Pex/mM" and isinstance(v, dict):
                return v.get("mean"), v.get("error")
            result = find_pex(v)
            if result is not None:
                return result
    elif isinstance(node, list):
        for item in node:
            result = find_pex(item)
            if result is not None:
                return result
    return None


def load_csv(folder, stem):
    gz = folder / f"{stem}.csv.gz"
    csv = folder / f"{stem}.csv"
    if gz.exists():
        with gzip.open(gz, "rt") as f:
            return pd.read_csv(f)
    elif csv.exists():
        return pd.read_csv(csv)
    return None


# ── load all data ─────────────────────────────────────────────────────────────

pex_data = {}  # label -> (mean, err)
energy_dfs = {}  # label -> df
rdf_dfs = {}  # label -> df
mp_dfs = {}  # label -> df

for d, lab in zip(sim_dirs, labels):
    yaml_path = d / "output.yaml"
    if yaml_path.exists():
        doc = yaml_load_faunus(yaml_path)
        if doc:
            pex = find_pex(doc)
            if pex and pex[0] is not None:
                pex_data[lab] = (float(pex[0]), float(pex[1]) if pex[1] is not None else 0.0)
                print(f"  {lab}: Pex = {pex_data[lab][0]:.2f} ± {pex_data[lab][1]:.2f} mM")
            else:
                print(f"  Warning: no Pex/mM found in {lab}")
        else:
            print(f"  Warning: could not parse output.yaml in {lab}")
    else:
        print(f"  Warning: no output.yaml in {lab}")

    df = load_csv(d, "energy")
    if df is not None:
        energy_dfs[lab] = df

    df = load_csv(d, "rdf")
    if df is not None:
        rdf_dfs[lab] = df

    df = d / "multipole_dist.csv"
    if df.exists():
        mp_dfs[lab] = pd.read_csv(df)


# ── 1. Excess pressure comparison ─────────────────────────────────────────────
if pex_data:
    fig, ax = plt.subplots(figsize=(max(4, len(pex_data) * 1.2), 4))
    xs = np.arange(len(pex_data))
    labs = list(pex_data.keys())
    means = [pex_data[l][0] for l in labs]
    errs = [pex_data[l][1] for l in labs]
    bars = ax.bar(
        xs,
        means,
        yerr=errs,
        capsize=5,
        color=colors[: len(labs)],
        alpha=0.8,
        edgecolor="black",
        linewidth=0.8,
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(labs, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(r"$P_{ex}$ (mM)")
    ax.set_title("Excess pressure comparison", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(plot_dir / "pex_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved pex_comparison.png")


# ── 2. RDF overlay ────────────────────────────────────────────────────────────
if rdf_dfs:
    fig, ax = plt.subplots(figsize=(7, 4))
    for (lab, df), col in zip(rdf_dfs.items(), colors):
        r_col = df.columns[0]
        if "g(r)" in df.columns:
            ax.plot(df[r_col], df["g(r)"], linewidth=1.5, label=lab, color=col)
    ax.axhline(1.0, color="gray", linewidth=0.6, linestyle="--")
    ax.set_xlim(left=40)
    ax.set_xlabel("r (Å)")
    ax.set_ylabel("g(r)")
    ax.set_title("Radial distribution function comparison", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "rdf_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved rdf_comparison.png")

    # RDF residuals relative to first simulation
    if len(rdf_dfs) > 1:
        ref_lab = labels[0]
        ref_df = rdf_dfs.get(ref_lab)
        if ref_df is not None and "g(r)" in ref_df.columns:
            r_ref = ref_df.iloc[:, 0].values
            g_ref = ref_df["g(r)"].values
            fig, ax = plt.subplots(figsize=(7, 4))
            for (lab, df), col in zip(list(rdf_dfs.items())[1:], colors[1:]):
                r_col = df.columns[0]
                if "g(r)" in df.columns:
                    g_interp = np.interp(r_ref, df[r_col].values, df["g(r)"].values)
                    ax.plot(
                        r_ref,
                        g_interp - g_ref,
                        linewidth=1.5,
                        label=f"{lab} − {ref_lab}",
                        color=col,
                    )
            ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
            ax.set_xlim(left=40)
            ax.set_xlabel("r (Å)")
            ax.set_ylabel(r"$\Delta g(r)$")
            ax.set_title(f"RDF residuals vs {ref_lab}", fontweight="bold")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(plot_dir / "rdf_residuals.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            print("Saved rdf_residuals.png")


# ── 3. Energy convergence overlay ─────────────────────────────────────────────
if energy_dfs:
    fig, ax = plt.subplots(figsize=(8, 4))
    for (lab, df), col in zip(energy_dfs.items(), colors):
        if "total" in df.columns:
            step_col = df.columns[0]
            ax.plot(df[step_col], df["total"], linewidth=0.8, label=lab, color=col, alpha=0.85)
    ax.set_xlabel("MC step")
    ax.set_ylabel("E (kT)")
    ax.set_title("Energy convergence comparison", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "energy_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved energy_comparison.png")


# ── 4. Multipole energy decomposition overlay ─────────────────────────────────
energy_cols = ["exact", "tot", "ii", "id", "dd", "iq"]
energy_titles = ["Exact", "Total", "Ion-Ion", "Ion-Dipole", "Dipole-Dipole", "Ion-Quadrupole"]

if mp_dfs:
    available_cols = [c for c in energy_cols if any(c in df.columns for df in mp_dfs.values())]
    if available_cols:
        n_cols = len(available_cols)
        ncols = min(3, n_cols)
        nrows = (n_cols + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

        for ax_idx, col in enumerate(available_cols):
            ax = axes[ax_idx // ncols][ax_idx % ncols]
            title = energy_titles[energy_cols.index(col)] if col in energy_cols else col
            for (lab, df), c in zip(mp_dfs.items(), colors):
                if col in df.columns:
                    r_col = df.columns[0]
                    ax.plot(df[r_col], df[col], linewidth=1.2, label=lab, color=c)
            ax.set_xlabel("R (Å)", fontsize=8)
            ax.set_ylabel("Energy (kT)", fontsize=8)
            ax.set_title(title, fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)

        for idx in range(n_cols, nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        fig.suptitle("Multipole energy decomposition comparison", fontweight="bold")
        fig.tight_layout()
        fig.savefig(plot_dir / "multipole_energy_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved multipole_energy_comparison.png")


# ── 5. Orientational correlations overlay ─────────────────────────────────────
corr_cols = ["mucorr", "p2", "quadcorr_norm", "long"]
corr_labels = [
    r"$\langle\hat{\mu}_i\cdot\hat{\mu}_j\rangle$",
    r"$P_2$ (nematic order)",
    r"$\langle\hat{\Phi}_i:\hat{\Phi}_j\rangle$",
    "Longitudinal projection",
]

if mp_dfs:
    avail_corr = [c for c in corr_cols if any(c in df.columns for df in mp_dfs.values())]
    if avail_corr:
        ncols = min(2, len(avail_corr))
        nrows = (len(avail_corr) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)

        for ax_idx, col in enumerate(avail_corr):
            ax = axes[ax_idx // ncols][ax_idx % ncols]
            clabel = corr_labels[corr_cols.index(col)] if col in corr_cols else col
            for (lab, df), c in zip(mp_dfs.items(), colors):
                if col in df.columns:
                    r_col = df.columns[0]
                    ax.plot(df[r_col], df[col], linewidth=1.2, label=lab, color=c)
            ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
            ax.set_xlabel("R (Å)", fontsize=8)
            ax.set_ylabel("Correlation", fontsize=8)
            ax.set_title(clabel, fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))

        for idx in range(len(avail_corr), nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        fig.suptitle("Orientational correlations comparison", fontweight="bold")
        fig.tight_layout()
        fig.savefig(plot_dir / "multipole_corr_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved multipole_corr_comparison.png")


print("Done.")
