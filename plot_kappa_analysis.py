#!/usr/bin/env python3
import argparse
import re
import gzip
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

parser = argparse.ArgumentParser(description='Plot kappa analysis results.')
parser.add_argument('path', help='Path to the folder containing simulation directories')
parser.add_argument('plots', help='Path to the output folder for plots')
args = parser.parse_args()
base_dir = Path(args.path).resolve()
plot_dir = Path(args.plots).resolve()
plot_dir.mkdir(parents=True, exist_ok=True)


def yaml_load_faunus(path):
    """Load a Faunus output.yaml, ignoring unknown custom tags."""
    loader = yaml.SafeLoader
    # Register a catch-all constructor for any unknown tag
    loader.add_multi_constructor('', lambda l, tag, node: l.construct_yaml_map(node) if isinstance(node, yaml.MappingNode)
                                  else (l.construct_sequence(node) if isinstance(node, yaml.SequenceNode)
                                        else l.construct_scalar(node)))
    with open(path) as f:
        return yaml.load(f, Loader=loader)


def iter_sim_folders():
    """Yield (sys_type, kappa, replica, folder) for every simulation directory."""
    pattern = re.compile(r'^(dipolar|isotropic|neutral|quadrupolar)-imp_([\d.]+?)(?:_(\d+))?$')
    for subdir in sorted(base_dir.iterdir()):
        if not subdir.is_dir():
            continue
        for folder in sorted(subdir.iterdir()):
            m = pattern.match(folder.name)
            if m:
                replica = int(m.group(3)) if m.group(3) else 1
                yield m.group(1), float(m.group(2)), replica, folder


def parse_folders():
    """Return dict: system_type -> list of (kappa, pressure_mean, pressure_err).

    Kappas with multiple replicas are averaged; error is std of replica means.
    Single-replica kappas keep their own mean/error unchanged.
    """
    # raw: sys_type -> kappa -> list of (mean, err)
    raw = defaultdict(lambda: defaultdict(list))

    for sys_type, kappa, replica, folder in iter_sim_folders():

        yaml_path = folder / 'output.yaml'
        if not yaml_path.exists():
            continue
        doc = yaml_load_faunus(yaml_path)
        if doc is None:
            continue

        # Walk all nested dicts/lists to find 'Pex/mM'
        def find_pex(node):
            if isinstance(node, dict):
                for k, v in node.items():
                    if k == 'Pex/mM' and isinstance(v, dict):
                        return v.get('mean'), v.get('error')
                    result = find_pex(v)
                    if result is not None:
                        return result
            elif isinstance(node, list):
                for item in node:
                    result = find_pex(item)
                    if result is not None:
                        return result
            return None

        pex = find_pex(doc)
        if pex is None:
            print(f"  Warning: no Pex/mM found in {folder.name}")
            continue
        raw[sys_type][kappa].append((pex[0], pex[1]))

    data = defaultdict(list)
    for sys_type, kappa_map in raw.items():
        for kappa, replicas in kappa_map.items():
            if len(replicas) == 1:
                data[sys_type].append((kappa, replicas[0][0], replicas[0][1]))
            else:
                means = np.array([r[0] for r in replicas])
                avg = np.mean(means)
                std = np.std(means, ddof=1)
                data[sys_type].append((kappa, avg, std))

    # Sort by kappa
    for sys_type in data:
        data[sys_type].sort(key=lambda x: x[0])
    return data


def load_energy(folder):
    gz = folder / 'energy.csv.gz'
    csv = folder / 'energy.csv'
    if gz.exists():
        with gzip.open(gz, 'rt') as f:
            return pd.read_csv(f)
    elif csv.exists():
        return pd.read_csv(csv)
    return None


def load_rdf(folder):
    rdf_path = folder / 'rdf.csv'
    if rdf_path.exists():
        return pd.read_csv(rdf_path)
    return None


# ── 1. Pressure vs kappa ──────────────────────────────────────────────────────
print("Parsing folders...")
pressure_data = parse_folders()

fig, axes = plt.subplots(1, len(pressure_data), figsize=(5 * len(pressure_data), 4), sharey=False)
if len(pressure_data) == 1:
    axes = [axes]

colors = {'dipolar': '#1f77b4', 'isotropic': '#2ca02c', 'neutral': '#ff7f0e', 'quadrupolar': '#d62728'}

for ax, sys_type in zip(axes, sorted(pressure_data)):
    rows = pressure_data[sys_type]
    kappas = [r[0] for r in rows]
    means  = [r[1] for r in rows]
    errs   = [r[2] for r in rows]
    ax.errorbar(kappas, means, yerr=errs, fmt='o-', color=colors.get(sys_type, 'k'),
                capsize=4, linewidth=1.5, markersize=5)
    ax.set_xlabel(r'$\kappa$ (Å$^{-1}$)')
    ax.set_ylabel(r'$P_{ex}$ (mM)')
    ax.set_title(sys_type.capitalize())
    ax.grid(True, alpha=0.3)

fig.suptitle('Excess pressure vs implicit-salt screening', fontweight='bold')
fig.tight_layout()
fig.savefig(plot_dir / 'pressure_vs_kappa.pdf', bbox_inches='tight')
fig.savefig(plot_dir / 'pressure_vs_kappa.png', dpi=150, bbox_inches='tight')
print("Saved pressure_vs_kappa.pdf/.png")

# ── 2. Energy convergence panels ─────────────────────────────────────────────
folders_by_type = defaultdict(list)
for sys_type, kappa, replica, folder in iter_sim_folders():
    label = f'κ={kappa:.4f}' if replica == 1 else f'κ={kappa:.4f} (r{replica})'
    folders_by_type[sys_type].append((kappa, label, folder))

for sys_type, items in sorted(folders_by_type.items()):
    items.sort(key=lambda x: x[0])
    n = len(items)
    ncols = min(5, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)

    for idx, (kappa, label, folder) in enumerate(items):
        ax = axes[idx // ncols][idx % ncols]
        df = load_energy(folder)
        if df is not None and 'total' in df.columns:
            ax.plot(df['step'], df['total'], linewidth=0.8, color=colors.get(sys_type, 'k'))
        ax.set_title(label, fontsize=8)
        ax.set_xlabel('MC step', fontsize=7)
        ax.set_ylabel('E (kT)', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(f'{sys_type.capitalize()} — energy convergence', fontweight='bold')
    fig.tight_layout()
    fname = f'energy_convergence_{sys_type}'
    fig.savefig(plot_dir / f'{fname}.pdf', bbox_inches='tight')
    fig.savefig(plot_dir / f'{fname}.png', dpi=150, bbox_inches='tight')
    print(f"Saved {fname}.pdf/.png")

# ── 3. RDF panels ────────────────────────────────────────────────────────────
for sys_type, items in sorted(folders_by_type.items()):
    items.sort(key=lambda x: x[0])
    n = len(items)
    ncols = min(5, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)

    for idx, (kappa, label, folder) in enumerate(items):
        ax = axes[idx // ncols][idx % ncols]
        df = load_rdf(folder)
        if df is not None:
            ax.plot(df['r'], df['g(r)'], linewidth=1.0, color=colors.get(sys_type, 'k'))
            ax.axhline(1.0, color='gray', linewidth=0.6, linestyle='--')
            ax.set_xlim(left=40)
        ax.set_title(label, fontsize=8)
        ax.set_xlabel('r (Å)', fontsize=7)
        ax.set_ylabel('g(r)', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(f'{sys_type.capitalize()} — radial distribution functions', fontweight='bold')
    fig.tight_layout()
    fname = f'rdf_{sys_type}'
    fig.savefig(plot_dir / f'{fname}.pdf', bbox_inches='tight')
    fig.savefig(plot_dir / f'{fname}.png', dpi=150, bbox_inches='tight')
    print(f"Saved {fname}.pdf/.png")

# ── 4. RDF overlay with colorbar per system ──────────────────────────────────
cmaps = {'dipolar': 'Blues', 'isotropic': 'Greens', 'neutral': 'Oranges', 'quadrupolar': 'Reds'}

for sys_type, items in sorted(folders_by_type.items()):
    items.sort(key=lambda x: x[0])
    kappas = [k for k, _, _ in items]
    kmin, kmax = kappas[0], kappas[-1]

    cmap = plt.get_cmap(cmaps.get(sys_type, 'viridis'))
    norm = plt.Normalize(vmin=kmin, vmax=kmax)

    fig, ax = plt.subplots(figsize=(6, 4))

    for kappa, label, folder in items:
        df = load_rdf(folder)
        if df is not None:
            color = cmap(norm(kappa))
            ax.plot(df['r'], df['g(r)'], linewidth=1.0, color=color)

    ax.axhline(1.0, color='gray', linewidth=0.8, linestyle='--')
    ax.set_xlim(left=40)
    ax.set_xlabel('r (Å)')
    ax.set_ylabel('g(r)')
    ax.set_title(f'{sys_type.capitalize()} — RDF overlay')
    ax.grid(True, alpha=0.3)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(r'$\kappa$ (Å$^{-1}$)')

    fig.tight_layout()
    fname = f'rdf_overlay_{sys_type}'
    fig.savefig(plot_dir / f'{fname}.pdf', bbox_inches='tight')
    fig.savefig(plot_dir / f'{fname}.png', dpi=150, bbox_inches='tight')
    print(f"Saved {fname}.pdf/.png")

print("Done.")
