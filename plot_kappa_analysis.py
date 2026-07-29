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
from scipy.interpolate import interp1d
from scipy import stats as scipy_stats

parser = argparse.ArgumentParser(description='Plot kappa analysis results.')
parser.add_argument('path', help='Path to directory containing implicit/ and explicit/ subdirectories')
parser.add_argument('plots', help='Path to the output folder for plots')
args = parser.parse_args()

top_dir      = Path(args.path).resolve()
implicit_dir = top_dir / 'implicit'
explicit_dir = top_dir / 'explicit'
base_dir     = implicit_dir
plot_dir     = Path(args.plots).resolve()
plot_dir.mkdir(parents=True, exist_ok=True)

colors = {'dipolar': '#1f77b4', 'isotropic': '#2ca02c', 'neutral': '#ff7f0e', 'quadrupolar': '#d62728'}
cmaps  = {'dipolar': 'Blues',   'isotropic': 'Greens',  'neutral': 'Oranges', 'quadrupolar': 'Reds'}


# ── helpers ───────────────────────────────────────────────────────────────────

def yaml_load_faunus(path):
    loader = yaml.SafeLoader
    loader.add_multi_constructor('', lambda l, tag, node:
        l.construct_yaml_map(node) if isinstance(node, yaml.MappingNode)
        else (l.construct_sequence(node) if isinstance(node, yaml.SequenceNode)
              else l.construct_scalar(node)))
    with open(path) as f:
        return yaml.load(f, Loader=loader)


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


def load_energy(folder):
    gz  = folder / 'energy.csv.gz'
    csv = folder / 'energy.csv'
    if gz.exists():
        with gzip.open(gz, 'rt') as f:
            return pd.read_csv(f)
    elif csv.exists():
        return pd.read_csv(csv)
    return None


def load_rdf(folder):
    gz  = folder / 'rdf.csv.gz'
    csv = folder / 'rdf.csv'
    if gz.exists():
        with gzip.open(gz, 'rt') as f:
            return pd.read_csv(f)
    elif csv.exists():
        return pd.read_csv(csv)
    return None


def load_multipole(folder):
    mp = folder / 'multipole_dist.csv'
    return pd.read_csv(mp) if mp.exists() else None


def replica_mean_sem(folders, load_fn, col):
    """Return (x, mean, sem) averaging col over DataFrames loaded from folders."""
    dfs = [load_fn(f) for f in folders]
    dfs = [df for df in dfs if df is not None and col in df.columns]
    if not dfs:
        return None, None, None
    x    = dfs[0].iloc[:, 0].values
    arr  = np.array([df[col].values for df in dfs])
    mean = arr.mean(axis=0)
    sem  = arr.std(axis=0, ddof=1) / np.sqrt(len(dfs)) if len(dfs) > 1 else np.zeros_like(mean)
    return x, mean, sem


def group_by_kappa(items):
    """Return dict kappa -> [folder, ...] in sorted kappa order."""
    groups = defaultdict(list)
    for kappa, _label, folder in items:
        groups[kappa].append(folder)
    return dict(sorted(groups.items()))


# ── iteration over simulation dirs ────────────────────────────────────────────

def iter_sim_folders():
    """Yield (sys_type, kappa, replica, folder) for every implicit simulation directory."""
    pattern = re.compile(r'^(dipolar|isotropic|neutral|quadrupolar)-imp_([\d.]+?)(?:_(\d+))?$')
    for subdir in sorted(base_dir.iterdir()):
        if not subdir.is_dir():
            continue
        for folder in sorted(subdir.iterdir()):
            m = pattern.match(folder.name)
            if m:
                replica = int(m.group(3)) if m.group(3) else 1
                yield m.group(1), float(m.group(2)), replica, folder


def iter_explicit_folders(sys_type):
    """Yield all explicit replica folders for a given system type."""
    if not explicit_dir.exists():
        return
    pattern = re.compile(rf'^{re.escape(sys_type)}_(\d+)$')
    for folder in sorted(explicit_dir.iterdir()):
        if folder.is_dir() and pattern.match(folder.name):
            yield folder


# ── parse implicit pressure data ──────────────────────────────────────────────

def parse_folders():
    """Return dict: system_type -> list of (kappa, pressure_mean, pressure_err)."""
    raw = defaultdict(lambda: defaultdict(list))
    for sys_type, kappa, replica, folder in iter_sim_folders():
        yaml_path = folder / 'output.yaml'
        if not yaml_path.exists():
            continue
        doc = yaml_load_faunus(yaml_path)
        if doc is None:
            continue
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
                data[sys_type].append((kappa, np.mean(means), np.std(means, ddof=1)))

    for sys_type in data:
        data[sys_type].sort(key=lambda x: x[0])
    return data


# ── parse explicit pressure data ──────────────────────────────────────────────

def parse_explicit():
    """Return dict: sys_type -> (pex_mean, pex_sem) from all discovered explicit replicas."""
    if not explicit_dir.exists():
        print("  Warning: explicit/ directory not found.")
        return {}
    sys_types = set()
    pattern = re.compile(r'^(dipolar|isotropic|neutral|quadrupolar|bulk)_(\d+)$')
    for folder in explicit_dir.iterdir():
        m = pattern.match(folder.name)
        if m:
            sys_types.add(m.group(1))

    result = {}
    for sys_type in sorted(sys_types):
        values = []
        for folder in iter_explicit_folders(sys_type):
            yaml_path = folder / 'output.yaml'
            if not yaml_path.exists():
                continue
            doc = yaml_load_faunus(yaml_path)
            if doc is None:
                continue
            pex = find_pex(doc)
            if pex is not None and pex[0] is not None:
                values.append(float(pex[0]))
        if not values:
            continue
        arr = np.array(values)
        sem = float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
        result[sys_type] = (float(np.mean(arr)), sem)
        print(f"  Explicit {sys_type}: Pex = {result[sys_type][0]:.2f} ± {sem:.2f} mM  (n={len(arr)})")
    return result


# ── find best kappa from RDF RMSD vs explicit ─────────────────────────────────

def find_best_kappa_rdf(folders_by_type):
    """Return dicts: sys_type -> best_discrete_kappa, sys_type -> interpolated_kappa."""
    best_kappas   = {}
    interp_kappas = {}

    for sys_type, items in folders_by_type.items():
        exp_folders = list(iter_explicit_folders(sys_type))
        r_exp, g_exp, _ = replica_mean_sem(exp_folders, load_rdf, 'g(r)')
        if r_exp is None:
            continue

        groups = group_by_kappa(items)
        kappa_list, rmsd_list = [], []

        for kappa, folders in groups.items():
            r_imp, g_imp, _ = replica_mean_sem(folders, load_rdf, 'g(r)')
            if r_imp is None:
                continue
            mask    = r_imp >= 40
            r_plot  = r_imp[mask]
            g_imp_p = g_imp[mask]
            g_exp_p = np.interp(r_plot, r_exp, g_exp)
            rmsd    = np.sqrt(np.mean((g_imp_p - g_exp_p) ** 2))
            kappa_list.append(kappa)
            rmsd_list.append(rmsd)

        if not kappa_list:
            continue

        kappa_arr = np.array(kappa_list)
        rmsd_arr  = np.array(rmsd_list)
        best_idx  = int(np.argmin(rmsd_arr))
        best_kappas[sys_type] = kappa_arr[best_idx]

        try:
            f_rmsd       = interp1d(kappa_arr, rmsd_arr, kind='cubic')
            k_fine       = np.linspace(kappa_arr[0], kappa_arr[-1], 10000)
            kappa_interp = k_fine[int(np.argmin(f_rmsd(k_fine)))]
            interp_kappas[sys_type] = kappa_interp
            print(f"  {sys_type}: best κ (RDF RMSD) discrete={kappa_arr[best_idx]:.4f}  "
                  f"interpolated={kappa_interp:.4f}  RMSD={rmsd_arr[best_idx]:.5f}")
        except Exception as exc:
            print(f"  Warning: RMSD interpolation failed for {sys_type}: {exc}")

    return best_kappas, interp_kappas


# ── parse data ────────────────────────────────────────────────────────────────
print("Parsing implicit folders...")
pressure_data = parse_folders()
print("Parsing explicit folders...")
explicit_pressure = parse_explicit()

folders_by_type = defaultdict(list)
for sys_type, kappa, replica, folder in iter_sim_folders():
    label = f'κ={kappa:.4f}' if replica == 1 else f'κ={kappa:.4f} (r{replica})'
    folders_by_type[sys_type].append((kappa, label, folder))

print("\nFinding best kappa from RDF vs explicit...")
best_kappas, interp_kappas = find_best_kappa_rdf(folders_by_type)

# ── 1. Pressure vs kappa with explicit overlay and best-kappa marker ──────────
print("\nPlotting pressure vs kappa...")
fig, axes = plt.subplots(1, len(pressure_data), figsize=(5 * len(pressure_data), 4), sharey=False)
if len(pressure_data) == 1:
    axes = [axes]

for ax, sys_type in zip(axes, sorted(pressure_data)):
    rows   = pressure_data[sys_type]
    kappas = np.array([r[0] for r in rows])
    means  = np.array([r[1] for r in rows])
    errs   = np.array([r[2] for r in rows])

    ax.errorbar(kappas, means, yerr=errs, fmt='o-', color=colors.get(sys_type, 'k'),
                capsize=4, linewidth=1.5, markersize=5, label='Implicit')

    if sys_type in explicit_pressure:
        pex_exp, pex_exp_sem = explicit_pressure[sys_type]
        ax.axhline(pex_exp, color='black', linewidth=1.5, linestyle='--', label='Explicit')
        ax.axhspan(pex_exp - pex_exp_sem, pex_exp + pex_exp_sem, alpha=0.15, color='black')

    if sys_type in best_kappas:
        bk = best_kappas[sys_type]
        ax.axvline(bk, color='red', linewidth=1.5, linestyle='--',
                   label=f'Best κ={bk:.4f}')
        if sys_type in interp_kappas:
            ik = interp_kappas[sys_type]
            ax.axvline(ik, color='red', linewidth=1.0, linestyle=':', alpha=0.6,
                       label=f'Interp κ={ik:.4f}')

    ax.legend(fontsize=7)
    ax.set_xlabel(r'$\kappa$ (Å$^{-1}$)')
    ax.set_ylabel(r'$P_{ex}$ (mM)')
    ax.set_title(sys_type.capitalize())
    ax.grid(True, alpha=0.3)

fig.suptitle('Excess pressure vs implicit-salt screening', fontweight='bold')
fig.tight_layout()
fig.savefig(plot_dir / 'pressure_vs_kappa.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved pressure_vs_kappa.png")

# ── 2. Best kappa comparison: RDF + residuals per system type ─────────────────
print("\nPlotting best kappa comparisons...")
for sys_type, best_kappa in best_kappas.items():
    groups      = group_by_kappa(folders_by_type[sys_type])
    imp_folders = groups.get(best_kappa, [])
    if not imp_folders:
        print(f"  Warning: no folders for {sys_type} at κ={best_kappa:.4f}")
        continue

    r_imp, g_imp, g_imp_sem = replica_mean_sem(imp_folders, load_rdf, 'g(r)')
    r_exp, g_exp, g_exp_sem = replica_mean_sem(list(iter_explicit_folders(sys_type)), load_rdf, 'g(r)')

    if r_imp is None or r_exp is None:
        print(f"  Warning: missing RDF data for {sys_type} comparison")
        continue

    mask     = r_imp >= 40
    r_plot   = r_imp[mask]
    g_imp_p  = g_imp[mask]
    g_sem_p  = g_imp_sem[mask]
    g_exp_p  = np.interp(r_plot, r_exp, g_exp)
    g_esem_p = np.interp(r_plot, r_exp, g_exp_sem)
    residuals = g_imp_p - g_exp_p
    res_err   = np.sqrt(g_sem_p**2 + g_esem_p**2)

    color_imp = colors.get(sys_type, '#1f77b4')
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(7, 6),
        gridspec_kw={'height_ratios': [2, 1]},
        sharex=True
    )

    ax_top.plot(r_plot, g_imp_p, color=color_imp, linewidth=1.5,
                label=f'Implicit κ={best_kappa:.4f}')
    ax_top.fill_between(r_plot, g_imp_p - g_sem_p, g_imp_p + g_sem_p,
                        color=color_imp, alpha=0.3)
    ax_top.plot(r_plot, g_exp_p, color='black', linewidth=1.5, linestyle='--',
                label='Explicit')
    ax_top.fill_between(r_plot, g_exp_p - g_esem_p, g_exp_p + g_esem_p,
                        color='black', alpha=0.15)
    ax_top.axhline(1.0, color='gray', linewidth=0.6, linestyle=':')
    ax_top.set_ylabel('g(r)')
    ax_top.legend(fontsize=8)
    ax_top.grid(True, alpha=0.3)

    title = f'{sys_type.capitalize()} — best κ={best_kappa:.4f}'
    if sys_type in interp_kappas:
        title += f'  (interpolated κ={interp_kappas[sys_type]:.4f})'
    ax_top.set_title(title, fontweight='bold')

    ax_bot.plot(r_plot, residuals, color=color_imp, linewidth=1.0)
    ax_bot.fill_between(r_plot, residuals - res_err, residuals + res_err,
                        color=color_imp, alpha=0.3)
    ax_bot.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax_bot.set_xlabel('r (Å)')
    ax_bot.set_ylabel(r'$\Delta g(r)$ (impl − expl)')
    ax_bot.grid(True, alpha=0.3)

    fig.tight_layout()
    fname = f'best_kappa_comparison_{sys_type}'
    fig.savefig(plot_dir / f'{fname}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {fname}.png")

# ── 3. Pex vs Energy correlation (implicit, all kappas) ──────────────────────
print("\nPlotting Pex vs Energy...")
sys_types_sorted = sorted(pressure_data.keys())
n_sys = len(sys_types_sorted)

if n_sys > 0:
    fig_pe, axes_pe = plt.subplots(1, n_sys, figsize=(5 * n_sys, 4), squeeze=False)

    for col_idx, sys_type in enumerate(sys_types_sorted):
        ax_pe    = axes_pe[0][col_idx]
        groups   = group_by_kappa(folders_by_type.get(sys_type, []))
        pex_dict = {row[0]: (row[1], row[2]) for row in pressure_data[sys_type]}

        kappa_pts, e_pts, e_err_pts, pex_pts, pex_err_pts = [], [], [], [], []

        for kappa, folders in groups.items():
            if kappa not in pex_dict:
                continue
            rep_E = []
            for f in folders:
                df = load_energy(f)
                if df is not None and 'total' in df.columns:
                    rep_E.append(float(df['total'].mean()))
            if not rep_E:
                continue
            kappa_pts.append(kappa)
            e_pts.append(float(np.mean(rep_E)))
            e_err_pts.append(float(np.std(rep_E, ddof=1)) if len(rep_E) > 1 else 0.0)
            pex_pts.append(pex_dict[kappa][0])
            pex_err_pts.append(pex_dict[kappa][1])

        if len(e_pts) < 2:
            ax_pe.set_title(f'{sys_type.capitalize()} — insufficient data')
            continue

        e_arr   = np.array(e_pts)
        pex_arr = np.array(pex_pts)
        k_arr   = np.array(kappa_pts)

        slope, intercept, r_val, p_val, _ = scipy_stats.linregress(e_arr, pex_arr)
        r2 = r_val**2

        cmap = plt.get_cmap(cmaps.get(sys_type, 'viridis'))
        norm = plt.Normalize(vmin=k_arr.min(), vmax=k_arr.max())

        sc = ax_pe.scatter(e_arr, pex_arr, c=k_arr, cmap=cmap, norm=norm, s=50, zorder=5)
        ax_pe.errorbar(e_arr, pex_arr, xerr=e_err_pts, yerr=pex_err_pts,
                       fmt='none', color='gray', alpha=0.5, capsize=3, linewidth=0.8)

        # Highlight best kappa point
        if sys_type in best_kappas:
            bk = best_kappas[sys_type]
            if bk in pex_dict and bk in {k: None for k in kappa_pts}:
                bidx = kappa_pts.index(bk)
                ax_pe.scatter([e_pts[bidx]], [pex_pts[bidx]], s=120, facecolors='none',
                              edgecolors='red', linewidths=2, zorder=6, label=f'Best κ={bk:.4f}')
                ax_pe.legend(fontsize=8)

        x_line = np.linspace(e_arr.min(), e_arr.max(), 200)
        ax_pe.plot(x_line, slope * x_line + intercept, 'k--', linewidth=1.2,
                   label=f'R²={r2:.3f},  p={p_val:.1e}')

        ax_pe.set_xlabel('Mean energy (kT)')
        ax_pe.set_ylabel(r'$P_{ex}$ (mM)')
        ax_pe.set_title(f'{sys_type.capitalize()}')
        ax_pe.legend(fontsize=8)
        ax_pe.grid(True, alpha=0.3)

        cb = fig_pe.colorbar(sc, ax=ax_pe)
        cb.set_label(r'$\kappa$ (Å$^{-1}$)')

    fig_pe.suptitle(r'$P_{ex}$ vs mean energy — implicit simulations', fontweight='bold')
    fig_pe.tight_layout()
    fig_pe.savefig(plot_dir / 'pex_vs_energy.png', dpi=150, bbox_inches='tight')
    plt.close(fig_pe)
    print("Saved pex_vs_energy.png")

# ── 4. Energy convergence panels ─────────────────────────────────────────────
for sys_type, items in sorted(folders_by_type.items()):
    items.sort(key=lambda x: x[0])
    n     = len(items)
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

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(f'{sys_type.capitalize()} — energy convergence', fontweight='bold')
    fig.tight_layout()
    fname = f'energy_convergence_{sys_type}'
    fig.savefig(plot_dir / f'{fname}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {fname}.png")

# ── 5. RDF panels (mean ± SEM across replicates per kappa) ───────────────────
for sys_type, items in sorted(folders_by_type.items()):
    items.sort(key=lambda x: x[0])
    groups        = group_by_kappa(items)
    unique_kappas = list(groups.keys())
    n     = len(unique_kappas)
    ncols = min(5, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)

    r_exp, g_exp, _ = replica_mean_sem(list(iter_explicit_folders(sys_type)), load_rdf, 'g(r)')

    for idx, kappa in enumerate(unique_kappas):
        folders = groups[kappa]
        ax      = axes[idx // ncols][idx % ncols]
        r, g_mean, g_sem = replica_mean_sem(folders, load_rdf, 'g(r)')
        if r is not None:
            color = colors.get(sys_type, 'k')
            ax.plot(r, g_mean, linewidth=1.0, color=color)
            ax.fill_between(r, g_mean - g_sem, g_mean + g_sem, color=color, alpha=0.3)
            if r_exp is not None:
                g_exp_interp = np.interp(r, r_exp, g_exp)
                ax.plot(r, g_exp_interp, linewidth=1.0, color='black', linestyle='--', alpha=0.5)
            ax.axhline(1.0, color='gray', linewidth=0.6, linestyle='--')
            ax.set_xlim(left=40)
        ax.set_title(f'κ={kappa:.4f} (n={len(folders)})', fontsize=8)
        ax.set_xlabel('r (Å)', fontsize=7)
        ax.set_ylabel('g(r)', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(f'{sys_type.capitalize()} — radial distribution functions', fontweight='bold')
    fig.tight_layout()
    fname = f'rdf_{sys_type}'
    fig.savefig(plot_dir / f'{fname}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {fname}.png")

# ── 6. RDF overlay with colorbar per system ───────────────────────────────────
for sys_type, items in sorted(folders_by_type.items()):
    items.sort(key=lambda x: x[0])
    groups = group_by_kappa(items)
    kappas = list(groups.keys())
    kmin, kmax = kappas[0], kappas[-1]
    bk = best_kappas.get(sys_type)

    cmap = plt.get_cmap(cmaps.get(sys_type, 'viridis'))
    norm = plt.Normalize(vmin=kmin, vmax=kmax)

    fig, ax = plt.subplots(figsize=(6, 4))
    for kappa, folders in groups.items():
        r, g_mean, _ = replica_mean_sem(folders, load_rdf, 'g(r)')
        if r is not None:
            ax.plot(r, g_mean, linewidth=1.0, color=cmap(norm(kappa)))

    # Overlay explicit RDF
    r_exp, g_exp, _ = replica_mean_sem(list(iter_explicit_folders(sys_type)), load_rdf, 'g(r)')
    if r_exp is not None:
        ax.plot(r_exp, g_exp, linewidth=1.5, color='black', linestyle='--',
                alpha=0.7, label='Explicit', zorder=5)
        ax.legend(fontsize=8)

    ax.axhline(1.0, color='gray', linewidth=0.8, linestyle='--')
    ax.set_xlim(left=40)
    ax.set_xlabel('r (Å)')
    ax.set_ylabel('g(r)')
    ax.set_title(f'{sys_type.capitalize()} — RDF overlay')
    ax.grid(True, alpha=0.3)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax).set_label(r'$\kappa$ (Å$^{-1}$)')

    fig.tight_layout()
    fname = f'rdf_overlay_{sys_type}'
    fig.savefig(plot_dir / f'{fname}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {fname}.png")

# ── 7. Multipole energy decomposition ─────────────────────────────────────────
energy_cols   = ['exact', 'tot', 'ii', 'id', 'dd', 'iq']
energy_titles = ['Exact', 'Total', 'Ion-Ion', 'Ion-Dipole', 'Dipole-Dipole', 'Ion-Quadrupole']

for sys_type, items in sorted(folders_by_type.items()):
    items.sort(key=lambda x: x[0])
    groups = group_by_kappa(items)
    kappas = list(groups.keys())
    kmin, kmax = kappas[0], kappas[-1]

    cmap = plt.get_cmap(cmaps.get(sys_type, 'viridis'))
    norm = plt.Normalize(vmin=kmin, vmax=kmax)

    ymin, ymax = np.inf, -np.inf
    cache = {}
    for col in energy_cols:
        for kappa, folders in groups.items():
            r, mean, sem = replica_mean_sem(folders, load_multipole, col)
            cache[(col, kappa)] = (r, mean, sem)
            if mean is not None:
                ymin = min(ymin, (mean - sem).min())
                ymax = max(ymax, (mean + sem).max())
    if np.isinf(ymin):
        continue
    ypad = (ymax - ymin) * 0.05

    r_exp_mp, mean_exp_mp = {}, {}
    for col in energy_cols:
        r_e, m_e, _ = replica_mean_sem(list(iter_explicit_folders(sys_type)), load_multipole, col)
        r_exp_mp[col], mean_exp_mp[col] = r_e, m_e

    fig, axes = plt.subplots(2, 3, figsize=(12, 6), squeeze=False)
    for ax_idx, (col, title) in enumerate(zip(energy_cols, energy_titles)):
        ax = axes[ax_idx // 3][ax_idx % 3]
        for kappa, folders in groups.items():
            r, mean, sem = cache[(col, kappa)]
            if r is None:
                continue
            ax.plot(r, mean, linewidth=1.0, color=cmap(norm(kappa)))
            ax.fill_between(r, mean - sem, mean + sem, color=cmap(norm(kappa)), alpha=0.25)
        if r_exp_mp[col] is not None:
            ax.plot(r_exp_mp[col], mean_exp_mp[col], linewidth=1.5, color='black',
                    linestyle='--', alpha=0.7, label='Explicit' if ax_idx == 0 else '_')
            if ax_idx == 0:
                ax.legend(fontsize=7)
        ax.set_ylim(ymin - ypad, ymax + ypad)
        ax.set_xlabel('R (Å)', fontsize=8)
        ax.set_ylabel('Energy (kT)', fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'{sys_type.capitalize()} — energy decomposition (mean ± SEM)', fontweight='bold')
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    cax = fig.add_axes([0.15, 0.02, 0.7, 0.025])
    sm  = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cax, orientation='horizontal', label=r'$\kappa$ (Å$^{-1}$)')
    fname = f'multipole_energy_{sys_type}'
    fig.savefig(plot_dir / f'{fname}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {fname}.png")

# ── 8. Orientational correlations ─────────────────────────────────────────────
corr_cols   = ['mucorr', 'p2', 'quadcorr_norm', 'long']
corr_labels = [
    r'$\langle\hat{\mu}_i\cdot\hat{\mu}_j\rangle$',
    r'$P_2$ (nematic order)',
    r'$\langle\hat{\Phi}_i:\hat{\Phi}_j\rangle$',
    'Longitudinal projection',
]

for sys_type, items in sorted(folders_by_type.items()):
    items.sort(key=lambda x: x[0])
    groups = group_by_kappa(items)
    kappas = list(groups.keys())
    kmin, kmax = kappas[0], kappas[-1]

    cmap       = plt.get_cmap(cmaps.get(sys_type, 'viridis'))
    norm_kappa = plt.Normalize(vmin=kmin, vmax=kmax)

    ymin, ymax = np.inf, -np.inf
    corr_cache = {}
    for col in corr_cols:
        for kappa, folders in groups.items():
            r, mean, sem = replica_mean_sem(folders, load_multipole, col)
            corr_cache[(col, kappa)] = (r, mean, sem)
            if mean is not None:
                ymin = min(ymin, (mean - sem).min())
                ymax = max(ymax, (mean + sem).max())
    if np.isinf(ymin):
        continue
    ypad = (ymax - ymin) * 0.05

    r_exp_cr, mean_exp_cr = {}, {}
    for col in corr_cols:
        r_e, m_e, _ = replica_mean_sem(list(iter_explicit_folders(sys_type)), load_multipole, col)
        r_exp_cr[col], mean_exp_cr[col] = r_e, m_e

    fig, axes = plt.subplots(2, 2, figsize=(9, 6), squeeze=False)
    for ax_idx, (col, label) in enumerate(zip(corr_cols, corr_labels)):
        ax = axes[ax_idx // 2][ax_idx % 2]
        for kappa, folders in groups.items():
            r, mean, sem = corr_cache[(col, kappa)]
            if r is None:
                continue
            ax.plot(r, mean, linewidth=1.0, color=cmap(norm_kappa(kappa)))
            ax.fill_between(r, mean - sem, mean + sem, color=cmap(norm_kappa(kappa)), alpha=0.25)
        if r_exp_cr[col] is not None:
            ax.plot(r_exp_cr[col], mean_exp_cr[col], linewidth=1.5, color='black',
                    linestyle='--', alpha=0.7, label='Explicit' if ax_idx == 0 else '_')
            if ax_idx == 0:
                ax.legend(fontsize=7)
        ax.axhline(0, color='gray', linewidth=0.6, linestyle='--')
        ax.set_ylim(ymin - ypad, ymax + ypad)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.2f}'))
        ax.set_xlabel('R (Å)', fontsize=8)
        ax.set_ylabel('Correlation', fontsize=8)
        ax.set_title(label, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'{sys_type.capitalize()} — orientational correlations (mean ± SEM)', fontweight='bold')
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    cax = fig.add_axes([0.15, 0.02, 0.7, 0.025])
    sm  = plt.cm.ScalarMappable(cmap=cmap, norm=norm_kappa)
    sm.set_array([])
    fig.colorbar(sm, cax=cax, orientation='horizontal', label=r'$\kappa$ (Å$^{-1}$)')
    fname = f'multipole_corr_{sys_type}'
    fig.savefig(plot_dir / f'{fname}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {fname}.png")

# ── 9. Summary table: RDF first-peak statistics ───────────────────────────────
print("\n── RDF first-peak summary ──")
table_rows = []
for sys_type, items in sorted(folders_by_type.items()):
    items.sort(key=lambda x: x[0])
    for kappa, folders in group_by_kappa(items).items():
        dfs = [load_rdf(f) for f in folders]
        dfs = [df for df in dfs if df is not None and 'g(r)' in df.columns]
        if not dfs:
            continue
        peak_vals = [df['g(r)'].max() for df in dfs]
        peak_rs   = [df['r'].iloc[df['g(r)'].idxmax()] for df in dfs]
        n         = len(dfs)
        gmax_mean = np.mean(peak_vals)
        gmax_sem  = np.std(peak_vals, ddof=1) / np.sqrt(n) if n > 1 else 0.0
        rmax_mean = np.mean(peak_rs)
        rmax_sem  = np.std(peak_rs,   ddof=1) / np.sqrt(n) if n > 1 else 0.0
        bk = best_kappas.get(sys_type)
        table_rows.append({
            'type':      sys_type,
            'kappa':     kappa,
            'best':      '★' if (bk is not None and np.isclose(kappa, bk)) else '',
            'n_rep':     n,
            'g(r)_max':  f'{gmax_mean:.4f} ± {gmax_sem:.4f}',
            'R_max (Å)': f'{rmax_mean:.3f} ± {rmax_sem:.3f}',
        })

table_df = pd.DataFrame(table_rows)
print(table_df.to_string(index=False))
table_df.to_csv(plot_dir / 'rdf_peak_summary.csv', index=False)
print("\nSaved rdf_peak_summary.csv")
print("Done.")
