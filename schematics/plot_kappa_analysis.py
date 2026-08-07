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
parser.add_argument('--no-titles', action='store_true', help='Suppress all plot titles and suptitles')
args = parser.parse_args()


def ax_title(ax, text, **kw):
    if not args.no_titles:
        ax.set_title(text, **kw)


def fig_title(fig, text, **kw):
    if not args.no_titles:
        fig.suptitle(text, **kw)

top_dir      = Path(args.path).resolve()
implicit_dir = top_dir / 'implicit'
explicit_dir = top_dir / 'explicit'
base_dir     = implicit_dir
plot_dir     = Path(args.plots).resolve()
plot_dir.mkdir(parents=True, exist_ok=True)

plt.rcParams['font.family'] = 'sans-serif'

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
    arr  = np.array([np.interp(x, df.iloc[:, 0].values, df[col].values) for df in dfs])
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
    pattern = re.compile(rf'^{re.escape(sys_type)}-exp(?:_(\d+))?$')
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
    pattern = re.compile(r'^(dipolar|isotropic|neutral|quadrupolar|bulk)-exp(?:_(\d+))?$')
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
    ax.set_xlabel(r'$\kappa$ [Å$^{-1}$]')
    ax.set_ylabel(r'$P_{ex}$ [mM]')
    ax_title(ax, sys_type.capitalize())
    ax.grid(True, alpha=0.3)

fig_title(fig, 'Excess pressure vs implicit-salt screening', fontweight='bold')
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
    ax_title(ax_top, title, fontweight='bold')

    ax_bot.plot(r_plot, residuals, color=color_imp, linewidth=1.0)
    ax_bot.fill_between(r_plot, residuals - res_err, residuals + res_err,
                        color=color_imp, alpha=0.3)
    ax_bot.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax_bot.set_xlabel('r [Å]')
    ax_bot.set_ylabel(r'$\Delta g(r)$ [impl − expl]')
    ax_bot.grid(True, alpha=0.3)

    fig.tight_layout()
    fname = f'best_kappa_comparison_{sys_type}'
    fig.savefig(plot_dir / f'{fname}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {fname}.png")

# ── 2b. Combined best-kappa comparison — all systems overlaid ─────────────────
print("\nPlotting combined best-kappa comparison...")
_bk_styles = {'isotropic': ':', 'dipolar': '--', 'quadrupolar': '-', 'neutral': '-.'}
_bk_order  = [t for t in ('isotropic', 'dipolar', 'quadrupolar', 'neutral') if t in best_kappas]

if _bk_order:
    _bfig, (_bax_t, _bax_b) = plt.subplots(
        2, 1, figsize=(6, 7),
        gridspec_kw={'height_ratios': [2, 1]},
        sharex=True
    )

    for _btype in _bk_order:
        _bkappa     = best_kappas[_btype]
        _bgroups    = group_by_kappa(folders_by_type[_btype])
        _bimp_fldrs = _bgroups.get(_bkappa, [])
        if not _bimp_fldrs:
            continue

        _br_imp, _bg_imp, _bg_imp_sem = replica_mean_sem(_bimp_fldrs, load_rdf, 'g(r)')
        _br_exp, _bg_exp, _bg_exp_sem = replica_mean_sem(
            list(iter_explicit_folders(_btype)), load_rdf, 'g(r)')
        if _br_imp is None or _br_exp is None:
            continue

        _bmask    = (_br_imp >= 40) & (_br_imp <= 60)
        _br       = _br_imp[_bmask]
        _bg_imp_p = _bg_imp[_bmask]
        _bg_sem_p = _bg_imp_sem[_bmask]
        _bg_exp_p = np.interp(_br, _br_exp, _bg_exp)
        _bg_esem  = np.interp(_br, _br_exp, _bg_exp_sem)
        _bres     = _bg_imp_p - _bg_exp_p
        _bres_err = np.sqrt(_bg_sem_p**2 + _bg_esem**2)

        _bcol = colors.get(_btype, '#1f77b4')
        _bls  = _bk_styles.get(_btype, '-')

        _bax_t.plot(_br, _bg_imp_p, color=_bcol, linewidth=1.5, linestyle=_bls,
                    label=_btype.capitalize())
        _bax_t.fill_between(_br, _bg_imp_p - _bg_sem_p, _bg_imp_p + _bg_sem_p,
                            color=_bcol, alpha=0.3)
        _bax_t.plot(_br, _bg_exp_p, color='black', linewidth=1.5, linestyle=_bls)
        _bax_t.fill_between(_br, _bg_exp_p - _bg_esem, _bg_exp_p + _bg_esem,
                            color='black', alpha=0.15)

        _bax_b.plot(_br, _bres, color=_bcol, linewidth=1.0, linestyle=_bls)
        _bax_b.fill_between(_br, _bres - _bres_err, _bres + _bres_err,
                            color=_bcol, alpha=0.3)

    _bax_t.axhline(1.0, color='gray', linewidth=0.6, linestyle=':')
    _bax_t.set_ylabel('g(r)', fontsize=11)
    _bax_t.set_xlim(40 - 0.4, 60 + 0.4)
    _bax_t.legend(fontsize=9)

    _bax_b.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    _bax_b.set_xlabel('r [Å]', fontsize=11)
    _bax_b.set_ylabel(r'$\Delta g(r)$ (imp-exp)', fontsize=11)

    fig_title(_bfig, 'Best-κ RDF comparison — all systems', fontweight='bold')
    _bfig.tight_layout()
    _bfig.savefig(plot_dir / 'best_kappa_comparison_combined.png', dpi=150, bbox_inches='tight')
    plt.close(_bfig)
    print("Saved best_kappa_comparison_combined.png")

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
            ax_title(ax_pe, f'{sys_type.capitalize()} — insufficient data')
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

        ax_pe.set_xlabel('Mean energy [kT]')
        ax_pe.set_ylabel(r'$P_{ex}$ [mM]')
        ax_title(ax_pe, f'{sys_type.capitalize()}')
        ax_pe.legend(fontsize=8)
        ax_pe.grid(True, alpha=0.3)

        cb = fig_pe.colorbar(sc, ax=ax_pe)
        cb.set_label(r'$\kappa$ [Å$^{-1}$]')

    fig_title(fig_pe, r'$P_{ex}$ vs mean energy — implicit simulations', fontweight='bold')
    fig_pe.tight_layout()
    fig_pe.savefig(plot_dir / 'pex_vs_energy.png', dpi=150, bbox_inches='tight')
    plt.close(fig_pe)
    print("Saved pex_vs_energy.png")

# ── 4. Energy convergence panels (implicit) ───────────────────────────────────
_rep_colors = ['tab:blue', 'tab:orange', 'tab:green']
_step_fmt   = plt.FuncFormatter(lambda x, _: f'{x/1e6:.0f}M' if x >= 1e6 else f'{x/1e3:.0f}k')

for sys_type, items in sorted(folders_by_type.items()):
    items.sort(key=lambda x: x[0])
    groups        = group_by_kappa(items)
    unique_kappas = list(groups.keys())

    ncols = 3
    nrows = 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False,
                             sharex='col', sharey=True)

    for idx, kappa in enumerate(unique_kappas):
        if idx >= nrows * ncols:
            break
        row, col = idx // ncols, idx % ncols
        ax = axes[row][col]

        for rep_idx, folder in enumerate(groups[kappa]):
            rep_color = _rep_colors[rep_idx % len(_rep_colors)]
            df = load_energy(folder)
            if df is not None and 'total' in df.columns:
                steps = df['step'].values
                U     = df['total'].values
                cumm  = np.cumsum(U) / np.arange(1, len(U) + 1)
                rep_label = f'replica {rep_idx + 1}' if idx == 0 else '_'
                ax.plot(steps, U,    color=rep_color, lw=0.6, alpha=0.35)
                ax.plot(steps, cumm, color=rep_color, lw=1.8, label=rep_label)
        ax.xaxis.set_major_formatter(_step_fmt)
        if idx == 0:
            ax.legend(fontsize=18, loc='lower right')
        ax.text(0.03, 0.04, rf'$\kappa$ = {kappa:.4f} Å$^{{-1}}$',
                transform=ax.transAxes, fontsize=17, va='bottom', ha='left',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7))
        if col == 0:
            ax.set_ylabel(r'$U_\mathrm{total}$ [$k_\mathrm{B}T$]', fontsize=19)
        if row == nrows - 1 or idx + ncols >= len(unique_kappas):
            ax.set_xlabel('MC step', fontsize=19)
        ax.tick_params(labelsize=18)

    for idx in range(len(unique_kappas), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    # Restore tick labels on last visible row per column (hidden axes break sharex)
    for c in range(ncols):
        for r in range(nrows - 1, -1, -1):
            if r * ncols + c < len(unique_kappas):
                axes[r][c].tick_params(labelbottom=True)
                break
    # Restore y tick labels on col 0 (sharey can suppress them)
    for r in range(nrows):
        if r * ncols < len(unique_kappas):
            axes[r][0].tick_params(labelleft=True)

    fig_title(fig, f'{sys_type.capitalize()} — energy convergence', fontweight='bold')
    fig.tight_layout()
    fname = f'energy_convergence_{sys_type}'
    fig.savefig(plot_dir / f'{fname}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {fname}.png")

# ── 4b. Energy convergence — explicit simulations across all system types ─────
print("\nPlotting explicit energy convergence...")

if list(iter_explicit_folders('isotropic')) or list(iter_explicit_folders('bulk')):
    from matplotlib.gridspec import GridSpec as _GS

    _efig = plt.figure(figsize=(10, 12))
    # Two GridSpecs: rows 0-1 tightly spaced, row 2 separated by a larger gap
    _gs_top = _GS(2, 4, figure=_efig, top=0.95, bottom=0.37, hspace=0.10, wspace=0.25)
    _gs_bot = _GS(1, 4, figure=_efig, top=0.30, bottom=0.06, wspace=0.25)

    _ax_iso  = _efig.add_subplot(_gs_top[0, 0:2])
    _ax_dip  = _efig.add_subplot(_gs_top[0, 2:4])
    _ax_quad = _efig.add_subplot(_gs_top[1, 0:2])
    _ax_neut = _efig.add_subplot(_gs_top[1, 2:4])
    _ax_bulk = _efig.add_subplot(_gs_bot[0, 1:3])

    # rows 0–1 share x within columns; bulk has its own x and y
    _ax_quad.sharex(_ax_iso)
    _ax_neut.sharex(_ax_dip)
    # rows 0–1 share y; bulk is independent
    _ax_dip.sharey(_ax_iso)
    _ax_quad.sharey(_ax_iso)
    _ax_neut.sharey(_ax_iso)
    # hide x tick labels on row 0 (shared with row 1)
    plt.setp(_ax_iso.get_xticklabels(), visible=False)
    plt.setp(_ax_dip.get_xticklabels(), visible=False)

    _exp_panel_map = [
        ('isotropic',   _ax_iso,  True),
        ('dipolar',     _ax_dip,  False),
        ('quadrupolar', _ax_quad, False),
        ('neutral',     _ax_neut, False),
        ('bulk',        _ax_bulk, False),
    ]

    for _eidx, (_t, _ax, _first) in enumerate(_exp_panel_map):
        _exp_fldrs = list(iter_explicit_folders(_t))
        if not _exp_fldrs:
            _ax.set_visible(False)
            continue
        for _rep_idx, _folder in enumerate(_exp_fldrs):
            _rep_color = _rep_colors[_rep_idx % len(_rep_colors)]
            _df = load_energy(_folder)
            if _df is not None and 'total' in _df.columns:
                _steps = _df['step'].values
                _U     = _df['total'].values
                _cumm  = np.cumsum(_U) / np.arange(1, len(_U) + 1)
                _ax.plot(_steps, _U,    color=_rep_color, lw=0.6, alpha=0.35)
                _ax.plot(_steps, _cumm, color=_rep_color, lw=1.8,
                         label=f'replica {_rep_idx + 1}' if _first else '_')
        _ax.xaxis.set_major_formatter(_step_fmt)
        if _first:
            _ax.legend(fontsize=14, loc='lower right')
        _ax.text(0.03, 0.04, f'{_t.capitalize()}\n0.1 M NaCl',
                 transform=_ax.transAxes, fontsize=9, va='bottom', ha='left',
                 bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7))
        _ax.text(0.97, 0.97, f'({chr(ord("a") + _eidx)})',
                 transform=_ax.transAxes, fontsize=12, va='top', ha='right')
        _ax.tick_params(labelsize=14)

    # y labels and ticks only on left column and bulk
    for _ax in (_ax_iso, _ax_quad, _ax_bulk):
        _ax.set_ylabel(r'$U_\mathrm{total}$ [$k_\mathrm{B}T$]', fontsize=15)
    for _ax in (_ax_dip, _ax_neut):
        _ax.tick_params(labelleft=False)
        _ax.set_ylabel('')
    # x labels only on bottom row of each shared-x column pair + bulk
    for _ax in (_ax_quad, _ax_neut, _ax_bulk):
        _ax.set_xlabel('MC step', fontsize=15)

    fig_title(_efig, 'Explicit simulations — energy convergence', fontweight='bold')
    _efig.savefig(plot_dir / 'energy_convergence_explicit.png', dpi=150, bbox_inches='tight')
    plt.close(_efig)
    print("Saved energy_convergence_explicit.png")

# ── pre-compute global RDF y-limit ───────────────────────────────────────────
_rdf_gmax = 0.0
for _sys, _items in folders_by_type.items():
    for _kappa, _folders in group_by_kappa(_items).items():
        _, _g, _s = replica_mean_sem(_folders, load_rdf, 'g(r)')
        if _g is not None:
            _rdf_gmax = max(_rdf_gmax, (_g + _s).max())
    _, _ge, _se = replica_mean_sem(list(iter_explicit_folders(_sys)), load_rdf, 'g(r)')
    if _ge is not None:
        _rdf_gmax = max(_rdf_gmax, (_ge + _se).max())
rdf_ymax = np.ceil(_rdf_gmax * 1.1 * 10) / 10  # 10 % headroom, rounded to 1 decimal

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
            ax.set_ylim(bottom=0, top=rdf_ymax)
        ax_title(ax, f'κ={kappa:.4f} (n={len(folders)})', fontsize=8)
        ax.set_xlabel('r [Å]', fontsize=7)
        ax.set_ylabel('g(r)', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig_title(fig, f'{sys_type.capitalize()} — radial distribution functions', fontweight='bold')
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
    ax.set_ylim(bottom=0, top=rdf_ymax)
    ax.set_xlabel('r [Å]')
    ax.set_ylabel('g(r)')
    ax_title(ax, f'{sys_type.capitalize()} — RDF overlay')
    ax.grid(True, alpha=0.3)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax).set_label(r'$\kappa$ [Å$^{-1}$]')

    fig.tight_layout()
    fname = f'rdf_overlay_{sys_type}'
    fig.savefig(plot_dir / f'{fname}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {fname}.png")

# ── 6b. RDF overlay — all systems on one plot with shared rainbow colorbar ────
print("\nPlotting combined RDF overlay (all systems, shared colorbar)...")
_overlay_types = [t for t in ('isotropic', 'dipolar', 'quadrupolar') if t in folders_by_type]

if len(_overlay_types) >= 2:
    # Collect global kappa range across all target systems
    _all_kappas = []
    for _t in _overlay_types:
        _all_kappas.extend(group_by_kappa(folders_by_type[_t]).keys())
    _kmin_global = min(_all_kappas)
    _kmax_global = max(_all_kappas)

    _cmap_combined = plt.get_cmap('rainbow')
    _norm_combined = plt.Normalize(vmin=_kmin_global, vmax=_kmax_global)

    fig_comb, ax_comb = plt.subplots(figsize=(7, 5))

    _sys_styles = {'isotropic': ':', 'quadrupolar': '-', 'dipolar': '--'}

    _rdf_xmin, _rdf_xmax, _rdf_pad = 40, 60, 0.4
    for _t in _overlay_types:
        _ls = _sys_styles.get(_t, '-')
        _groups = group_by_kappa(folders_by_type[_t])
        for _kappa, _folders in _groups.items():
            _r, _g_mean, _ = replica_mean_sem(_folders, load_rdf, 'g(r)')
            if _r is not None:
                _m = (_r >= _rdf_xmin) & (_r <= _rdf_xmax)
                ax_comb.plot(_r[_m], _g_mean[_m], linewidth=1.0, alpha=1, linestyle=_ls,
                             color=_cmap_combined(_norm_combined(_kappa)))

    # Explicit overlays — same linestyle as their system, black to distinguish from implicit
    for _t in _overlay_types:
        _ls = _sys_styles.get(_t, '-')
        _r_exp, _g_exp, _ = replica_mean_sem(list(iter_explicit_folders(_t)), load_rdf, 'g(r)')
        if _r_exp is not None:
            _m = (_r_exp >= _rdf_xmin) & (_r_exp <= _rdf_xmax)
            ax_comb.plot(_r_exp[_m], _g_exp[_m], linewidth=1.0, color='black',
                         linestyle=_ls, alpha=0.75,
                         label=f'{_t.capitalize()}')

    ax_comb.axhline(1.0, color='gray', linewidth=0.6, linestyle='--')
    ax_comb.set_xlim(_rdf_xmin - _rdf_pad, _rdf_xmax + _rdf_pad)
    ax_comb.xaxis.set_major_locator(plt.MultipleLocator(4))
    ax_comb.set_ylim(bottom=-rdf_ymax * 0.04, top=rdf_ymax)
    ax_comb.set_xlabel('r [Å]', fontsize=13)
    ax_comb.set_ylabel('g(r)', fontsize=13)
    ax_comb.tick_params(axis='both', labelsize=12)
    ax_title(ax_comb, 'RDF overlay — isotropic, quadrupolar & dipolar')
    ax_comb.legend(fontsize=11)
    ax_comb.grid(False)

    _sm_comb = plt.cm.ScalarMappable(cmap=_cmap_combined, norm=_norm_combined)
    _sm_comb.set_array([])
    _cb_comb = fig_comb.colorbar(_sm_comb, ax=ax_comb)
    _cb_comb.set_label(r'$\kappa$ [Å$^{-1}$]', fontsize=13)
    _cb_comb.ax.tick_params(labelsize=12)

    fig_comb.tight_layout()
    fig_comb.savefig(plot_dir / 'rdf_overlay_combined.png', dpi=150, bbox_inches='tight')
    plt.close(fig_comb)
    print("Saved rdf_overlay_combined.png")

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
                ax.legend(fontsize=9, )
        ax.set_ylim(ymin - ypad, ymax + ypad)
        ax.set_xlabel('R [Å]', fontsize=8)
        ax.set_ylabel('Energy [kT]', fontsize=8)
        ax_title(ax, title, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    fig_title(fig, f'{sys_type.capitalize()} — energy decomposition (mean ± SEM)', fontweight='bold')
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    cax = fig.add_axes([0.15, 0.02, 0.7, 0.025])
    sm  = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cax, orientation='horizontal', label=r'$\kappa$ [Å$^{-1}$]')
    fname = f'multipole_energy_{sys_type}'
    fig.savefig(plot_dir / f'{fname}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {fname}.png")

# ── 7b. Combined multipole energy — all systems, shared rainbow colorbar ──────
print("\nPlotting combined multipole energy decomposition...")
_me_types  = [t for t in ('isotropic', 'dipolar', 'quadrupolar') if t in folders_by_type]
_me_styles = {'isotropic': ':', 'dipolar': '--', 'quadrupolar': '-'}

if len(_me_types) >= 2:
    _me_all_k = []
    for _t in _me_types:
        _me_all_k.extend(group_by_kappa(folders_by_type[_t]).keys())
    _me_kmin = min(_me_all_k)
    _me_kmax = max(_me_all_k)
    _me_cmap = plt.get_cmap('rainbow')
    _me_norm = plt.Normalize(vmin=_me_kmin, vmax=_me_kmax)

    # Cache data and find global y range per energy column
    _mc, _mce = {}, {}
    _me_ymin = {col: np.inf  for col in energy_cols}
    _me_ymax = {col: -np.inf for col in energy_cols}
    for _t in _me_types:
        _mg = group_by_kappa(folders_by_type[_t])
        for _col in energy_cols:
            for _kappa, _folders in _mg.items():
                _r, _m, _s = replica_mean_sem(_folders, load_multipole, _col)
                _mc[(_t, _col, _kappa)] = (_r, _m, _s)
                if _m is not None:
                    _me_ymin[_col] = min(_me_ymin[_col], (_m - _s).min())
                    _me_ymax[_col] = max(_me_ymax[_col], (_m + _s).max())
            _re, _me2, _ = replica_mean_sem(list(iter_explicit_folders(_t)), load_multipole, _col)
            _mce[(_t, _col)] = (_re, _me2)
            if _me2 is not None:
                _me_ymin[_col] = min(_me_ymin[_col], _me2.min())
                _me_ymax[_col] = max(_me_ymax[_col], _me2.max())

    if not np.isinf(_me_ymin[energy_cols[0]]):
        fig_me, axes_me = plt.subplots(2, 3, figsize=(13, 7),
                                       sharex=True, sharey=False, squeeze=False)
        fig_me.subplots_adjust(hspace=0.16, wspace=0.35, bottom=0.21)

        for _mi, (_col, _etitle) in enumerate(zip(energy_cols, energy_titles)):
            _row, _pcol = divmod(_mi, 3)
            _ax = axes_me[_row][_pcol]

            _me_xmin, _me_xmax, _me_xpad = 41, 60, 0.4
            for _t in _me_types:
                _ls = _me_styles.get(_t, '-')
                for _kappa in group_by_kappa(folders_by_type[_t]):
                    _r, _m, _ = _mc[(_t, _col, _kappa)]
                    if _r is not None:
                        _xmask = (_r >= _me_xmin) & (_r <= _me_xmax)
                        _ax.plot(_r[_xmask], _m[_xmask], linewidth=1.5, alpha=1, linestyle=_ls,
                                 color=_me_cmap(_me_norm(_kappa)))

            for _t in _me_types:
                _ls = _me_styles.get(_t, '-')
                _re, _me2 = _mce[(_t, _col)]
                if _re is not None:
                    _xmask = (_re >= _me_xmin) & (_re <= _me_xmax)
                    _ax.plot(_re[_xmask], _me2[_xmask], linewidth=1.5, color='black',
                             linestyle=_ls, alpha=1,
                             label=_t.capitalize() if _mi == 0 else '_')

            _ypad = (_me_ymax[_col] - _me_ymin[_col]) * 0.05
            _ax.set_ylim(_me_ymin[_col] - _ypad, _me_ymax[_col] + _ypad)
            _ax.set_xlim(_me_xmin - _me_xpad, _me_xmax + _me_xpad)
            _ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=4))
            _ax.tick_params(axis='both', labelsize=15)
            _ax.text(0.15, 0.05, f'({chr(ord("a") + _mi)})', transform=_ax.transAxes,
                     fontsize=12, ha='right', va='bottom')
            if _row == 1:
                _ax.set_xlabel('R [Å]', fontsize=16)
            _ax.set_ylabel(_etitle + ' [k$_\mathrm{B}$T]', fontsize=16)

        axes_me[0][0].legend(fontsize=13)
        axes_me[0][0].set_xticks(range(42, 61, 4))

        _sm_me = plt.cm.ScalarMappable(cmap=_me_cmap, norm=_me_norm)
        _sm_me.set_array([])
        _pl_me = axes_me[1][0].get_position()
        _pr_me = axes_me[1][2].get_position()
        _cax_me = fig_me.add_axes([_pl_me.x0, 0.06, _pr_me.x1 - _pl_me.x0, 0.03])
        _cb_me = fig_me.colorbar(_sm_me, cax=_cax_me, orientation='horizontal')
        _cb_me.set_label(r'$\kappa$ [Å$^{-1}$]', fontsize=16)
        _cb_me.ax.tick_params(labelsize=15)

        fig_title(fig_me, 'Combined multipole energy decomposition — all systems', fontweight='bold')
        fig_me.align_ylabels(axes_me.flatten())
        fig_me.savefig(plot_dir / 'multipole_energy_combined.png', dpi=150, bbox_inches='tight')
        plt.close(fig_me)
        print("Saved multipole_energy_combined.png")

# ── 8. Orientational correlations ─────────────────────────────────────────────
corr_cols   = ['mucorr', 'long', 'quadcorr_norm', 'p2']
corr_labels = [
    r'Dipole correlation',#, $\langle\hat{\mu}_i\cdot\hat{\mu}_j\rangle$',
    r'Longitudinal projection',#, $\langle(\hat{\mu}_i\cdot\hat{R})(\hat{\mu}_i\cdot\hat{R})\rangle$',
    r'Quadrupole correlation',#, $\langle\hat{\Phi}_i:\hat{\Phi}_j\rangle$',
    r'Nematic order',#, $\langle{(3}\mathrm{cos}^2\,\theta-1)/{2}\rangle$',
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
        ax.set_xlabel('R [Å]', fontsize=8)
        ax.set_ylabel('Correlation', fontsize=8)
        ax_title(ax, label, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    fig_title(fig, f'{sys_type.capitalize()} — orientational correlations (mean ± SEM)', fontweight='bold')
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    cax = fig.add_axes([0.15, 0.02, 0.7, 0.025])
    sm  = plt.cm.ScalarMappable(cmap=cmap, norm=norm_kappa)
    sm.set_array([])
    fig.colorbar(sm, cax=cax, orientation='horizontal', label=r'$\kappa$ [Å$^{-1}$]')
    fname = f'multipole_corr_{sys_type}'
    fig.savefig(plot_dir / f'{fname}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {fname}.png")

# ── 8b. Combined orientational correlations — all systems, shared rainbow colorbar ──
print("\nPlotting combined orientational correlations...")
_corr_types = [t for t in ('isotropic', 'dipolar', 'quadrupolar') if t in folders_by_type]

if len(_corr_types) >= 2:
    _corr_all_k = []
    for _t in _corr_types:
        _corr_all_k.extend(group_by_kappa(folders_by_type[_t]).keys())
    _corr_kmin = min(_corr_all_k)
    _corr_kmax = max(_corr_all_k)
    _corr_cmap = plt.get_cmap('rainbow')
    _corr_norm = plt.Normalize(vmin=_corr_kmin, vmax=_corr_kmax)
    _corr_styles = {'isotropic': ':', 'dipolar': '--', 'quadrupolar': '-'}

    # Cache all data and find global y range
    _cc, _ce = {}, {}
    _cymin, _cymax = np.inf, -np.inf
    for _t in _corr_types:
        _cg = group_by_kappa(folders_by_type[_t])
        for _col in corr_cols:
            for _kappa, _folders in _cg.items():
                _r, _m, _s = replica_mean_sem(_folders, load_multipole, _col)
                _cc[(_t, _col, _kappa)] = (_r, _m, _s)
                if _m is not None:
                    _cymin = min(_cymin, (_m - _s).min())
                    _cymax = max(_cymax, (_m + _s).max())
            _re, _me, _ = replica_mean_sem(list(iter_explicit_folders(_t)), load_multipole, _col)
            _ce[(_t, _col)] = (_re, _me)
            if _me is not None:
                _cymin = min(_cymin, _me.min())
                _cymax = max(_cymax, _me.max())

    if not np.isinf(_cymin):
        _cypad = (_cymax - _cymin) * 0.05

        fig_cc, axes_cc = plt.subplots(2, 2, figsize=(9, 7),
                                       sharex=True, sharey='row', squeeze=False)
        fig_cc.subplots_adjust(hspace=0.16, wspace=0.12, bottom=0.19)

        for _ci, (_col, _clabel) in enumerate(zip(corr_cols, corr_labels)):
            _row, _pcol = divmod(_ci, 2)
            _ax = axes_cc[_row][_pcol]

            _cr_xmin, _cr_xmax, _cr_xpad = 41, 59, 0.4
            for _t in _corr_types:
                _ls = _corr_styles.get(_t, '-')
                for _kappa in group_by_kappa(folders_by_type[_t]):
                    _r, _m, _ = _cc[(_t, _col, _kappa)]
                    if _r is not None:
                        _xmask = (_r >= _cr_xmin) & (_r <= _cr_xmax)
                        _ax.plot(_r[_xmask], _m[_xmask], linewidth=1, alpha=1, linestyle=_ls,
                                 color=_corr_cmap(_corr_norm(_kappa)))

            for _t in _corr_types:
                _ls = _corr_styles.get(_t, '-')
                _re, _me = _ce[(_t, _col)]
                if _re is not None:
                    _xmask = (_re >= _cr_xmin) & (_re <= _cr_xmax)
                    _ax.plot(_re[_xmask], _me[_xmask], linewidth=1, color='black',
                             linestyle=_ls, alpha=1, label=_t.capitalize())

            _ax.axhline(0, color='gray', linewidth=1, linestyle='--')
            _ax.set_ylim((-0.1, 0.28) if _row == 0 else (-0.28, 0.1))
            _ax.set_xlim(_cr_xmin - _cr_xpad, _cr_xmax + _cr_xpad)
            _ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.1f}'))
            _ax.tick_params(axis='both', labelsize=12)
            #_ax.grid(True, alpha=0.2)
            _ax.text(0.97, 0.05, f'({chr(ord("a") + _ci)})', transform=_ax.transAxes,
                     fontsize=12, ha='right', va='bottom')

            if _row == 1:
                _ax.set_xlabel('R [Å]', fontsize=13)
            if _pcol == 1:
                _ax.set_ylabel(_clabel, fontsize=12)
                _ax.yaxis.set_label_position('right')
                _ax.yaxis.tick_right()
                #_ax.tick_params(axis='y', which='both', labelright=True, labelleft=False)
            else:
                _ax.set_ylabel(_clabel, fontsize=12)

        # x ticks 42–58 step 2, pruning 40 and 60 from the visible range
        axes_cc[0][0].set_xticks(range(42, 60, 4))

        # half the default number of y ticks on each row
        for _pc in range(2):
            axes_cc[0][_pc].yaxis.set_major_locator(plt.MaxNLocator(nbins=4))
            axes_cc[1][_pc].yaxis.set_major_locator(plt.MaxNLocator(nbins=4))

        axes_cc[0][0].legend(fontsize=11)

        _sm_cc = plt.cm.ScalarMappable(cmap=_corr_cmap, norm=_corr_norm)
        _sm_cc.set_array([])
        # colorbar spanning exactly from left-plot left edge to right-plot right edge
        _pl = axes_cc[1][0].get_position()
        _pr = axes_cc[1][1].get_position()
        _cax_cc = fig_cc.add_axes([_pl.x0, 0.06, _pr.x1 - _pl.x0, 0.03])
        _cb_cc = fig_cc.colorbar(_sm_cc, cax=_cax_cc, orientation='horizontal')
        _cb_cc.set_label(r'$\kappa$ [Å$^{-1}$]', fontsize=13)
        _cb_cc.ax.tick_params(labelsize=12)

        fig_title(fig_cc, 'Combined orientational correlations — all systems', fontweight='bold')
        fig_cc.savefig(plot_dir / 'multipole_corr_combined.png', dpi=150, bbox_inches='tight')
        plt.close(fig_cc)
        print("Saved multipole_corr_combined.png")

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
