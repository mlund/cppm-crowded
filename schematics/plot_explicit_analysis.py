#!/usr/bin/env python3
"""Explicit-simulation analysis: bulk thermodynamics, pressure, ion counts, and RDFs."""
import argparse
import gzip
import re
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

parser = argparse.ArgumentParser(description='Analyse explicit CPPM simulations.')
parser.add_argument('path',  help='Path to the explicit/ directory')
parser.add_argument('plots', help='Output directory for plots')
parser.add_argument('--no-titles', action='store_true', help='Suppress all plot titles')
args = parser.parse_args()

root_dir = Path(args.path).resolve()
plot_dir = Path(args.plots).resolve()
plot_dir.mkdir(parents=True, exist_ok=True)

plt.rcParams['font.family'] = 'sans-serif'

colors = {
    'bulk':        'tab:gray',
    'isotropic':   'tab:blue',
    'dipolar':     'tab:orange',
    'neutral':     'tab:purple',
    'quadrupolar': 'tab:red',
}

SYS_ORDER = ['bulk', 'isotropic', 'dipolar', 'quadrupolar', 'neutral']


def ax_title(ax, text, **kw):
    if not args.no_titles:
        ax.set_title(text, **kw)


def fig_title(fig, text, **kw):
    if not args.no_titles:
        fig.suptitle(text, **kw)


# ── I/O helpers ───────────────────────────────────────────────────────────────

class _FaunusLoader(yaml.SafeLoader):
    pass

_FaunusLoader.add_multi_constructor(
    '!',
    lambda loader, tag, node: (
        loader.construct_mapping(node) if isinstance(node, yaml.MappingNode)
        else loader.construct_sequence(node) if isinstance(node, yaml.SequenceNode)
        else loader.construct_scalar(node)
    ),
)


def load_output(path):
    with open(path) as f:
        return yaml.load(f, Loader=_FaunusLoader)


def load_input(path):
    with open(path) as f:
        return yaml.load(f, Loader=_FaunusLoader)


def load_rdf(folder, filename='rdf.csv'):
    for name in (filename + '.gz', filename):
        p = folder / name
        if p.exists():
            opener = gzip.open(p, 'rt') if name.endswith('.gz') else open(p)
            with opener as f:
                return pd.read_csv(f)
    return None


def load_energy(folder):
    for name in ('energy.csv.gz', 'energy.csv'):
        p = folder / name
        if p.exists():
            opener = gzip.open(p, 'rt') if name.endswith('.gz') else open(p)
            with opener as f:
                return pd.read_csv(f)
    return None


def replica_mean_sem(folders, load_fn, col):
    """Average *col* over DataFrames loaded from *folders*; return (x, mean, sem)."""
    dfs = [load_fn(f) for f in folders]
    dfs = [df for df in dfs if df is not None and col in df.columns]
    if not dfs:
        return None, None, None
    x   = dfs[0].iloc[:, 0].values
    arr = np.array([np.interp(x, df.iloc[:, 0].values, df[col].values) for df in dfs])
    mean = arr.mean(axis=0)
    sem  = arr.std(axis=0, ddof=1) / np.sqrt(len(dfs)) if len(dfs) > 1 else np.zeros_like(mean)
    return x, mean, sem


# ── folder discovery ──────────────────────────────────────────────────────────

def iter_sim_folders(sys_type):
    """Yield all replica folders for *sys_type* (e.g. 'isotropic' → isotropic-exp, isotropic-exp_2, …)."""
    pat = re.compile(rf'^{re.escape(sys_type)}-exp(?:_(\d+))?$')
    for folder in sorted(root_dir.iterdir()):
        if folder.is_dir() and pat.match(folder.name) and (folder / 'output.yaml').exists():
            yield folder


def discover_all():
    """Return DataFrame with columns [type, replicate, folder] for every discovered simulation."""
    rows = []
    pat  = re.compile(r'^(bulk|isotropic|dipolar|neutral|quadrupolar)-exp(?:_(\d+))?$')
    for folder in sorted(root_dir.iterdir()):
        if not folder.is_dir():
            continue
        m = pat.match(folder.name)
        if m and (folder / 'output.yaml').exists():
            rows.append({'type': m.group(1), '_suffix': m.group(2) or '0', 'folder': folder})
    if not rows:
        return pd.DataFrame(columns=['type', 'replicate', 'folder'])
    df = pd.DataFrame(rows).sort_values(['type', '_suffix']).reset_index(drop=True)
    df['replicate'] = df.groupby('type').cumcount() + 1
    return df[['type', 'replicate', 'folder']]


# ── output.yaml parsers ───────────────────────────────────────────────────────

def _find_analysis(doc, kind, **filters):
    for entry in doc.get('analysis', []):
        if kind not in entry:
            continue
        val = entry[kind]
        if all(val.get(k) == v for k, v in filters.items()):
            return val
    return None


def molarity(doc, atomtype):
    cv = _find_analysis(doc, 'collective_variable',
                        property='Molarity',
                        description=f'selection: atomtype {atomtype}')
    return cv['mean'] if cv else 0.0


def molarity_std(doc, atomtype):
    cv = _find_analysis(doc, 'collective_variable',
                        property='Molarity',
                        description=f'selection: atomtype {atomtype}')
    if cv is None:
        return 0.0
    return max(cv['rms'] ** 2 - cv['mean'] ** 2, 0.0) ** 0.5


def ion_count(doc, atomtype):
    cv = _find_analysis(doc, 'collective_variable',
                        property='Count',
                        description=f'selection: atomtype {atomtype}')
    if cv is None:
        return None, None
    mean = cv['mean']
    std  = max(cv['rms'] ** 2 - mean ** 2, 0.0) ** 0.5
    return mean, std


def excess_pressure(doc):
    vvm = _find_analysis(doc, 'virtual_volume_move')
    pex = vvm['Pex/mM']
    return pex['mean'], pex.get('error', 0.0)


def macroion_molarity(doc):
    n = 0
    for block in doc.get('blocks', []):
        if block['molecule'] == 'macroion':
            n = block.get('active') or block['N']
            break
    cell = doc['cell']
    Lx, Ly, Lz = (cell if isinstance(cell, list) else [cell] * 3)[:3]
    return n / (Lx * Ly * Lz) * 1e27 / 6.022e23


def ionic_strength(doc):
    return 0.5 * (molarity(doc, 'Na') + molarity(doc, 'Cl'))


def debye_length(doc):
    ionic = ionic_strength(doc)
    if ionic <= 0:
        return float('inf')
    T     = doc['medium']['temperature']
    eps_r = (78.36 if doc['medium']['permittivity'] == 'Water'
             else float(doc['medium']['permittivity']))
    lB    = 1.671e5 / (eps_r * T)
    kappa = (8 * np.pi * lB * ionic * 6.022e-4) ** 0.5
    return 1.0 / kappa


def pressure_breakdown(folder):
    doc  = load_output(folder / 'output.yaml')
    p_na = molarity(doc, 'Na') * 1000
    p_cl = molarity(doc, 'Cl') * 1000
    p_mp = macroion_molarity(doc) * 1000
    p_ex, p_ex_err = excess_pressure(doc)
    e_na = molarity_std(doc, 'Na') * 1000
    e_cl = molarity_std(doc, 'Cl') * 1000
    e_tot = (e_na ** 2 + e_cl ** 2 + p_ex_err ** 2) ** 0.5
    return dict(Na=p_na, Cl=p_cl, macroion=p_mp, excess=p_ex,
                total=p_na + p_cl + p_mp + p_ex,
                Na_err=e_na, Cl_err=e_cl, macroion_err=0.0,
                excess_err=p_ex_err, total_err=e_tot)


# ── parse all simulations ─────────────────────────────────────────────────────

print("Discovering simulation folders…")
sim_dirs = discover_all()
print(sim_dirs.to_string(index=False))

# ── 1. Molarity table ─────────────────────────────────────────────────────────
print("\n── Molarity by directory ──")
_mol_rows = []
for _, row in sim_dirs.iterrows():
    doc = load_output(row['folder'] / 'output.yaml')
    _mol_rows.append({
        'type': row['type'], 'replicate': row['replicate'],
        '[Na+] / M': molarity(doc, 'Na'),
        '[Cl-] / M': molarity(doc, 'Cl'),
        'I / M': ionic_strength(doc),
    })
mol_df = pd.DataFrame(_mol_rows).set_index(['type', 'replicate'])
pd.set_option('display.precision', 4)
print(mol_df)

# ── 2. Bulk thermodynamic parameters ──────────────────────────────────────────
print("\n── Bulk thermodynamics ──")
bulk_folders = list(iter_sim_folders('bulk'))

bulk_docs = [load_output(f / 'output.yaml') for f in bulk_folders]
_ref_doc  = bulk_docs[0]

_inp_path = bulk_folders[0] / 'input.yml'
_inp      = load_input(_inp_path)

def _atom_sigma(inp, name):
    for atom in inp.get('atoms', []):
        if atom.get('name') == name:
            return atom.get('sigma')
    return None

def _mol_activity(inp, name):
    for mol in inp.get('molecules', []):
        if mol.get('name') == name:
            return mol.get('activity')
    return None

activity = _mol_activity(_inp, 'Cations')
ion_size = 0.5 * (_atom_sigma(_inp, 'Na') + _atom_sigma(_inp, 'Cl'))
T        = _ref_doc['medium']['temperature']
eps_r    = (78.36 if _ref_doc['medium']['permittivity'] == 'Water'
            else float(_ref_doc['medium']['permittivity']))

c_na_vals  = [molarity(d, 'Na') for d in bulk_docs]
c_cl_vals  = [molarity(d, 'Cl') for d in bulk_docs]
pex_vals   = [excess_pressure(d)[0] for d in bulk_docs]

c_na       = np.mean(c_na_vals)
c_cl       = np.mean(c_cl_vals)
ionic      = 0.5 * (c_na + c_cl)
P_ex_mean  = np.mean(pex_vals)
P_ideal    = (c_na + c_cl) * 1000
phi        = 1 + P_ex_mean / P_ideal
gamma_pm   = ((activity / c_na) * (activity / c_cl)) ** 0.5
lambda_B   = 1.671e5 / (eps_r * T)
kappa      = (8 * np.pi * lambda_B * ionic * 6.022e-4) ** 0.5
ln_g_LL    = -kappa * lambda_B / 2
ln_g_DH    = -kappa * lambda_B / (2 * (1 + kappa * ion_size))

print(f"Bulk averages over {len(bulk_folders)} replicates:")
print(f"  prescribed activity  = {activity} M")
print(f"  ⟨[Na+]⟩ = {c_na:.4g} M,  ⟨[Cl-]⟩ = {c_cl:.4g} M  (I = {ionic:.4g} M)")
print(f"  γ_± = {gamma_pm:.4g}")
print(f"  P_ideal = {P_ideal:.4g} mM,  ⟨P_ex⟩ = {P_ex_mean:.4g} mM")
print(f"  osmotic coeff ϕ = {phi:.4g}")
print(f"  λ_B = {lambda_B:.3f} Å,  κ⁻¹ = {1/kappa:.3f} Å,  ion size a = {ion_size:.2f} Å")
print(f"  ln γ_± (measured)       = {np.log(gamma_pm):+.4f}")
print(f"  ln γ_± (DH limit law)   = {ln_g_LL:+.4f}")
print(f"  ln γ_± (DH + size)      = {ln_g_DH:+.4f}")
print(f"  3·(ϕ − 1)               = {3*(phi-1):+.4f}")

# ── 3. Pressure breakdown — bar chart ─────────────────────────────────────────
print("\nPlotting pressure components…")
comps = ['Na', 'Cl', 'macroion', 'excess', 'total']
err_c = [c + '_err' for c in comps]

pres_rows = []
for _, row in sim_dirs.iterrows():
    try:
        p = pressure_breakdown(row['folder'])
    except Exception as e:
        print(f"  Skipping {row['folder'].name}: {e}")
        continue
    pres_rows.append({**p, 'type': row['type'], 'replicate': row['replicate']})

pres_df = pd.DataFrame(pres_rows)

# Bulk reference (mean over all bulk replicates)
_bulk_mean = pres_df[pres_df['type'] == 'bulk'][comps].mean()
_bulk_std  = pres_df[pres_df['type'] == 'bulk'][comps].std().fillna(0)

# ΔP relative to bulk
delta_rows = []
for _, row in pres_df[pres_df['type'] != 'bulk'].iterrows():
    d = {c: row[c] - _bulk_mean[c] for c in comps}
    d.update({ec: (row[ec] ** 2 + _bulk_std[c] ** 2) ** 0.5
              for c, ec in zip(comps, err_c)})
    d['type'] = row['type']
    delta_rows.append(d)
delta_df = pd.DataFrame(delta_rows)

non_bulk_types = [t for t in SYS_ORDER if t != 'bulk' and t in delta_df['type'].values]
delta_mean = delta_df.groupby('type')[comps].mean()
delta_std  = delta_df.groupby('type')[comps].std().fillna(0)

n_types = len(non_bulk_types)
width   = 0.6 / n_types if n_types else 0.6
x       = np.arange(len(comps))

fig, ax = plt.subplots(figsize=(12, 5))
for i, sys_type in enumerate(non_bulk_types):
    vals   = [delta_mean.loc[sys_type, c] for c in comps]
    errs   = [delta_std.loc[sys_type, c] for c in comps]
    offset = (i - n_types / 2 + 0.5) * width
    ax.bar(x + offset, vals, width, yerr=errs, capsize=3,
           color=colors[sys_type], label=sys_type,
           edgecolor='black', linewidth=0.5,
           error_kw=dict(elinewidth=0.8, ecolor='black'))

ax.axhline(0, color='k', lw=0.5, ls=':')
ax.set_xticks(x)
ax.set_xticklabels(comps, fontsize=12)
ax.set_ylabel(r'$\Delta P$ / mM  (relative to bulk)', fontsize=12)
ax.legend(frameon=False, fontsize=10)
ax_title(ax, 'Pressure components (ΔP relative to bulk) — triplicate averages')
fig.tight_layout()
fig.savefig(plot_dir / 'pressure_components.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved pressure_components.png")

# ── 4. Ion counts — bar chart ─────────────────────────────────────────────────
print("Plotting ion counts…")
count_rows = []
for _, row in sim_dirs[sim_dirs['type'] != 'bulk'].iterrows():
    doc = load_output(row['folder'] / 'output.yaml')
    for ion in ('Na', 'Cl'):
        mean, std = ion_count(doc, ion)
        if mean is not None:
            count_rows.append({'ion': ion, 'type': row['type'],
                               'replicate': row['replicate'], 'mean': mean, 'std': std})

count_df  = pd.DataFrame(count_rows)
count_avg = (count_df.groupby(['type', 'ion'])
             .agg(mean_N=('mean', 'mean'), std_N=('mean', 'std'))
             .reset_index().fillna(0))

# Bulk reference counts from molarity × box volume
_cell = _ref_doc['cell']
_V    = _cell[0] * _cell[1] * _cell[2]
_N_A  = 6.022e23
bulk_N_Na = c_na * _V / 1e27 * _N_A
bulk_N_Cl = c_cl * _V / 1e27 * _N_A

_ion_types = [t for t in SYS_ORDER if t != 'bulk' and t in count_avg['type'].values]
bar_w = 0.35
x2    = np.arange(len(_ion_types))

fig, ax = plt.subplots(figsize=(9, 5))
for j_idx, (ion, hatch, label) in enumerate([('Na', '', 'Na$^+$'), ('Cl', '///', 'Cl$^-$')]):
    _sub = count_avg[count_avg['ion'] == ion].set_index('type').reindex(_ion_types).reset_index()
    for k, (_, row) in enumerate(_sub.iterrows()):
        ax.bar(x2[k] + (j_idx - 0.5) * bar_w, row['mean_N'], bar_w,
               yerr=row['std_N'], capsize=4,
               color=colors[row['type']], hatch=hatch,
               edgecolor='black', linewidth=0.5,
               label=label if k == 0 else '_',
               error_kw=dict(elinewidth=0.8, ecolor='black'))

ax.axhline(bulk_N_Na, color='black', lw=1.2, ls='--',
           label=fr'bulk Na$^+$=Cl$^-$ ({bulk_N_Na:.0f})')
ax.set_xticks(x2)
ax.set_xticklabels(_ion_types)
ax.set_ylabel(r'$\langle N \rangle$')
ax.legend(frameon=False, fontsize=10)
ax_title(ax, 'Ion counts — triplicate averages')
fig.tight_layout()
fig.savefig(plot_dir / 'ion_counts.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved ion_counts.png")

# ── 5. Ionic strength and Debye length table ──────────────────────────────────
print("\n── Ionic strength / Debye length by type ──")
ionic_rows = []
for _, row in sim_dirs.iterrows():
    doc  = load_output(row['folder'] / 'output.yaml')
    lD   = debye_length(doc)
    ionic_rows.append({
        'type': row['type'], 'replicate': row['replicate'],
        '[Na+] / M': molarity(doc, 'Na'),
        '[Cl-] / M': molarity(doc, 'Cl'),
        'I / M':    ionic_strength(doc),
        'λ_D / Å':  lD,
    })
ionic_raw = pd.DataFrame(ionic_rows)
_cols     = ['[Na+] / M', '[Cl-] / M', 'I / M', 'λ_D / Å']
_mean_i   = ionic_raw.groupby('type')[_cols].mean()
_std_i    = ionic_raw.groupby('type')[_cols].std().fillna(0)
ionic_sum = _mean_i.copy().astype(object)
for c in _cols:
    ionic_sum[c] = _mean_i[c].map('{:.4f}'.format) + ' ± ' + _std_i[c].map('{:.4f}'.format)
print(ionic_sum.reindex([t for t in SYS_ORDER if t in ionic_sum.index]))

# ── 6. Macroion–macroion RDF overlay ─────────────────────────────────────────
print("\nPlotting macroion–macroion RDF overlay…")
fig, ax = plt.subplots(figsize=(8, 5))

for sys_type in [t for t in SYS_ORDER if t != 'bulk']:
    folders = list(iter_sim_folders(sys_type))
    r, g_mean, g_sem = replica_mean_sem(folders, load_rdf, 'g(r)')
    if r is None:
        continue
    col = colors[sys_type]
    ax.plot(r, g_mean, color=col, lw=1.8, label=sys_type.capitalize())
    ax.fill_between(r, g_mean - g_sem, g_mean + g_sem, color=col, alpha=0.2)

ax.axhline(1.0, color='gray', lw=0.6, ls='--')
ax.set_xlabel('r [Å]', fontsize=12)
ax.set_ylabel('g(r)', fontsize=12)
ax.legend(frameon=False, fontsize=10)
ax_title(ax, 'Macroion–macroion RDF — triplicate mean ± SEM')
fig.tight_layout()
fig.savefig(plot_dir / 'rdf_macroion_overlay.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved rdf_macroion_overlay.png")

# ── 7. Ion-pair RDF overlays ──────────────────────────────────────────────────
print("Plotting ion-pair RDF overlays…")
ion_rdf_specs = [
    ('rdf-na-np.csv', 'Na–NP',  'rdf_na-np'),
    ('rdf-cl-pp.csv', 'Cl–PP',  'rdf_cl-pp'),
    ('rdf-na-cl.csv', 'Na–Cl',  'rdf_na-cl'),
]

fig, axes = plt.subplots(len(ion_rdf_specs), 1,
                         figsize=(8, 5 * len(ion_rdf_specs)), sharex=False)

for ax, (fname, label, _) in zip(axes, ion_rdf_specs):
    for sys_type in [t for t in SYS_ORDER if t != 'bulk']:
        folders = list(iter_sim_folders(sys_type))
        r, g_mean, g_sem = replica_mean_sem(
            folders, lambda f, fn=fname: load_rdf(f, fn), 'g(r)')
        if r is None:
            continue
        col = colors[sys_type]
        ax.plot(r, g_mean, color=col, lw=1.5, label=sys_type.capitalize())
        ax.fill_between(r, g_mean - g_sem, g_mean + g_sem, color=col, alpha=0.2)
    ax.set_xlabel('r [Å]', fontsize=12)
    ax.set_ylabel(f'{label}  g(r)', fontsize=11)
    ax.legend(frameon=False, fontsize=9)

fig_title(fig, 'Ion–NP pair RDFs — triplicate mean ± SEM')
fig.tight_layout()
fig.savefig(plot_dir / 'rdf_ion_pairs.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved rdf_ion_pairs.png")

# ── 8. Energy convergence ─────────────────────────────────────────────────────
print("Plotting energy convergence…")
rep_colors = ['tab:blue', 'tab:orange', 'tab:green']
step_fmt   = plt.FuncFormatter(lambda x, _: f'{x/1e6:.0f}M' if x >= 1e6 else f'{x/1e3:.0f}k')

sys_types_all = [t for t in SYS_ORDER if any(sim_dirs['type'] == t)]
n_types = len(sys_types_all)
n_reps  = int(sim_dirs.groupby('type')['replicate'].max().max())

fig, axes = plt.subplots(n_types, n_reps, figsize=(5 * n_reps, 4 * n_types),
                         sharex=False, sharey='row')

for row_idx, sys_type in enumerate(sys_types_all):
    subset = sim_dirs[sim_dirs['type'] == sys_type].sort_values('replicate')
    for col_idx in range(n_reps):
        ax = axes[row_idx, col_idx]
        rep_row = subset[subset['replicate'] == col_idx + 1]
        if rep_row.empty:
            ax.set_visible(False)
            continue
        df = load_energy(rep_row.iloc[0]['folder'])
        if df is None or 'total' not in df.columns:
            ax.set_visible(False)
            continue
        steps = df['step'].values
        U     = df['total'].values
        cumm  = np.cumsum(U) / np.arange(1, len(U) + 1)
        col   = rep_colors[col_idx % len(rep_colors)]
        ax.plot(steps, U,    color=col, lw=0.6, alpha=0.35)
        ax.plot(steps, cumm, color=col, lw=1.8)
        if col_idx == 0:
            ax.set_ylabel(r'$U_\mathrm{total}$ / $k_\mathrm{B}T$', fontsize=11)
        ax.set_xlabel('MC step', fontsize=11)
        ax.set_title(f'{sys_type} — replica {col_idx + 1}', fontsize=10)
        ax.xaxis.set_major_formatter(step_fmt)

fig_title(fig, 'Energy vs MC steps — convergence check', fontweight='bold')
fig.tight_layout()
fig.savefig(plot_dir / 'energy_convergence.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved energy_convergence.png")

print("\nDone.")
