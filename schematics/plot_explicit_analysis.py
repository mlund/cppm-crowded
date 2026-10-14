#!/usr/bin/env python3
"""Explicit-simulation analysis: bulk thermodynamics, pressure, ion counts, and RDFs."""
import argparse
import gzip
import re
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

SYS_LABEL = {
    'bulk':        'Bulk',
    'isotropic':   r'Isotropic, $P^{0}_{0}$',
    'dipolar':     r'Dipolar, $P^{1}_{8}$',
    'quadrupolar': r'Quadrupolar, $P^{2}_{8}$',
    'neutral':     r'Neutral, $P^{\mathrm{n}}$',
}


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
ax.set_xticklabels(comps, fontsize=13)
ax.set_ylabel(r'$\Delta P$ / mM  (relative to bulk)', fontsize=14)
ax.legend(frameon=False, fontsize=12)
ax_title(ax, 'Pressure components (ΔP relative to bulk) — triplicate averages')
fig.tight_layout()
fig.savefig(plot_dir / 'pressure_components.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved pressure_components.png")

# ── 3b. Pressure breakdown — diverging stacked bar ────────────────────────────
# (col: DataFrame key, label: legend display, color)
stack_comps = [
    ('Na',       r'Na$^+$', '#6ddc9e'),
    ('Cl',       r'Cl$^-$', '#ffb347'),
    ('macroion', 'Macro-ion',    '#60b4ff'),
    ('excess',   'Excess',  '#ff7f7f'),
]

x3 = np.arange(len(non_bulk_types))
fig, ax = plt.subplots(figsize=(7, 8))

bottoms_pos = np.zeros(len(non_bulk_types))
bottoms_neg = np.zeros(len(non_bulk_types))
bar_segs = []  # (col, i, actual_val, sem, rect)

for col, label, color in stack_comps:
    vals = np.array([delta_mean.loc[t, col] for t in non_bulk_types])
    sems = np.array([delta_std.loc[t, col] if t in delta_std.index else 0.0
                     for t in non_bulk_types])
    pos  = np.where(vals > 0, vals, 0.0)
    neg  = np.where(vals < 0, vals, 0.0)
    bars_pos = ax.bar(x3, pos, bottom=bottoms_pos, color=color,
                      label=label, edgecolor='white', linewidth=0.4, alpha=0.95)
    bars_neg = ax.bar(x3, neg, bottom=bottoms_neg, color=color,
                      edgecolor='white', linewidth=0.4, alpha=0.95)
    for i in range(len(non_bulk_types)):
        if pos[i] != 0:
            bar_segs.append((col, i, float(vals[i]), float(sems[i]), bars_pos[i]))
        if neg[i] != 0:
            bar_segs.append((col, i, float(vals[i]), float(sems[i]), bars_neg[i]))
    bottoms_pos += pos
    bottoms_neg += neg

tot_vals = np.array([delta_mean.loc[t, 'total'] for t in non_bulk_types])
tot_errs = np.array([delta_std.loc[t,  'total'] if t in delta_std.index else 0.0
                     for t in non_bulk_types])
ax.errorbar(x3, tot_vals, yerr=tot_errs, fmt='D', color='black',
            markersize=7, capsize=3, lw=1.2, label='Total', zorder=5,
            )

_ytxt_pad = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
for _i, (_tv, _te) in enumerate(zip(tot_vals, tot_errs)):
    ax.text(x3[_i], _tv + _te + _ytxt_pad, f'{_tv:.1f} $\\pm$ {_te:.1f}',
            ha='center', va='bottom', color='black', fontsize=9,
            fontweight='normal', zorder=6)

ax.set_axisbelow(True)
ax.axhline(0, color='black', lw=1.5, ls='--',zorder=0)
ax.set_xticks(x3)
ax.set_xticklabels([SYS_LABEL.get(t, t) for t in non_bulk_types], fontsize=12)
ax.set_ylabel(r'$p_\mathrm{osm} = p_\mathrm{osm}^{\mathrm{system}} - p_\mathrm{osm}^{\mathrm{bulk}}$ [mM]', fontsize=14)

# auto-extend negative y limit by 15 % to avoid bar/legend overlap
ymin, ymax = ax.get_ylim()
ax.set_ylim(ymin * 1.2, ymax * 1.2)

# Black outline around the combined ideal (Na + Cl + macroion) block
_ideal_cols = ['Na', 'Cl', 'macroion']
_bar_w = bar_segs[0][4].get_width() if bar_segs else 0.8
ylo, yhi = ax.get_ylim()
_txt_pad = (yhi - ylo) * 0.01
for i, t in enumerate(non_bulk_types):
    _ideal_pos = sum(max(delta_mean.loc[t, c], 0) for c in _ideal_cols)
    _ideal_neg = sum(min(delta_mean.loc[t, c], 0) for c in _ideal_cols)
    _ideal_sum = _ideal_pos + _ideal_neg
    _ideal_sem = np.sqrt(sum(
        (delta_std.loc[t, c] if t in delta_std.index else 0.0) ** 2
        for c in _ideal_cols))
    _x_left = x3[i] - _bar_w / 2
    ax.add_patch(mpatches.Rectangle(
    (_x_left, _ideal_neg), _bar_w, _ideal_pos - _ideal_neg,
    fill=False, edgecolor='black', linewidth=1.5, zorder=6))
    _y_text = min(0, _ideal_neg) - _txt_pad
    ax.text(x3[i], _y_text, f'{_ideal_sum:.1f} $\\pm$ {_ideal_sem:.1f}',
            ha='center', va='top', color='black', fontsize=9)

for col, i, val, sem, rect in bar_segs:
    h = rect.get_height()
    label_str = (
    f'$\\mathbf{{{val:.1f}}}$' 
    if col == 'macroion' 
    else f'$\\mathbf{{{val:.1f} \\pm {sem:.1f}}}$'
    )
    if col == 'excess':
        ylo, yhi = ax.get_ylim()
        _pad = (yhi - ylo) * 0.01
        ax.text(x3[i], rect.get_y() + h - _pad, label_str,
                ha='center', va='top', color='white', fontsize=9)
    else:
        ax.text(x3[i], rect.get_y() + h / 2, label_str,
                ha='center', va='center', color='white', fontsize=9)

ax2 = ax.twinx()
ax2.set_ylim(ax.get_ylim())
ax2.yaxis.set_major_locator(ax.yaxis.get_major_locator())
ax2.set_yticklabels([])
ax2.set_ylabel('')

_handles, _labels = ax.get_legend_handles_labels()
_ldict = dict(zip(_labels, _handles))

_ideal_lbls = [r'Na$^+$', r'Cl$^-$', 'Macro-ion']
_other_lbls  = ['Total', 'Excess']

leg_other = ax.legend(
    [_ldict[l] for l in _other_lbls if l in _ldict], _other_lbls,
    frameon=False, fontsize=11, ncol=2,
    loc='upper left', bbox_to_anchor=(0.22, 0.96),
    columnspacing=0.8, handlelength=1.2)

leg_ideal = ax.legend(
    [_ldict[l] for l in _ideal_lbls if l in _ldict], _ideal_lbls,
    title='Ideal', title_fontsize=11,
    frameon=True, edgecolor='black', fancybox=False,
    fontsize=11, ncol=3,
    loc='upper right', bbox_to_anchor=(1.0, 1.0),
    columnspacing=0.8, handlelength=1.2)

ax.add_artist(leg_other)
ax_title(ax, 'Pressure components (ΔP relative to bulk)')
fig.tight_layout()
fig.subplots_adjust(top=0.82)
fig.savefig(plot_dir / 'pressure_stacked.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved pressure_stacked.png")

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
ax.set_xticklabels([SYS_LABEL.get(t, t) for t in _ion_types])
ax.set_ylabel(r'$\langle N \rangle$')
ax.legend(frameon=False, fontsize=12)
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
    c_na_i = molarity(doc, 'Na')
    c_cl_i = molarity(doc, 'Cl')
    I_i    = 0.5 * (c_na_i + c_cl_i)
    try:
        pb = pressure_breakdown(row['folder'])
        p_ideal_i = pb['Na'] + pb['Cl'] + pb['macroion']
        p_ex_i    = pb['excess']
        p_tot_i   = pb['total']
    except Exception:
        p_ideal_i = p_ex_i = p_tot_i = float('nan')
    ionic_rows.append({
        'type': row['type'], 'replicate': row['replicate'],
        '[Na+] / mM':   c_na_i * 1000,
        '[Cl-] / mM':   c_cl_i * 1000,
        'I / mM':       I_i    * 1000,
        'γ_±':          (activity * 1000) / (I_i * 1000) if I_i > 0 else float('nan'),
        'P_ideal / mM': p_ideal_i,
        'P_ex / mM':    p_ex_i,
        'P_tot / mM':   p_tot_i,
        'λ_D / Å':      lD,
    })
ionic_raw = pd.DataFrame(ionic_rows)
_cols     = ['[Na+] / mM', '[Cl-] / mM', 'I / mM', 'γ_±',
             'P_ideal / mM', 'P_ex / mM', 'P_tot / mM', 'λ_D / Å']
_mean_i   = ionic_raw.groupby('type')[_cols].mean()
_std_i    = ionic_raw.groupby('type')[_cols].std().fillna(0)
ionic_sum = _mean_i.copy().astype(object)
for c in _cols:
    ionic_sum[c] = _mean_i[c].map('{:.4f}'.format) + ' ± ' + _std_i[c].map('{:.4f}'.format)
print(ionic_sum.reindex([t for t in SYS_ORDER if t in ionic_sum.index]))

# Save mean ± std table as CSV (separate mean and error columns, ASCII headers)
_csv_col_map = {
    '[Na+] / mM':   'cNa_mM',
    '[Cl-] / mM':   'cCl_mM',
    'I / mM':       'I_mM',
    'γ_±':          'gamma_pm',
    'P_ideal / mM': 'P_ideal_mM',
    'P_ex / mM':    'P_ex_mM',
    'P_tot / mM':   'P_tot_mM',
    'λ_D / Å':      'lambda_D_A',
}
_csv_rows = []
for t in [t for t in SYS_ORDER if t in _mean_i.index]:
    row_out = {'type': t}
    for c in _cols:
        key = _csv_col_map[c]
        row_out[key]          = _mean_i.loc[t, c]
        row_out[key + '_err'] = _std_i.loc[t, c]
    _csv_rows.append(row_out)
pd.DataFrame(_csv_rows).to_csv(plot_dir / 'ionic_pressure_table.csv', index=False, float_format='%.6g')
print("Saved ionic_pressure_table.csv")

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
ax.set_xlabel('r [Å]', fontsize=14)
ax.set_ylabel('g(r)', fontsize=14)
ax.legend(frameon=False, fontsize=12)
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
    ax.set_xlabel('r [Å]', fontsize=14)
    ax.set_ylabel(f'{label}  g(r)', fontsize=13)
    ax.legend(frameon=False, fontsize=11)

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
            ax.set_ylabel(r'$U_\mathrm{total}$ / $k_\mathrm{B}T$', fontsize=13)
        ax.set_xlabel('MC step', fontsize=13)
        ax.set_title(f'{sys_type} — replica {col_idx + 1}', fontsize=12)
        ax.xaxis.set_major_formatter(step_fmt)

fig_title(fig, 'Energy vs MC steps — convergence check', fontweight='bold')
fig.tight_layout()
fig.savefig(plot_dir / 'energy_convergence.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved energy_convergence.png")

print("\nDone.")
