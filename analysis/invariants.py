"""Structural and thermodynamic invariants for matching implicit- to explicit-ion runs.

Each invariant is an integral over the centre-of-mass pair structure that a correct
implicit-ion model must reproduce:

- KB:  G = 4π∫[g−1] r² dr, the Kirkwood–Buff integral. Fixes the osmotic compressibility,
       β(∂Π/∂ρ) = 1/(1+ρG), and needs only the orientation-averaged g(r).
- mu:  4π∫ g ⟨μ̂ᵢ·μ̂ⱼ⟩_r r² dr, the l=1 projection; ρ·mu = g_K − 1 (Kirkwood factor).
- Q:   4π∫ g ⟨Φ̂ᵢ:Φ̂ⱼ⟩_r r² dr, the analogous quadrupolar (l=2) projection.

All integrals are truncated at L with the finite-volume weight of Krüger et al.
(J. Phys. Chem. Lett. 4, 235 (2013)); a plain cut does not converge because g(r) tails
below 1 in a closed box.
"""

import gzip
import re
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel


def finite_size_weight(x):
    return 1 - 1.5 * x + 0.5 * x**3


def weighted_integral(r, f, L):
    """4π∫₀ᴸ f(r) r² w(r/L) dr."""
    m = r <= L
    return 4 * np.pi * np.trapezoid(f[m] * r[m] ** 2 * finite_size_weight(r[m] / L), r[m])


def kb_integral(r, g, L):
    return weighted_integral(r, g - 1, L)


def rmsd_r2(r, dg):
    """Shell-volume weighted RMSD, [<r² Δg²>]^½; ∝ [∫Δg² r² dr]^½ on a uniform grid."""
    return np.sqrt(np.mean(r**2 * dg**2))


# Cached: every analysis reads the same few hundred files several times. Callers must not
# mutate the returned frames.
@cache
def _read_csv(path):
    path = Path(path)
    gz = path.with_name(path.name + ".gz")
    if gz.exists():
        with gzip.open(gz, "rt") as f:
            return pd.read_csv(f)
    return pd.read_csv(path) if path.exists() else None


def load_rdf(folder):
    return _read_csv(Path(folder) / "rdf.csv")


def load_multipole(folder):
    return _read_csv(Path(folder) / "multipole_dist.csv")


def mean_sem(values, axis=None):
    """Replica mean and standard error (ddof=1; zero for a single replica), ignoring NaN."""
    v = np.asarray(values, dtype=float)
    if axis is None:
        v = v[np.isfinite(v)]
        if len(v) == 0:
            return np.nan, np.nan
    n = v.shape[0] if axis is not None else len(v)
    sem = v.std(axis=axis, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(v.mean(axis=axis))
    return v.mean(axis=axis), sem


def replica_mean_sem(folders, load_fn, col, x=None):
    """(x, mean, sem) of column col across replica folders, interpolated onto x.

    x defaults to the first replica's grid.
    """
    dfs = [df for df in map(load_fn, folders) if df is not None and col in df.columns]
    if not dfs:
        return None, None, None
    if x is None:
        x = dfs[0].iloc[:, 0].values
    arr = np.array([np.interp(x, df.iloc[:, 0].values, df[col].values) for df in dfs])
    return (x, *mean_sem(arr, axis=0))


def percus_yevick_gr(r, sigma, phi, q_max=5.0, n_q=50000):
    """
    PY hard-sphere g(r) via sine-transform of the analytical S(q).

    g(r) - 1 = 1/(2π²ρ) ∫ q² [S(q)-1] sin(qr)/(qr) dq
    """
    rho = 6.0 * phi / (np.pi * sigma**3)
    q = np.linspace(1e-6, q_max, n_q)
    # analytical PY S(q)
    x = q * sigma
    a = (1 + 2 * phi) ** 2 / (1 - phi) ** 4
    b = -6 * phi * (1 + phi / 2) ** 2 / (1 - phi) ** 4
    g = phi * a / 2
    A = a * (np.sin(x) - x * np.cos(x)) / x**3
    B = b * (2 * x * np.sin(x) - (x**2 - 2) * np.cos(x) - 2) / x**4
    G = (
        g
        * (-(x**4) * np.cos(x) + 4 * ((3 * x**2 - 6) * np.cos(x) + (x**3 - 6 * x) * np.sin(x) + 6))
        / x**6
    )
    Sq = 1.0 / (1.0 + 24.0 * phi * (A + B + G))
    lorch = np.sinc(q / q_max)  # Lorch window: sin(πq/q_max)/(πq/q_max)
    r = np.asarray(r, dtype=float)
    gr = np.array(
        [
            1.0
            + np.trapezoid(q**2 * (Sq - 1.0) * np.sin(q * ri) / (q * ri) * lorch, q)
            / (2.0 * np.pi**2 * rho)
            for ri in r
        ]
    )
    gr[r < sigma] = 0.0
    return gr


BJERRUM = 7.15  # Å; water at 298.15 K, εr = 78.36, as in the Faunus runs


def debye_length(molar):
    """Debye length [Å] of a 1:1 salt at the given molarity."""
    return 1 / np.sqrt(8 * np.pi * BJERRUM * molar * 6.02214076e-4)


def molarity(folder, atomtype):
    """Mean molarity of atomtype over the box, from Faunus' collective-variable output.

    For Cl⁻ this is the free salt left after Donnan partitioning: every Cl⁻ pairs with a free
    Na⁺, and the remaining Na⁺ neutralise the macroions.
    """
    text = (Path(folder) / "output.yaml").read_text()
    m = re.search(
        rf"property: Molarity\s+description: 'selection: atomtype {atomtype}'\s+"
        r"num_samples: \d+\s+mean: ([\d.eE+-]+)",
        text,
    )
    return float(m.group(1)) if m else np.nan


def implicit_groups(implicit_dir, sys_type):
    """{κ: [replica folders]} for one variant's implicit runs, sorted by κ.

    Includes the *_exp runs (κ from the explicit-ion ionic strength): their inputs match
    the scan apart from debye_length, so they extend it to short Debye lengths.
    """
    pat = re.compile(rf"^{re.escape(sys_type)}-imp_([\d.]+?)(?:_exp)?(?:_\d+)?$")
    groups = {}
    for f in sorted((Path(implicit_dir) / sys_type).glob(f"{sys_type}-imp_*")):
        m = pat.match(f.name)
        if m and f.is_dir():
            groups.setdefault(float(m.group(1)), []).append(f)
    return dict(sorted(groups.items()))


def explicit_folders(explicit_dir, sys_type):
    pat = re.compile(rf"^{re.escape(sys_type)}-exp(?:_\d+)?$")
    return [f for f in sorted(Path(explicit_dir).iterdir()) if f.is_dir() and pat.match(f.name)]


def common_cutoff(folders):
    """Largest L covered by every rdf.csv and multipole_dist.csv among folders."""
    L = np.inf
    for f in folders:
        rdf, mp = load_rdf(f), load_multipole(f)
        if rdf is not None:
            L = min(L, rdf.iloc[:, 0].max())
        if mp is not None:
            L = min(L, mp["R"].max())
    return L


def replica_invariants(folder, L):
    """Return {'KB', 'mu', 'Q'} for one replica folder; missing data give NaN."""
    rdf, mp = load_rdf(folder), load_multipole(folder)
    out = {"KB": np.nan, "mu": np.nan, "Q": np.nan}
    if rdf is None:
        return out
    r, g = rdf.iloc[:, 0].values, rdf["g(r)"].values
    out["KB"] = kb_integral(r, g, L)
    if mp is not None:
        R = mp["R"].values
        g_R = np.interp(R, r, g)
        out["mu"] = weighted_integral(R, g_R * mp["mucorr"].values, L)
        out["Q"] = weighted_integral(R, g_R * mp["quadcorr_norm"].values, L)
    return out


def delta_table(groups, exp_folders, keys, L=None):
    """Per-κ ΔX = X_imp − X_exp and its SEM for each invariant in keys.

    Columns: kappa, debye, n, d<key>, s<key>, plus the implicit mean and SEM alone, x<key>
    and e<key>; df.attrs["exp"] holds the explicit (mean, SEM) per key. L defaults to the
    common cutoff of all runs.
    A single-replica κ has no SEM of its own; it gets the pooled per-run SD of the
    replicated κ, so the GP does not treat it as exact.
    """
    if L is None:
        L = common_cutoff([f for fs in groups.values() for f in fs] + exp_folders)
    exp_inv = [replica_invariants(f, L) for f in exp_folders]
    exp_stats = {k: mean_sem([d[k] for d in exp_inv]) for k in keys}
    imp_inv = {kappa: [replica_invariants(f, L) for f in fs] for kappa, fs in groups.items()}
    pooled_sd = {
        k: np.sqrt(
            np.nanmean(
                [np.var([d[k] for d in inv], ddof=1) for inv in imp_inv.values() if len(inv) > 1]
            )
        )
        for k in keys
    }
    rows = []
    for kappa, inv in imp_inv.items():
        row = {"kappa": kappa, "debye": 1 / kappa, "n": len(inv)}
        for k in keys:
            mi, si = mean_sem([d[k] for d in inv])
            if len(inv) == 1:
                si = pooled_sd[k]
            me, se = exp_stats[k]
            row[f"d{k}"], row[f"s{k}"] = mi - me, np.hypot(si, se)
            row[f"x{k}"], row[f"e{k}"] = mi, si
        rows.append(row)
    df = pd.DataFrame(rows)
    df.attrs["exp"] = exp_stats
    return df


def gp_curve(x, y, yerr, grid):
    """Latent mean and sd of y(x) on grid from a GP with per-point noise yerr.

    A fitted white-noise term absorbs scatter that replica SEMs (n=3) underestimate; it is
    then folded into the noise and excluded from the returned latent curve, so roots and
    intervals reflect the smooth trend rather than per-run noise. y is normalised by hand
    so the noise and the data are in the same units.
    """
    x = np.asarray(x, float)[:, None]
    y, yerr = np.asarray(y, float), np.asarray(yerr, float)
    m, s = y.mean(), y.std() or 1.0
    noise = np.maximum((yerr / s) ** 2, 1e-12)
    span = np.ptp(x) or 1.0
    # Length-scale floor of span/4: ΔX(λ) is smooth, and shorter scales chase replica noise
    signal = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(span / 2, (span / 4, span * 5))
    fit = GaussianProcessRegressor(
        signal + WhiteKernel(1e-2, (1e-6, 1e1)),
        alpha=noise,
        n_restarts_optimizer=5,
        random_state=0,
    ).fit(x, (y - m) / s)
    latent = GaussianProcessRegressor(
        fit.kernel_.k1, alpha=noise + fit.kernel_.k2.noise_level, optimizer=None
    ).fit(x, (y - m) / s)
    mu, sd = latent.predict(grid[:, None], return_std=True)
    return latent, m + s * mu, s * sd, (m, s)


def zero_crossings(x, y):
    """Linear-interpolated roots of y(x)."""
    s = np.sign(y)
    idx = np.where(s[:-1] * s[1:] < 0)[0]
    return x[idx] - y[idx] * (x[idx + 1] - x[idx]) / (y[idx + 1] - y[idx])


def kappa_star(debye, dmean, dsem, ref_sem=0.0, n_samples=500, n_grid=400, seed=0):
    """Debye length where ΔX = 0, with a 68% interval from GP posterior samples.

    dsem is the implicit-run SEM only: the explicit reference is subtracted at every λ, so
    its SEM (ref_sem) is one shared offset per posterior sample, not independent noise.
    Returns dict with lam (posterior-mean root), lo/hi, bracketed, frac_bracketed (share of
    samples with a root inside the scan), and the grid with latent mean/sd.
    """
    debye, dmean, dsem = (np.asarray(a, float) for a in (debye, dmean, dsem))
    ok = np.isfinite(dmean)
    debye, dmean, dsem = debye[ok], dmean[ok], dsem[ok]
    grid = np.linspace(debye.min(), debye.max(), n_grid)
    gp, mu, sd, (m, s) = gp_curve(debye, dmean, dsem, grid)

    roots = zero_crossings(grid, mu)
    # Several roots: keep the one where the mean is flattest in |ΔX| sense, i.e. nearest argmin
    lam = (
        roots[np.argmin(np.abs(roots - grid[np.argmin(np.abs(mu))]))]
        if len(roots)
        else grid[np.argmin(np.abs(mu))]
    )

    rng = np.random.default_rng(seed)
    samples = m + s * gp.sample_y(grid[:, None], n_samples, random_state=seed)
    samples += rng.normal(0.0, ref_sem, n_samples)
    sample_roots = []
    for s in samples.T:
        rs = zero_crossings(grid, s)
        if len(rs):
            sample_roots.append(rs[np.argmin(np.abs(rs - lam))])
    frac = len(sample_roots) / n_samples
    lo, hi = np.percentile(sample_roots, [16, 84]) if sample_roots else (np.nan, np.nan)
    return {
        "lam": lam,
        "lo": lo,
        "hi": hi,
        "bracketed": len(roots) > 0,
        "frac_bracketed": frac,
        "grid": grid,
        "mu": mu,
        "sd": sd,
        "sd_ref": np.hypot(sd, ref_sem),  # including the shared reference
    }
