#!/usr/bin/env python3
"""
Topographic visualization of Faunus SDF (.dx) files.

Each (lon, lat) direction from the NP centre is assigned a scalar ρ derived
from the 3-D volumetric grid by one of three projection modes:
  shell      : sample ρ at a fixed radius r_ref  [default]
  peak       : maximum ρ along each ray in [r_min, r_max]
  integrated : ∫ρ(r) dr along each ray in [r_min, r_max]

Two display projections (--projection):
  sphere     : 3-D bumped globe — colour + surface displacement encode ρ [default]
  mollweide  : 2-D Mollweide equal-area map with hill-shading — shows the full
               sphere at once with topographic texture; no information hidden

All panels share one colour scale (global vmin/vmax across all species) so
relative density differences between species are directly comparable.
Panels are arranged in a fixed 2×2 grid.

Input: individual .dx files, a directory (all sdf-*.dx inside it), or both.

Usage
  python plot_sdf_sphere.py /path/to/dipolar-exp --projection mollweide
  python plot_sdf_sphere.py sdf-na.dx sdf-cl.dx --mode peak
  python plot_sdf_sphere.py /path/to/dir sdf-extra.dx --cmap RdBu_r --vcenter 1.0
"""

import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.colors import LightSource
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline
from scipy.ndimage import gaussian_filter


# ── Species label map ─────────────────────────────────────────────────────────

SPECIES_LABELS = {
    "sdf-na": r"Na$^+$",
    "sdf-cl": r"Cl$^-$",
    "sdf-np": "NP",
    "sdf-pp": "PP",
}


def stem_label(stem: str) -> str:
    return SPECIES_LABELS.get(stem, stem)


# ── DX parser ─────────────────────────────────────────────────────────────────

def read_dx(path: str):
    """Parse an OpenDX regular-grid file (Faunus SDF format).

    Returns
    -------
    coords : (x, y, z)  — three 1-D position arrays in Å
    data   : (nx, ny, nz) float array  — relative density, C-order (z fastest)
    """
    with open(path) as fh:
        lines = fh.readlines()

    nx = ny = nz = None
    origin = None
    deltas = []
    data_start = None

    for i, raw in enumerate(lines):
        line = raw.strip()
        if line.startswith("object 1"):
            m = re.search(r"counts\s+(\d+)\s+(\d+)\s+(\d+)", line)
            nx, ny, nz = int(m.group(1)), int(m.group(2)), int(m.group(3))
        elif line.startswith("origin"):
            origin = np.array([float(v) for v in line.split()[1:4]])
        elif line.startswith("delta"):
            deltas.append([float(v) for v in line.split()[1:4]])
        elif "data follows" in line:
            data_start = i + 1
            break

    tokens = " ".join(lines[data_start:]).split()
    flat = np.array([float(t) for t in tokens[: nx * ny * nz]], dtype=float)
    data = flat.reshape((nx, ny, nz))

    dx, dy, dz = deltas[0][0], deltas[1][1], deltas[2][2]
    x = origin[0] + np.arange(nx) * dx
    y = origin[1] + np.arange(ny) * dy
    z = origin[2] + np.arange(nz) * dz
    return (x, y, z), data


def collect_dx_paths(inputs: list[str]) -> list[Path]:
    """Expand directories to sdf-*.dx globs; keep individual .dx paths as-is."""
    paths = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            found = sorted(p.glob("sdf-*.dx"))
            if not found:
                print(f"  Warning: no sdf-*.dx files found in {p}")
            paths.extend(found)
        elif p.suffix == ".dx":
            paths.append(p)
        else:
            print(f"  Warning: skipping {p} (not a .dx file or directory)")
    return paths


# ── Angular sampling ──────────────────────────────────────────────────────────

def make_interpolator(coords, data):
    return RegularGridInterpolator(
        coords, data, method="linear", bounds_error=False, fill_value=0.0
    )


def _xyz(r, theta_g, phi_g):
    x = r * np.sin(theta_g) * np.cos(phi_g)
    y = r * np.sin(theta_g) * np.sin(phi_g)
    z = r * np.cos(theta_g)
    return np.stack([x, y, z], axis=-1)


def sample_shell(interp, theta_g, phi_g, r_ref: float) -> np.ndarray:
    return interp(_xyz(r_ref, theta_g, phi_g))


def sample_peak(interp, theta_g, phi_g, r_min: float, r_max: float,
                n: int = 80) -> np.ndarray:
    rho = np.zeros(theta_g.shape)
    for r in np.linspace(r_min, r_max, n):
        rho = np.maximum(rho, interp(_xyz(r, theta_g, phi_g)))
    return rho


def sample_integrated(interp, theta_g, phi_g, r_min: float, r_max: float,
                      n: int = 80) -> np.ndarray:
    r_vals = np.linspace(r_min, r_max, n)
    stack = np.array([interp(_xyz(r, theta_g, phi_g)) for r in r_vals])
    return np.trapz(stack, r_vals, axis=0)


# ── Shared norm builder ───────────────────────────────────────────────────────

def build_norm(vmin, vmax, vcenter):
    if vcenter is not None:
        vc = float(np.clip(vcenter, vmin + 1e-6, vmax - 1e-6))
        return mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vc, vmax=vmax)
    return mcolors.Normalize(vmin=vmin, vmax=vmax)


# ── 3-D bumped sphere ─────────────────────────────────────────────────────────

def draw_sphere(ax, theta_g, phi_g, rho, r_base, bump_frac, cmap, norm,
                elev, azim, smooth_sigma, title):
    if smooth_sigma > 0:
        rho = gaussian_filter(rho, sigma=smooth_sigma)

    excess = np.clip(rho - 1.0, 0, None)
    excess_norm = excess / max(excess.max(), 1e-9)
    r_surf = r_base * (1.0 + bump_frac * excess_norm)
    x = r_surf * np.sin(theta_g) * np.cos(phi_g)
    y = r_surf * np.sin(theta_g) * np.sin(phi_g)
    z = r_surf * np.cos(theta_g)

    face_rgba = plt.get_cmap(cmap)(norm(rho))
    ls = LightSource(azdeg=315, altdeg=45)
    shaded_rgb = ls.shade_rgb(face_rgba[..., :3], r_surf,
                               blend_mode="soft", fraction=0.6)
    shaded_rgba = np.dstack([shaded_rgb, face_rgba[..., 3]])

    ax.plot_surface(x, y, z, facecolors=shaded_rgba,
                    rstride=1, cstride=1, antialiased=False, shade=False)

    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    xs = r_base * np.outer(np.cos(u), np.sin(v))
    ys = r_base * np.outer(np.sin(u), np.sin(v))
    zs = r_base * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, color="silver", alpha=0.15, shade=True,
                    rstride=2, cstride=2, linewidth=0)

    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=15, pad=2)


# ── Mollweide hillshading ─────────────────────────────────────────────────────

def _moll_forward(lat, lon):
    """Mollweide forward projection: (lat, lon) → (x, y)."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    lat, lon = np.broadcast_arrays(lat, lon)
    t = lat.copy()
    for _ in range(12):
        f  = 2.0 * t + np.sin(2.0 * t) - np.pi * np.sin(lat)
        df = np.where(np.abs(2.0 + 2.0 * np.cos(2.0 * t)) < 1e-12,
                      1e-12, 2.0 + 2.0 * np.cos(2.0 * t))
        t -= f / df
    return (2.0 * np.sqrt(2.0) / np.pi) * lon * np.cos(t), np.sqrt(2.0) * np.sin(t)


def mollweide_rasterize(rho_map, theta_1d, phi_1d, out_W=800, out_H=400):
    """Project an (H, W) angular density map onto a Mollweide pixel grid."""
    # RectBivariateSpline needs strictly increasing first axis
    lat_1d = (np.pi / 2.0 - theta_1d)[::-1]
    rho_flip = rho_map[::-1]
    rbs = RectBivariateSpline(lat_1d, phi_1d, rho_flip, kx=3, ky=3)

    xs = np.linspace(-2.0 * np.sqrt(2), 2.0 * np.sqrt(2), out_W)
    ys = np.linspace(-np.sqrt(2), np.sqrt(2), out_H)
    x_img, y_img = np.meshgrid(xs, ys)

    sin_t = np.clip(y_img / np.sqrt(2), -1.0, 1.0)
    t = np.arcsin(sin_t)
    lat_img = np.arcsin(np.clip((2.0 * t + np.sin(2.0 * t)) / np.pi, -1.0, 1.0))
    cos_t = np.cos(t)
    with np.errstate(invalid="ignore", divide="ignore"):
        lon_img = np.where(np.abs(cos_t) > 1e-10,
                           np.pi * x_img / (2.0 * np.sqrt(2.0) * cos_t), 0.0)

    # Proper Mollweide ellipse: (x/2√2)² + (y/√2)² ≤ 1
    inside = (x_img ** 2 / 8.0 + y_img ** 2 / 2.0) <= 1.0

    eps = 1e-6
    lat_q = np.clip(lat_img[inside], lat_1d[0] + eps, lat_1d[-1] - eps)
    lon_q = np.clip(lon_img[inside], phi_1d[0] + eps, phi_1d[-1] - eps)

    rho_img = np.full((out_H, out_W), np.nan)
    rho_img[inside] = rbs(lat_q, lon_q, grid=False)
    return rho_img, inside


def draw_mollweide(ax, theta_1d, phi_1d, rho_map, cmap, norm,
                   vert_exag, smooth_sigma, title, out_W=800, out_H=400):
    """Mollweide map with LightSource hill-shading — full sphere, no hidden hemisphere."""
    if smooth_sigma > 0:
        rho_map = gaussian_filter(rho_map, sigma=smooth_sigma)

    rho_img, inside = mollweide_rasterize(rho_map, theta_1d, phi_1d, out_W, out_H)

    vmin = norm.vmin if hasattr(norm, "vmin") else rho_img[inside].min()
    rho_fill = np.where(np.isnan(rho_img), vmin, rho_img)

    ls = LightSource(azdeg=225, altdeg=45)
    rgba = ls.shade(rho_fill, cmap=plt.get_cmap(cmap), norm=norm,
                    vert_exag=vert_exag, blend_mode="soft")
    rgba[~inside, 3] = 0.0

    extent = [-2.0 * np.sqrt(2), 2.0 * np.sqrt(2), -np.sqrt(2), np.sqrt(2)]
    ax.imshow(rgba, origin="lower", extent=extent, aspect="equal",
              interpolation="bilinear", zorder=1)

    from matplotlib.patches import Ellipse
    ell = Ellipse((0, 0), width=4.0 * np.sqrt(2), height=2.0 * np.sqrt(2),
                  fill=False, edgecolor="0.2", linewidth=1.0, zorder=3)
    ax.add_patch(ell)

    ax.set_xlim(-2.0 * np.sqrt(2) * 1.02, 2.0 * np.sqrt(2) * 1.02)
    ax.set_ylim(-np.sqrt(2) * 1.05, np.sqrt(2) * 1.05)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_axis_off()

    if title:
        ax.text(0.03, 0.95, title, transform=ax.transAxes, fontsize=15,
                va="top", ha="left",)
                #bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75))


# ── Figure assembly ───────────────────────────────────────────────────────────

CBAR_LABELS = {
    "shell":      "Relative density  ρ",
    "peak":       "Peak relative density  ρ",
    "integrated": "Integrated density  ∫ρ dr  [Å]",
}


def make_figure(results, args):
    N = len(results)
    if N > 4:
        print(f"  Warning: {N} files found; only the first 4 will be plotted.")
        results = results[:4]
        N = 4

    # Global colour scale across all species
    all_rho = np.concatenate([r["rho"].ravel() for r in results])
    norm = build_norm(all_rho.min(), all_rho.max(), args.vcenter)

    is_3d = (args.projection == "sphere")

    # Choose 1×2 or 2×2 grid depending on panel count
    ncols = 2
    nrows = 1 if N <= 2 else 2

    # Figure size: Mollweide panels are 2:1 wide, spheres are square
    pw, ph = (4.5, 4.5) if is_3d else (5.5, 3.2)
    fig_h = ph * nrows + 0.6
    CBAR_FRAC = 1.2 / fig_h   # fixed ~1.2 inch absolute strip for colorbar + label
    fig = plt.figure(figsize=(pw * ncols, fig_h), dpi=args.dpi)

    gs = gridspec.GridSpec(
        nrows, ncols,
        figure=fig,
        hspace=0.05 if not is_3d else 0.0,
        wspace=0.05,
        top=0.97,
        bottom=CBAR_FRAC,
        left=0.02,
        right=0.98,
    )

    proj_kw = {"projection": "3d"} if is_3d else {}
    letters = "abcdefgh"
    axes = []
    for i, res in enumerate(results):
        row, col = divmod(i, ncols)
        ax = fig.add_subplot(gs[row, col], **proj_kw)
        axes.append(ax)
        panel_title = fr"$\mathbf{{({letters[i]})}}$ {res['label']}"
        if is_3d:
            draw_sphere(ax, res["theta_g"], res["phi_g"], res["rho"],
                        r_base=args.r_ref, bump_frac=args.bump,
                        cmap=args.cmap, norm=norm,
                        elev=args.elev, azim=args.azim,
                        smooth_sigma=args.smooth, title=panel_title)
        else:
            draw_mollweide(ax, res["theta_1d"], res["phi_1d"], res["rho"],
                           cmap=args.cmap, norm=norm,
                           vert_exag=args.vert_exag,
                           smooth_sigma=args.smooth, title=panel_title)

    # Hide unused slots in the last row (only relevant for 2×2 with 3 panels)
    for i in range(N, nrows * ncols):
        row, col = divmod(i, ncols)
        fig.add_subplot(gs[row, col]).set_visible(False)

    # Shared colorbar — span the full width of the panel grid.
    # canvas.draw() freezes GridSpec positions so get_position() is accurate.
    fig.canvas.draw()
    visible_axes = [ax for ax in axes if ax.get_visible()]
    x0_cb = min(ax.get_position().x0 for ax in visible_axes)
    x1_cb = max(ax.get_position().x1 for ax in visible_axes)

    cbar_h = 0.028
    cbar_y = CBAR_FRAC * 0.55   # sits in the upper portion of the reserved strip
    cax = fig.add_axes([x0_cb, cbar_y, x1_cb - x0_cb, cbar_h])

    import math
    sm = plt.cm.ScalarMappable(cmap=args.cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")

    # Equidistant ticks in each half for TwoSlopeNorm
    if isinstance(norm, mcolors.TwoSlopeNorm):
        n_half = 3
        left  = np.linspace(norm.vmin,    norm.vcenter, n_half + 1)
        right = np.linspace(norm.vcenter, norm.vmax,    n_half + 1)[1:]
        tick_vals = np.concatenate([left, right])
    else:
        rng   = norm.vmax - norm.vmin
        step  = max(1, round(rng / 5))
        start = math.floor(norm.vmin / step) * step
        tick_vals = np.arange(start, norm.vmax + step * 0.5, step)

    cbar.set_ticks(tick_vals)
    cbar.set_ticklabels(
        [f"{v:.0f}" if v == int(v) else f"{v:.1f}" for v in tick_vals],
        fontsize=15,
    )
    cbar.set_label(CBAR_LABELS[args.mode], fontsize=17, labelpad=4)
    cbar.ax.tick_params(axis="x", length=4, direction="out", width=0.8, labelsize=15)
    for spine in ["top", "left", "right"]:
        cbar.ax.spines[spine].set_visible(False)
    cbar.ax.spines["bottom"].set_linewidth(0.8)

    return fig


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Topographic SDF visualization from Faunus .dx files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "inputs", nargs="+",
        help=".dx file(s) or directory containing sdf-*.dx files (or both)",
    )

    g = p.add_argument_group("Angular projection")
    g.add_argument("--mode", choices=["shell", "peak", "integrated"],
                   default="shell")
    g.add_argument("--r-ref", type=float, default=24.0,
                   help="Shell radius [Å] (mode=shell)")
    g.add_argument("--r-min", type=float, default=20.0,
                   help="Inner ray radius [Å] (mode=peak/integrated)")
    g.add_argument("--r-max", type=float, default=27.0,
                   help="Outer ray radius [Å] (mode=peak/integrated)")

    g = p.add_argument_group("Display projection")
    g.add_argument("--projection", choices=["sphere", "mollweide"],
                   default="mollweide",
                   help="'sphere' = 3-D globe; 'mollweide' = full-sphere 2-D map")
    g.add_argument("--bump", type=float, default=0.20,
                   help="[sphere] Topographic bump as fraction of r_ref")
    g.add_argument("--elev", type=float, default=25.0,
                   help="[sphere] View elevation angle [°]")
    g.add_argument("--azim", type=float, default=-60.0,
                   help="[sphere] View azimuth angle [°]")
    g.add_argument("--vert-exag", type=float, default=8.0,
                   help="[mollweide] Vertical exaggeration for hill-shading")

    g = p.add_argument_group("Colour scale")
    g.add_argument("--cmap", default="hot_r")
    g.add_argument("--vcenter", type=float, default=None,
                   help="Diverging norm pivot (e.g. 1.0 = bulk); use with RdBu_r")

    g = p.add_argument_group("Common")
    g.add_argument("--smooth", type=float, default=1.5,
                   help="Gaussian smoothing σ on angular map (0 = off)")
    g.add_argument("--resolution", "-r", type=int, default=150,
                   help="Latitude grid points (longitude = 2×)")
    g.add_argument("--dpi", type=int, default=200)
    g.add_argument("-o", "--output", default=None)
    args = p.parse_args()

    dx_paths = collect_dx_paths(args.inputs)
    if not dx_paths:
        p.error("No .dx files found.")

    H = args.resolution
    W = 2 * H
    theta = np.linspace(1e-4, np.pi - 1e-4, H)
    phi   = np.linspace(-np.pi, np.pi, W)
    phi_g, theta_g = np.meshgrid(phi, theta)

    results = []
    for path in dx_paths:
        coords, data = read_dx(path)
        interp = make_interpolator(coords, data)

        if args.mode == "shell":
            rho = sample_shell(interp, theta_g, phi_g, args.r_ref)
        elif args.mode == "peak":
            rho = sample_peak(interp, theta_g, phi_g, args.r_min, args.r_max)
        else:
            rho = sample_integrated(interp, theta_g, phi_g, args.r_min, args.r_max)

        stem  = path.stem
        label = stem_label(stem)
        print(f"{stem:12s}  ρ ∈ [{rho.min():.3f}, {rho.max():.3f}]")
        results.append({"label": label,
                         "theta_g": theta_g, "phi_g": phi_g,
                         "theta_1d": theta, "phi_1d": phi,
                         "rho": rho})

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 16})
    fig = make_figure(results, args)

    # Resolve output path: explicit -o flag wins; else use directory basename;
    # else fall back to panel_sdf.png
    if args.output:
        outpath = Path(args.output)
    else:
        dir_names = [Path(i).resolve().name for i in args.inputs if Path(i).is_dir()]
        stem = dir_names[0] if dir_names else "panel_sdf"
        outpath = Path(stem + ".png")

    fig.savefig(outpath, dpi=args.dpi)
    print(f"Saved → {outpath}")


if __name__ == "__main__":
    main()
