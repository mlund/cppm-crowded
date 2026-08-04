#!/usr/bin/env python3
"""Mollweide panel comparing surface charge density across multiple NP configurations.

All panels share a common colour scale so differences in surface charge density
are directly comparable across NPs.  For exactly 3 panels the --layout flag
selects a triangular arrangement that uses publication space more efficiently.
"""

import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path

CHARGES = {"NP": -1.0, "PP": +1.0}

PANEL_WIDTH = 7.0   # inches — double-column width
DPI         = 300


def parse_xyz(path):
    atoms = {}
    with open(path) as f:
        n = int(f.readline())
        f.readline()
        for _ in range(n):
            parts = f.readline().split()
            atoms.setdefault(parts[0], []).append([float(v) for v in parts[1:4]])
    return {k: np.array(v) for k, v in atoms.items()}


def to_spherical(coords):
    c = coords / np.linalg.norm(coords, axis=1, keepdims=True)
    return np.arctan2(c[:, 1], c[:, 0]), np.arcsin(np.clip(c[:, 2], -1.0, 1.0))


def surface_charge_density(lon_g, lat_g, lon_a, lat_a, charges, lam, R_nm):
    """Exact vMF surface charge density in e/Å²."""
    lo_a = lon_a[:, None, None]
    la_a = lat_a[:, None, None]
    q    = charges[:, None, None]

    cos_gamma = (np.sin(la_a) * np.sin(lat_g[None])
                 + np.cos(la_a) * np.cos(lat_g[None]) * np.cos(lon_g[None] - lo_a))

    kernel = (lam / (2.0 * np.pi)) * np.exp(lam * (cos_gamma - 1.0)) / (1.0 - np.exp(-2.0 * lam))

    return np.sum(q * kernel, axis=0) / R_nm**2 / 100.0  # e/Å²


def label_from_stem(stem, letter):
    """Return '(a) $P^{1}_{8}$'-style label parsed from a filename stem like 'P18' or 'ref_P18'."""
    m = re.search(r'[Pp](\d+)', stem)
    if m:
        digits = m.group(1)
        sup, sub = digits[0], digits[1:]
        body = fr"P^{{{sup}}}_{{{sub}}}" if sub else fr"P^{{{sup}}}"
        return fr"$\mathbf{{({letter})}}$ ${body}$"
    return fr"$\mathbf{{({letter})}}$"


def process_file(path, lam, radius, resolution):
    atoms = parse_xyz(path)

    if radius is None:
        all_coords = np.vstack([v for k, v in atoms.items() if k != "MP"])
        R_nm = np.mean(np.linalg.norm(all_coords, axis=1)) / 10.0
    else:
        R_nm = radius

    lons, lats, qs = [], [], []
    p_vec = np.zeros(3)       # dipole accumulator, e·Å
    Q_ten = np.zeros((3, 3))  # quadrupole accumulator, e·Å²
    for kind, coords in atoms.items():
        q = CHARGES.get(kind)
        if q is None:
            continue
        lo, la = to_spherical(coords)
        lons.append(lo)
        lats.append(la)
        qs.extend([q] * len(lo))
        p_vec += q * coords.sum(axis=0)
        for r in coords:
            Q_ten += 0.5 * q * (3.0 * np.outer(r, r) - np.dot(r, r) * np.eye(3))

    if not qs:
        raise ValueError(f"No charged atoms found in {path}")

    lon_a   = np.concatenate(lons)
    lat_a   = np.concatenate(lats)
    charges = np.array(qs)

    W = resolution
    H = resolution // 2
    lon_g, lat_g = np.meshgrid(
        np.linspace(-np.pi, np.pi, W),
        np.linspace(-np.pi / 2, np.pi / 2, H),
    )

    scd    = surface_charge_density(lon_g, lat_g, lon_a, lat_a, charges, lam, R_nm)
    p_mag  = np.linalg.norm(p_vec)          # e·Å
    p_hat  = p_vec / p_mag

    evals, evecs = np.linalg.eigh(Q_ten)
    idx    = np.argmax(np.abs(evals))
    q1_hat = evecs[:, idx]                   # principal quadrupole axis
    q_frob = np.linalg.norm(Q_ten, "fro")    # Frobenius norm, e·Å²

    return lon_g, lat_g, scd, R_nm, float(charges.sum()), p_hat, p_mag, q1_hat, q_frob


def dipole_great_circle(p_hat, n=600):
    """(lon, lat) masked arrays tracing the great circle perpendicular to p_hat."""
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(p_hat, ref)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    u = np.cross(p_hat, ref);  u /= np.linalg.norm(u)
    v = np.cross(p_hat, u)

    t   = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.outer(np.cos(t), u) + np.outer(np.sin(t), v)
    lon = np.arctan2(pts[:, 1], pts[:, 0])
    lat = np.arcsin(np.clip(pts[:, 2], -1.0, 1.0))

    # Mask the point after each ±π wrap so matplotlib draws two separate arcs
    jumps = np.abs(np.diff(lon)) > np.pi
    mask  = np.r_[False, jumps]
    return np.ma.array(lon, mask=mask), np.ma.array(lat, mask=mask)


def draw_dipole_equator(ax, p_hat, p_mag):
    """Overlay dipole equatorial great circle and mu label on a Mollweide axes."""
    lon, lat = dipole_great_circle(p_hat)
    ax.plot(lon, lat, color="k", lw=1.2, ls="--", zorder=5)

    text = fr"$\mu\!=\!{p_mag:.1f}\ e\,\mathrm{{\AA}}$"
    ax.text(-29.0, -1.55, text, fontsize=9, ha="center", va="center", color="k", zorder=6)


def draw_quadrupole_axis(ax, q1_hat, q_frob):
    """Draw solid antipodal triangle markers at the principal quadrupole axis poles."""
    lon1 = np.arctan2(q1_hat[1], q1_hat[0])
    lat1 = np.arcsin(np.clip(q1_hat[2], -1.0, 1.0))
    lon2 = lon1 - np.pi if lon1 > 0 else lon1 + np.pi
    lat2 = -lat1

    ax.plot(lon1, lat1, marker="^", ms=6, mfc="white", mec="k", mew=1.0, ls="none", zorder=7)
    ax.plot(lon2, lat2, marker="v", ms=6, mfc="white", mec="k", mew=1.0, ls="none", zorder=7)

    text = fr"$\Theta\!=\!{q_frob:.1f}\ e\,\mathrm{{\AA}}^2$"
    ax.text(29.0, -1.55, text, fontsize=9, ha="center", va="center", color="k", zorder=6)


def _add_mollweide(fig, spec, label):
    ax = fig.add_subplot(spec, projection="mollweide")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(False)
    ax.text(
        -0.0, 0.99, label,
        transform=ax.transAxes, fontsize=9,
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.6),
    )
    return ax


def build_axes(fig, results, layout):
    """Create Mollweide axes according to *layout* and return them in order."""
    N      = len(results)
    labels = [r[0] for r in results]
    MARGIN = dict(left=0.02, right=0.98)

    if layout == "vertical" or N != 3:
        # ── original stacked layout ───────────────────────────────────────────
        CBAR_PAD  = 0.02
        GS_TOP, GS_BOTTOM = 0.99, 0.12
        available = GS_TOP - GS_BOTTOM
        hspace    = CBAR_PAD * N / (available - CBAR_PAD * (N - 1))
        gs = gridspec.GridSpec(
            N, 1, hspace=hspace,
            top=GS_TOP, bottom=GS_BOTTOM, **MARGIN,
        )
        return [_add_mollweide(fig, gs[i], labels[i]) for i in range(N)]

    # ── triangular layouts (exactly 3 panels) ─────────────────────────────────
    # All three panels are the same half-width size.  A 4-column GridSpec lets
    # us place the two side-by-side panels in cols 0-1 and 2-3, and the lone
    # centred panel in cols 1-2 — identical width, visually centred.
    CBAR_BOTTOM = 0.10
    gs = gridspec.GridSpec(
        2, 4,
        height_ratios=[1, 1],
        hspace=0.00,
        wspace=0.40,
        top=0.99, bottom=CBAR_BOTTOM, **MARGIN,
    )

    if layout == "pyramid":
        # (a) centred on top; (b) and (c) side-by-side on bottom
        axes = [
            _add_mollweide(fig, gs[0, 1:3], labels[0]),
            _add_mollweide(fig, gs[1, 0:2], labels[1]),
            _add_mollweide(fig, gs[1, 2:4], labels[2]),
        ]
    else:  # "inverted" — default for 3 panels
        # (a) and (b) side-by-side on top; (c) centred on bottom
        axes = [
            _add_mollweide(fig, gs[0, 0:2], labels[0]),
            _add_mollweide(fig, gs[0, 2:4], labels[1]),
            _add_mollweide(fig, gs[1, 1:3], labels[2]),
        ]

    return axes


def main():
    parser = argparse.ArgumentParser(
        description="Mollweide panel of surface charge density for multiple NPs."
    )
    parser.add_argument("xyz", nargs="+", help="XYZ files to compare (one panel each)")
    parser.add_argument(
        "--lam", "-l", type=float, default=20.0,
        help="vMF concentration λ (default 20, half-width ~15°, matching Ref. 44)",
    )
    parser.add_argument(
        "--radius", type=float, default=None,
        help="Sphere radius in nm (auto-detected from coordinates if omitted)",
    )
    parser.add_argument(
        "--resolution", "-r", type=int, default=300,
        help="Longitude grid points (latitude = resolution//2, default 300)",
    )
    parser.add_argument(
        "--layout",
        choices=["vertical", "pyramid", "inverted"],
        default="inverted",
        help=(
            "Panel arrangement: 'vertical' stacks all panels (original), "
            "'pyramid' puts one panel on top and two below, "
            "'inverted' puts two panels on top and one below (default for 3 panels)."
        ),
    )
    parser.add_argument("-o", "--output", default=None, help="Save figure to file")
    parser.add_argument(
        "--no-dipole", action="store_true",
        help="Suppress the dipole equatorial great circle overlay",
    )
    parser.add_argument(
        "--no-quadrupole", action="store_true",
        help="Suppress the quadrupole principal axis markers overlay",
    )
    args = parser.parse_args()

    letters = "abcdefghijklmnopqrstuvwxyz"
    panel_labels = [
        label_from_stem(Path(p).stem, letters[i]) for i, p in enumerate(args.xyz)
    ]

    # ── compute SCD for every file ────────────────────────────────────────────
    results = []
    for path, label in zip(args.xyz, panel_labels):
        lon_g, lat_g, scd, R_nm, net_q, p_hat, p_mag, q1_hat, q_frob = process_file(
            path, args.lam, args.radius, args.resolution
        )
        results.append((label, lon_g, lat_g, scd, p_hat, p_mag, q1_hat, q_frob))
        print(f"{label}: R = {R_nm:.3f} nm | net charge = {net_q:.0f} e | mu = {p_mag:.2f} e·Å | Theta_F = {q_frob:.2f} e·Å²")

    # ── shared colour scale ───────────────────────────────────────────────────
    vmax = max(np.percentile(np.abs(r[3]), 98) for r in results)  # r[3] = scd
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    # ── figure size ───────────────────────────────────────────────────────────
    N = len(results)
    layout = args.layout if N == 3 else "vertical"

    if layout == "vertical":
        fig_height = N * (PANEL_WIDTH / 2) + 0.7
    else:  # pyramid or inverted: two equal rows of half-width panels
        fig_height = 2 * (PANEL_WIDTH / 4) + 0.9

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9})
    fig = plt.figure(figsize=(PANEL_WIDTH, fig_height), dpi=DPI)

    axes = build_axes(fig, results, layout)

    # ── plot data ─────────────────────────────────────────────────────────────
    mesh_ref = None
    for ax, (label, lon_g, lat_g, scd, p_hat, p_mag, q1_hat, q_frob) in zip(axes, results):
        mesh = ax.pcolormesh(
            lon_g, lat_g, scd,
            cmap="RdBu_r", norm=norm, shading="auto", rasterized=True,
        )
        if mesh_ref is None:
            mesh_ref = mesh
        if not args.no_dipole:
            draw_dipole_equator(ax, p_hat, p_mag)
        if not args.no_quadrupole:
            draw_quadrupole_axis(ax, q1_hat, q_frob)

    # ── shared colourbar below all panels ────────────────────────────────────
    fig.canvas.draw()
    # Span the colourbar across the full GridSpec width using the outermost axes.
    x0   = min(ax.get_position().x0 for ax in axes)
    x1   = max(ax.get_position().x1 for ax in axes)
    ybot = min(ax.get_position().y0 for ax in axes)

    cbar_height = 0.025
    cbar_pad    = 0.05
    cax = fig.add_axes([x0, ybot - cbar_height - cbar_pad, x1 - x0, cbar_height])
    cbar = fig.colorbar(mesh_ref, cax=cax, orientation="horizontal")
    cbar.set_label(
        r"Surface charge density, $\sigma\;[e\,\mathrm{\AA}^{-2}]$",
        fontsize=9, labelpad=6,
    )

    if args.output:
        plt.savefig(args.output, dpi=DPI, bbox_inches="tight")
        print(f"Saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
