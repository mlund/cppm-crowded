#!/usr/bin/env python3
"""Mollweide panel comparing surface charge density across multiple NP configurations.

All panels share a common colour scale so differences in surface charge density
are directly comparable across NPs.  For exactly 3 panels the --layout flag
selects a triangular arrangement that uses publication space more efficiently.
"""

import argparse
import re
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

CHARGES = {"NP": -1.0, "PP": +1.0}

PANEL_WIDTH = 2.2  # inches — double-column width
DPI = 600


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
    q = charges[:, None, None]

    cos_gamma = np.sin(la_a) * np.sin(lat_g[None]) + np.cos(la_a) * np.cos(lat_g[None]) * np.cos(
        lon_g[None] - lo_a
    )

    kernel = (lam / (2.0 * np.pi)) * np.exp(lam * (cos_gamma - 1.0)) / (1.0 - np.exp(-2.0 * lam))

    return np.sum(q * kernel, axis=0) / R_nm**2 / 100.0  # e/Å²


def label_from_stem(stem, letter):
    """Return '(a) $P^{1}_{8}$'-style label parsed from a filename stem like 'P18' or 'ref_P18'."""
    m = re.search(r"[Pp](\d+)", stem)
    if m:
        digits = m.group(1)
        sup, sub = digits[0], digits[1:]
        body = rf"P^{{{sup}}}_{{{sub}}}" if sub else rf"P^{{{sup}}}"
        return rf"$\mathbf{{({letter})}}$ ${body}$"
    return rf"$\mathbf{{({letter})}}$"


def process_file(path, lam, radius, resolution):
    atoms = parse_xyz(path)

    if radius is None:
        all_coords = np.vstack([v for k, v in atoms.items() if k != "MP"])
        R_nm = np.mean(np.linalg.norm(all_coords, axis=1)) / 10.0
    else:
        R_nm = radius

    lons, lats, qs = [], [], []
    p_vec = np.zeros(3)  # dipole accumulator, e·Å
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

    lon_a = np.concatenate(lons)
    lat_a = np.concatenate(lats)
    charges = np.array(qs)

    W = resolution
    H = resolution // 2
    lon_g, lat_g = np.meshgrid(
        np.linspace(-np.pi, np.pi, W),
        np.linspace(-np.pi / 2, np.pi / 2, H),
    )

    scd = surface_charge_density(lon_g, lat_g, lon_a, lat_a, charges, lam, R_nm)
    p_mag = np.linalg.norm(p_vec)  # e·Å
    p_hat = p_vec / p_mag

    evals, evecs = np.linalg.eigh(Q_ten)
    idx = np.argmax(np.abs(evals))
    q1_hat = evecs[:, idx]  # principal (prolate) quadrupole axis
    idx_obl = np.argmin(evals)
    q_oblate = evecs[:, idx_obl]  # oblate axis (most negative eigenvalue)
    q_frob = np.linalg.norm(Q_ten, "fro")

    return lon_g, lat_g, scd, R_nm, float(charges.sum()), p_hat, p_mag, q1_hat, q_oblate, q_frob


def draw_dipole_equator(ax, p_hat, p_mag, ms=6, show_text=True):
    """Overlay dipole pole markers and optional mu label."""
    lon1 = np.arctan2(p_hat[1], p_hat[0])
    lat1 = np.arcsin(np.clip(p_hat[2], -1.0, 1.0))
    lon2 = lon1 - np.pi if lon1 > 0 else lon1 + np.pi
    lat2 = -lat1
    ax.plot(lon1, lat1, marker="s", ms=ms, mfc="white", mec="k", mew=1.0, ls="none", zorder=7)
    ax.plot(lon2, lat2, marker="s", ms=ms, mfc="white", mec="k", mew=1.0, ls="none", zorder=7)

    if show_text:
        text = rf"$\mu\!=\!{p_mag:.1f}\ e\,\mathrm{{\AA}}$"
        ax.text(-29.0, -1.55, text, fontsize=9, ha="center", va="center", color="k", zorder=6)


def draw_quadrupole_axis(ax, q1_hat, q_oblate, q_frob, ms=6, show_text=True):
    """Draw triangle markers at prolate (white) and oblate (black) quadrupole poles."""
    lon1 = np.arctan2(q1_hat[1], q1_hat[0])
    lat1 = np.arcsin(np.clip(q1_hat[2], -1.0, 1.0))
    lon2 = lon1 - np.pi if lon1 > 0 else lon1 + np.pi
    lat2 = -lat1
    ax.plot(lon1, lat1, marker="^", ms=ms, mfc="white", mec="k", mew=1.0, ls="none", zorder=7)
    ax.plot(lon2, lat2, marker="v", ms=ms, mfc="white", mec="k", mew=1.0, ls="none", zorder=7)

    lon3 = np.arctan2(q_oblate[1], q_oblate[0])
    lat3 = np.arcsin(np.clip(q_oblate[2], -1.0, 1.0))
    lon4 = lon3 - np.pi if lon3 > 0 else lon3 + np.pi
    lat4 = -lat3
    ax.plot(lon3, lat3, marker="^", ms=ms, mfc="black", mec="k", mew=1.0, ls="none", zorder=7)
    ax.plot(lon4, lat4, marker="v", ms=ms, mfc="black", mec="k", mew=1.0, ls="none", zorder=7)

    if show_text:
        text = rf"$\Theta\!=\!{q_frob:.1f}\ e\,\mathrm{{\AA}}^2$"
        ax.text(29.0, -1.55, text, fontsize=9, ha="center", va="center", color="k", zorder=6)


def _add_mollweide(fig, spec, label, show_index=True):
    ax = fig.add_subplot(spec, projection="mollweide")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(False)
    if show_index:
        ax.text(
            -0.0,
            0.99,
            label,
            transform=ax.transAxes,
            fontsize=9,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.6),
        )
    return ax


def build_axes(fig, results, layout, show_index=True):
    """Create Mollweide axes according to *layout* and return them in order."""
    N = len(results)
    labels = [r[0] for r in results]
    MARGIN = dict(left=0.02, right=0.98)

    if layout == "horizontal":
        # ── single row, labels placed below in figure space by caller ─────────
        gs = gridspec.GridSpec(
            1,
            N,
            wspace=0.05,
            top=0.97,
            bottom=0.30,
            left=0.01,
            right=0.99,
        )
        return [_add_mollweide(fig, gs[0, i], labels[i], show_index=False) for i in range(N)]

    if layout == "vertical" or N != 3:
        # ── original stacked layout ───────────────────────────────────────────
        CBAR_PAD = 0.02
        GS_TOP, GS_BOTTOM = 0.88, 0.02
        available = GS_TOP - GS_BOTTOM
        hspace = CBAR_PAD * N / (available - CBAR_PAD * (N - 1))
        gs = gridspec.GridSpec(
            N,
            1,
            hspace=hspace,
            top=GS_TOP,
            bottom=GS_BOTTOM,
            **MARGIN,
        )
        return [_add_mollweide(fig, gs[i], labels[i], show_index=show_index) for i in range(N)]

    # ── triangular layouts (exactly 3 panels) ─────────────────────────────────
    # All three panels are the same half-width size.  A 4-column GridSpec lets
    # us place the two side-by-side panels in cols 0-1 and 2-3, and the lone
    # centred panel in cols 1-2 — identical width, visually centred.
    gs = gridspec.GridSpec(
        2,
        4,
        height_ratios=[1, 1],
        hspace=0.00,
        wspace=0.40,
        top=0.88,
        bottom=0.02,
        **MARGIN,
    )

    if layout == "pyramid":
        # (a) centred on top; (b) and (c) side-by-side on bottom
        axes = [
            _add_mollweide(fig, gs[0, 1:3], labels[0], show_index=show_index),
            _add_mollweide(fig, gs[1, 0:2], labels[1], show_index=show_index),
            _add_mollweide(fig, gs[1, 2:4], labels[2], show_index=show_index),
        ]
    else:  # "inverted" — default for 3 panels
        # (a) and (b) side-by-side on top; (c) centred on bottom
        axes = [
            _add_mollweide(fig, gs[0, 0:2], labels[0], show_index=show_index),
            _add_mollweide(fig, gs[0, 2:4], labels[1], show_index=show_index),
            _add_mollweide(fig, gs[1, 1:3], labels[2], show_index=show_index),
        ]

    return axes


def main():
    parser = argparse.ArgumentParser(
        description="Mollweide panel of surface charge density for multiple NPs."
    )
    parser.add_argument("xyz", nargs="+", help="XYZ files to compare (one panel each)")
    parser.add_argument(
        "--lam",
        "-l",
        type=float,
        default=20.0,
        help="vMF concentration λ (default 20, half-width ~15°, matching Ref. 44)",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=None,
        help="Sphere radius in nm (auto-detected from coordinates if omitted)",
    )
    parser.add_argument(
        "--resolution",
        "-r",
        type=int,
        default=300,
        help="Longitude grid points (latitude = resolution//2, default 300)",
    )
    parser.add_argument(
        "--layout",
        choices=["vertical", "pyramid", "inverted", "horizontal"],
        default="inverted",
        help=(
            "Panel arrangement: 'vertical' stacks all panels, "
            "'horizontal' places all panels in a row (works for any N), "
            "'pyramid' puts one panel on top and two below, "
            "'inverted' puts two panels on top and one below (default for 3 panels)."
        ),
    )
    parser.add_argument("-o", "--output", default=None, help="Save figure to file")
    parser.add_argument(
        "--no-dipole",
        action="store_true",
        help="Suppress the dipole pole markers overlay",
    )
    parser.add_argument(
        "--no-quadrupole",
        action="store_true",
        help="Suppress the quadrupole principal axis markers overlay",
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Hide the µ/Θ text annotations and the panel index labels",
    )
    parser.add_argument(
        "--marker-size",
        type=float,
        default=6.0,
        help="Base marker size (default 6); scaled proportionally by dipole/quadrupole magnitude",
    )
    args = parser.parse_args()

    letters = "abcdefghijklmnopqrstuvwxyz"
    panel_labels = [label_from_stem(Path(p).stem, letters[i]) for i, p in enumerate(args.xyz)]

    # ── compute SCD for every file ────────────────────────────────────────────
    results = []
    for path, label in zip(args.xyz, panel_labels):
        lon_g, lat_g, scd, R_nm, net_q, p_hat, p_mag, q1_hat, q_oblate, q_frob = process_file(
            path, args.lam, args.radius, args.resolution
        )
        results.append((label, lon_g, lat_g, scd, p_hat, p_mag, q1_hat, q_oblate, q_frob))
        print(
            f"{label}: R = {R_nm:.3f} nm | net charge = {net_q:.0f} e | mu = {p_mag:.2f} e·Å | Theta_F = {q_frob:.2f} e·Å²"
        )

    # ── shared colour scale ───────────────────────────────────────────────────
    vmax = max(np.percentile(np.abs(r[3]), 98) for r in results)  # r[3] = scd
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    # ── figure size ───────────────────────────────────────────────────────────
    N = len(results)
    if args.layout == "horizontal":
        layout = "horizontal"
    elif N == 3:
        layout = args.layout
    else:
        layout = "vertical"

    if layout == "horizontal":
        fig_width = N * PANEL_WIDTH
        fig_height = PANEL_WIDTH / 2 + 0.7
    elif layout == "vertical":
        fig_width = PANEL_WIDTH
        fig_height = N * (PANEL_WIDTH / 2) + 0.7
    else:  # pyramid or inverted
        fig_width = PANEL_WIDTH
        fig_height = 2 * (PANEL_WIDTH / 4) + 0.9

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9})
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=DPI)

    axes = build_axes(fig, results, layout, show_index=not args.no_labels)

    # ── marker size scaling (base = midpoint of the value range across panels) ─
    MS_BASE = args.marker_size
    p_mags = np.array([r[5] for r in results])
    q_frobs = np.array([r[8] for r in results])
    mid_p = float(np.mean(p_mags)) if p_mags.size > 0 else 1.0
    mid_q = float(np.mean(q_frobs)) if q_frobs.size > 0 else 1.0

    show_text = not args.no_labels

    # ── plot data ─────────────────────────────────────────────────────────────
    mesh_ref = None
    for ax, (_label, lon_g, lat_g, scd, p_hat, p_mag, q1_hat, q_oblate, q_frob) in zip(
        axes, results
    ):
        mesh = ax.pcolormesh(
            lon_g,
            lat_g,
            scd,
            cmap="RdBu_r",
            norm=norm,
            shading="auto",
            rasterized=True,
        )
        if mesh_ref is None:
            mesh_ref = mesh
        if not args.no_dipole:
            ms_d = MS_BASE * (p_mag / mid_p)
            draw_dipole_equator(ax, p_hat, p_mag, ms=ms_d, show_text=show_text)
        if not args.no_quadrupole:
            ms_q = MS_BASE * (q_frob / mid_q)
            draw_quadrupole_axis(ax, q1_hat, q_oblate, q_frob, ms=ms_q, show_text=show_text)

    # ── shared colourbar ─────────────────────────────────────────────────────
    fig.canvas.draw()
    x0 = min(ax.get_position().x0 for ax in axes)
    x1 = max(ax.get_position().x1 for ax in axes)

    cbar_h = 0.025

    if layout == "horizontal":
        # Labels sit just below each projection; colorbar below the labels.
        ybot = min(ax.get_position().y0 for ax in axes)
        if not args.no_labels:
            for ax, (label, *_) in zip(axes, results):
                cx = (ax.get_position().x0 + ax.get_position().x1) / 2
                fig.text(cx, ybot - 0.03, label, ha="center", va="top", fontsize=9)
            cbar_y = 0.05
        else:
            cbar_y = ybot - cbar_h - 0.08
        cax = fig.add_axes([x0, cbar_y, x1 - x0, cbar_h])
        cbar = fig.colorbar(mesh_ref, cax=cax, orientation="horizontal")
        cax.xaxis.set_ticks_position("bottom")
        cax.xaxis.set_label_position("bottom")
    else:
        # Colorbar above all panels.
        ytop = max(ax.get_position().y1 for ax in axes)
        cax = fig.add_axes([x0, ytop + 0.07, x1 - x0, cbar_h])
        cbar = fig.colorbar(mesh_ref, cax=cax, orientation="horizontal")
        cax.xaxis.set_ticks_position("bottom")
        cax.xaxis.set_label_position("top")

    cbar.set_label(
        r"Surface charge density, $\sigma\;[e\,\mathrm{\AA}^{-2}]$",
        fontsize=9,
        labelpad=6,
    )

    if args.output:
        plt.savefig(args.output, dpi=DPI, bbox_inches="tight")
        print(f"Saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
