#!/usr/bin/env python3
"""Mollweide surface charge density map of a nanoparticle from an XYZ file.

Each charged bead is smeared with the exact von Mises-Fisher (vMF) kernel,
giving a continuous surface charge density in e/nm².

  lam    : vMF concentration parameter λ (larger = narrower patch; λ ≈ 1/σ²
            where σ is the Gaussian bandwidth in radians)
  R_nm   : sphere radius in nm (auto-detected from coordinates if not given)

MP beads are the NP centre and are ignored.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

CHARGES = {"NP": -1.0, "PP": +1.0}

# Publication layout constants
FIGSIZE = (7.0, 4.0)  # inches — double-column width, ~2:1 Mollweide ratio
DPI = 300


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
    """Exact von Mises-Fisher surface charge density in e/nm².

    Kernel K(γ) = λ / (4π sinh λ) · exp(λ cos γ)
    normalized so that integral over the unit sphere = 1 sr^{-1}.
    Numerically stable form used:
        K = λ/(2π) · exp(λ(cos γ − 1)) / (1 − exp(−2λ))
    Dividing by R_nm^2 converts sr^{-1} -> nm^{-2}.

    Parameters
    ----------
    lon_g, lat_g : (H, W) grid in radians
    lon_a, lat_a : (N,) atom positions in radians
    charges      : (N,) charge in elementary charges
    lam          : vMF concentration parameter λ (λ ≈ 1/σ² for small σ)
    R_nm         : sphere radius in nm
    """
    lo_a = lon_a[:, None, None]
    la_a = lat_a[:, None, None]
    q = charges[:, None, None]

    cos_gamma = np.sin(la_a) * np.sin(lat_g[None]) + np.cos(la_a) * np.cos(lat_g[None]) * np.cos(
        lon_g[None] - lo_a
    )

    kernel = (lam / (2.0 * np.pi)) * np.exp(lam * (cos_gamma - 1.0)) / (1.0 - np.exp(-2.0 * lam))

    return np.sum(q * kernel, axis=0) / R_nm**2 / 100.0  # e/Å²  (1 nm² = 100 Å²)


def main():
    parser = argparse.ArgumentParser(
        description="Mollweide surface charge density (e/nm²) of NP patches."
    )
    parser.add_argument("xyz", help="Path to .xyz file")
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
    parser.add_argument("-o", "--output", default=None, help="Save figure to file")
    args = parser.parse_args()

    atoms = parse_xyz(args.xyz)

    # Auto-detect radius from all non-MP beads (coordinates assumed in Angstroms)
    if args.radius is None:
        all_coords = np.vstack([v for k, v in atoms.items() if k != "MP"])
        R_nm = np.mean(np.linalg.norm(all_coords, axis=1)) / 10.0  # Å -> nm
    else:
        R_nm = args.radius

    lons, lats, qs = [], [], []
    for kind, coords in atoms.items():
        q = CHARGES.get(kind)
        if q is None:
            continue
        lo, la = to_spherical(coords)
        lons.append(lo)
        lats.append(la)
        qs.extend([q] * len(lo))

    if not qs:
        raise ValueError("No charged atoms found.")

    lon_a = np.concatenate(lons)
    lat_a = np.concatenate(lats)
    charges = np.array(qs)

    W = args.resolution
    H = args.resolution // 2
    lon_g, lat_g = np.meshgrid(
        np.linspace(-np.pi, np.pi, W),
        np.linspace(-np.pi / 2, np.pi / 2, H),
    )

    scd = surface_charge_density(lon_g, lat_g, lon_a, lat_a, charges, lam=args.lam, R_nm=R_nm)

    # Net charge sanity check: integrate scd over sphere area
    # dA = R^2 * cos(lat) * dlat * dlon
    dlat = np.pi / H
    dlon = 2 * np.pi / W
    dA = R_nm**2 * np.cos(lat_g) * dlat * dlon
    net_charge = np.sum(scd * dA)
    expected = float(charges.sum())
    print(f"R = {R_nm:.3f} nm | λ = {args.lam:.1f}")
    print(f"Net charge: integral = {net_charge:.2f} e  (expected {expected:.0f} e)")

    vmax = np.percentile(np.abs(scd), 98)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
        }
    )

    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    ax = fig.add_subplot(111, projection="mollweide")

    mesh = ax.pcolormesh(
        lon_g, lat_g, scd, cmap="RdBu_r", norm=norm, shading="auto", rasterized=True
    )
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(False)

    # Reserve bottom margin for the colorbar + label before resolving positions.
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.22)
    fig.canvas.draw()

    pos = ax.get_position()
    cbar_height = 0.05
    cbar_pad = 0.04  # gap between plot bottom and colorbar top
    cax = fig.add_axes(
        [
            pos.x0,
            pos.y0 - cbar_height - cbar_pad,
            pos.width,
            cbar_height,
        ]
    )
    cbar = fig.colorbar(mesh, cax=cax, orientation="horizontal")
    cbar.set_label(
        r"Surface charge density, $\sigma\;[\mathit{e}/\mathrm{\AA}^{2}]$",
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
