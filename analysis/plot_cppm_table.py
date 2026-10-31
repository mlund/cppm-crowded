"""Fig. 1: CPPM variants with their multipole moments, renders and surface-charge maps.

The renders and Mollweide maps come from an Inkscape composite (figures/schematics/
cppm-table_MW.png) whose serif table clashes with the other figures; this crops the table off
and redraws it in the matplotlib font.

Usage: python analysis/plot_cppm_table.py figures/schematics/cppm-table_MW.png figures
"""

import argparse
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

TITLES = [r"Isotropic ($P^{0}_{0}$)", r"Dipolar ($P^{1}_{8}$)", r"Quadrupolar ($P^{2}_{8}$)"]
ROWS = {  # label → one value per variant; net charge, dipole and quadrupole moment
    r"$q/e$": ["−8", "−8", "−8"],
    r"$\mu/e$Å": ["33.0", "186.6", "43.0"],
    r"$\Theta/e$Å$^2$": ["389.1", "4489.4", "6869.2"],
}
WIDTH = 7.0  # in; spans both columns of a two-column page
TABLE_H = 0.85  # in


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("composite", type=Path, help="Inkscape PNG with the table on top")
    p.add_argument("plots", type=Path, help="Output folder")
    args = p.parse_args()

    img = mpimg.imread(args.composite)
    w = img.shape[1]
    # The table is bounded by two horizontal rules; its column spans give the panel centres
    dark = (img[..., :3].max(axis=2) < 0.3) & (img[..., 3] > 0.5)
    rows = np.flatnonzero(dark.sum(axis=1) > 0.25 * w)
    groups = np.split(rows, np.flatnonzero(np.diff(rows) > 1) + 1)
    first, last = groups[0][0], groups[1][-1]  # the two table rules; renders follow below
    xs = np.flatnonzero(dark[first])
    spans = np.split(xs, np.flatnonzero(np.diff(xs) > 5) + 1)
    spans = [(s[0] / w, s[-1] / w) for s in spans if len(s) > w / 20]
    assert len(spans) == len(TITLES), spans
    ink = (img[..., :3].min(axis=2) < 0.98) & (img[..., 3] > 0.02)  # include anti-aliasing
    blank = last + int(np.argmax(ink[last:].sum(axis=1) == 0))
    img = img[blank:]

    h_img = WIDTH * img.shape[0] / w
    fig = plt.figure(figsize=(WIDTH, h_img + TABLE_H))
    frac = h_img / (h_img + TABLE_H)
    ax_img = fig.add_axes([0, 0, 1, frac])
    ax_img.imshow(img, interpolation="lanczos")
    ax_img.axis("off")

    tab = fig.add_axes([0, frac, 1, 1 - frac])
    tab.set_xlim(0, 1)
    tab.set_ylim(0, 1)
    tab.axis("off")
    n = len(ROWS)
    y_title, y_rule1, y_rule2 = 0.9, 0.74, 0.05
    ys = np.linspace(y_rule1, y_rule2, n + 2)[1:-1]
    for (x0, x1), title, *vals in zip(spans, TITLES, *ROWS.values()):
        xc = (x0 + x1) / 2
        tab.text(xc, y_title, title, ha="center", va="center", fontsize=9.5)
        for y in (y_rule1, y_rule2):
            tab.plot([x0, x1], [y, y], color="black", lw=0.9)
        for y, v in zip(ys, vals):
            tab.text(xc, y, v, ha="center", va="center", fontsize=9)
    for y, label in zip(ys, ROWS):
        tab.text(0.005, y, label, ha="left", va="center", fontsize=9)

    args.plots.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(args.plots / f"cppm_table.{ext}", dpi=600)


if __name__ == "__main__":
    main()
