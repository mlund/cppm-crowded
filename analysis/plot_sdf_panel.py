"""Title the VMD spatial-distribution renders with the variant symbols.

The renders come from VMD/Inkscape as one PNG (figures/schematics/sdfs.png) whose serif titles
clash with the other figures; this crops them off and sets the titles in the matplotlib font.

Usage: python analysis/plot_sdf_panel.py figures/schematics/sdfs.png figures
"""

import argparse
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

SYMBOLS = [r"$P^{0}_{0}$", r"$P^{1}_{8}$", r"$P^{2}_{8}$", r"$P^\mathrm{n}$"]
WIDTH = 7.0  # in; spans both columns of a two-column page
TITLE_BAND = 0.3  # in


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("render", type=Path, help="Four-panel PNG with titles on top")
    p.add_argument("plots", type=Path, help="Output folder")
    args = p.parse_args()

    img = mpimg.imread(args.render)
    ink = (img[..., :3].min(axis=2) < 0.9) & (img[..., 3] > 0.1)
    # The render's own titles end at the first fully blank row; the panels start below it
    rows = ink.sum(axis=1)
    top = int(np.argmax(rows > 0))
    cut = top + int(np.argmax(rows[top:] == 0))
    img = img[cut:]
    # Panel centres from the blank columns separating the renders
    cols = ink[cut:].any(axis=0)
    edges = np.flatnonzero(np.diff(cols.astype(int)))
    starts, ends = edges[::2] + 1, edges[1::2] + 1
    keep = (ends - starts) > img.shape[1] / 20  # ignore specks
    centres = (starts[keep] + ends[keep]) / 2 / img.shape[1]
    assert len(centres) == len(SYMBOLS), centres

    h_img = WIDTH * img.shape[0] / img.shape[1]
    fig = plt.figure(figsize=(WIDTH, h_img + TITLE_BAND))
    ax = fig.add_axes([0, 0, 1, h_img / (h_img + TITLE_BAND)])
    ax.imshow(img, interpolation="lanczos")
    ax.axis("off")
    for x, sym in zip(centres, SYMBOLS):
        fig.text(x, 0.99, sym, ha="center", va="top", fontsize=11)
    args.plots.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(args.plots / f"sdfs.{ext}", dpi=600)


if __name__ == "__main__":
    main()
