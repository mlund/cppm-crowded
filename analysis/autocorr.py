"""Correlation times of the pressure estimator and ion counts (ESI, sampling).

Production runs store only block averages, so this reads per-step series from
separate runs with the same Hamiltonian and moves (vvm.csv.gz, n_na/n_cl.csv.gz)
and checks that blocks of 1000 samples are long compared with τ.

Usage: python autocorr.py <runs dir>
"""

import io
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

BLOCK = 1000


def read_gz(path):
    """CSV from a gzip stream that may still be open for writing."""
    text = zlib.decompressobj(31).decompress(path.read_bytes()).decode(errors="ignore")
    return pd.read_csv(io.StringIO(text[: text.rfind("\n")]))


def tau_int(x, c=5.0):
    """Integrated autocorrelation time with Sokal's self-consistent window."""
    x = x - x.mean()
    f = np.fft.rfft(x, 2 * len(x))
    acf = np.fft.irfft(f * np.conj(f))[: len(x)]
    acf /= acf[0]
    tau = 0.5
    for m in range(1, len(x)):
        tau += acf[m]
        if m >= c * tau:
            break
    return tau


def block_error(x, b):
    n = len(x) // b
    means = x[: n * b].reshape(n, b).mean(axis=1)
    return means.std(ddof=1) / np.sqrt(n)


def main(runs):
    rows = []
    for d in sorted(p for p in runs.iterdir() if p.is_dir() and "-exp" in p.name):
        w = np.exp(-read_gz(d / "vvm.csv.gz")["dU/kT"].to_numpy())  # estimator averages e^{-ΔU}
        rows.append(
            {
                "run": d.name,
                "samples": len(w),
                "tau_P": tau_int(w),
                "tau_Na": tau_int(read_gz(d / "n_na.csv.gz")["value"].to_numpy()),
                "tau_Cl": tau_int(read_gz(d / "n_cl.csv.gz")["value"].to_numpy()),
                # Error should not grow with block length once blocks exceed τ
                "err(2B)/err(B)": block_error(w, 2 * BLOCK) / block_error(w, BLOCK),
            }
        )
    print(pd.DataFrame(rows).round(2).to_string(index=False))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
