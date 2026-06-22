import sys
import numpy as np
from itertools import combinations
from pathlib import Path


def parse_xyz(path):
    atoms = {"PP": [], "NP": [], "MP": []}
    with open(path) as f:
        n = int(f.readline())
        f.readline()  # comment
        for _ in range(n):
            parts = f.readline().split()
            name, x, y, z = parts[0], float(parts[1]), float(parts[2]), float(parts[3])
            if name in atoms:
                atoms[name].append([x, y, z])
    return {k: np.array(v) for k, v in atoms.items() if v}


def avg_distance(coords_a, coords_b, same=False):
    dists = []
    if same:
        for i, j in combinations(range(len(coords_a)), 2):
            dists.append(np.linalg.norm(coords_a[i] - coords_a[j]))
    else:
        for a in coords_a:
            for b in coords_b:
                dists.append(np.linalg.norm(a - b))
    return np.mean(dists), np.std(dists), len(dists)


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("P18.xyz")
    atoms = parse_xyz(path)

    print(f"File: {path}")
    for k, v in atoms.items():
        print(f"  {k}: {len(v)} beads")
    print()

    pairs = [
        ("PP", "PP", True),
        ("NP", "NP", True),
        ("PP", "NP", False),
    ]
    if "MP" in atoms:
        pairs += [("PP", "MP", False), ("NP", "MP", False)]

    print(f"{'Pair':<12} {'N pairs':>8} {'Mean dist':>12} {'Std dev':>12}")
    print("-" * 48)
    for a, b, same in pairs:
        if a not in atoms or b not in atoms:
            continue
        mean, std, n = avg_distance(atoms[a], atoms[b], same=same)
        label = f"{a}-{b}"
        print(f"{label:<12} {n:>8} {mean:>12.4f} {std:>12.4f}")


if __name__ == "__main__":
    main()
