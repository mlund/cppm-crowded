#!/usr/bin/env python3
"""Inputs for the cutoff test: screened Fanourgakis vs Yukawa Ewald, implicit salt, P18.

One trajectory per κ is sampled with the production Hamiltonian, then replayed
(`faunus rerun`) through each test Hamiltonian, so all are compared on identical
frames and can be reweighted. Short MC runs time each Hamiltonian.
"""

import argparse
import math
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BOX = 149.64407707272974
HALF_BOX = BOX / 2

# CODATA 2018, as in the coulomb crate, so the medium reproduces 1/κ exactly
E = 1.602176634e-19
EPS0 = 8.8541878128e-12
KB = 1.380649e-23
NA = 6.02214076e23
T = 298.15
EPS_R_WATER = 78.35565171480539  # coulomb::Medium::neat_water(298.15)

FANOURGAKIS_CUTOFFS = [HALF_BOX, 60.0, 50.0, 40.0, 30.0, 25.0, 20.0, 15.0]
EWALD_REF = [(40.0, 1e-6), (30.0, 1e-7)]  # second setting checks the reference itself
# Looser ε lets optimize() cut k-vectors; reruns check that structure survives it
EWALD_OPT = [(40.0, 1e-3), (40.0, 1e-4), (20.0, 1e-3), (20.0, 1e-4)]
# Fixed α from the yukawa-ewald-cppm fit, inside its range L/r_c ∈ [5, 30]
EWALD_FIT = [(20.0, 1e-3), (15.0, 1e-3)]
SPEED = {
    "fan": [HALF_BOX, 40.0, 30.0, 20.0],
    "ewald": [40.0, 30.0, 20.0, 15.0],
}
# Accelerated variants of the accurate settings: exact group culling, or splines + cell list
ACCELERATED = {
    "fan": [HALF_BOX, 40.0],
    "ewaldopt": [(20.0, 1e-3), (40.0, 1e-4)],
}
# Culling and splines must cover WCA between macroion centres, 2^(1/6)·σ_MP
WCA_RANGE = 2 ** (1 / 6) * 40.0


def nacl_molarity(debye_length):
    """Molarity whose Debye length equals `debye_length` (Å) in water at T."""
    bjerrum = E**2 / (4 * math.pi * EPS0 * EPS_R_WATER * KB * T) * 1e10
    return 1 / (8 * math.pi * bjerrum * NA * 1e-27 * debye_length**2)


def fitted_alpha(cutoff, kappa):
    """α from the CPPM P18 parameter-space fit (~/github/yukawa-ewald-cppm, R² = 0.875).

    η_min = 1.43 ln L* − 1.59 κr_c + 0.75 ln L*·κr_c − 2.25 with L* = L/r_c, floored at 0.8
    and raised 20% as that report recommends.
    """
    ln_l, kr = math.log(BOX / cutoff), kappa * cutoff
    eta = 1.43 * ln_l - 1.59 * kr + 0.75 * ln_l * kr - 2.25
    return 1.2 * max(eta, 0.8) / cutoff


def header(structure, medium_extra=""):
    return f"""\
atoms:
  - {{name: MP, mass: 1.0e7,  charge:  0.0, sigma: 40.0, epsilon: 2.479}}
  - {{name: NP, mass: 1.0e-3, charge: -1.0, sigma:  4.0, epsilon: 2.479}}
  - {{name: PP, mass: 1.0e-3, charge:  1.0, sigma:  4.0, epsilon: 2.479}}
  - {{name: UP, mass: 1.0e-3, charge:  0.0, sigma:  4.0, epsilon: 2.479}}

molecules:
  - name: macroion
    degrees_of_freedom: Rigid
    has_com: true
    from_structure: "{structure}"

system:
  cell: !Cuboid [{BOX}, {BOX}, {BOX}]
  medium:
    permittivity: !Water
    temperature: {T}{medium_extra}
  blocks:
    - molecule: macroion
      N: 30
      insert: !GridCOM {{rotate: true}}
"""


def energy(
    scheme, cutoff, debye_length=None, accuracy=None, optimize=False, alpha=None, accel=None
):
    """Nonbonded block; κ comes from medium.salt unless `debye_length` is given.

    `accel`: "cull" (exact group-pair culling) or "spline" (tabulated, with cell list).
    """
    fixed_alpha = f"\n      alpha: {alpha:.6g}" if alpha is not None else ""
    reach = max(cutoff, WCA_RANGE)
    nonbonded = {
        None: "",
        "cull": f"\n      cutoff: {reach!r}\n      bounding_spheres: true",
        "spline": f"\n      spline:\n        cutoff: {reach!r}",
    }[accel]
    if scheme == "fan":
        screening = f", debye_length: {debye_length}" if debye_length else ""
        return f"""\
  energy:
    - !Nonbonded{nonbonded}
      default:
        - !Fanourgakis {{cutoff: {cutoff}{screening}}}
        - !WeeksChandlerAndersen {{mixing: LB}}
"""
    return f"""\
  energy:
    - !Nonbonded{nonbonded}
      default:
        - !WeeksChandlerAndersen {{mixing: LB}}
    - !Ewald
      cutoff: {cutoff}
      accuracy: {accuracy}
      policy: PBC
      optimize: {str(optimize).lower()}{fixed_alpha}
"""


def propagate(steps, seed="!Hardware"):
    return f"""\
propagate:
  seed: {seed}
  criterion: Metropolis
  steps: {steps}
  collections:
    - !Stochastic
      moves:
        - !TranslateMolecule {{molecule: macroion, dp: 4.0, repeat: 30}}
        - !RotateMolecule    {{molecule: macroion, dprot: 0.4,  repeat: 30}}
"""


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kappas", nargs="+", default=["0.0264", "0.0526", "0.0950"])
    parser.add_argument("--out", type=Path, default=REPO / "data/runs/cutoff")
    parser.add_argument("--traj-steps", type=int, default=20000)
    parser.add_argument("--traj-every", type=int, default=20)
    parser.add_argument("--speed-steps", type=int, default=500)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / "simulation/structures/P18.xyz", args.out / "P18.xyz")

    for kappa in args.kappas:
        debye = 1 / float(kappa)
        salt = f"\n    salt: [!NaCl, {nacl_molarity(debye):.12g}]"
        root = args.out / f"P18_{kappa}"
        src = REPO / f"data/implicit/dipolar/dipolar-imp_{kappa}/state.yaml"
        root.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, root / "start_state.yaml")

        # Production Hamiltonian (explicit debye_length), with frames for rerun
        write(
            root / "traj/input.yml",
            header("../../P18.xyz")
            + energy("fan", HALF_BOX, debye_length=debye)
            + f"""
analysis:
  - !Energy
    file: energy.csv
    frequency: !Every {args.traj_every}
  - !Trajectory
    file: traj.xtc
    frequency: !Every {args.traj_every}
    save_frame_state: true
  - !Trajectory
    file: traj.xyz
    frequency: !Every {args.traj_every}

"""
            + propagate(args.traj_steps),
        )

        # Rerun Hamiltonians share κ via medium.salt
        reruns = {f"fan_{rc:.1f}": energy("fan", rc) for rc in FANOURGAKIS_CUTOFFS}
        reruns |= {
            f"ewald_{rc:.0f}_{acc:.0e}": energy("ewald", rc, accuracy=acc) for rc, acc in EWALD_REF
        }
        reruns |= {
            f"ewaldopt_{rc:.0f}_{acc:.0e}": energy("ewald", rc, accuracy=acc, optimize=True)
            for rc, acc in EWALD_OPT
        }
        fit = {
            f"ewaldfit_{rc:.0f}_{acc:.0e}": energy(
                "ewald", rc, accuracy=acc, alpha=fitted_alpha(rc, float(kappa))
            )
            for rc, acc in EWALD_FIT
        }
        reruns |= fit
        accelerated = {}
        for accel in ("cull", "spline"):
            for rc in ACCELERATED["fan"]:
                accelerated[f"fan{accel}_{rc:.1f}"] = energy("fan", rc, accel=accel)
            for rc, acc in ACCELERATED["ewaldopt"]:
                accelerated[f"ewaldopt{accel}_{rc:.0f}_{acc:.0e}"] = energy(
                    "ewald", rc, accuracy=acc, optimize=True, accel=accel
                )
        reruns |= {k: v for k, v in accelerated.items() if "spline" in k}
        for label, block in reruns.items():
            write(
                root / f"rerun/{label}/input.yml",
                header("../../../P18.xyz", salt)
                + block
                + """
analysis:
  - !Energy
    file: energy.csv
    frequency: !Every 1
  - !VirtualVolumeMove
    volume_displacement: 0.1
    method: Isotropic
    block_size: 50
    file: pressure.csv
    frequency: !Every 1

"""
                + propagate(1),
            )

        # Timing: same start state and seed for every Hamiltonian
        timed = {f"fan_{rc:.1f}": energy("fan", rc) for rc in SPEED["fan"]}
        for rc in SPEED["ewald"]:
            timed[f"ewald_{rc:.0f}"] = energy("ewald", rc, accuracy=1e-5)
            timed[f"ewaldopt_{rc:.0f}"] = energy("ewald", rc, accuracy=1e-5, optimize=True)
        for rc, acc in EWALD_OPT:
            timed[f"ewaldopt_{rc:.0f}_{acc:.0e}"] = energy("ewald", rc, accuracy=acc, optimize=True)
        timed |= fit | accelerated
        for label, block in timed.items():
            write(
                root / f"speed/{label}/input.yml",
                header("../../../P18.xyz", salt)
                + block
                + "\nanalysis: []\n\n"
                + propagate(args.speed_steps, seed="!Fixed 1"),
            )
        print(f"κ = {kappa}: λ_D = {debye:.4f} Å, NaCl {nacl_molarity(debye):.6g} M -> {root}")


if __name__ == "__main__":
    main()
