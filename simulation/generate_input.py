#!/usr/bin/env python3
"""Render template_2.yaml into a new simulation folder."""

import argparse
import math
import os
import sys
from pathlib import Path

try:
    from jinja2 import Environment, FileSystemLoader, StrictUndefined
except ImportError:
    sys.exit("jinja2 not found — install with: pip install jinja2")


def main():
    parser = argparse.ArgumentParser(description="Generate faunus input from template")
    parser.add_argument("output_dir", help="Folder to create (e.g. isotropic_20_NP)")
    parser.add_argument(
        "--type",
        dest="system_type",
        default="isotropic",
        choices=["isotropic", "dipolar", "quadrupolar", "neutral", "bulk"],
        help="System type (default: isotropic)",
    )
    parser.add_argument(
        "--N",
        dest="n_macroions",
        type=int,
        default=30,
        help="Number of macroions (default: 30)",
    )
    parser.add_argument(
        "--phi",
        dest="volume_fraction",
        type=float,
        default=0.3,
        help="Volume fraction (default: 0.3)",
    )
    parser.add_argument(
        "--kappa",
        type=float,
        default=0.0,
        help="Inverse Debye length 1/Å (default: 0 = no added salt)",
    )
    parser.add_argument(
        "--micro-steps",
        type=int,
        default=4000,
        help="Production steps = 10 * micro_steps (default: 4000 → 40000 steps)",
    )
    parser.add_argument(
        "--equil-micro-steps",
        type=int,
        default=2000,
        help="Equilibration steps = 10 * equil_micro_steps (default: 2000 → 20000 steps)",
    )
    parser.add_argument(
        "--potential",
        default="fanourgakis",
        choices=["coulomb", "fanourgakis"],
        help="Electrostatic potential to use (default: fanourgakis)",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=None,
        help="Electrostatic cutoff in Å (default: half the box length)",
    )
    parser.add_argument(
        "--template",
        default="template_2.yaml",
        help="Template file relative to simulation/ (default: template_2.yaml)",
    )
    parser.add_argument(
        "--traj",
        action="store_true",
        default=False,
        help="Save trajectory to traj.xtc during production (default: off)",
    )
    parser.add_argument(
        "--sdf",
        action="store_true",
        default=False,
        help="Save spatial distribution function to sdf.dx during production (default: off)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files without prompting (for batch/scripted use)",
    )
    args = parser.parse_args()

    sim_root = Path(__file__).resolve().parent
    template_path = sim_root / args.template

    if not template_path.exists():
        sys.exit(f"Template not found: {template_path}")

    env = Environment(
        loader=FileSystemLoader(str(sim_root)),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template(args.template)

    pi = math.pi
    macroion_volume = 4.0 * pi / 3.0 * 20.0**3
    box_length = (args.n_macroions * macroion_volume / args.volume_fraction) ** (1 / 3)

    cutoff = args.cutoff if args.cutoff is not None else box_length / 2.0

    common = dict(
        system_type=args.system_type,
        n_macroions=args.n_macroions,
        volume_fraction=args.volume_fraction,
        kappa=args.kappa,
        potential=args.potential,
        cutoff=cutoff,
    )

    out_dir = sim_root.parent / "data" / "runs" / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    # Relative, so rendered inputs stay valid if the repo is moved
    structures_dir = os.path.relpath(sim_root / "structures", out_dir)

    files = [
        (out_dir / "input_equi.yml", args.equil_micro_steps, True),
        (out_dir / "input.yml", args.micro_steps, False),
    ]

    for out_file, micro_steps, is_equilibration in files:
        if out_file.exists() and not args.force:
            answer = input(f"{out_file} already exists. Overwrite? [y/N] ").strip().lower()
            if answer != "y":
                sys.exit("Aborted.")
        rendered = template.render(
            **common,
            structures_dir=structures_dir,
            micro_steps=micro_steps,
            is_equilibration=is_equilibration,
            save_traj=args.traj,
            save_sdf=args.sdf,
        )
        out_file.write_text(rendered)

    n_cations = 2 if args.system_type == "neutral" else 2 + args.n_macroions * 8
    n_anions = 2

    print(
        f"Written: {out_dir / 'input_equi.yml'} ({10 * args.equil_micro_steps} steps, equilibration only)"
    )
    print(f"Written: {out_dir / 'input.yml'} ({10 * args.micro_steps} steps)")
    if args.system_type != "bulk":
        print(f"  system_type    : {args.system_type}")
        print(f"  n_macroions    : {args.n_macroions}")
        print(f"  volume_fraction: {args.volume_fraction}")
        print(f"  box_length     : {box_length:.4f} Å")
        if args.kappa == 0:
            print(f"  n_Cations      : {n_cations}")
            print(f"  n_Anions       : {n_anions}")
        else:
            print(
                f"  ions           : implicit (Yukawa, κ = {args.kappa:.4f} Å⁻¹, λD = {1 / args.kappa:.4f} Å)"
            )
    print(f"  potential      : {args.potential}")
    print(f"  cutoff         : {cutoff:.4f} Å")


if __name__ == "__main__":
    main()
