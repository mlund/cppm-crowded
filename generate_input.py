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
        choices=["isotropic", "dipolar","quadrupolar", "neutral", "bulk"],
        help="System type (default: isotropic)",
    )
    parser.add_argument(
        "--N", dest="n_macroions", type=int, default=10,
        help="Number of macroions (default: 10)",
    )
    parser.add_argument(
        "--phi", dest="volume_fraction", type=float, default=0.3,
        help="Volume fraction (default: 0.3)",
    )
    parser.add_argument(
        "--kappa", type=float, default=0.0,
        help="Inverse Debye length 1/Å (default: 0 = no added salt)",
    )
    parser.add_argument(
        "--micro-steps", type=int, default=1000,
        help="MC micro-steps multiplier for production (repeat = 10 * micro_steps, default: 1000 → 10000 repeats)",
    )
    parser.add_argument(
        "--equil-micro-steps", type=int, default=200,
        help="MC micro-steps multiplier for equilibration (repeat = 10 * equil_micro_steps, default: 200 → 2000 repeats)",
    )
    parser.add_argument(
        "--potential",
        default="coulomb",
        choices=["coulomb", "fanourgakis"],
        help="Electrostatic potential to use (default: coulomb)",
    )
    parser.add_argument(
        "--cutoff", type=float, default=None,
        help="Electrostatic cutoff in Å (default: half the box length)",
    )
    parser.add_argument(
        "--template", default="template_2.yaml",
        help="Template file relative to project root (default: template_2.yaml)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing files without prompting (for batch/scripted use)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent
    template_path = project_root / args.template

    if not template_path.exists():
        sys.exit(f"Template not found: {template_path}")

    env = Environment(
        loader=FileSystemLoader(str(project_root)),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template(args.template)

    pi = math.pi
    macroion_volume = 4.0 * pi / 3.0 * 20.0 ** 3
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

    out_dir = project_root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    files = [
        (out_dir / "input_equi.yml", args.equil_micro_steps, True),
        (out_dir / "input.yml", args.micro_steps, False),
    ]

    for out_file, micro_steps, is_equilibration in files:
        if out_file.exists() and not args.force:
            answer = input(f"{out_file} already exists. Overwrite? [y/N] ").strip().lower()
            if answer != "y":
                sys.exit("Aborted.")
        rendered = template.render(**common, micro_steps=micro_steps, is_equilibration=is_equilibration)
        out_file.write_text(rendered)

    n_cations = 2 if args.system_type == "neutral" else 2 + args.n_macroions * 8
    n_anions = 2

    print(f"Written: {out_dir / 'input_equi.yml'} ({10 * args.equil_micro_steps} repeats, equilibration only)")
    print(f"Written: {out_dir / 'input.yml'} ({10 * args.micro_steps} repeats)")
    if args.system_type != "bulk":
        print(f"  system_type    : {args.system_type}")
        print(f"  n_macroions    : {args.n_macroions}")
        print(f"  volume_fraction: {args.volume_fraction}")
        print(f"  box_length     : {box_length:.4f} Å")
        if args.kappa == 0:
            print(f"  n_Cations      : {n_cations}")
            print(f"  n_Anions       : {n_anions}")
        else:
            print(f"  ions           : implicit (Yukawa, κ = {args.kappa:.4f} Å⁻¹, λD = {1/args.kappa:.4f} Å)")
    print(f"  potential      : {args.potential}")
    print(f"  cutoff         : {cutoff:.4f} Å")


if __name__ == "__main__":
    main()
