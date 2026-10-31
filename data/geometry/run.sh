#!/usr/bin/env bash
# 12 runs: isotropic implicit ions with bead charges scaled so pair energies carry the
# Debye–Hückel amplitude of an ion-excluded sphere; equilibrate, then produce.
set -u
cd "$(dirname "$0")/data/runs"
F=/home/mikael/github/faunus-rs/target/release/faunus
for d in *-geo*; do
  (
    cd "$d"
    git -C /home/mikael/github/faunus-rs describe --tags --dirty > faunus_build.txt
    nice -n 19 "$F" run -i input_equi.yml -s state.yaml > equi.log 2>&1 &&
    nice -n 19 "$F" run -i input.yml -s state.yaml > prod.log 2>&1
    echo "$d exit $? $(date +%FT%T)"
  ) &
done
wait
echo ALL DONE $(date +%FT%T)
