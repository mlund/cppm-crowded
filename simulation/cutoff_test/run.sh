#!/usr/bin/env bash
# Run the cutoff test in a tree made by make_inputs.py: trajectories, then reruns, then timings.
# Usage: run.sh <run-tree> <faunus-binary> [jobs] [stages] [label-glob]
#   stages: any of traj, rerun, speed (default: all three); label-glob filters rerun and
#   speed folders, e.g. 'ewald*' to redo only the Ewald Hamiltonians on existing frames.
set -euo pipefail
tree=$(realpath "$1")
faunus=$(realpath "$2")
jobs=${3:-32}
stages=${4:-traj rerun speed}
glob=${5:-*}
export faunus

# Each run gets its own copy of the start state, since `faunus run -s` overwrites it
traj() { cd "$1" && cp ../start_state.yaml state.yaml && "$faunus" run -i input.yml -s state.yaml >run.log 2>&1; }
rerun() { cd "$1" && "$faunus" rerun -i input.yml --traj ../../traj/traj.xtc >run.log 2>&1; }
speed() { cd "$1" && cp ../../start_state.yaml state.yaml && "$faunus" run -i input.yml -s state.yaml >run.log 2>&1; }
export -f traj rerun speed

cd "$tree"
for stage in $stages; do
    case $stage in
        traj) ls -d P18_*/traj | xargs -P "$jobs" -I{} bash -c 'traj {}' ;;
        rerun) ls -d P18_*/rerun/$glob | xargs -P "$jobs" -I{} bash -c 'rerun {}' ;;
        # Few at a time, so timings are not skewed by a loaded machine
        speed) ls -d P18_*/speed/$glob | xargs -P 4 -I{} bash -c 'speed {}' ;;
    esac
done
