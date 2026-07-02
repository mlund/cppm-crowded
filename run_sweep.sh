#!/usr/bin/env bash
# run_sweep.sh — kappa sweep across system types with equil+production in same directory.
# Runs locally or inside a SLURM job (called by lunarc_sweep.sh).

set -euo pipefail

SCRIPT_START_TIME=$(date +%s)
SCRIPT_START_DATE=$(date)
echo "Script started at: $SCRIPT_START_DATE"

#######################################
# Environment auto-detection
#######################################

if [[ -d "$HOME/cppm-crowded" ]]; then
    PROJ_ROOT="$HOME/cppm-crowded"
else
    PROJ_ROOT="$HOME/projects/cppm-crowded"
fi
FAUNUS="$(dirname "$PROJ_ROOT")/faunus-rs/target/release/faunus"
GENERATE="$PROJ_ROOT/generate_input.py"

#######################################
# Help
#######################################

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Kappa sweep (pick one form):
  --kappas <v1,v2,...>                     Explicit list, e.g. 0.07,0.08,0.10
  --kappa-min <v> --kappa-max <v> --kappa-n <N>   Linspace of N points

System types:
  --systems <s1,s2,...>    Comma-separated; choices: isotropic,dipolar,quadrupolar,neutral
                           (default: isotropic)

Simulation parameters (passed to generate_input.py):
  --N <int>                Number of macroions (default: 10)
  --phi <float>            Volume fraction (default: 0.3)
  --micro-steps <int>      Production micro-steps multiplier (default: 1000)
  --equil-micro-steps <int>  Equilibration micro-steps multiplier (default: 200)
  --potential <str>        coulomb or fanourgakis (default: fanourgakis)
  --traj                   Save trajectory to traj.xtc during production (default: off)
  --replicas <int>         Number of replicas per condition (default: 1, no suffix).
                           First replica keeps the base name; subsequent ones get _2, _3, ...

  -h, --help

Examples:
  $0 --kappas 0.070,0.080,0.0981 --systems isotropic,quadrupolar
  $0 --kappa-min 0.07 --kappa-max 0.12 --kappa-n 6 --systems isotropic
EOF
}

#######################################
# Defaults
#######################################

SYSTEMS="isotropic"
N_MACROIONS=10
PHI=0.3
MICRO_STEPS=1000
EQUIL_MICRO_STEPS=200
POTENTIAL="fanourgakis"
TRAJ=""
REPLICAS=1
USER_KAPPAS=""
KAPPA_MIN="" KAPPA_MAX="" KAPPA_N=""

#######################################
# Parse arguments
#######################################

while [[ $# -gt 0 ]]; do
    case "$1" in
        --kappas)            USER_KAPPAS="$2";       shift 2 ;;
        --kappa-min)         KAPPA_MIN="$2";          shift 2 ;;
        --kappa-max)         KAPPA_MAX="$2";          shift 2 ;;
        --kappa-n)           KAPPA_N="$2";            shift 2 ;;
        --systems)           SYSTEMS="$2";            shift 2 ;;
        --N)                 N_MACROIONS="$2";        shift 2 ;;
        --phi)               PHI="$2";                shift 2 ;;
        --micro-steps)       MICRO_STEPS="$2";        shift 2 ;;
        --equil-micro-steps) EQUIL_MICRO_STEPS="$2";  shift 2 ;;
        --potential)         POTENTIAL="$2";          shift 2 ;;
        --traj)              TRAJ="--traj";           shift 1 ;;
        --replicas)          REPLICAS="$2";           shift 2 ;;
        -h|--help)           usage; exit 0 ;;
        *)                   echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

#######################################
# Build kappa array
#######################################

KAPPA_VALS=()
if [[ -n "$USER_KAPPAS" ]]; then
    IFS=',' read -ra KAPPA_VALS <<< "$USER_KAPPAS"
elif [[ -n "$KAPPA_MIN" && -n "$KAPPA_MAX" && -n "$KAPPA_N" ]]; then
    mapfile -t KAPPA_VALS < <(python3 -c "
import numpy as np
for v in np.linspace($KAPPA_MIN, $KAPPA_MAX, $KAPPA_N):
    print(f'{v:.6g}')
")
else
    echo "ERROR: provide --kappas or --kappa-min/--kappa-max/--kappa-n" >&2
    usage; exit 1
fi

IFS=',' read -ra SYSTEM_VALS <<< "$SYSTEMS"

TOTAL=$(( ${#SYSTEM_VALS[@]} * ${#KAPPA_VALS[@]} * REPLICAS ))
echo "Systems  : ${SYSTEM_VALS[*]}"
echo "Kappas   : ${KAPPA_VALS[*]}"
echo "Replicas : $REPLICAS"
echo "Total    : $TOTAL simulation(s)"
echo

#######################################
# Per-simulation function
#   Runs equil then production in the same directory.
#   The state.yaml written by equil is picked up by production via -s state.yaml.
#######################################

run_one() {
    local SYSTEM="$1"
    local KAPPA="$2"
    local REPLICA="$3"   # empty string for the first (base) replica
    local KAPPA_FMT
    KAPPA_FMT=$(printf "%.4f" "$KAPPA")
    local DIR_NAME="${SYSTEM}-imp_${KAPPA_FMT}${REPLICA}"
    local SIM_DIR="$PROJ_ROOT/$DIR_NAME"

    echo "  [${SYSTEM} κ=${KAPPA_FMT}${REPLICA}] Generating input files"
    python3 "$GENERATE" "$DIR_NAME" \
        --type "$SYSTEM" \
        --N "$N_MACROIONS" \
        --phi "$PHI" \
        --kappa "$KAPPA" \
        --micro-steps "$MICRO_STEPS" \
        --equil-micro-steps "$EQUIL_MICRO_STEPS" \
        --potential "$POTENTIAL" \
        ${TRAJ:+--traj} \
        --force

    cd "$SIM_DIR"

    echo "  [${SYSTEM} κ=${KAPPA_FMT}${REPLICA}] Equilibration"
    "$FAUNUS" run -i input_equi.yml -s state.yaml

    echo "  [${SYSTEM} κ=${KAPPA_FMT}${REPLICA}] Production (resuming from state.yaml)"
    "$FAUNUS" run -i input.yml -s state.yaml

    echo "  [${SYSTEM} κ=${KAPPA_FMT}${REPLICA}] Done"
}

#######################################
# Launch all simulations in parallel
#######################################

PIDS=()
for SYSTEM in "${SYSTEM_VALS[@]}"; do
    for KAPPA in "${KAPPA_VALS[@]}"; do
        for (( R=1; R<=REPLICAS; R++ )); do
            SUFFIX=""
            [[ $R -gt 1 ]] && SUFFIX="_${R}"
            ( run_one "$SYSTEM" "$KAPPA" "$SUFFIX" ) &
            PIDS+=($!)
        done
    done
done

echo "Waiting for ${#PIDS[@]} parallel simulation(s) to finish..."
FAILED=0
for PID in "${PIDS[@]}"; do
    wait "$PID" || { echo "ERROR: simulation PID $PID failed" >&2; FAILED=1; }
done
[[ $FAILED -eq 0 ]] || exit 1

#######################################
# Execution time
#######################################

SCRIPT_END_TIME=$(date +%s)
ELAPSED=$((SCRIPT_END_TIME - SCRIPT_START_TIME))
echo "========================================"
echo "Execution Time Summary"
echo "========================================"
echo "Script started : $SCRIPT_START_DATE"
echo "Script ended   : $(date)"
printf "Total elapsed  : %dh %dm %ds (%d seconds)\n" \
    $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60)) "$ELAPSED"
echo "========================================"
