#!/usr/bin/env bash
# run_sweep.sh — kappa sweep (implicit ions) or explicit-ion run across system types.
# Runs locally or inside a SLURM job (called by lunarc_sweep.sh).
# When no kappa args are given, explicit-ion mode is assumed and dirs are named <system>-exp[_N].

set -euo pipefail

SCRIPT_START_TIME=$(date +%s)
SCRIPT_START_DATE=$(date)
echo "Script started at: $SCRIPT_START_DATE"

#######################################
# Environment auto-detection
#######################################

# Repo root is the parent of this script's directory; faunus-rs is expected next to the repo
PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FAUNUS="$(dirname "$PROJ_ROOT")/faunus-rs/target/release/faunus"
GENERATE="$PROJ_ROOT/simulation/generate_input.py"
# uv pins the Python environment locally; clusters without uv fall back to python3
if command -v uv >/dev/null; then
    PYTHON=(uv run --project "$PROJ_ROOT" python)
else
    PYTHON=(python3)
fi

#######################################
# Help
#######################################

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Kappa sweep — implicit ions (pick one form; omit both for explicit-ion mode):
  --kappas <v1,v2,...>                     Explicit list, e.g. 0.07,0.08,0.10
  --kappa-min <v> --kappa-max <v> --kappa-n <N>   Linspace of N points

  If no kappa argument is provided the script runs in explicit-ion mode:
  directories are named <system>-exp (first replica) and <system>-exp_2, _3, ...

System types:
  --systems <s1,s2,...>    Comma-separated; choices: isotropic,dipolar,quadrupolar,neutral
                           (default: isotropic)

Simulation parameters (passed to generate_input.py):
  --N <int>                Number of macroions (default: 30)
  --phi <float>            Volume fraction (default: 0.3)
  --micro-steps <int>      Production steps / 10 (default: 4000)
  --equil-micro-steps <int>  Equilibration steps / 10 (default: 2000)
  --potential <str>        coulomb or fanourgakis (default: fanourgakis)
  --traj                   Save trajectory to traj.xtc during production (default: off)
  --sdf                    Save Spatial distribution function to sdf.dx during production (default: off)
  --replicas <int>         Number of replicas per condition (default: 1, no suffix).
                           First replica keeps the base name; subsequent ones get _2, _3, ...
  --submit                 Actually run the simulations. Without this flag only input files
                           are generated (dry-run mode).

  -h, --help

Examples:
  $0 --kappas 0.070,0.080,0.0981 --systems isotropic,quadrupolar
  $0 --kappa-min 0.07 --kappa-max 0.12 --kappa-n 6 --systems isotropic
  $0 --systems isotropic,dipolar --replicas 3          # explicit-ion mode
EOF
}

#######################################
# Defaults
#######################################

SYSTEMS="isotropic"
N_MACROIONS=30
PHI=0.3
MICRO_STEPS=4000
EQUIL_MICRO_STEPS=2000
POTENTIAL="fanourgakis"
TRAJ=""
SDF=""
REPLICAS=1
SUBMIT=false
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
        --sdf)               SDF="--sdf";             shift 1 ;;
        --replicas)          REPLICAS="$2";           shift 2 ;;
        --submit)            SUBMIT=true;             shift 1 ;;
        -h|--help)           usage; exit 0 ;;
        *)                   echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

#######################################
# Build kappa array (or set explicit-ion mode)
#######################################

EXPLICIT_MODE=false
KAPPA_VALS=()
if [[ -n "$USER_KAPPAS" ]]; then
    IFS=',' read -ra KAPPA_VALS <<< "$USER_KAPPAS"
elif [[ -n "$KAPPA_MIN" && -n "$KAPPA_MAX" && -n "$KAPPA_N" ]]; then
    mapfile -t KAPPA_VALS < <("${PYTHON[@]}" -c "
import numpy as np
for v in np.linspace($KAPPA_MIN, $KAPPA_MAX, $KAPPA_N):
    print(f'{v:.6g}')
")
else
    EXPLICIT_MODE=true
    KAPPA_VALS=("explicit")   # sentinel — loop runs once
fi

IFS=',' read -ra SYSTEM_VALS <<< "$SYSTEMS"

TOTAL=$(( ${#SYSTEM_VALS[@]} * ${#KAPPA_VALS[@]} * REPLICAS ))
echo "Systems  : ${SYSTEM_VALS[*]}"
if [[ "$EXPLICIT_MODE" == "true" ]]; then
    echo "Mode     : explicit ions (no kappa)"
else
    echo "Kappas   : ${KAPPA_VALS[*]}"
fi
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

    local DIR_NAME LOG_TAG
    local KAPPA_ARGS=()

    if [[ "$EXPLICIT_MODE" == "true" ]]; then
        DIR_NAME="${SYSTEM}-exp${REPLICA}"
        LOG_TAG="${SYSTEM} exp${REPLICA}"
    else
        local KAPPA_FMT
        KAPPA_FMT=$(printf "%.4f" "$KAPPA")
        DIR_NAME="${SYSTEM}-imp_${KAPPA_FMT}${REPLICA}"
        LOG_TAG="${SYSTEM} κ=${KAPPA_FMT}${REPLICA}"
        KAPPA_ARGS=(--kappa "$KAPPA")
    fi

    local SIM_DIR="$PROJ_ROOT/data/runs/$DIR_NAME"

    echo "  [${LOG_TAG}] Generating input files"
    "${PYTHON[@]}" "$GENERATE" "$DIR_NAME" \
        --type "$SYSTEM" \
        --N "$N_MACROIONS" \
        --phi "$PHI" \
        ${KAPPA_ARGS[@]+"${KAPPA_ARGS[@]}"} \
        --micro-steps "$MICRO_STEPS" \
        --equil-micro-steps "$EQUIL_MICRO_STEPS" \
        --potential "$POTENTIAL" \
        ${TRAJ:+--traj} \
        ${SDF:+--sdf} \
        --force

    if [[ "$SUBMIT" == "true" ]]; then
        cd "$SIM_DIR"

        echo "  [${LOG_TAG}] Equilibration"
        "$FAUNUS" run -i input_equi.yml -s state.yaml

        echo "  [${LOG_TAG}] Production (resuming from state.yaml)"
        "$FAUNUS" run -i input.yml -s state.yaml

        echo "  [${LOG_TAG}] Done"
    else
        echo "  [${LOG_TAG}] Input files generated (dry-run; use --submit to run)"
    fi
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
