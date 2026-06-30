#!/usr/bin/env bash
# launch_sweep.sh — count simulations, then submit lunarc_sweep.sh with the right --cpus-per-task.
# Accepts the same arguments as run_sweep.sh.
# For local runs, use run_sweep.sh directly.

set -euo pipefail

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]
Same options as run_sweep.sh. Counts total simulations and submits to SLURM
with --cpus-per-task=N so each parallel job gets its own CPU.

Examples:
  $0 --kappas 0.070,0.080,0.0981 --systems isotropic,quadrupolar
  $0 --kappa-min 0.07 --kappa-max 0.12 --kappa-n 6 --systems isotropic,dipolar,quadrupolar
EOF
}

#######################################
# Parse just the sweep args to count N
#######################################

USER_KAPPAS=""
KAPPA_MIN="" KAPPA_MAX="" KAPPA_N=""
SYSTEMS="isotropic"

_args=("$@")
i=0
while [[ $i -lt ${#_args[@]} ]]; do
    case "${_args[$i]}" in
        --kappas)    USER_KAPPAS="${_args[$((i+1))]}"; ((i+=2)) ;;
        --kappa-min) KAPPA_MIN="${_args[$((i+1))]}";   ((i+=2)) ;;
        --kappa-max) KAPPA_MAX="${_args[$((i+1))]}";   ((i+=2)) ;;
        --kappa-n)   KAPPA_N="${_args[$((i+1))]}";     ((i+=2)) ;;
        --systems)   SYSTEMS="${_args[$((i+1))]}";     ((i+=2)) ;;
        -h|--help)   usage; exit 0 ;;
        *)           ((i+=1)) ;;
    esac
done

#######################################
# Count kappa points
#######################################

if [[ -n "$USER_KAPPAS" ]]; then
    N_KAPPA=$(echo "$USER_KAPPAS" | tr ',' '\n' | wc -l)
elif [[ -n "$KAPPA_MIN" && -n "$KAPPA_MAX" && -n "$KAPPA_N" ]]; then
    N_KAPPA="$KAPPA_N"
else
    echo "ERROR: provide --kappas or --kappa-min/--kappa-max/--kappa-n" >&2
    usage; exit 1
fi

N_SYSTEMS=$(echo "$SYSTEMS" | tr ',' '\n' | wc -l)
N=$(( N_KAPPA * N_SYSTEMS ))

echo "Submitting $N parallel simulation(s) (${N_SYSTEMS} system(s) x ${N_KAPPA} kappa point(s))"
echo "  --cpus-per-task=$N"

sbatch --cpus-per-task="$N" \
    "$(dirname "$0")/lunarc_sweep.sh" "$@"
