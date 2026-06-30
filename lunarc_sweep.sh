#!/bin/bash

# Project information
#SBATCH -A lu2026-2-71

# Walltime HH:MM:SS
#SBATCH -t 08:00:00

# Job name and output files
#SBATCH -J cppm_sweep_quadrupolar
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

# print the sbatch submission file
cat $0

# Resources request
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
# Note: --cpus-per-task is set dynamically by launch_sweep.sh

##SBATCH -p gpua100
##SBATCH --mail-user=sebastian.pineda_pineda@chem.lu.se
##SBATCH --mail-type=END

#######################################
# Log job resources and configuration
#######################################

echo "========================================"
echo "SLURM Job Information"
echo "========================================"
echo "Job ID:            $SLURM_JOB_ID"
echo "Job Name:          $SLURM_JOB_NAME"
echo "Job Submit Time:   $(date)"
echo "User:              $USER"
echo "Working Directory: $(pwd)"
echo ""
echo "========================================"
echo "CPU Configuration"
echo "========================================"
echo "Number of Nodes:   $SLURM_NNODES"
echo "Tasks per Node:    $SLURM_NTASKS_PER_NODE"
echo "Total Tasks:       $SLURM_NTASKS"
echo "CPUs per Task:     $SLURM_CPUS_PER_TASK"
echo "Walltime Limit:    $SLURM_TIMELIMIT"
echo ""
echo "========================================"
echo "Node Information"
echo "========================================"
echo "Allocated Nodes:   $SLURM_NODELIST"
echo ""
echo "========================================"
echo "Memory Configuration"
echo "========================================"
echo "Memory per Node:   $SLURM_MEM_PER_NODE MB"
echo ""
echo "========================================"
echo "System Hardware Info"
echo "========================================"
lscpu | head -20
echo ""

module purge
module add GCCcore/12.3.0
module add Python/3.11.3

time "$SLURM_SUBMIT_DIR/run_sweep.sh" "$@"

#######################################
# Log final job statistics
#######################################

echo ""
echo "========================================"
echo "Job Completion Information"
echo "========================================"
echo "Job End Time: $(date)"
echo "Exit Status:  $?"
echo ""
echo "Run: seff $SLURM_JOB_ID   for CPU efficiency and memory stats."
echo "========================================"

mv *.err *.out "$SLURM_SUBMIT_DIR/" 2>/dev/null || true
