#!/bin/bash

# Project information

#SBATCH -A lu2026-2-71

# Walltime HH:MM:SS
#SBATCH -t 15:00:00

# Job name and output files

#SBATCH -J cppm_tripolar
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

# print the sbatch submission file

cat $0

# Resources request

#SBATCH -N 1
#SBATCH --ntasks-per-node=1

# NVIDIA A100 GPUs with AMD (Uncomment these line if GPU is needed)
##SBATCH -p gpua100

# Job notification (Uncomment these line if need)
##SBATCH --mail-user=sebastian.pineda_pineda@chem.lu.se
##SBATCH --mail-type=END # other optinal types:BEGIN,END,FAIL,REQUEUE,ALL

#######################################
# Log job resources and configuration
#######################################

echo "========================================"
echo "SLURM Job Information"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Job Submit Time: $(date)"
echo "User: $USER"
echo "Working Directory: $(pwd)"
echo ""
echo "========================================"
echo "CPU Configuration"
echo "========================================"
echo "Number of Nodes: $SLURM_NNODES"
echo "Tasks per Node: $SLURM_NTASKS_PER_NODE"
echo "Total Tasks: $SLURM_NTASKS"
echo "CPUs per Task: $SLURM_CPUS_PER_TASK"
echo "Walltime Limit: $SLURM_TIMELIMIT"
echo ""
echo "========================================"
echo "Node Information"
echo "========================================"
echo "Allocated Nodes: $SLURM_NODELIST"
echo ""
echo "========================================"
echo "Memory Configuration"
echo "========================================"
echo "Memory per Node: $SLURM_MEM_PER_NODE MB"
echo ""
echo "========================================"
echo "System Hardware Info"
echo "========================================"
lscpu | head -20
echo ""

module purge
module add GCCcore/12.3.0 
module add Python/3.11.3

time ../../faunus-rs/target/release/faunus run -i input.yml -s state.yaml

#######################################
# Log final job statistics
#######################################

echo ""
echo "========================================"
echo "Job Completion Information"
echo "========================================"
echo "Job End Time: $(date)"
echo "Exit Status: $?"
echo ""
echo "Note: To get detailed CPU time and efficiency statistics after job completion,"
echo "run the following command with your job ID:"
echo "  seff $SLURM_JOB_ID"
echo ""
echo "This will show:"
echo "  - CPU Efficiency"
echo "  - Memory Usage"
echo "  - Allocated Resources"
echo "  - Actual Usage"
echo "========================================"

