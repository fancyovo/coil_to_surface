#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=adam-dirty-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:RTX5090:4
#SBATCH --mem=128G
#SBATCH --time=00:20:00
#SBATCH --array=0-6%1
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

project="${PROJECT:?PROJECT is required}"
run_root="${RUN_ROOT:?RUN_ROOT is required}"
initial_case="${INITIAL_CASE:?INITIAL_CASE is required}"
task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

names=(
  beta09_raw_c010
  beta07_raw_c010
  beta05_raw_c010
  beta09_robust_c010
  beta05_robust_c010
  beta05_robust_c005
  beta05_robust_c015
)
beta1=(0.9 0.7 0.5 0.9 0.5 0.5 0.5)
robust=(0 0 0 1 1 1 1)
perturbation=(0.01 0.01 0.01 0.01 0.01 0.005 0.015)

if (( task_id < 0 || task_id >= ${#names[@]} )); then
  printf 'unsupported array task: %s\n' "$task_id" >&2
  exit 2
fi

export PROJECT="$project"
export RUN_ROOT="$run_root/${names[$task_id]}"
export INITIAL_CASE="$initial_case"
export ITERATIONS=16
export MAX_WALL_S=1100
export LEARNING_RATE=0.01
export SEED=2026080101
export BETA1="${beta1[$task_id]}"
export BETA2=0.999
export ROBUST_DIRECTION_FILTER="${robust[$task_id]}"
export PERTURBATION="${perturbation[$task_id]}"

exec bash "$project/scripts/slurm_flow_prior_standard_adam.sh"
