#!/usr/bin/env bash
#SBATCH --account=stu
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_default
#SBATCH --job-name=qh-equals-launch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
: "${PROJECT:?PROJECT is required}"
: "${EXPERIMENT_ROOT:?EXPERIMENT_ROOT is required}"
: "${GPU_LIB:?GPU_LIB is required}"

cd "$PROJECT"
dependency=""
sbatch --test-only --account=competition --partition=P107-RTX5090 \
  --qos=qos_p107-rtx5090 --cpus-per-task=4 --gres=gpu:RTX5090:1 \
  --array=0-3 --job-name=qh-equals-data --export=ALL,SHARD_COUNT=4,N_PHI=64,N_THETA=64 \
  scripts/slurm_qh_equal_s_surface_qs.sh
equal_job=$(sbatch --parsable --account=competition --partition=P107-RTX5090 \
  --qos=qos_p107-rtx5090 --cpus-per-task=4 --gres=gpu:RTX5090:1 \
  --array=0-3 --job-name=qh-equals-data --export=ALL,SHARD_COUNT=4,N_PHI=64,N_THETA=64 \
  scripts/slurm_qh_equal_s_surface_qs.sh)

sbatch --test-only --dependency="afterok:$equal_job" --export=ALL \
  scripts/slurm_analyze_qh_equal_s_surface_qs.sh
analysis_job=$(sbatch --parsable --dependency="afterok:$equal_job" --export=ALL \
  scripts/slurm_analyze_qh_equal_s_surface_qs.sh)
printf 'equal_s=%s\nequal_s_analysis=%s\n' "$equal_job" "$analysis_job" \
  | tee "$EXPERIMENT_ROOT/equal_s_submitted_jobs.txt"
