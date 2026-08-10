#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=score-nsys-stat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

profile_dir="${PROFILE_DIR:?PROFILE_DIR must name the completed profiling output directory}"
nsys_bin="${NSYS_BIN:-/public/app/cuda/13.0/bin/nsys}"
for report in nvtx_sum nvtx_gpu_proj_sum cuda_gpu_kern_sum cuda_api_sum; do
    "$nsys_bin" stats --force-export=true --report "$report" --format csv \
        "$profile_dir/profile.nsys-rep" > "$profile_dir/${report}.csv"
done
printf '{"status":"ok","profile_dir":"%s"}\n' "$profile_dir" \
    > "$profile_dir/nsys_stats_done.json"
