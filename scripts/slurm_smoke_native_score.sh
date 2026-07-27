#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=smoke-native-score
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --exclude=anode02
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd /home/scc/pb24511935/local_surface_evaluator
mapfile -t compute_processes < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits |
        sed '/^[[:space:]]*$/d'
)
if (( ${#compute_processes[@]} != 0 )); then
    printf 'allocated GPU is not idle; compute PIDs: %s\n' "${compute_processes[*]}" >&2
    exit 42
fi

source /home/scc/pb24511935/coil/.venv/bin/activate
python scripts/smoke_native_score.py \
    /home/scc/pb24511935/local_surface_evaluator_data/volume_score_2000/cases/id_0206752.json \
    --metadata /home/scc/pb24511935/local_surface_evaluator_data/volume_score_2000/metadata_selected.json \
    --output runs/native_score/smoke_0206752.json
