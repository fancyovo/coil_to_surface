#!/usr/bin/env bash
#SBATCH --account=competition
#SBATCH --partition=P107-RTX5090
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --job-name=lse-vqs-40
#SBATCH --chdir=/home/scc/pb24511935/local_surface_evaluator
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
project_root=/home/scc/pb24511935/local_surface_evaluator
cd "$project_root"

module load miniconda/py312 cuda/13.0 2>/dev/null || true
source /home/scc/pb24511935/coil/.venv/bin/activate
export CUDA_HOME=/public/app/cuda/13.0
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python - <<'PY'
import scipy
import simsopt
import torch

print("scipy", scipy.__version__)
print("simsopt", simsopt.__version__)
print("torch", torch.__version__, "cuda", torch.version.cuda)
PY

for build_dir in build_mixed build_volume_qs; do
    cmake -S "$project_root/gpu_backend" -B "$project_root/gpu_backend/$build_dir" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_COMPILER="$CUDACXX" \
        -DCUDAToolkit_ROOT="$CUDA_HOME" \
        -DCMAKE_CUDA_ARCHITECTURES=120
    cmake --build "$project_root/gpu_backend/$build_dir" -j "$SLURM_CPUS_PER_TASK"
done

output_dir="$project_root/runs/volume_qs_quasr_idle_final_40_cluster_threads1"
mkdir -p "$output_dir"
{
    date -Iseconds
    echo "SLURM_JOB_ID=$SLURM_JOB_ID"
    echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
    nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
        --format=csv,noheader,nounits
    echo "__COMPUTE__"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name \
        --format=csv,noheader,nounits
    echo "__LIBRARIES__"
    ldd "$project_root/gpu_backend/build_mixed/libstellarator_gpu.so"
    ldd "$project_root/gpu_backend/build_volume_qs/libstellarator_gpu.so"
} > "$output_dir/gpu_preflight.txt"

python "$project_root/scripts/validate_field_gradient.py" \
    --case "$project_root/examples/01.json" \
    --key raw \
    --current-unit MA \
    --gpu-lib "$project_root/gpu_backend/build_volume_qs/libstellarator_gpu.so" \
    --device 0 \
    --points 128 \
    --output "$output_dir/field_gradient_audit.json"

python "$project_root/scripts/batch_volume_qs_quasr.py" \
    --case-dir /home/scc/pb24511935/local_surface_evaluator_data/volume_qs_40/cases \
    --metadata /home/scc/pb24511935/local_surface_evaluator_data/volume_qs_40/metadata_selected.json \
    --output-dir "$output_dir" \
    --per-helicity 20 \
    --points 100000 \
    --alpha-fit-points 30000 \
    --alpha-order 12 \
    --precision fp32 \
    --threads 1

{
    date -Iseconds
    nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
        --format=csv,noheader,nounits
} > "$output_dir/gpu_postflight.txt"
