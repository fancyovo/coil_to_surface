#!/usr/bin/env bash
#SBATCH --account=stu
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_default
#SBATCH --job-name=prepare-desc-env
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

project=/home/scc/pb24511935/local_surface_evaluator
archive=/home/scc/pb24511935/local_surface_evaluator_data/desc_wheelhouse_linux.tar
wheelhouse=$project/.wheelhouse-desc016
bootstrap=$project/.virtualenv-bootstrap
environment=$project/.venv-desc016-py312

cd "$project"
if [[ ! -d "$wheelhouse" ]]; then
    mkdir -p "$wheelhouse"
    tar -xf "$archive" -C "$wheelhouse"
fi
if [[ ! -x "$environment/bin/python" ]]; then
    /home/scc/pb24511935/coil/.venv/bin/python -m pip install \
        --no-index --find-links "$wheelhouse" --target "$bootstrap" virtualenv
    PYTHONPATH="$bootstrap" /home/scc/pb24511935/coil/.venv/bin/python \
        -m virtualenv "$environment"
fi
source "$environment/bin/activate"
python -m pip install --no-index --find-links "$wheelhouse" \
    "desc-opt==0.16.0" "simsopt==1.10.6"
python - <<'PY'
import desc
import simsopt

print(f"DESC={desc.__version__}")
print(f"simsopt={simsopt.__version__}")
PY
