#!/usr/bin/env bash

set -euo pipefail

: "${REPORT:?REPORT is required}"
: "${DESC_DIR:?DESC_DIR is required}"

project=${PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
target_helicity=${TARGET_HELICITY:-QH}
asset_dir=$(dirname "$DESC_DIR")/assets

required=(
    "$DESC_DIR/summary.json"
    "$DESC_DIR/input.check"
    "$DESC_DIR/equilibrium.h5"
    "$asset_dir/poincare.png"
    "$asset_dir/boozer_b.png"
    "$asset_dir/boozer_b.html"
    "$asset_dir/coils_surface.png"
    "$asset_dir/coils_surface.html"
)
for path in "${required[@]}"; do
    test -s "$path" || {
        printf 'missing or empty required artifact: %s\n' "$path" >&2
        exit 1
    }
done

python3 "$project/scripts/validate_desc_report_artifacts.py" --report "$REPORT" --desc-dir "$DESC_DIR" --target-helicity "$target_helicity"
