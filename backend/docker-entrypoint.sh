#!/bin/sh
set -eu

python -c 'from src.models.artifacts import require_artifacts; require_artifacts()'

if [ -n "${ARCHIGEN_FACADE_EXPERIMENT:-}" ]; then
    test -s "$ARCHIGEN_FACADE_EXPERIMENT" || { echo "Selected experimental facade weights are missing" >&2; exit 1; }
fi

exec "$@"
