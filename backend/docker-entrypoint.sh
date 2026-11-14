#!/bin/sh
set -eu

if ! python -c 'from src.models.artifacts import require_artifacts; require_artifacts()'; then
    echo 'Restore Docker models with: docker compose run --rm model-init' >&2
    exit 1
fi

if [ -n "${ARCHIGEN_FACADE_EXPERIMENT:-}" ]; then
    test -s "$ARCHIGEN_FACADE_EXPERIMENT" || { echo "Selected experimental facade weights are missing" >&2; exit 1; }
fi

exec "$@"
