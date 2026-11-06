#!/bin/sh
set -eu

for model in \
    floorplan_unetpp/floorplan_multitask.onnx \
    floorplan_unetpp/floorplan_fallback.onnx \
    generator_perceptual_ema.onnx \
    facade512_experiment/generator.onnx \
    facade512_experiment/generator.json
do
    if [ ! -s "/app/models/saved/$model" ]; then
        printf 'Missing model %s. Fetch the DVC weights as described in README.md.\n' "$model" >&2
        exit 1
    fi
done

exec "$@"
