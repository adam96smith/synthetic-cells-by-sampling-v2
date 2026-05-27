#!/bin/bash

DATASET_NAME='cell_doublets'
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATASET_PATH="$(realpath "$SCRIPT_DIR/../../..")"/

# Get Sampler for all labelled data
python data_generator/default/GeneratorSampler.py \
    --data-root $DATASET_PATH \
    --data-folder Fluo-lifeactgfp-train/ \
    --dataset-id cell_doublets \
    --output-dir data_lifeactgfp/

# Get Sampler for all labelled data
python data_generator/default/GeneratorSampler.py \
    --data-root $DATASET_PATH \
    --data-folder Fluo-tdTomato-train/ \
    --dataset-id cell_doublets \
    --output-dir data_tdTomato/