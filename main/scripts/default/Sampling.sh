#!/bin/bash

DATASET_NAME='DICTY'
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATASET_PATH="$(realpath "$SCRIPT_DIR/../../..")"/

# Get Sampler for all labelled data
python data_generator/default/GeneratorSampler.py \
    --data-root $DATASET_PATH \
    --data-folder Fluo-$DATASET_NAME-train/ \
    --dataset-id $DATASET_NAME 