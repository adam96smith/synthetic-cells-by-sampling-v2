#!/bin/bash

DATASET_NAME='SHAINY1'
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATASET_PATH="$(realpath "$SCRIPT_DIR/../../..")"/

# Run Final for CTC
python model_codes/default/ModelRun.py \
    --data-root $DATASET_PATH \
    --dataset-id $DATASET_NAME \
    --test-dir Fluo-SHAINY1-control/ \
    --model-dir models/$DATASET_NAME/ \
    --model-name model_v3 \
    --model-version model_best.pt \
    --config config/$DATASET_NAME/model_train.yaml 