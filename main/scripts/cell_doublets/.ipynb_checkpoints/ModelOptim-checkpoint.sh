#!/bin/bash

DATASET_NAME='cell_doublets'
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATASET_PATH="$(realpath "$SCRIPT_DIR/../../..")"/
MODEL_VERSION='model_best.pt'

# Submitted Model
python model_codes/default/ModelOptim.py \
    --data-root $DATASET_PATH \
    --dataset-id $DATASET_NAME \
    --test-dir Fluo-lifeactgfp-train/ \
    --model models/$DATASET_NAME/model_lifeactgfp/model_best.pt \
    --config config/$DATASET_NAME/model_train.yaml 