#!/bin/bash

DATASET_NAME='cell_doublets'
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATASET_PATH="$(realpath "$SCRIPT_DIR/../../..")"/

# Run Final for CTC
python model_codes/default/ModelRun.py \
    --data-root $DATASET_PATH \
    --dataset-id $DATASET_NAME \
    --test-dir Fluo-lifeactgfp-train/ \
    --model-dir models/$DATASET_NAME/ \
    --model-name model_lifeactgfp \
    --model-version model_best.pt \
    --config config/$DATASET_NAME/model_train.yaml 

# python model_codes/default/ModelRun.py \
#     --data-root $DATASET_PATH \
#     --dataset-id $DATASET_NAME \
#     --test-dir Fluo-tdTomato-train/ \
#     --model-dir models/$DATASET_NAME/ \
#     --model-name model_tdTomato \
#     --model-version model_best.pt \
#     --config config/$DATASET_NAME/model_train.yaml 