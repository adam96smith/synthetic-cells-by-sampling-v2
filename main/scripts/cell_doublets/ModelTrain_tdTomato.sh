#!/bin/bash

DATASET_NAME='cell_doublets'
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATASET_PATH="$(realpath "$SCRIPT_DIR/../../..")"/

# # Train Model

# python model_codes/default/ModelTrain.py \
#     --data-root $DATASET_PATH \
#     --dataset-id $DATASET_NAME \
#     --train-dir synthetic_data/$DATASET_NAME/tdTomato_texture/ \
#     --eval-dir Fluo-tdTomato-train/ \
#     --model-dir models/$DATASET_NAME/ \
#     --model-name model_tdTomato \
#     --config config/$DATASET_NAME/model_train.yaml

python model_codes/default/ModelTrain.py \
    --data-root $DATASET_PATH \
    --dataset-id $DATASET_NAME \
    --train-dir synthetic_data/$DATASET_NAME/tdTomato_texture/ \
    --eval-dir Fluo-tdTomato-train/ \
    --model-dir models/$DATASET_NAME/ \
    --model-name model_tdTomato \
    --config config/$DATASET_NAME/model_train_resume_tdT.yaml