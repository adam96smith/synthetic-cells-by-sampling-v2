#!/bin/bash

DATASET_NAME='cell_doublets'
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATASET_PATH="$(realpath "$SCRIPT_DIR/../../..")"/

# # Training Data
# python data_generator/default/GeneratorMask.py \
#     --save-path synthetic_data/$DATASET_NAME/ \
#     --N 100 \
#     --dataset-id $DATASET_NAME \
#     --config config/$DATASET_NAME/synth_parameters.yaml \
#     --global-config config/$DATASET_NAME/global_parameters.yaml

# python data_generator/default/GeneratorImage.py \
#     --dataset-id $DATASET_NAME \
#     --mask-dir synthetic_data/$DATASET_NAME/ \
#     --sampler-dir data_lifeactgfp/ \
#     --sub-folder lifeactgfp_texture/ \
#     --config config/$DATASET_NAME/synth_parameters.yaml

# python data_generator/default/GeneratorImage.py \
#     --dataset-id $DATASET_NAME \
#     --mask-dir synthetic_data/$DATASET_NAME/ \
#     --sampler-dir data_tdTomato/ \
#     --sub-folder tdTomato_texture/ \
#     --config config/$DATASET_NAME/synth_parameters.yaml

python data_generator/default/GeneratorSignedDistance.py \
    --input-dir synthetic_data/$DATASET_NAME/ \
    --config config/$DATASET_NAME/synth_parameters.yaml \
    --global-config config/$DATASET_NAME/global_parameters.yaml 