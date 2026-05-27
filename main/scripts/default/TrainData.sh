#!/bin/bash

DATASET_NAME='DICTY'
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATASET_PATH="$(realpath "$SCRIPT_DIR/../../..")"/

# # Training Data
python data_generator/default/GeneratorMask.py \
    --save-path synthetic_data/TOY_A549/ \
    --N 100 \
    --dataset-id $DATASET_NAME \
    --config config/$DATASET_NAME/A549/synth_parameters_A.yaml \
    --global-config config/$DATASET_NAME/A549/global_parameters.yaml

python data_generator/default/GeneratorImage.py \
    --dataset-id $DATASET_NAME \
    --mask-dir synthetic_data/$DATASET_NAME/ \
    --sampler-dir data_$DATASET_NAME/ \
    --sub-folder aug_texture/ \
    --config config/$DATASET_NAME/synth_parameters_B.yaml
