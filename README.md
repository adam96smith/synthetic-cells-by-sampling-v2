# Synthetic Cells by Sampling: Full Pipeline for Cell Segmentation

This repository provides a complete pipeline for generating **synthetic training images of A549 cells with branching filopodia**, training a **segmentation model**, and evaluating the trained model on **real data**.
The goal is to leverage synthetic data to improve segmentation performance on real-world images.

The pipeline is organized into **four main execution stages**, each controlled by a Bash script and parameterized via configuration files.

---

## Repository Structure

```
.
├── Fluo-C3DH-A549/               # Test data from Cell Tracking Challenge
├── Fluo-C3DH-A549-train/         # Annotated training data from Cell Tracking Challenge
├── main/                         # All executable scripts (run commands from here)
│   ├── scripts/
│   │   └── default/
│   │       ├── Sampling.sh       # Sample intensities from annotated data (eg. Fluo-C3DH-A549-train/)
│   │       ├── ModelTrain.sh     # Train segmentation model on dataset stored in synthetic_data/
│   │       ├── RunFinal.sh       # Run inference on real data (Fluo-C3DH-A549/ or Fluo-C3DH-A549-train/)
│   │       └── ...
│   │   └── A549/
│   │       ├── TrainData.sh      # Generate dataset with synthetic images with labels
│   │       └── ...
│   ├── config/                   # Configuration files for each stage
│   ├── data_generator/           # Scripts for sampling, and shape + image generation
│   ├── model_codes/              # Scripts for model training and inference
│   ├── synthetic_data/           # Where synthetic training images are stored
│   ├── models/                   # Where trained models are stored
│   ├── utils/
│   └── ...
```

---

## Pipeline Overview

The full workflow consists of the following steps:

1. **Synthetic Data Sampling**
2. **Training Data Generation**
3. **Model Training**
4. **Final Evaluation / Inference**

Each step:

* Is executed via a Bash script
* Loads parameters from configuration files located in `main/config/`
* Produces outputs that feed into the next stage
---

## 0. Requirements

Execute the bash script for setting up the `synth-cell-env`:

```
cd main
bash prepare_software.sh
```

Alternatively, use `uv` to install all dependencies (much faster!).

```
cd main
uv pip install -r requirements.txt
```

Then, activate the environment

```
source synth-cell-env/bin/activate
```

**Important:** All scripts must be executed **from the `main/` directory**

---

## 1. Synthetic Data Sampling

**Script:**

```bash
bash scripts/default/Sampling.sh
```

### Purpose

Using ground truth annotations, the intensity values from real images are grouped based on intervals of a distance map.
The formatted data is used to texture synthetic images (stored in `data_generator/sampled_data/data_A549/`)

### Parameters

Located in:

```
main/config/global_parameters.yaml
```

| Parameter      | Description                          |
| -------------- | ------------------------------------ |
| `SAMPLING`     | Voxel size in $\mu m$                |
| `DX`           | Distance interval width              |
| `MIN_DIST`     | Minimum distance sampled             |
| `MAX_DIST`     | Maximum distance sampled             |

**Recommended**: DX should be no lower than 2x the in-plane sampling.

## 2. Training Data Preparation

**Script:**

```bash
bash scripts/default/TrainData.sh
```

### Purpose

This script runs GeneratorMask.py to get cell shapes, GeneratorImage.py to get synthetic images and GeneratorCurveSeg.py for curvature mask.
All images are stored in `synthetic_data/A549/`, with images inside `custom_texture/`. 
**Note:**: Custom script for A549 cells is 'data_generator/custom_A549/GeneratorMask.py`, the alternative in `default/` generates generic masks with specified shape.


### Parameters

Located in:

```
main/config/synth_parameters.yaml
```

| Parameter        | Description                                      |
| ---------------- | ------------------------------------------------ |
|  `IMAGE_SIZE`    | Size of synthetic images                         |
|  `DISTMAP_BLUR`  | If True, blur the distance map used in sampling  |
|  `DISTMAP_SIG`   | Sigma used to blur distances (**in $\mu m$**)    |
|  `GAUSSIAN_BLUR` | If True, apply blur to the final image           |
|  `GAUSSIAN_SIG`  | Sigma used to blur image (**in pixels**)         |

**Note:** Applying a blur to the distance map reduces the block effect caused by sampling at distance intervals.

## 3. Model Training

**Script:**

```bash
bash scripts/default/ModelTrain.sh
```

### Purpose

Trains the segmentation model using the synthetic dataset.

### Key Outputs

* Trained model checkpoints
* Training logs
* Loss and metric curves

### Parameters

Located in:

```
main/config/model_train.yaml
```

| Parameter       | Description                                                                        | 
| --------------- | ---------------------------------------------------------------------------------- | 
| `PATCH_SHAPE'   | Size of patches sampled during training                                            |
| `WEIGHTS'       | Weights for custom loss function (Background, Foreground, high Curvature Regions)  |
| `STEP_SIZE'     | Number of training iterations between **evaluation** on real data                  | 
| `BURN_IN'       | Number of training iterations before first **evaluation**                          |
| `EVAL_PATIENCE' | Number of evaluations averaged to assess training progress                         | 
| `EVAL_TARGET'   | Target IoU score average that prompts termination of model training                |

**Note:** Default `EVAL_TARGET' set high so training not terminated.

## 4. Final Evaluation / Inference

**Script:**

```bash
bash scripts/default/RunFinal.sh
```

### Purpose

Runs the trained model on **real data** to generate final predictions.

#### Args (see RunFinal.sh)

| Parameter         | Description                                                              | 
| ----------------- | ------------------------------------------------------------------------ |
| `VERSION`         | Specify the checkpoint to use for inference                              |
| `TEST_DIR`        | Test directory in root folder (eg. `Fluo-C3Dh-A549/`)                    |
| '--model-name'    | Model name (found in `models/`)                                          |
| `--final`         | Prompts the script to save output compatible for Cell Tracking Challenge | 

**Note:** Training script saves 3 model outputs: model.pt (final model), and model_peak.pt (peak evaluation score) and model_best.pt (peak moving average).

---

## Execution Order Summary

```bash
cd main

bash prepare_software.sh
source synth-cell-env/bin/activate
bash scripts/default/Sampling.sh
bash scripts/default/TrainData.sh
bash scripts/default/ModelTrain.sh
bash scripts/default/RunFinal.sh
```


