from pathlib import Path
import numpy as np
from tifffile import imread
import glob
import os
import sys
import random

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

if os.getcwd() not in sys.path:
    (sys.path).append(os.getcwd())
    
random.seed(0)

from custom_model.unet import UNet3D
from custom_model.loss import BAILoss
from custom_model.cellpatchdataset import CellPatchDataset
from custom_model.augmentations import Compose3D, RandomFlip3D, ColorJitter3D, GaussianNoise3D, GaussianBlur3D

from utils import load_config

import argparse

parser = argparse.ArgumentParser(description='Run a network on Test Dataset.')
parser.add_argument('--data-root', type=str, required=True, help='Path to the base directory containing all datasets.')
parser.add_argument('--dataset-id', type=str, required=True, help='Short identifier for the dataset (e.g., H157). Used for logging, config selection, etc.')
parser.add_argument('--train-dir', type=str, required=True, help='Directory with Training Data in default structure.')
parser.add_argument('--config', type=str, required=True, help='YAML Config. File')
parser.add_argument('--global-config', type=str, default=None, help='Config. File for Sampler')

# redundant args
parser.add_argument('--disable-cuda', action='store_true', help='If True, train on CPU')
parser.add_argument('--eval-dir', default=None, help='Directory with Training Data in default structure.')
parser.add_argument('--model-dir', type=str, required=True, help='Directory to Save Trained Model.')
parser.add_argument('--model-name', type=str, required=True, help='Model Name.')

args = parser.parse_args()


# ---------------------------------------------------
# Functions
# ---------------------------------------------------


def train_epoch(model, loader, optimizer, criterion, scaler, device="cuda"):

    model.train()

    running_loss = 0.0

    pbar = tqdm(loader)

    for batch in pbar:

        x = batch["image"].to(device, non_blocking=True)
        y = batch["target"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device):

            pred = model(x)

            loss = criterion(pred, y)

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        running_loss += loss.item()

        pbar.set_description(f"loss={loss.item():.4f}")

    return running_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device="cuda"):

    model.eval()

    running_loss = 0.0

    for batch in tqdm(loader):

        x = batch["image"].to(device, non_blocking=True)
        y = batch["target"].to(device, non_blocking=True)

        with autocast(device):

            pred = model(x)

            loss = criterion(pred, y)

        running_loss += loss.item()

    return running_loss / len(loader)


def save_checkpoint(model, optimizer, scheduler, epoch, path):

    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch
    }, path)


# ---------------------------------------------------
# Parameters
# ---------------------------------------------------
config = load_config(args.config)
if args.global_config is None:
    global_params = load_config(f'config/{args.dataset_id}/global_parameters.yaml')
else:
    global_params = load_config(args.global_config)

sf = config['evaluation']['EVAL_SCALE']

sampling = global_params['SAMPLING']
anisotropy = sampling[0]/(sf*sampling[1])

patch_size=(32,64,64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------
# Load data
# ---------------------------------------------------

label_dir = '' ## image labels 1 level above textured images
for s in (args.train_dir).split('/')[:-2]:
    label_dir += s + '/'

# inputs
images = sorted(glob.glob(f'{args.train_dir}image_*.tif'))
    
# ground truth
labels = sorted(glob.glob(f'{label_dir}SDF/dist_map_*.tif'))

print(len(images), 'Images')
print(len(labels), 'Targets')

# Patch Creators    
valid_indices = random.sample(list(np.arange(len(images))), int(.3*len(images)))    


train_images = [img for i, img in enumerate(images) if i not in valid_indices]
train_labels = [lab for i, lab in enumerate(labels) if i not in valid_indices]

val_images = [img for i, img in enumerate(images) if i in valid_indices]
val_labels = [lab for i, lab in enumerate(labels) if i in valid_indices]

# aumentations 
transforms = Compose3D([RandomFlip3D(p=0.5),
                        ColorJitter3D(brightness=0.2, contrast=0.2, p=0.8),
                        GaussianNoise3D(sigma=(0.0, 0.05), p=0.5),
                        GaussianBlur3D(sigma=(2.5, 1.5, 1.5), p=0.75)
                       ])

train_dataset = CellPatchDataset([img for i, img in enumerate(images) if i not in valid_indices],
                                 [lab for i, lab in enumerate(labels) if i not in valid_indices],
                                 patch_size=patch_size, 
                                 transform=transforms,
                                )

val_dataset = CellPatchDataset([img for i, img in enumerate(images) if i in valid_indices],
                               [lab for i, lab in enumerate(labels) if i in valid_indices],
                               patch_size=patch_size,   
                               transform=transforms,
                              )


# Dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=2,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=2,
)

# ---------------------------------------------------
# Create model
# ---------------------------------------------------

model = UNet3D(depth=5, planar_layers=1).to(device).float()

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=.05,  # Learning rate is set by the lr_sched below
    weight_decay=0.5e-4,
)

# LR Scheduler
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
                                              base_lr=1e-6, # 0.0001
                                              max_lr=1e-3, # 0.01
                                              step_size_up=500,
                                              step_size_down=1500,
                                              cycle_momentum=True if 'momentum' in optimizer.defaults else False
                                              )

criterion = BAILoss(alpha=4, beta=0)

scaler = GradScaler("cuda")

num_epochs = 100

best_val = float("inf")

checkpoint_dir = "checkpoints/"
os.makedirs(checkpoint_dir, exist_ok=True)

for epoch in range(num_epochs):

    print(f"\nEpoch {epoch}")

    train_loss = train_epoch(
        model,
        train_loader,
        optimizer,
        criterion,
        scaler
    )

    val_loss = validate(
        model,
        val_loader,
        criterion
    )

    scheduler.step()

    print(f"train_loss: {train_loss:.4f}")
    print(f"val_loss:   {val_loss:.4f}")

    # latest checkpoint
    save_checkpoint(
        model,
        optimizer,
        scheduler,
        epoch,
        checkpoint_dir + "latest.pt"
    )

    # best checkpoint
    if val_loss < best_val:

        best_val = val_loss

        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch,
            checkpoint_dir + "best.pt"
        )

        print("saved best checkpoint")