import os
import sys
import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

# Project paths
ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT / "preprocessing"))
sys.path.insert(0, str(ROOT / "M3-P4" / "models" / "cnn"))
sys.path.insert(0, str(ROOT / "training"))

from dataset import SpeechEnhancementDataset
from model import SpeechEnhancementCNN


# --------------------------------------------------
# Configuration
# --------------------------------------------------

SEED = 42
BATCH_SIZE = 4
EPOCHS = 5
LEARNING_RATE = 0.001

DATASET_DIR = Path(
    Path(os.environ.get("M3_P2_DATASET_DIR", ROOT / "dataset" / "data"))
)

MIXTURE_DIR = DATASET_DIR / "mixtures"
SPEECH_DIR = DATASET_DIR / "speech"

MODEL_DIR = ROOT / "M3-P4" / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "trained_model.pth"
METRICS_PATH = MODEL_DIR / "training_metrics.csv"


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device("cpu")

print("M3-P4 CNN TRAINING")
print("==================")
print("Device       :", device)
print("Random seed  :", SEED)
print("Batch size   :", BATCH_SIZE)
print("Epochs       :", EPOCHS)
print("Learning rate:", LEARNING_RATE)


# --------------------------------------------------
# Dataset
# --------------------------------------------------

dataset = SpeechEnhancementDataset(
    MIXTURE_DIR,
    SPEECH_DIR
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)

print()
print("Total samples:", len(dataset))
print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))


# --------------------------------------------------
# DataLoaders
# --------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# --------------------------------------------------
# Model
# --------------------------------------------------

model = SpeechEnhancementCNN().to(device)

parameter_count = sum(
    p.numel() for p in model.parameters()
)

print("Model parameters:", parameter_count)


# --------------------------------------------------
# Loss + optimizer
# --------------------------------------------------

criterion = torch.nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# --------------------------------------------------
# Metrics file
# --------------------------------------------------

with open(METRICS_PATH, "w", newline="", encoding="utf-8") as f:

    writer = csv.writer(f)

    writer.writerow([
        "epoch",
        "train_loss",
        "validation_loss"
    ])


# --------------------------------------------------
# Training
# --------------------------------------------------

for epoch in range(1, EPOCHS + 1):

    model.train()

    train_loss = 0.0

    for inputs, targets in train_loader:

        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, targets)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)


    # ----------------------------------------------
    # Validation
    # ----------------------------------------------

    model.eval()

    validation_loss = 0.0

    with torch.no_grad():

        for inputs, targets in val_loader:

            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, targets)

            validation_loss += loss.item()

    validation_loss /= len(val_loader)


    print(
        f"Epoch {epoch}/{EPOCHS} "
        f"| Train Loss: {train_loss:.6f} "
        f"| Validation Loss: {validation_loss:.6f}"
    )


    with open(
        METRICS_PATH,
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            epoch,
            train_loss,
            validation_loss
        ])


# --------------------------------------------------
# Save model
# --------------------------------------------------

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "sample_rate": 16000,
        "n_fft": 512,
        "win_length": 512,
        "hop_length": 128,
        "parameter_count": parameter_count,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "random_seed": SEED
    },
    MODEL_PATH
)


print()
print("Training complete.")
print("Model saved  :", MODEL_PATH)
print("Metrics saved:", METRICS_PATH)


