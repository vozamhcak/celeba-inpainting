import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from unet import UNet
import json

DATA_DIR = "data/processed"
MODEL_PATH = "models/inpainting_unet.pt"
OUTPUT_DIR = "results"
BATCH_SIZE = 64
LR = 1e-4
EPOCHS = 20
IMG_SIZE = 64
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(root=os.path.join(DATA_DIR, "train"), transform=transform)
test_dataset = datasets.ImageFolder(root=os.path.join(DATA_DIR, "test"), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

def train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    model = UNet().to(DEVICE)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_loss = float("inf")
    train_log = []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for masked, real in train_loader:
            masked = masked.to(DEVICE)
            real = real.to(DEVICE)

            output = model(masked)
            loss = criterion(output, real)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f}")

        train_log.append({"epoch": epoch + 1, "loss": avg_loss})

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Model saved with loss: {best_loss:.4f}")

    with open(os.path.join(OUTPUT_DIR, "train_log.json"), "w") as f:
        json.dump(train_log, f, indent=2)

if __name__ == "__main__":
    train()
