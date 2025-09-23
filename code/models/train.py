# code/models/train_debug.py
import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image
from unet import UNet

DATA_DIR = "data/processed"
IMG_SIZE = 64
BATCH_SIZE = 10
EPOCHS = 5
SUBSET_SIZE = 100
MASK_MAX_AREA_RATIO = 0.3
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")

random.seed(SEED)
torch.manual_seed(SEED)

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
dataset = Subset(dataset, range(min(SUBSET_SIZE, len(dataset))))

def mask_random_square(img, max_area_ratio=MASK_MAX_AREA_RATIO, fill=0.0):
    c, h, w = img.shape
    mask_img = img.clone()
    max_area = int(max_area_ratio * h * w)
    side_max = int(max_area ** 0.5)
    size_min = max(1, h // 8)
    size_max = max(size_min, side_max)
    size = random.randint(size_min, min(size_max, min(h, w)))
    y = random.randint(0, h - size)
    x = random.randint(0, w - size)
    mask_img[:, y:y+size, x:x+size] = fill
    return mask_img

class MaskedDataset(torch.utils.data.Dataset):
    def __init__(self, base_ds):
        self.ds = base_ds
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        img, _ = self.ds[idx]
        masked = mask_random_square(img)
        return masked, img

loader = DataLoader(MaskedDataset(dataset), batch_size=BATCH_SIZE, shuffle=True)

model = UNet().to(DEVICE)
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for masked, real in loader:
        masked = masked.to(DEVICE)
        real = real.to(DEVICE)

        output = model(masked)
        loss = criterion(output, real)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"[AIRFLOW][Epoch {epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f}")
