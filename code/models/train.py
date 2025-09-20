import os
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image

from unet import UNet  

DATA_DIR    = "data/celeba"
OUTPUT_DIR  = "results"
MODEL_PATH  = "models/inpainting_unet.pt"
SEED        = 42

IMG_SIZE    = 32              
BATCH_SIZE  = 64
LR          = 1e-4
EPOCHS      = 20              
USE_SUBSET  = True
SUBSET_SIZE = 20000           

MASK_MAX_AREA_RATIO = 0.30    

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = get_device()
print(f"[INFO] Using device: {DEVICE}")


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
set_seed(SEED)

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),            
    
])

dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
if USE_SUBSET:
    dataset = Subset(dataset, range(min(SUBSET_SIZE, len(dataset))))

def mask_random_square(img, max_area_ratio=MASK_MAX_AREA_RATIO, fill=0.0):
    """
    img: Tensor [C,H,W] в [0,1]
    Рисуем чёрный квадрат со случайным размером и позицией.
    """
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

class MaskedCelebA(torch.utils.data.Dataset):
    def __init__(self, base_ds):
        self.ds = base_ds
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        img, _ = self.ds[idx]
        masked = mask_random_square(img)
        return masked, img

def train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    masked_dataset = MaskedCelebA(dataset)
    loader = DataLoader(
        masked_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,         
        pin_memory=False
    )

    model = UNet().to(DEVICE)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for masked, real in loader:
            masked = masked.to(DEVICE, non_blocking=True)
            real   = real.to(DEVICE,   non_blocking=True)

            output = model(masked)
            loss = criterion(output, real)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(loader))
        print(f"Epoch [{epoch+1}/{EPOCHS}]  Loss: {avg_loss:.4f}")

        model.eval()
        with torch.no_grad():
            sample_masked, sample_real = next(iter(loader))
            sample_masked = sample_masked.to(DEVICE)
            sample_out = model(sample_masked).detach().cpu()
            grid = torch.cat([sample_masked.detach().cpu()[:8], sample_out[:8], sample_real[:8]], dim=0)
            save_image(grid, f"{OUTPUT_DIR}/epoch_{epoch+1:03d}.png", nrow=8)

        if avg_loss < best_loss:
            best_loss = avg_loss
            payload = {
                "state_dict": model.state_dict(),
                "config": {
                    "img_size": IMG_SIZE,
                    "seed": SEED,
                    "arch": "UNet",
                }
            }
            torch.save(payload, MODEL_PATH)
            with open(os.path.join(OUTPUT_DIR, "train_config.json"), "w") as f:
                json.dump({
                    "epoch": epoch+1,
                    "best_loss": best_loss,
                    "img_size": IMG_SIZE,
                    "batch_size": BATCH_SIZE,
                    "lr": LR,
                    "subset": USE_SUBSET,
                    "subset_size": SUBSET_SIZE,
                    "device": str(DEVICE)
                }, f, indent=2)
            print(f"Saved best model to {MODEL_PATH} (loss={best_loss:.4f})")

if __name__ == "__main__":
    train()
