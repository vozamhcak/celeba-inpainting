import os
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from PIL import Image

to_pil = transforms.ToPILImage()

RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
IMG_SIZE = 64
BATCH_SIZE = 64
TEST_SIZE = 0.2

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(PROCESSED_DATA_DIR, "train"), exist_ok=True)
os.makedirs(os.path.join(PROCESSED_DATA_DIR, "test"), exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(root=RAW_DATA_DIR, transform=transform)

train_size = int((1 - TEST_SIZE) * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

def save_dataset(dataset, folder):
    for i, (img, _) in enumerate(dataset):
        img = to_pil(img)
        img_path = os.path.join(folder, f"{i:04d}.png")
        img.save(img_path)

save_dataset(train_dataset, os.path.join(PROCESSED_DATA_DIR, "train"))
save_dataset(test_dataset, os.path.join(PROCESSED_DATA_DIR, "test"))

print(f"train - {len(train_dataset)}, test - {len(test_dataset)}")
