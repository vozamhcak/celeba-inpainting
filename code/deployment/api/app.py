import io
import base64
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import torch
from torchvision import transforms
from PIL import Image

import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.join(BASE_DIR, "code", "models"))

from code.models.unet import UNet

# load model 

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
    
model_path = "models/inpainting_unet.pt"

payload = torch.load(model_path, map_location=DEVICE)
model = UNet().to(DEVICE)
model.load_state_dict(payload["state_dict"])
model.eval()

IMG_SIZE = payload["config"].get("img_size", 64)

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

to_pil = transforms.ToPILImage()




# fastapi app

app = FastAPI(title = "Celeba Inpainting API")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(tensor).cpu().squeeze(0)

        out_img = to_pil(output)

        buffer = io.BytesIO()
        out_img.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return JSONResponse(content={"result": img_str})
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)