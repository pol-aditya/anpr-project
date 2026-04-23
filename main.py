from fastapi import FastAPI, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import cv2
import numpy as np
import os

from model import CNN_Transformer
from utils import preprocess, decode
from cv_module import detect_plate_region

app = FastAPI(
    title="ANPR - License Plate Recognition",
    description="Automatic Number Plate Recognition System"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# MODEL LOADING (SAFE)
# =========================

model = CNN_Transformer(37)

MODEL_PATH = "plate_model.pth"

if os.path.exists(MODEL_PATH):
    print("Model found, loading...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
else:
    print("Model not found, skipping loading...")
    model = None


# =========================
# ROUTES
# =========================

@app.get("/")
async def get_home():
    return FileResponse("index.html", media_type="text/html")


@app.post("/predict")
async def predict(file: UploadFile):

    # 🔴 IMPORTANT: handle missing model
    if model is None:
        return {"plate": "Model not loaded (file missing)"}

    contents = await file.read()

    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        return {"plate": "ERROR: Invalid image"}

    # Plate detection
    plate = detect_plate_region(img)

    if plate is not None:
        img = preprocess(plate)
    else:
        img = preprocess(img)

    with torch.no_grad():
        pred = model(img)

    text = decode(pred)

    return {"plate": text}