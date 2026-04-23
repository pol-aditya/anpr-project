from fastapi import FastAPI, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import cv2
import numpy as np
from pathlib import Path

from model import CNN_Transformer
from utils import preprocess, decode
from cv_module import detect_plate_region

app = FastAPI(title="ANPR - License Plate Recognition", description="Automatic Number Plate Recognition System")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load model
model = CNN_Transformer(37)
model.load_state_dict(torch.load("plate_model.pth", map_location="cpu"))
model.eval()


@app.get("/")
async def get_home():
    return FileResponse("index.html", media_type="text/html")


@app.post("/predict")
async def predict(file: UploadFile):

    contents = await file.read()

    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        return {"plate": "ERROR: Invalid image"}

    # ✅ segmentation
    plate = detect_plate_region(img)

    if plate is not None:
        img = preprocess(plate)
    else:
        img = preprocess(img)

    with torch.no_grad():
        pred = model(img)

    text = decode(pred)

    return {"plate": text}