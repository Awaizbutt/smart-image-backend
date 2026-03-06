from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
import io
import numpy as np

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File is not an image")

    contents = await file.read()

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Cannot read image. Please upload a valid JPG/PNG.")

    width, height = image.size

    # Convert to numpy for simple analysis
    arr = np.array(image)  # shape (H, W, 3), values 0-255
    mean_brightness = float(arr.mean())  # 0-255

    # Very simple rule-based "model"
    if mean_brightness > 170:
        label = "bright image"
    elif mean_brightness < 80:
        label = "dark image"
    else:
        label = "normal image"

    # confidence as a simple function of distance from middle
    confidence = abs(mean_brightness - 128) / 128
    confidence = float(min(max(confidence, 0.0), 1.0))

    return JSONResponse({
        "filename": file.filename,
        "width": width,
        "height": height,
        "prediction": label,
        "confidence": confidence,
        "mean_brightness": mean_brightness
    })
