from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("runs/detect/train-2/weights/best.pt")

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    print("📩 Request received")

    image_bytes = await file.read()
    print(f"📦 Image size: {len(image_bytes)} bytes")

    image = Image.open(io.BytesIO(image_bytes))
    print("🖼 Image loaded")

    # Lower confidence slightly to avoid missing fists in difficult lighting.
    results = model.predict(image, conf=0.15, imgsz=640, verbose=False)[0]
    print("🤖 Model inference done")

    predictions = []

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls = int(box.cls[0])

        predictions.append({
            "x": (x1 + x2) / 2,
            "y": (y1 + y2) / 2,
            "width": x2 - x1,
            "height": y2 - y1,
            "confidence": conf,
            "class": model.names[cls]
        })

    print(f"🎯 Predictions: {predictions}")

    return {"predictions": predictions}