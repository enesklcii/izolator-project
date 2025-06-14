from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from PIL import Image
import io
import torch
import torchvision.transforms as transforms
from pymongo import MongoClient
import base64

app = FastAPI()

# CORS ayarları - React localhost:3000'dan gelen istekleri kabul et
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

@app.get("/")
def root():
    return {"message": "API çalışıyor!"}

@app.get("/records/")
def get_records():
    records = list(collection.find({}, {"_id": 0}))  # _id alanını gizle
    return JSONResponse(content={"records": records})


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB bağlantısı
client = MongoClient("mongodb://localhost:27017/")
db = client["izolatorDB"]
collection = db["tahminler"]

class_names = ["Kırık", "Sağlam"]

# Modeli yükle (model dosyan uygun yerde olmalı)
model = torch.jit.load("efficientnet_b0_isolator_cpu.pt")
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted.item()].item()
    return class_names[predicted.item()], confidence

def save_to_mongo(filename, image_bytes, prediction, confidence):
    encoded_img = base64.b64encode(image_bytes).decode("utf-8")
    document = {
        "filename": filename,
        "prediction": prediction,
        "confidence": round(confidence * 100, 2),
        "image": encoded_img
    }
    collection.insert_one(document)
    print("MongoDB'ye kayıt yapıldı.")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    prediction, confidence = predict_image(contents)
    save_to_mongo(file.filename, contents, prediction, confidence)
    return JSONResponse(content={
        "filename": file.filename,
        "prediction": prediction,
        "confidence": f"{confidence*100:.2f}%"
    })
