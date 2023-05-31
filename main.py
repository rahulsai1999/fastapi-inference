from typing import List
from fastapi import FastAPI, UploadFile, File
from predict import get_prediction

app = FastAPI()


@app.get("/")
async def read_root():
    return {"Status": "normal"}


@app.post("/predict")
async def predict(files: List[UploadFile] = File(...)):
    file = files[0]
    img_bytes = await file.read()
    class_id, class_name = get_prediction(image_bytes=img_bytes)
    return {"class_id": class_id, "class_name": class_name}
