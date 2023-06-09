from PIL import Image
from typing import List
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse

# Import functions
from image.classify import get_prediction
from image.caption import generate_captions
from image.segment import segment_image
from audio.transcribe import transcribe

app = FastAPI()


@app.get("/")
async def read_root():
    return {"Status": "normal"}


@app.post("/predict/image/class")
async def predict(files: List[UploadFile] = File(...)):
    file = files[0]
    img_bytes = await file.read()
    class_id, class_name = get_prediction(image_bytes=img_bytes)
    return {"class_id": class_id, "class_name": class_name}


@app.post("/predict/image/caption")
async def predict(files: List[UploadFile] = File(...)):
    file = files[0]
    img = Image.open(file.file)
    caption = generate_captions(image=img)
    return {"caption": caption}


@app.post("/predict/image/segment")
async def predict(files: List[UploadFile] = File(...)):
    file = files[0]
    img = Image.open(file.file)
    segment_image(image=img)
    return FileResponse("overlaid_image.png")


@app.post("/predict/audio/transcribe")
async def predict(files: List[UploadFile] = File(...)):
    file = files[0]
    audio_bytes = await file.read()
    transcribed_text = transcribe(audio_bytes=audio_bytes)
    return {"transcribed_text": transcribed_text}
