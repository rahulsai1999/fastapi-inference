import os
import whisper

model = whisper.load_model("base")


def transcribe(audio_bytes):
    audio = open("audio.wav", "wb")
    audio.write(audio_bytes)
    audio.close()
    res = whisper.transcribe(model, "audio.wav", fp16=False)
    os.remove("audio.wav")

    return res