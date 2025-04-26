import os
import gdown
import tensorflow as tf
import zipfile
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import BertTokenizer, TFBertForSequenceClassification
from deep_translator import GoogleTranslator

# إعداد التطبيق
app = FastAPI()

# السماح بالـ CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = "."
ZIP_PATH = "assets.zip"
DOWNLOAD_URL = "https://drive.google.com/uc?export=download&id=1kIrOwZfT4zqXjZQvVdRobCUDAt-wA4bR"

# تحميل وفك الموديل
def setup_model():
    if not os.path.exists("tf_model.h5"):
        print("🔽 Downloading model...")
        gdown.download(DOWNLOAD_URL, ZIP_PATH, quiet=False)
        print("✅ Download complete.")

        print("📦 Extracting model...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall()
        print("✅ Extraction done.")

        os.remove(ZIP_PATH)

setup_model()

# تحميل الموديل والتوكنيزر
model = TFBertForSequenceClassification.from_pretrained(MODEL_DIR)
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
labels = ["happy", "sad", "angry", "normal"]

# راوت البريديكت
@app.post("/predict")
async def predict(request: Request):
    data = await request.json()

    if "text" not in data:
        return JSONResponse(content={"error": "No text provided"}, status_code=400)

    try:
        translated_text = GoogleTranslator(source='auto', target='en').translate(data["text"])
    except Exception as e:
        return JSONResponse(content={"error": f"Translation failed: {str(e)}"}, status_code=500)

    inputs = tokenizer(translated_text, return_tensors="tf", truncation=True, padding=True)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = tf.argmax(logits, axis=1).numpy()[0]
    sentiment = labels[predicted_class]

    return {
        "original_text": data["text"],
        "translated_text": translated_text,
        "sentiment": sentiment,
        "label_index": int(predicted_class),
        "logits": logits.numpy().tolist()
    }
