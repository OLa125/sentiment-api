import os
import requests
import tensorflow as tf
import zipfile
import shutil
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import BertTokenizer, TFBertForSequenceClassification
from deep_translator import GoogleTranslator

app = Flask(__name__)
CORS(app)

MODEL_DIR = "."
ZIP_PATH = "assets.zip"
DOWNLOAD_URL = "https://drive.google.com/uc?export=download&id=1kIrOwZfT4zqXjZQvVdRobCUDAt-wA4bR"


# -------------------------
# تحميل وفك الضغط للموديل
# -------------------------
def setup_model():
    if not os.path.exists(MODEL_DIR):
        print("🔽 Downloading model...")
        with requests.get(DOWNLOAD_URL, stream=True) as r:
            with open(ZIP_PATH, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
        print("✅ Download complete.")

        print("📦 Extracting model...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall()
        print("✅ Extraction done.")

        os.remove(ZIP_PATH)

setup_model()

# -------------------------
# تحميل الموديل والتوكنيزر
# -------------------------
model = TFBertForSequenceClassification.from_pretrained(MODEL_DIR)
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
labels = ["happy", "sad", "angry", "normal"]

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    try:
        translated_text = GoogleTranslator(source='auto', target='en').translate(data["text"])
    except Exception as e:
        return jsonify({"error": f"Translation failed: {str(e)}"}), 500

    inputs = tokenizer(translated_text, return_tensors="tf", truncation=True, padding=True)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = tf.argmax(logits, axis=1).numpy()[0]
    sentiment = labels[predicted_class]

    return jsonify({
        "original_text": data["text"],
        "translated_text": translated_text,
        "sentiment": sentiment,
        "label_index": int(predicted_class),
        "logits": logits.numpy().tolist()
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
