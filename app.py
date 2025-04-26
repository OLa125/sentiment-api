import os
import gdown
import tensorflow as tf
import zipfile
import gradio as gr
from transformers import BertTokenizer, TFBertForSequenceClassification
from deep_translator import GoogleTranslator

MODEL_DIR = "."
ZIP_PATH = "assets.zip"
DOWNLOAD_URL = "https://drive.google.com/uc?export=download&id=1kIrOwZfT4zqXjZQvVdRobCUDAt-wA4bR"

# -------------------------
# تحميل وفك الضغط للموديل
# -------------------------
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

# تحميل الموديل والتوكنيزر
setup_model()

# تحميل الموديل والتوكنيزر
model = TFBertForSequenceClassification.from_pretrained(MODEL_DIR)
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
labels = ["happy", "sad", "angry", "normal"]

# دالة التنبؤ
def predict(text):
    try:
        translated_text = GoogleTranslator(source='auto', target='en').translate(text)
    except Exception as e:
        return f"Translation failed: {str(e)}"
    
    inputs = tokenizer(translated_text, return_tensors="tf", truncation=True, padding=True)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = tf.argmax(logits, axis=1).numpy()[0]
    sentiment = labels[predicted_class]

    return {
        "original_text": text,
        "translated_text": translated_text,
        "sentiment": sentiment,
        "label_index": int(predicted_class),
        "logits": logits.numpy().tolist()
    }

# واجهة Gradio
iface = gr.Interface(fn=predict, inputs="text", outputs="json")

iface.launch(share=True)  # share=True هيسمحلك بمشاركة الرابط العام
