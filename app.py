from flask import Flask, render_template, request
import cv2
import numpy as np
import tensorflow as tf
import os
import gdown
from werkzeug.utils import secure_filename

# === Download model from Google Drive if not present ===
MODEL_PATH = "model.keras"
if not os.path.exists(MODEL_PATH):
    print("🔽 Downloading model from Google Drive...")
    gdown.download("https://drive.google.com/uc?id=15bQqvOX3rk3nXztosHgwJ0-Kdd36Wc2v", MODEL_PATH, quiet=False)

# === Load the downloaded model ===
model = tf.keras.models.load_model(MODEL_PATH)

# === Flask App ===
app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_path = None

    if request.method == 'POST':
        if 'file' not in request.files:
            prediction = "No file part in the request."
            return render_template('index.html', prediction=prediction)

        file = request.files['file']

        if file.filename == '':
            prediction = "No selected file."
            return render_template('index.html', prediction=prediction)

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            image_path = filepath

            # ✅ Preprocess and predict
            try:
                img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (224, 224))
                img = img.reshape(1, 224, 224, 1).astype('float32') / 255.0

                pred = model.predict(img)[0][0]
                prediction = "Abnormal" if pred > 0.5 else "Normal"
            except Exception as e:
                prediction = f"Error during prediction: {str(e)}"

    return render_template('index.html', prediction=prediction, image_path=image_path)

if __name__ == '__main__':
    # For local testing only; Render will use Gunicorn
    app.run(host='0.0.0.0', port=5000)
