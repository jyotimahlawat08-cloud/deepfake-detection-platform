from flask import Flask, render_template, request
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import preprocess_input
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = "app/static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load trained model
# model = load_model("GAN_Model/saved_model/detector.keras")
model = load_model("GAN_Model/saved_model/best_xception.keras")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():

    if "image" not in request.files:
        return "No file selected"

    file = request.files["image"]

    if file.filename == "":
        return "No file selected"

    # Save uploaded file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    print("Saved Path:", filepath)
    print("File Exists:", os.path.exists(filepath))

    # Read image
    img = cv2.imread(filepath)

    if img is None:
        return f"Image load nahi hui.\nPath = {filepath}"

    # Preprocess image
    img = cv2.resize(img, (299, 299))
    img = np.array(img, dtype=np.float32)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img, verbose=0)
    score = float(prediction[0][0])

    print("Prediction Score:", score)

    # Fake = 0, Real = 1
    if score >= 0.70:
        result = f"✅ Real Image\nConfidence: {score*100:.2f}%"
    else:
        result = f"❌ Fake Image\nConfidence: {(1-score)*100:.2f}%"

    return render_template(
        "index.html",
        image=filename,
        result=result
    )


if __name__ == "__main__":
    app.run(debug=True)