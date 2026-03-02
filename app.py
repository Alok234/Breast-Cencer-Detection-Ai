from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)

# Dummy model: always predicts random 0 or 1
def dummy_model(image_array):
    return np.random.randint(0, 2)  # 0 = Normal, 1 = Abnormal

@app.route("/api/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    # Just ignore the actual image for now
    result = dummy_model(None)

    return jsonify({
        "prediction": "Abnormal" if result == 1 else "Normal",
        "confidence": float(np.random.rand()),  # random confidence
        "disclaimer": "Educational tool only. Not medical advice."
    })

@app.route("/")
def home():
    return "Backend Running ✅"

if __name__ == "__main__":
    app.run()
