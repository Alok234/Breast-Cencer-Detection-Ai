from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app)

# === Fix: Load old .h5 model safely ===
try:
    # Load model without compiling
    old_model = tf.keras.models.load_model("test_model.h5", compile=False)

    # Rebuild input layer to avoid batch_shape issues
    new_input = tf.keras.layers.Input(shape=(224, 224, 3), name="input_layer")  # match your resized input
    model = tf.keras.models.Model(inputs=new_input, outputs=old_model(new_input))

    # Optionally save in SavedModel format for future use
    model.save("test_model_saved")
    print("Model loaded and ready.")
except Exception as e:
    print("Error loading model:", e)
    model = None

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        # Get image from request
        file = request.files.get("image")
        if file is None:
            return jsonify({"error": "No image provided"}), 400

        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        image = image.resize((224, 224))  # resize to model input
        img_array = np.expand_dims(np.array(image)/255.0, axis=0)

        # Predict
        prediction = model.predict(img_array)
        result = int(np.argmax(prediction))  # assuming classification
        return jsonify({"result": "Abnormal" if result == 1 else "Normal"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def home():
    return "Backend Running"

if __name__ == "__main__":
    # Dynamic port for Render
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
