from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os  # <-- for environment variables

app = Flask(__name__)
CORS(app)

# Load your old .h5 model
model = tf.keras.models.load_model("test_model.h5")

# Save in TensorFlow SavedModel format (folder)
model.save("test_model")  # this creates a folder

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get image from request
        file = request.files["image"]
        image = Image.open(io.BytesIO(file.read())).resize((224,224))  # resize to model input
        img_array = np.expand_dims(np.array(image)/255.0, axis=0)
        
        # Predict using model
        prediction = model.predict(img_array)
        result = int(np.argmax(prediction))  # 0=Normal, 1=Abnormal
        
        return jsonify({"result": "Abnormal" if result==1 else "Normal"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def home():
    return "Backend Running"

if __name__ == "__main__":
    # Use Render's dynamic port
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
