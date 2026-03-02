from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

model = tf.keras.models.load_model("model.h5")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    image = Image.open(io.BytesIO(file.read())).resize((224,224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    result = int(np.argmax(prediction))
    
    return jsonify({"result": "Abnormal" if result==1 else "Normal"})

@app.route("/")
def home():
    return "Backend Running"

if __name__ == "__main__":
    app.run()
