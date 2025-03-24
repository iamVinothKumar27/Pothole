from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

TFLITE_MODEL_PATH = "Pothole.tflite"

# Load TFLite model once
def load_tflite_interpreter():
    if not hasattr(load_tflite_interpreter, "interpreter"):
        interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
        interpreter.allocate_tensors()
        load_tflite_interpreter.interpreter = interpreter
    return load_tflite_interpreter.interpreter

# Run TFLite inference
def run_tflite_inference(img_array):
    interpreter = load_tflite_interpreter()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], img_array.astype(np.float32))
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
    return prediction

@app.route("/", methods=["GET", "POST"])
def index():
    result = {}
    filename = None

    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename:
            img = Image.open(file.stream).convert("RGB")
            img = img.resize((224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0) / 255.0

            filename = "uploaded_image.jpg"
            filepath = os.path.join("static", filename)
            img.save(filepath)

            pothole_pred = run_tflite_inference(x)
            result["pothole"] = "Pothole Detected" if pothole_pred >= 0.5 else "No Pothole Detected"
            result["confidence"] = float(pothole_pred)

    return render_template("index.html", result=result, filename=filename)

if __name__ == "__main__":
    app.run(debug=True)
