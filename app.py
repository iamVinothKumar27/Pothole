from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
import cv2
import math
import scipy.ndimage

app = Flask(__name__)
TFLITE_MODEL_PATH = "Pothole.tflite"

def load_tflite_interpreter():
    if not hasattr(load_tflite_interpreter, "interpreter"):
        interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
        interpreter.allocate_tensors()
        load_tflite_interpreter.interpreter = interpreter
    return load_tflite_interpreter.interpreter

def run_tflite_inference(img_array):
    interpreter = load_tflite_interpreter()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], img_array.astype(np.float32))
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
    return prediction

def detect_cracks(image_path, output_path):
    gray_image = cv2.imread(image_path, 0)
    gray_image = gray_image / 255.0

    fudgefactor = 1.3
    sigma = 21
    kernel = 2 * math.ceil(2 * sigma) + 1

    blur = cv2.GaussianBlur(gray_image, (kernel, kernel), sigma)
    gray_image = cv2.subtract(gray_image, blur)

    sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.hypot(sobelx, sobely)
    ang = np.arctan2(sobely, sobelx)

    threshold = 4 * fudgefactor * np.mean(mag)
    mag[mag < threshold] = 0

    mag = orientated_non_max_suppression(mag, ang)
    mag[mag > 0] = 255
    mag = mag.astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    result = cv2.morphologyEx(mag, cv2.MORPH_CLOSE, kernel)

    cv2.imwrite(output_path, result)

def orientated_non_max_suppression(mag, ang):
    ang_quant = np.round(ang / (np.pi/4)) % 4
    winE = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]])
    winSE = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    winS = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
    winSW = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])

    magE = non_max_suppression(mag, winE)
    magSE = non_max_suppression(mag, winSE)
    magS = non_max_suppression(mag, winS)
    magSW = non_max_suppression(mag, winSW)

    mag[ang_quant == 0] = magE[ang_quant == 0]
    mag[ang_quant == 1] = magSE[ang_quant == 1]
    mag[ang_quant == 2] = magS[ang_quant == 2]
    mag[ang_quant == 3] = magSW[ang_quant == 3]
    return mag

def non_max_suppression(data, win):
    data_max = scipy.ndimage.maximum_filter(data, footprint=win, mode='constant')
    data_max[data != data_max] = 0
    return data_max

@app.route("/", methods=["GET", "POST"])
def index():
    result = {}
    filename = None
    crack_filename = None
    crack_status = None

    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename:
            img = Image.open(file.stream).convert("RGB")
            img = img.resize((224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0) / 255.0

            # Save original image
            filename = "uploaded_image.jpg"
            filepath = os.path.join("static", filename)
            img.save(filepath)

            # Run pothole detection
            pothole_pred = run_tflite_inference(x)
            result["pothole"] = "Pothole Detected" if pothole_pred >= 0.5 else "No Pothole Detected"
            result["confidence"] = float(pothole_pred)

            # Crack Detection
            crack_filename = "crack_detected.jpg"
            crack_path = os.path.join("static", crack_filename)
            detect_cracks(filepath, crack_path)

            # Load crack result image and check if there's white (crack) pixels
            crack_img = cv2.imread(crack_path, 0)
            white_pixel_count = cv2.countNonZero(crack_img)
            crack_status = "Crack Detected" if white_pixel_count > 100 else "No Crack Detected"

    return render_template("index.html", result=result, filename=filename,
                           crack_filename=crack_filename, crack_status=crack_status)


if __name__ == "__main__":
    app.run(debug=True)
