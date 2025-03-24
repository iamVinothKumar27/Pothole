from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
import cv2
import math
import scipy.ndimage
import uuid
import shutil

app = Flask(__name__)
TFLITE_MODEL_PATH = "Pothole.tflite"

# Load TFLite model
def load_tflite_interpreter():
    if not hasattr(load_tflite_interpreter, "interpreter"):
        interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
        interpreter.allocate_tensors()
        load_tflite_interpreter.interpreter = interpreter
    return load_tflite_interpreter.interpreter

# Run model inference
def run_tflite_inference(img_array):
    interpreter = load_tflite_interpreter()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], img_array.astype(np.float32))
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index']).copy()[0][0]

# Crack detection
def detect_cracks(image_path, output_path):
    gray_image = cv2.imread(image_path, 0) / 255.0
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
    result = cv2.morphologyEx(mag.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cv2.imwrite(output_path, result)

def orientated_non_max_suppression(mag, ang):
    ang_quant = np.round(ang / (np.pi/4)) % 4
    magE = non_max_suppression(mag, np.array([[0,0,0],[1,1,1],[0,0,0]]))
    magSE = non_max_suppression(mag, np.array([[1,0,0],[0,1,0],[0,0,1]]))
    magS = non_max_suppression(mag, np.array([[0,1,0],[0,1,0],[0,1,0]]))
    magSW = non_max_suppression(mag, np.array([[0,0,1],[0,1,0],[1,0,0]]))
    mag[ang_quant == 0] = magE[ang_quant == 0]
    mag[ang_quant == 1] = magSE[ang_quant == 1]
    mag[ang_quant == 2] = magS[ang_quant == 2]
    mag[ang_quant == 3] = magSW[ang_quant == 3]
    return mag

def non_max_suppression(data, win):
    data_max = scipy.ndimage.maximum_filter(data, footprint=win, mode='constant')
    data_max[data != data_max] = 0
    return data_max

# 🖼️ IMAGE detection function (used by both image and video analysis)
def process_image(img_pil, save_path="static/uploaded_image.jpg"):
    # Resize and convert image to RGB, then save it.
    img_pil = img_pil.resize((224, 224)).convert("RGB")
    x = image.img_to_array(img_pil)
    x = np.expand_dims(x, axis=0) / 255.0
    img_pil.save(save_path)

    # Run pothole detection.
    pothole_pred = run_tflite_inference(x)
    pothole_result = "Pothole Detected" if pothole_pred >= 0.5 else "No Pothole Detected"

    # Set up crack detection filename and path.
    crack_filename = "crack_detected.jpg"
    crack_path = os.path.join("static", crack_filename)
    
    # Run crack detection and save the result.
    detect_cracks(save_path, crack_path)
    crack_img = cv2.imread(crack_path, 0)
    
    # Determine crack status.
    crack_status = "Crack Detected" if cv2.countNonZero(crack_img) > 100 else "No Crack Detected"

    return pothole_result, float(pothole_pred), crack_filename, crack_status

# Preprocessing function to enhance video frames.
def preprocess_frame(frame):
    # Convert frame to LAB color space for contrast enhancement.
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge((l_enhanced, a, b))
    enhanced_frame = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    # Apply a sharpening filter.
    kernel_sharpening = np.array([[-1, -1, -1],
                                  [-1,  9, -1],
                                  [-1, -1, -1]])
    sharpened = cv2.filter2D(enhanced_frame, -1, kernel_sharpening)
    return sharpened

# 🎥 VIDEO detection function with improved frame preprocessing.
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps * 3)  # Extract a frame every 3 seconds
    frame_index, pothole_frames = 0, 0
    results = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame only at the specified interval
        if frame_index % interval == 0:
            processed_frame = preprocess_frame(frame)  # Apply enhancements
            temp_img = Image.fromarray(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
            filename_only = f"frame_{uuid.uuid4().hex}.jpg"
            temp_path = os.path.join("static", filename_only)
            pothole_result, conf, crack_filename, crack_status = process_image(temp_img, save_path=temp_path)
            # Record the results with the timestamp
            sec = round(frame_index / fps) if fps > 0 else frame_index
            results.append({
                "time": f"{sec}s",
                "path": filename_only,  # only filename passed to template
                "pothole": pothole_result,
                "confidence": conf
            })



        frame_index += 1

    cap.release()
    os.remove(video_path)  # Clean up the video file from storage

    return results


# 🌐 Main Route
@app.route("/", methods=["GET", "POST"])
def index():
    result, filename, crack_filename, crack_status, video_result = {}, None, None, None, None

    if request.method == "POST":
        submit_type = request.form.get("submit_type")
        if submit_type == "image":
            file = request.files.get("file")
            if file and file.filename:
                img_pil = Image.open(file.stream)
                filename = "uploaded_image.jpg"
                result["pothole"], result["confidence"], crack_filename, crack_status = process_image(img_pil)
        elif submit_type == "video":
            video = request.files.get("video")
            if video and video.filename:
                os.makedirs("static", exist_ok=True)
                unique_name = f"video_{uuid.uuid4().hex}.mp4"
                video_path = os.path.join("static", unique_name)
                video.save(video_path)
                video_result = process_video(video_path)

    return render_template("index.html", result=result, filename=filename,
                           crack_filename=crack_filename, crack_status=crack_status,
                           video_result=video_result)

if __name__ == "__main__":
    app.run(debug=True)
