from flask import Flask, render_template, request, redirect, url_for, flash, session
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
import cv2
import uuid
import json
import requests
import datetime
import pytz
from collections import Counter

app = Flask(__name__)
app.secret_key = 'supersecretkey'

TFLITE_MODEL_PATH = "Pothole.tflite"
REPORTS_FILE = "reports.json"
USERS_FILE = "users.json"

ADMIN_EMAIL = "admin@gmail.com"
ADMIN_PASSWORD = "admin123"

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
    output_data = interpreter.get_tensor(output_details[0]['index']).copy()
    return output_data[0][0]

def detect_cracks(image_path, pixels_per_meter=1000):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    crack_lengths_meters = []
    for contour in contours:
        perimeter_pixels = cv2.arcLength(contour, True)
        length_meters = perimeter_pixels / pixels_per_meter
        crack_lengths_meters.append(length_meters)

    binary_path = os.path.join("static", "binary_crack.jpg")
    cv2.imwrite(binary_path, binary)

    total_crack_length = sum(crack_lengths_meters)
    return total_crack_length, "binary_crack.jpg"

def estimate_pothole_diameter(image_path, pixels_per_meter=1000):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    largest_contour = max(contours, key=cv2.contourArea)
    ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
    diameter_pixels = radius * 2
    diameter_meters = diameter_pixels / pixels_per_meter
    return diameter_meters


# Process a single image
def process_image(img_pil, save_path="static/uploaded_image.jpg"):
    img_pil = img_pil.resize((224, 224)).convert("RGB")
    x = image.img_to_array(img_pil)
    x = np.expand_dims(x, axis=0) / 255.0
    img_pil.save(save_path)
    pothole_pred = run_tflite_inference(x)
    pothole_result = "Pothole Detected" if pothole_pred >= 0.5 else "No Pothole Detected"
    total_crack_length, crack_binary_filename = detect_cracks(save_path)
    crack_status = "Crack Detected" if total_crack_length > 0 else "No Crack Detected"

    pothole_diameter = 0.0
    if pothole_result == "Pothole Detected":
        pothole_diameter = estimate_pothole_diameter(save_path)

    return pothole_result, float(pothole_pred), crack_binary_filename, crack_status, total_crack_length, pothole_diameter

# Preprocess frame for video
def preprocess_frame(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge((l_enhanced, a, b))
    enhanced_frame = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    kernel_sharpening = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(enhanced_frame, -1, kernel_sharpening)

# Process video
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps * 3)
    frame_index = 0
    results = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index % interval == 0:
            processed_frame = preprocess_frame(frame)
            temp_img = Image.fromarray(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
            filename_only = f"frame_{uuid.uuid4().hex}.jpg"
            temp_path = os.path.join("static", filename_only)
            pothole_result, conf, crack_filename, crack_status, crack_length, pothole_diameter = process_image(temp_img, save_path=temp_path)
            sec = round(frame_index / fps) if fps > 0 else frame_index
            results.append({
                "time": f"{sec}s",
                "path": filename_only,
                "pothole": pothole_result,
                "confidence": conf,
                "pothole_diameter": pothole_diameter,
                "crack_status": crack_status,
                "crack_length": crack_length
            })
        frame_index += 1
    cap.release()
    os.remove(video_path)
    return results



def reverse_geocode(lat, lon):
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}&zoom=18&addressdetails=1"
        headers = {"User-Agent": "RoadDefectDetector"}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json().get("display_name", f"{lat},{lon}")
        return f"{lat},{lon}"
    except:
        return f"{lat},{lon}"

def load_reports():
    if os.path.exists(REPORTS_FILE):
        try:
            with open(REPORTS_FILE, "r") as f:
                content = f.read().strip()
                if not content:
                    return []
                reports = json.loads(content)
            return reports
        except json.JSONDecodeError:
            return []
    return []

def save_report(image, location, crack_length=0.0, pothole_diameter=0.0):
    if location and "," in location:
        lat, lon = location.split(",")
        address = reverse_geocode(lat.strip(), lon.strip())
    else:
        address = "Unknown Location"

    defect_type = "both"
    if "frame_" in image or image == "uploaded_image.jpg":
        defect_type = "pothole"

    reports = load_reports()

    india_timezone = pytz.timezone("Asia/Kolkata")
    current_time = datetime.datetime.now(india_timezone).strftime("%Y-%m-%d %H:%M:%S")

    reports.append({
        "id": len(reports) + 1,
        "time": current_time,
        "image": image,
        "location": address,
        "status": "pending",
        "type": defect_type,
        "crack_length": crack_length,
        "pothole_diameter": pothole_diameter
    })
    with open(REPORTS_FILE, "w") as f:
        json.dump(reports, f)

@app.route("/")
def home():
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        if email == ADMIN_EMAIL and password == ADMIN_PASSWORD:
            session["admin_logged_in"] = True
            return redirect(url_for("admin"))
        else:
            if os.path.exists(USERS_FILE):
                with open(USERS_FILE, "r") as f:
                    try:
                        users = json.load(f)
                    except json.JSONDecodeError:
                        users = []
            else:
                users = []

            for user in users:
                if user["email"] == email and user["password"] == password:
                    session["user_logged_in"] = True
                    return redirect(url_for("index"))

            flash("Invalid Credentials")
            return redirect(url_for("login"))

    return render_template("login.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        password = request.form.get("password")

        if os.path.exists(USERS_FILE):
            with open(USERS_FILE, "r") as f:
                try:
                    users = json.load(f)
                except json.JSONDecodeError:
                    users = []
        else:
            users = []

        for user in users:
            if user["email"] == email:
                flash("Email already exists")
                return redirect(url_for("signup"))

        users.append({"name": name, "email": email, "password": password})

        with open(USERS_FILE, "w") as f:
            json.dump(users, f, indent=4)

        flash("Signup successful. Please login.")
        return redirect(url_for("login"))

    return render_template("signup.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/index", methods=["GET", "POST"])
def index():
    if not session.get("user_logged_in"):
        flash("Please login first.")
        return redirect(url_for("login"))

    result, filename, crack_filename, crack_status, video_result, total_crack_length, pothole_diameter = {}, None, None, None, None, 0.0, 0.0
    if request.method == "POST":
        submit_type = request.form.get("submit_type")
        if submit_type == "image":
            file = request.files.get("file")
            if file and file.filename:
                img_pil = Image.open(file.stream)
                filename = "uploaded_image.jpg"
                result["pothole"], result["confidence"], crack_filename, crack_status, total_crack_length, pothole_diameter = process_image(img_pil)
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
                           video_result=video_result, total_crack_length=total_crack_length,
                           pothole_diameter=pothole_diameter)

@app.route("/admin")
def admin():
    if not session.get("admin_logged_in"):
        flash("Unauthorized access. Please login as Admin.")
        return redirect(url_for("login"))

    reports = load_reports()
    return render_template("admin.html", reports=reports)

@app.route("/report", methods=["POST"])
def report():
    image = request.form.get("image")
    location = request.form.get("location")
    crack_length = float(request.form.get("crack_length", "0.0"))
    pothole_diameter = float(request.form.get("pothole_diameter", "0.0"))
    save_report(image, location, crack_length, pothole_diameter)
    return "Reported Successfully"
@app.route("/update_status", methods=["POST"])
def update_status():
    report_id = int(request.form.get("id"))
    action = request.form.get("action")
    reports = load_reports()
    for report in reports:
        if report["id"] == report_id:
            if action == "close":
                report["status"] = "in_progress"
            elif action == "complete":
                report["status"] = "complete"
            break
    with open(REPORTS_FILE, "w") as f:
        json.dump(reports, f)
    return render_template("admin.html", reports=reports)

@app.route("/delete_report", methods=["POST"])
def delete_report():
    report_id = int(request.form.get("id"))
    reports = load_reports()
    reports = [r for r in reports if r["id"] != report_id]
    for idx, r in enumerate(reports):
        r["id"] = idx + 1
    with open(REPORTS_FILE, "w") as f:
        json.dump(reports, f)
    return render_template("admin.html", reports=reports)

@app.route("/severe")
def severe():
    reports = load_reports()
    locations = [report["location"] for report in reports]
    location_counts = Counter(locations)
    severe_locations = [loc for loc, count in location_counts.items() if count > 5]
    return render_template("severe.html", severe_locations=severe_locations)

if __name__ == "__main__":
    app.run(debug=True)
