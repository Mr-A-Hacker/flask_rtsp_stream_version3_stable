print("\n=== STARTING APP ===")

import os
os.environ["ULTRALYTICS_HUB"] = "False"
os.environ["ULTRALYTICS_CHECK"] = "False"
os.environ["WANDB_DISABLED"] = "true"
os.environ["YOLO_VERBOSE"] = "False"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TORCH_DEVICE"] = "cpu"

print("Importing cv2...")
import cv2
print("cv2 OK")

print("Importing turbojpeg...")
from turbojpeg import TurboJPEG
print("TurboJPEG OK")

print("Importing torch...")
import torch
print("Torch OK")

print("Importing ultralytics YOLO...")
from ultralytics import YOLO
print("YOLO import OK")

print("Importing numpy...")
import numpy as np
print("numpy OK")

print("Importing flask...")
from flask import Flask, Response, render_template, request, jsonify
print("Flask OK")

import threading
import time

print("=== IMPORTS COMPLETE ===")

app = Flask(__name__)

# -----------------------------
# CONFIG
# -----------------------------
RTSP_URL = "rtsp://192.168.2.10:554/stream1"
jpeg = TurboJPEG()

latest_frame = None
alarm_box_enabled = False
alarm_box = None
alarm_sound = "default.wav"

ALARM_FOLDER = "/home/admin/my_backup/camera/cam2/alarms"
os.makedirs(ALARM_FOLDER, exist_ok=True)

# -----------------------------
# LOAD YOLOv8-NANO (auto-download)
# -----------------------------
model = YOLO("yolov8n.pt")  # auto-downloads if missing

# -----------------------------
# CAMERA THREAD
# -----------------------------
def camera_reader():
    global latest_frame
    cap = None

    while True:
        try:
            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                time.sleep(1)

            ret, frame = cap.read()
            if ret:
                latest_frame = frame
            else:
                cap.release()
                cap = None

        except:
            time.sleep(1)


threading.Thread(target=camera_reader, daemon=True).start()

# -----------------------------
# ALARM TRIGGER (ffplay, 30s cooldown)
# -----------------------------
alarm_cooldown = 0  # timestamp when next alarm is allowed

def trigger_alarm():
    global alarm_cooldown
    now = time.time()

    # 30-second cooldown
    if now < alarm_cooldown:
        return

    alarm_path = os.path.join(ALARM_FOLDER, alarm_sound)
    if os.path.exists(alarm_path):
        os.system(f"ffplay -nodisp -autoexit -loglevel quiet '{alarm_path}' &")

    alarm_cooldown = now + 30  # next allowed alarm time


# -----------------------------
# YOLO HUMAN/CAR MOTION DETECTION
# -----------------------------
prev_person_centers = []  # store previous frame centers

def generate_frames():
    global alarm_box, prev_person_centers

    while True:
        if latest_frame is None:
            time.sleep(0.01)
            continue

        frame = cv2.resize(latest_frame, (960, 540))
        current_person_centers = []

        # Run YOLO only if alarm box is enabled
        if alarm_box_enabled and alarm_box is not None:
            results = model(frame, verbose=False)[0]

            bx1, by1, bx2, by2 = alarm_box

            for box in results.boxes:
                cls = int(box.cls[0])
                name = model.names[cls]

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # ALWAYS draw green boxes for all detections
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    name,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                # Only allow alarm logic for humans and cars
                if name not in ["person", "car"]:
                    continue

                # Check if detection is inside alarm box
                if not (x1 > bx1 and y1 > by1 and x2 < bx2 and y2 < by2):
                    continue

                # Track center for movement detection
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                current_person_centers.append((cx, cy))

            # Detect movement (medium sensitivity)
            movement_detected = False
            movement_threshold = 20  # pixels

            for cx, cy in current_person_centers:
                if prev_person_centers:
                    dists = [np.hypot(cx - pcx, cy - pcy) for (pcx, pcy) in prev_person_centers]
                    if min(dists) > movement_threshold:
                        movement_detected = True
                        break

            prev_person_centers = current_person_centers

            # Trigger alarm if movement detected inside the box
            if movement_detected:
                trigger_alarm()

        else:
            prev_person_centers = []

        # Draw alarm box (red)
        if alarm_box is not None:
            bx1, by1, bx2, by2 = alarm_box
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 255), 2)

        jpeg_bytes = jpeg.encode(frame, quality=40)

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            jpeg_bytes +
            b"\r\n"
        )


# -----------------------------
# ROUTES
# -----------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/save_box", methods=["POST"])
def save_box():
    global alarm_box
    data = request.get_json()
    alarm_box = data.get("box_pix")
    return jsonify({"status": "ok"})


@app.route("/toggle_alarm_box", methods=["POST"])
def toggle_alarm_box():
    global alarm_box_enabled
    alarm_box_enabled = not alarm_box_enabled
    return jsonify({"enabled": alarm_box_enabled})


@app.route("/list_alarms")
def list_alarms():
    files = [f for f in os.listdir(ALARM_FOLDER) if f.endswith(".wav")]
    return jsonify(files)


@app.route("/set_alarm", methods=["POST"])
def set_alarm():
    global alarm_sound
    data = request.get_json()
    alarm_sound = data.get("filename")
    return jsonify({"status": "ok"})


# -----------------------------
# RUN SERVER
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
