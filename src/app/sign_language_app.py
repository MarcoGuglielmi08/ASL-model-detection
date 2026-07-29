import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import customtkinter as ctk

from PIL import Image, ImageTk

import pandas as pd
import numpy as np
import joblib
import time
import math
import threading

from src.util.paths import MODELS_DIR
from util.paths import ASSETS_DIR

# Window, model, and camera settings.
WINDOW_WIDTH = 1500
WINDOW_HEIGHT = 900

LANDMARK_MODEL_PATH = ASSETS_DIR / "hand_landmarker.task"
CLASSIFIER_MODEL_PATH = MODELS_DIR / "asl_model.pkl"

USE_Z = False

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

PREDICTION_INTERVAL = 0.7
TEXT_UPDATE_INTERVAL = 0.3

# Theme colors.
BG_DARK      = "#0a0c10"
BG_PANEL     = "#111520"
BG_CARD      = "#161b27"
ACCENT_CYAN  = "#00e5ff"
ACCENT_BLUE  = "#2979ff"
TEXT_PRIMARY = "#e8eaf0"
TEXT_DIM     = "#5a6070"
BORDER       = "#1e2535"

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


print("Loading classifier model...")
model = joblib.load(CLASSIFIER_MODEL_PATH)
print("Classifier loaded!")


# MediaPipe hand detector used by the live webcam loop.
BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(LANDMARK_MODEL_PATH)),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
)

landmarker = HandLandmarker.create_from_options(options)


# MediaPipe hand landmark indices.
WRIST = 0
THUMB  = [1, 2, 3, 4]
INDEX  = [5, 6, 7, 8]
MIDDLE = [9, 10, 11, 12]
RING   = [13, 14, 15, 16]
PINKY  = [17, 18, 19, 20]

FINGERS = {
    "thumb": THUMB, "index": INDEX, "middle": MIDDLE,
    "ring": RING,   "pinky": PINKY,
}

FINGERTIPS = {
    "thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20,
}


def normalize_landmarks(landmarks):
    # Apply the same wrist-centered normalization used during extraction.
    wrist = landmarks[0]
    middle_mcp = landmarks[9]
    dx = middle_mcp.x - wrist.x
    dy = middle_mcp.y - wrist.y
    dz = middle_mcp.z - wrist.z
    scale = math.sqrt(dx*dx + dy*dy + dz*dz) or 1.0
    normalized = []
    for point in landmarks:
        normalized.append([
            (point.x - wrist.x) / scale,
            (point.y - wrist.y) / scale,
            (point.z - wrist.z) / scale,
        ])
    return np.array(normalized)


def euclidean(a, b):
    # Compute distance between two landmark points.
    return np.linalg.norm(a - b)


def compute_angle(a, b, c):
    # Compute the angle at joint b formed by points a-b-c.
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-8:
        return 0.0
    return float(np.arccos(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)))


def extract_features(raw_landmarks):
    # Build one feature row with the same schema used for model training.
    points_3d = normalize_landmarks(raw_landmarks)
    points = points_3d[:, :2] if not USE_Z else points_3d
    features = {}

    flat = points_3d.flatten()
    for i, value in enumerate(flat):
        features[f"lm_{i}"] = float(value)

    fingertip_pairs = [
        ("thumb","index"), ("thumb","middle"), ("thumb","ring"), ("thumb","pinky"),
        ("index","middle"), ("middle","ring"), ("ring","pinky"),
    ]
    for a, b in fingertip_pairs:
        features[f"{a}_{b}_tip_distance"] = float(euclidean(points[FINGERTIPS[a]], points[FINGERTIPS[b]]))

    wrist = points[WRIST]
    for finger, tip_idx in FINGERTIPS.items():
        features[f"wrist_to_{finger}_tip"] = float(euclidean(wrist, points[tip_idx]))

    for finger_name, joints in FINGERS.items():
        features[f"{finger_name}_angle_1"] = float(compute_angle(points[joints[0]], points[joints[1]], points[joints[2]]))
        features[f"{finger_name}_angle_2"] = float(compute_angle(points[joints[1]], points[joints[2]], points[joints[3]]))

    for finger, tip_idx in FINGERTIPS.items():
        vec = points[tip_idx] - wrist
        features[f"{finger}_vector_x"] = float(vec[0])
        features[f"{finger}_vector_y"] = float(vec[1])
        if USE_Z:
            features[f"{finger}_vector_z"] = float(vec[2])

    return features


def predict_letter_with_confidence(features):
    # Run the trained sklearn pipeline on a single feature dictionary.
    input_df = pd.DataFrame([features])
    return str(model.predict(input_df)[0])



app = ctk.CTk()
app.title("ASL Translator")
app.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
app.configure(fg_color=BG_DARK)
app.resizable(True, True)

FONT_MONO   = ("Courier New", 12)
FONT_HEAD   = ("Courier New", 13, "bold")
FONT_TITLE  = ("Courier New", 20, "bold")
FONT_LETTER = ("Courier New", 110, "bold")
FONT_STATUS = ("Courier New", 11)
FONT_LABEL  = ("Courier New", 11)

# Top bar.

topbar = ctk.CTkFrame(app, height=56, corner_radius=0, fg_color=BG_PANEL,
                      border_width=1, border_color=BORDER)
topbar.pack(fill="x")
topbar.pack_propagate(False)

# App title.
title_lbl = ctk.CTkLabel(topbar, text="ASL SIGN TRANSLATOR",
                          font=FONT_TITLE, text_color=ACCENT_CYAN)
title_lbl.pack(side="left", padx=24, pady=14)

# Version tag.
ctk.CTkLabel(topbar, text="v2.0", font=FONT_STATUS, text_color=TEXT_DIM
             ).pack(side="left", pady=14)

# Live status indicator.
status_frame = ctk.CTkFrame(topbar, fg_color="#0d1f15", corner_radius=6,
                             border_width=1, border_color="#1a3a20")
status_frame.pack(side="right", padx=24, pady=12)

status_dot = ctk.CTkLabel(status_frame, text="*", text_color="#00e676",
                           font=("Courier New", 14))
status_dot.pack(side="left", padx=(10, 4), pady=4)

status_label = ctk.CTkLabel(status_frame, text="LIVE", text_color="#00e676",
                             font=FONT_HEAD)
status_label.pack(side="left", padx=(0, 10), pady=4)


# Main layout.

body = ctk.CTkFrame(app, fg_color=BG_DARK)
body.pack(fill="both", expand=True, padx=16, pady=16)


left_panel = ctk.CTkFrame(body, fg_color=BG_PANEL, corner_radius=10,
                           border_width=1, border_color=BORDER)
left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))

# camera label bar
cam_bar = ctk.CTkFrame(left_panel, fg_color=BG_CARD, corner_radius=0, height=36)
cam_bar.pack(fill="x")
cam_bar.pack_propagate(False)
ctk.CTkLabel(cam_bar, text="CAMERA FEED", font=FONT_HEAD,
             text_color=TEXT_DIM).pack(side="left", padx=16, pady=8)
ctk.CTkLabel(cam_bar, text="CAM:0  *  30fps", font=FONT_STATUS,
             text_color=TEXT_DIM).pack(side="right", padx=16)

video_label = ctk.CTkLabel(left_panel, text="")
video_label.pack(padx=16, pady=16, expand=True)


right_panel = ctk.CTkFrame(body, fg_color=BG_PANEL, corner_radius=10,
                            border_width=1, border_color=BORDER, width=340)
right_panel.pack(side="right", fill="y")
right_panel.pack_propagate(False)


pred_card = ctk.CTkFrame(right_panel, fg_color=BG_CARD, corner_radius=8,
                          border_width=1, border_color=BORDER)
pred_card.pack(fill="x", padx=16, pady=(16, 8))

ctk.CTkLabel(pred_card, text="DETECTED LETTER", font=FONT_HEAD,
             text_color=TEXT_DIM).pack(pady=(14, 0))

prediction_label = ctk.CTkLabel(pred_card, text="-",
                                  font=FONT_LETTER,
                                  text_color=ACCENT_CYAN)
prediction_label.pack(pady=(0, 14))

# thin cyan underline accent
accent_bar = ctk.CTkFrame(pred_card, height=3, fg_color=ACCENT_CYAN, corner_radius=0)
accent_bar.pack(fill="x")


feat_card = ctk.CTkFrame(right_panel, fg_color=BG_CARD, corner_radius=8,
                          border_width=1, border_color=BORDER)
feat_card.pack(fill="both", expand=True, padx=16, pady=(8, 8))

feat_bar = ctk.CTkFrame(feat_card, fg_color=BG_DARK, corner_radius=0, height=32)
feat_bar.pack(fill="x")
feat_bar.pack_propagate(False)
ctk.CTkLabel(feat_bar, text="HAND FEATURES", font=FONT_HEAD,
             text_color=TEXT_DIM).pack(side="left", padx=14, pady=6)

coords_textbox = ctk.CTkTextbox(
    feat_card,
    font=FONT_MONO,
    fg_color=BG_DARK,
    text_color="#7ecfff",
    corner_radius=0,
    border_width=0,
    scrollbar_button_color=BORDER,
    scrollbar_button_hover_color=ACCENT_BLUE,
)
coords_textbox.pack(fill="both", expand=True, padx=2, pady=(0, 2))


ctrl_card = ctk.CTkFrame(right_panel, fg_color=BG_CARD, corner_radius=8,
                          border_width=1, border_color=BORDER)
ctrl_card.pack(fill="x", padx=16, pady=(0, 16))

running = True


def toggle_camera():
    global running
    running = not running
    if running:
        status_dot.configure(text_color="#00e676")
        status_label.configure(text="LIVE", text_color="#00e676")
        status_frame.configure(fg_color="#0d1f15", border_color="#1a3a20")
        toggle_btn.configure(text="PAUSE", fg_color="#0d1a2e",
                              border_color=ACCENT_BLUE, text_color=ACCENT_BLUE)
    else:
        status_dot.configure(text_color="#ff5252")
        status_label.configure(text="PAUSED", text_color="#ff5252")
        status_frame.configure(fg_color="#1f0d0d", border_color="#3a1a1a")
        toggle_btn.configure(text="RESUME", fg_color="#1a1a0d",
                              border_color="#ffea00", text_color="#ffea00")


toggle_btn = ctk.CTkButton(
    ctrl_card,
    text="PAUSE",
    height=46,
    font=FONT_HEAD,
    fg_color="#0d1a2e",
    hover_color="#0a1525",
    border_width=1,
    border_color=ACCENT_BLUE,
    text_color=ACCENT_BLUE,
    corner_radius=6,
    command=toggle_camera,
)
toggle_btn.pack(fill="x", padx=14, pady=14)


# Webcam capture.

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 30)


# Camera capture thread.

current_frame = None
frame_lock = threading.Lock()
running_thread = True


def camera_loop():
    global current_frame
    while running_thread:
        if running:
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                with frame_lock:
                    current_frame = frame.copy()
        time.sleep(0.01)


camera_thread = threading.Thread(target=camera_loop, daemon=True)
camera_thread.start()


# Hand skeleton drawing.

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

def draw_hand(frame, landmarks, w, h):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    # connections
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 100, 180), 1, cv2.LINE_AA)

    # joints
    for i, (x, y) in enumerate(pts):
        is_tip = i in FINGERTIPS.values()
        color = (0, 229, 255) if is_tip else (41, 121, 255)
        radius = 6 if is_tip else 4
        cv2.circle(frame, (x, y), radius, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), radius, (255, 255, 255), 1, cv2.LINE_AA)


# GUI update loop.

last_prediction_time = 0
last_text_update = 0


def update_frame():
    global last_prediction_time, last_text_update

    with frame_lock:
        if current_frame is None:
            app.after(33, update_frame)
            return
        frame = current_frame.copy()

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp_ms = int(time.time() * 1000)

    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    if result.hand_landmarks:
        hand_landmarks = result.hand_landmarks[0]
        h, w, _ = frame.shape
        draw_hand(frame, hand_landmarks, w, h)

        try:
            features = extract_features(hand_landmarks)
            now = time.time()

            if now - last_prediction_time > PREDICTION_INTERVAL:
                prediction = predict_letter_with_confidence(features)
                prediction_label.configure(text=prediction)
                last_prediction_time = now

            if now - last_text_update > TEXT_UPDATE_INTERVAL:
                keys = list(features.keys())[:20]
                preview = "\n".join([f"{k:<28} {features[k]:+.4f}" for k in keys])
                coords_textbox.delete("1.0", "end")
                coords_textbox.insert("1.0", preview)
                last_text_update = now

        except Exception as e:
            print("Prediction error:", e)
            prediction_label.configure(text="-")

    else:
        prediction_label.configure(text="-")
        coords_textbox.delete("1.0", "end")
        coords_textbox.insert("1.0", "[ no hand detected ]")

    # Display the processed frame.
    # CTkImage keeps the frame scaled correctly on HiDPI displays.
    display_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(display_rgb).resize((860, 660))

    ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img,
                            size=(860, 660))

    video_label.configure(image=ctk_img)
    video_label.image = ctk_img        # keep reference

    app.after(33, update_frame)


# Close handler.

def on_closing():
    global running_thread
    running_thread = False
    cap.release()
    cv2.destroyAllWindows()
    app.destroy()


app.protocol("WM_DELETE_WINDOW", on_closing)



update_frame()
app.mainloop()

