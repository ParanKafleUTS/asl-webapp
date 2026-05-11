"""
ASL Hand Sign Recognition - Flask Backend
Supports: *_skeleton.h5, *_crop.h5, *_raw.h5, *_fast.h5, landmark_mlp.h5
"""

import os
import cv2
import numpy as np
import base64
import threading
import time
import logging

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TF warnings
import tensorflow as tf
from tensorflow import keras

import mediapipe as mp

# ─────────────────────────── App setup ───────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["SECRET_KEY"] = "asl-recognition-secret-key"
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    max_http_buffer_size=20 * 1024 * 1024,  # 20 MB for frames
    async_mode="threading",
)

# ─────────────────────────── Constants ───────────────────────────
MODEL_DIR = r"C:\Users\kafle\Documents\VS_AI\Hand_sign_to_voice"

AVAILABLE_MODELS = [
    "efficientnetb0_skeleton.h5",
    "mobilenetv2_skeleton.h5",
    "efficientnetb0_crop.h5",
    "mobilenetv2_crop.h5",
    "efficientnetb0_fast.h5",
    "mobilenetv2_raw.h5",
    "landmark_mlp.h5",
]

# ASL class label sets
ASL_26  = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
ASL_29  = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["DEL", "NOTHING", "SPACE"]
ASL_27  = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["SPACE"]

# ─────────────────────────── MediaPipe ───────────────────────────
mp_hands = mp.solutions.hands

_hands_instance = None
_hands_lock = threading.Lock()

def get_hands():
    global _hands_instance
    if _hands_instance is None:
        _hands_instance = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    return _hands_instance

# ─────────────────────────── Global model state ──────────────────
current_model      = None
current_model_name = None
current_labels     = None
model_lock         = threading.Lock()

# ─────────────────────────── Helpers ─────────────────────────────

def model_type(name: str) -> str:
    """Infer preprocessing strategy from filename."""
    n = name.lower()
    if "skeleton" in n:
        return "skeleton"
    if "crop" in n:
        return "crop"
    if "landmark" in n or "mlp" in n:
        return "landmark"
    # raw / fast
    return "raw"


def labels_for(model) -> list:
    n = model.output_shape[-1]
    if n == 26:
        return ASL_26
    if n == 27:
        return ASL_27
    if n == 29:
        return ASL_29
    # Fallback: numeric labels
    return [str(i) for i in range(n)]


def load_model_file(model_name: str) -> dict:
    """Load a .h5 model and cache it globally. Returns status dict."""
    global current_model, current_model_name, current_labels
    path = os.path.join(MODEL_DIR, model_name)
    if not os.path.isfile(path):
        return {"success": False, "error": f"File not found: {path}"}
    try:
        logger.info(f"Loading model: {path}")
        model = keras.models.load_model(path, compile=False)
        labels = labels_for(model)
        with model_lock:
            current_model      = model
            current_model_name = model_name
            current_labels     = labels
        logger.info(
            f"Loaded {model_name} | type={model_type(model_name)} "
            f"| input={model.input_shape} | classes={len(labels)}"
        )
        return {
            "success"     : True,
            "model"       : model_name,
            "model_type"  : model_type(model_name),
            "input_shape" : str(model.input_shape),
            "n_classes"   : len(labels),
            "labels_sample": labels[:5],
        }
    except Exception as e:
        logger.exception("Model load failed")
        return {"success": False, "error": str(e)}

# ─────────────────────────── Frame helpers ───────────────────────

def decode_frame(data_url: str) -> np.ndarray:
    """Base64 data-URL → BGR numpy array."""
    _, encoded = data_url.split(",", 1)
    buf = base64.b64decode(encoded)
    arr = np.frombuffer(buf, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def get_landmarks(frame_bgr):
    """Run MediaPipe on a BGR frame. Returns (landmark_obj | None, hand_bbox | None)."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    with _hands_lock:
        results = get_hands().process(rgb)
    if not results.multi_hand_landmarks:
        return None, None
    lm = results.multi_hand_landmarks[0]
    h, w = frame_bgr.shape[:2]
    xs = [p.x for p in lm.landmark]
    ys = [p.y for p in lm.landmark]
    margin = 0.15
    bbox = (
        max(0, int((min(xs) - margin) * w)),
        max(0, int((min(ys) - margin) * h)),
        min(w, int((max(xs) + margin) * w)),
        min(h, int((max(ys) + margin) * h)),
    )
    return lm, bbox

# ─────────────────────────── Preprocessing ───────────────────────

def preprocess_crop(frame, model) -> tuple:
    """Detect hand, crop, resize to model's spatial input size."""
    lm, bbox = get_landmarks(frame)
    if lm is None:
        return None, False
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return None, False
    crop = frame[y1:y2, x1:x2]
    # Determine target size from model
    in_h, in_w = model.input_shape[1], model.input_shape[2]
    crop = cv2.resize(crop, (in_w, in_h))
    # Normalise to [0, 1]
    inp = crop.astype(np.float32) / 255.0
    return np.expand_dims(inp, 0), True


def preprocess_raw(frame, model) -> tuple:
    """Resize full frame (no hand detection required)."""
    in_h, in_w = model.input_shape[1], model.input_shape[2]
    resized = cv2.resize(frame, (in_w, in_h))
    inp = resized.astype(np.float32) / 255.0
    return np.expand_dims(inp, 0), True


def preprocess_skeleton(frame, model) -> tuple:
    """
    Extract 21-keypoint coords from MediaPipe and reshape to match
    whatever the skeleton model expects.
    """
    lm, _ = get_landmarks(frame)
    if lm is None:
        return None, False

    raw = np.array(
        [[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32
    )  # (21, 3)

    in_shape = model.input_shape[1:]  # everything after batch dim

    if len(in_shape) == 1:
        flat_len = in_shape[0]
        if flat_len == 42:
            flat = raw[:, :2].flatten()
        else:  # 63 or other
            flat = raw.flatten()[:flat_len]
        inp = flat.reshape(1, flat_len)

    elif len(in_shape) == 2:
        n_pts, n_dim = in_shape
        inp = raw[:n_pts, :n_dim].reshape(1, n_pts, n_dim)

    elif len(in_shape) == 3:
        # e.g. (21, 3, 1) – image-style input for skeleton
        n_pts, n_dim, _ = in_shape
        inp = raw[:n_pts, :n_dim].reshape(1, n_pts, n_dim, 1)

    else:
        inp = raw.flatten().reshape(1, -1)

    return inp.astype(np.float32), True


def preprocess_landmark(frame, model) -> tuple:
    """
    Normalised landmark vector (wrist-relative, unit-scaled).
    Matches landmark_mlp.h5 which expects a 1-D feature vector.
    """
    lm, _ = get_landmarks(frame)
    if lm is None:
        return None, False

    raw = np.array(
        [[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32
    )  # (21, 3)

    # Wrist-relative normalisation
    wrist = raw[0]
    centered = raw - wrist
    scale = np.max(np.abs(centered)) + 1e-8
    normed = (centered / scale).flatten()  # (63,)

    in_shape = model.input_shape[1:]
    n = int(np.prod(in_shape))

    if len(normed) < n:
        normed = np.pad(normed, (0, n - len(normed)))
    else:
        normed = normed[:n]

    return normed.reshape(1, *in_shape).astype(np.float32), True

# ─────────────────────────── Inference ───────────────────────────

PREPROCESS_MAP = {
    "crop"    : preprocess_crop,
    "raw"     : preprocess_raw,
    "skeleton": preprocess_skeleton,
    "landmark": preprocess_landmark,
}


def run_inference(frame, model, name, labels) -> dict:
    mtype = model_type(name)
    fn    = PREPROCESS_MAP.get(mtype, preprocess_raw)

    inp, hand_found = fn(frame, model)

    if inp is None:
        return {"hand_detected": False, "label": None, "confidence": 0.0}

    with model_lock:
        preds = model.predict(inp, verbose=0)[0]

    idx        = int(np.argmax(preds))
    confidence = float(preds[idx])
    label      = labels[idx] if idx < len(labels) else str(idx)

    return {
        "hand_detected": hand_found,
        "label"        : label,
        "confidence"   : round(confidence * 100, 1),
        "all_top3"     : _top3(preds, labels),
    }


def _top3(preds, labels):
    top = np.argsort(preds)[::-1][:3]
    return [
        {"label": labels[i] if i < len(labels) else str(i),
         "prob" : round(float(preds[i]) * 100, 1)}
        for i in top
    ]

# ─────────────────────────── Routes ──────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", models=AVAILABLE_MODELS)


@app.route("/api/models")
def api_models():
    existing = []
    for m in AVAILABLE_MODELS:
        path  = os.path.join(MODEL_DIR, m)
        mtype = model_type(m)
        existing.append({
            "name"   : m,
            "type"   : mtype,
            "exists" : os.path.isfile(path),
        })
    return jsonify(existing)


@app.route("/api/load_model", methods=["POST"])
def api_load_model():
    data       = request.get_json()
    model_name = (data or {}).get("model_name", "")
    if model_name not in AVAILABLE_MODELS:
        return jsonify({"success": False, "error": "Unknown model name"}), 400
    result = load_model_file(model_name)
    return jsonify(result)


@app.route("/api/status")
def api_status():
    with model_lock:
        loaded = current_model_name
    return jsonify({
        "model_loaded": loaded is not None,
        "model_name"  : loaded,
        "model_type"  : model_type(loaded) if loaded else None,
    })

# ─────────────────────────── WebSocket ───────────────────────────

@socketio.on("connect")
def on_connect():
    logger.info(f"Client connected: {request.sid}")
    with model_lock:
        loaded = current_model_name
    emit("status", {"model_loaded": loaded is not None, "model_name": loaded})


@socketio.on("disconnect")
def on_disconnect():
    logger.info(f"Client disconnected: {request.sid}")


@socketio.on("frame")
def handle_frame(data):
    with model_lock:
        model  = current_model
        name   = current_model_name
        labels = current_labels

    if model is None:
        emit("prediction", {"error": "No model loaded. Please select a model first."})
        return

    try:
        frame  = decode_frame(data["image"])
        result = run_inference(frame, model, name, labels)
        emit("prediction", result)
    except Exception as e:
        logger.exception("Inference error")
        emit("prediction", {"error": str(e)})


# ─────────────────────────── Entry point ─────────────────────────

if __name__ == "__main__":
    logger.info("Starting ASL Recognition Server on http://0.0.0.0:5000")
    socketio.run(app, host="0.0.0.0", port=5000, debug=False, use_reloader=False)
