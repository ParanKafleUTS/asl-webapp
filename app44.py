"""
ASL Hand Sign Recognition - Flask Backend
Preprocessing matches training scripts exactly:

  *_crop.h5 / *_skeleton.h5 / *_fast.h5
      Rescaling layer is BAKED INSIDE the model.
      Feed raw float32 [0-255].

  *_raw.h5
      Trained with ImageDataGenerator(rescale=1/255) which is EXTERNAL.
      Divide by 255 manually -> [0, 1].

  landmark_mlp.h5
      Raw [x, y, z] coordinates, no normalization at all.
      Load label order from landmark_label_encoder.pkl.

Label order (image models) = Python sorted() on Kaggle folder names:
  A-Z (uppercase, ASCII 65-90 sort first) then del, nothing, space (lowercase).
"""

import os
import pickle
import cv2
import numpy as np
import base64
import threading
import logging

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
import mediapipe as mp

# ── App setup ────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["SECRET_KEY"] = "asl-recognition-secret-key"
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    max_http_buffer_size=20 * 1024 * 1024,
    async_mode="threading",
)

# ── Paths ────────────────────────────────────────────────────────
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

# ── Label sets ───────────────────────────────────────────────────
# sorted() on Kaggle ASL Alphabet class names (A-Z uppercase + del/nothing/space lowercase)
ASL_29 = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "nothing", "space"]
# skeleton models: "nothing" images have no hand -> skipped -> 28 classes
ASL_28 = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "space"]
ASL_26 = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# ── MediaPipe ────────────────────────────────────────────────────
mp_hands    = mp.solutions.hands
_hands_inst = None
_hands_lock = threading.Lock()

def get_hands():
    global _hands_inst
    if _hands_inst is None:
        _hands_inst = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.65,
        )
    return _hands_inst

# ── Model state ──────────────────────────────────────────────────
current_model      = None
current_model_name = None
current_labels     = None
model_lock         = threading.Lock()

# ── Helpers ──────────────────────────────────────────────────────

def model_type(name: str) -> str:
    n = name.lower()
    if "skeleton" in n:               return "skeleton"
    if "crop" in n:                   return "crop"
    if "landmark" in n or "mlp" in n: return "landmark"
    return "raw"


def labels_for(model_name: str, model) -> list:
    n_cls = model.output_shape[-1]

    # landmark MLP: load saved LabelEncoder for guaranteed correct order
    if "mlp" in model_name.lower() or "landmark" in model_name.lower():
        enc_path = os.path.join(MODEL_DIR, "landmark_label_encoder.pkl")
        if os.path.isfile(enc_path):
            try:
                with open(enc_path, "rb") as f:
                    le = pickle.load(f)
                logger.info(f"Loaded landmark encoder classes: {list(le.classes_)}")
                return list(le.classes_)
            except Exception as e:
                logger.warning(f"Could not load label encoder: {e}")
        logger.warning("landmark_label_encoder.pkl not found — using sorted fallback")

    if n_cls == 29: return ASL_29
    if n_cls == 28: return ASL_28
    if n_cls == 26: return ASL_26
    return [str(i) for i in range(n_cls)]


def load_model_file(model_name: str) -> dict:
    global current_model, current_model_name, current_labels
    path = os.path.join(MODEL_DIR, model_name)
    if not os.path.isfile(path):
        return {"success": False, "error": f"File not found: {path}"}
    try:
        logger.info(f"Loading model: {path}")
        model  = keras.models.load_model(path, compile=False)
        labels = labels_for(model_name, model)
        with model_lock:
            current_model      = model
            current_model_name = model_name
            current_labels     = labels
        logger.info(
            f"Loaded {model_name} | type={model_type(model_name)} "
            f"| input={model.input_shape} | classes={len(labels)} | first5={labels[:5]}"
        )
        return {
            "success"      : True,
            "model"        : model_name,
            "model_type"   : model_type(model_name),
            "input_shape"  : str(model.input_shape),
            "n_classes"    : len(labels),
            "labels_sample": labels[:5],
        }
    except Exception as e:
        logger.exception("Model load failed")
        return {"success": False, "error": str(e)}

# ── Frame helpers ─────────────────────────────────────────────────

def decode_frame(data_url: str) -> np.ndarray:
    _, encoded = data_url.split(",", 1)
    buf = base64.b64decode(encoded)
    arr = np.frombuffer(buf, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)   # BGR


def get_landmarks(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    with _hands_lock:
        results = get_hands().process(rgb)
    if not results.multi_hand_landmarks:
        return None, None
    lm   = results.multi_hand_landmarks[0]
    h, w = frame_bgr.shape[:2]
    xs   = [p.x for p in lm.landmark]
    ys   = [p.y for p in lm.landmark]
    m    = 0.15
    bbox = (
        max(0, int((min(xs) - m) * w)),
        max(0, int((min(ys) - m) * h)),
        min(w, int((max(xs) + m) * w)),
        min(h, int((max(ys) + m) * h)),
    )
    return lm, bbox

# ── Preprocessing ─────────────────────────────────────────────────

def preprocess_crop(frame, model) -> tuple:
    """
    Crop hand region, resize, BGR->RGB.
    Pass raw [0-255]: Rescaling(1/127.5, -1) is BAKED INSIDE the model
    (see create_fast_model in 05_fast_crop.py).
    EfficientNetB0 variant has no Rescaling layer — also feed raw [0-255].
    """
    lm, bbox = get_landmarks(frame)
    if lm is None:
        return None, False
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return None, False
    crop     = frame[y1:y2, x1:x2]
    H, W     = model.input_shape[1], model.input_shape[2]
    crop     = cv2.resize(crop, (W, H))
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
    return np.expand_dims(crop_rgb, 0), True


def preprocess_raw(frame, model) -> tuple:
    """
    Crop to hand FIRST, then resize, BGR->RGB, divide by 255 -> [0, 1].

    The Kaggle training images are isolated hands (like I_test.jpg).
    Feeding the full webcam frame (face + body + background) causes the
    domain gap that breaks recognition. Cropping bridges that gap without
    retraining.

    Training used ImageDataGenerator(rescale=1/255) externally (04_raw_cnn.py).
    No Rescaling layer baked inside this model.
    """
    lm, bbox = get_landmarks(frame)
    if lm is None:
        return None, False
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return None, False
    hand        = frame[y1:y2, x1:x2]
    H, W        = model.input_shape[1], model.input_shape[2]
    resized     = cv2.resize(hand, (W, H))
    resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    return np.expand_dims(resized_rgb / 255.0, 0), True


def preprocess_skeleton(frame, model) -> tuple:
    """
    Reproduce the skeleton image exactly as generated during training.

    Training script draws on a WHITE canvas with cv2:
        canvas = np.ones((224,224,3), uint8) * 255   <- white background
        cv2.line(...)  color=(0,0,0)                 <- black bones
        cv2.circle(...)  color=(0,0,255)             <- RED in cv2 BGR

    Images saved with cv2.imwrite (BGR) and loaded by tf.data as RGB, so the
    red circles (BGR 0,0,255) appear as RGB (255,0,0) = red in the model input.

    Here: draw with cv2 (BGR), then convert BGR->RGB before feeding.
    Rescaling BAKED IN -> pass raw [0-255].
    """
    lm, _ = get_landmarks(frame)
    if lm is None:
        return None, False

    H = model.input_shape[1]
    W = model.input_shape[2]

    canvas = np.ones((H, W, 3), dtype=np.uint8) * 255  # white background

    CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
        (5,9),(9,13),(13,17),
    ]
    pts = [(int(p.x * W), int(p.y * H)) for p in lm.landmark]

    for a, b in CONNECTIONS:
        cv2.line(canvas, pts[a], pts[b], (0, 0, 0), 2, cv2.LINE_AA)   # black

    for x, y in pts:
        cv2.circle(canvas, (x, y), 3, (0, 0, 255), -1, cv2.LINE_AA)  # red (BGR)

    # BGR->RGB to match what tf.image.decode_jpeg produces from the saved file
    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32)
    return np.expand_dims(canvas_rgb, 0), True


def preprocess_landmark(frame, model) -> tuple:
    """
    Raw [x, y, z] coordinates — NO normalization.
    Training (landmark MLP script):
        for lm in landmarks.landmark:
            coords.extend([lm.x, lm.y, lm.z])   <- no scaling applied
    """
    lm, _ = get_landmarks(frame)
    if lm is None:
        return None, False

    coords = []
    for p in lm.landmark:
        coords.extend([p.x, p.y, p.z])
    return np.array(coords, dtype=np.float32).reshape(1, 63), True

# ── Inference ─────────────────────────────────────────────────────

PREPROCESS_MAP = {
    "crop"    : preprocess_crop,
    "raw"     : preprocess_raw,
    "skeleton": preprocess_skeleton,
    "landmark": preprocess_landmark,
}


def run_inference(frame, model, name, labels) -> dict:
    fn = PREPROCESS_MAP.get(model_type(name), preprocess_raw)
    inp, hand_found = fn(frame, model)

    if inp is None:
        return {"hand_detected": False, "label": None, "confidence": 0.0}

    with model_lock:
        preds = model.predict(inp, verbose=0)[0]

    idx        = int(np.argmax(preds))
    confidence = float(preds[idx])
    label      = labels[idx] if idx < len(labels) else str(idx)

    # Compute normalised hand centre from bbox so the frontend can do zone
    # hit-testing entirely in the browser (no extra round-trip needed).
    hand_cx, hand_cy = 0.5, 0.5   # fallback centre
    try:
        fn2 = PREPROCESS_MAP.get(model_type(name), preprocess_raw)
        # Re-run get_landmarks (cached by MediaPipe internal tracking, very cheap)
        lm2, bbox2 = get_landmarks(frame)
        if lm2 is not None and bbox2 is not None:
            x1b, y1b, x2b, y2b = bbox2
            fh, fw = frame.shape[:2]
            hand_cx = ((x1b + x2b) / 2) / fw
            hand_cy = ((y1b + y2b) / 2) / fh
    except Exception:
        pass

    return {
        "hand_detected": hand_found,
        "label"        : label,
        "confidence"   : round(confidence * 100, 1),
        "all_top3"     : _top3(preds, labels),
        "hand_cx"      : round(hand_cx, 4),
        "hand_cy"      : round(hand_cy, 4),
    }


def _top3(preds, labels):
    top = np.argsort(preds)[::-1][:3]
    return [
        {"label": labels[i] if i < len(labels) else str(i),
         "prob" : round(float(preds[i]) * 100, 1)}
        for i in top
    ]

# ── Routes ────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", models=AVAILABLE_MODELS)


@app.route("/api/models")
def api_models():
    return jsonify([
        {"name": m, "type": model_type(m),
         "exists": os.path.isfile(os.path.join(MODEL_DIR, m))}
        for m in AVAILABLE_MODELS
    ])


@app.route("/api/load_model", methods=["POST"])
def api_load_model():
    data       = request.get_json()
    model_name = (data or {}).get("model_name", "")
    if model_name not in AVAILABLE_MODELS:
        return jsonify({"success": False, "error": "Unknown model name"}), 400
    return jsonify(load_model_file(model_name))


@app.route("/api/status")
def api_status():
    with model_lock:
        loaded = current_model_name
    return jsonify({
        "model_loaded": loaded is not None,
        "model_name"  : loaded,
        "model_type"  : model_type(loaded) if loaded else None,
    })


@app.route("/api/debug_frame", methods=["POST"])
def api_debug_frame():
    """
    Return a JPEG of what the model actually receives after preprocessing.
    Used by the frontend debug canvas so the user can see the exact input.
    """
    with model_lock:
        model = current_model
        name  = current_model_name

    if model is None:
        return jsonify({"error": "no model"}), 400

    data  = request.get_json()
    frame = decode_frame(data["image"])
    mtype = model_type(name)

    try:
        if mtype in ("crop", "raw"):
            lm, bbox = get_landmarks(frame)
            if lm is None:
                return jsonify({"hand": False})
            x1, y1, x2, y2 = bbox
            crop = frame[y1:y2, x1:x2]
            H, W = model.input_shape[1], model.input_shape[2]
            vis  = cv2.resize(crop, (W, H))

        elif mtype == "skeleton":
            lm, _ = get_landmarks(frame)
            if lm is None:
                return jsonify({"hand": False})
            H, W = model.input_shape[1], model.input_shape[2]
            vis  = np.ones((H, W, 3), dtype=np.uint8) * 255
            CONN = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),
                    (15,16),(0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]
            pts  = [(int(p.x*W), int(p.y*H)) for p in lm.landmark]
            for a, b in CONN:
                cv2.line(vis, pts[a], pts[b], (0,0,0), 2, cv2.LINE_AA)
            for x, y in pts:
                cv2.circle(vis, (x,y), 3, (0,0,255), -1, cv2.LINE_AA)

        else:  # landmark — show the crop as visual aid
            lm, bbox = get_landmarks(frame)
            if lm is None:
                return jsonify({"hand": False})
            x1, y1, x2, y2 = bbox
            vis = cv2.resize(frame[y1:y2, x1:x2], (200, 200))

        _, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64    = base64.b64encode(buf).decode()
        return jsonify({"hand": True, "image": f"data:image/jpeg;base64,{b64}"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── WebSocket ─────────────────────────────────────────────────────

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

# ── Entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Starting ASL Recognition Server on http://0.0.0.0:5000")
    socketio.run(app, host="0.0.0.0", port=5000, debug=False, use_reloader=False)
