"""
ASL Hand Sign Recognition - Flask Backend (Streamlined)

New architecture: MediaPipe runs in the browser (JS).
The browser sends pre-processed data; the backend only runs the Keras model.

Endpoints:
  POST /api/predict   — receives landmarks or crop, returns prediction
  POST /api/load_model
  GET  /api/models
  GET  /api/status
"""

import os, pickle, base64, threading, logging
import numpy as np
import cv2

from flask import Flask, render_template, request, jsonify

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tensorflow import keras

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["SECRET_KEY"] = "asl-recognition-secret-key"

# ── Paths ─────────────────────────────────────────────────────────
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

# ── Labels ────────────────────────────────────────────────────────
ASL_29 = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "nothing", "space"]
ASL_28 = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "space"]
ASL_26 = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# ── Model state ───────────────────────────────────────────────────
current_model      = None
current_model_name = None
current_labels     = None
model_lock         = threading.Lock()


def model_type(name: str) -> str:
    n = name.lower()
    if "skeleton" in n:                    return "skeleton"
    if "crop"     in n:                    return "crop"
    if "landmark" in n or "mlp" in n:      return "landmark"
    return "raw"


def labels_for(model_name: str, model) -> list:
    n = model.output_shape[-1]
    if "mlp" in model_name.lower() or "landmark" in model_name.lower():
        enc = os.path.join(MODEL_DIR, "landmark_label_encoder.pkl")
        if os.path.isfile(enc):
            try:
                with open(enc, "rb") as f:
                    return list(pickle.load(f).classes_)
            except Exception:
                pass
    if n == 29: return ASL_29
    if n == 28: return ASL_28
    return ASL_26

# ── Inference helpers ─────────────────────────────────────────────

def run_landmark(coords: list, model, labels: list) -> dict:
    """
    coords: flat list of 63 floats [x0,y0,z0, x1,y1,z1, ... x20,y20,z20]
    No normalisation — matches training exactly.
    """
    inp   = np.array(coords, dtype=np.float32).reshape(1, 63)
    with model_lock:
        preds = model.predict(inp, verbose=0)[0]
    return _make_result(preds, labels)


def run_image(b64_image: str, model, name: str, labels: list) -> dict:
    """
    b64_image: base64 JPEG/PNG of the hand crop sent from the browser.
    Browser already cropped to the hand region, so we just resize + normalise.
    """
    buf    = base64.b64decode(b64_image.split(",", 1)[-1])
    arr    = np.frombuffer(buf, np.uint8)
    img    = cv2.imdecode(arr, cv2.IMREAD_COLOR)          # BGR
    H, W   = model.input_shape[1], model.input_shape[2]
    img    = cv2.resize(img, (W, H))
    rgb    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

    mtype = model_type(name)
    if mtype == "skeleton":
        # Browser sends the skeleton canvas it drew — feed raw [0-255]
        inp = np.expand_dims(rgb, 0)
    elif mtype in ("crop", "fast"):
        # Rescaling baked inside model — feed raw [0-255]
        inp = np.expand_dims(rgb, 0)
    else:
        # raw: external /255 normalisation
        inp = np.expand_dims(rgb / 255.0, 0)

    with model_lock:
        preds = model.predict(inp, verbose=0)[0]
    return _make_result(preds, labels)


def _make_result(preds, labels) -> dict:
    idx  = int(np.argmax(preds))
    conf = float(preds[idx])
    top3 = [
        {"label": labels[i] if i < len(labels) else str(i),
         "prob":  round(float(preds[i]) * 100, 1)}
        for i in np.argsort(preds)[::-1][:3]
    ]
    return {
        "label"     : labels[idx] if idx < len(labels) else str(idx),
        "confidence": round(conf * 100, 1),
        "top3"      : top3,
    }

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
    global current_model, current_model_name, current_labels
    name = (request.get_json() or {}).get("model_name", "")
    if name not in AVAILABLE_MODELS:
        return jsonify({"success": False, "error": "Unknown model"}), 400
    path = os.path.join(MODEL_DIR, name)
    if not os.path.isfile(path):
        return jsonify({"success": False, "error": f"File not found: {path}"}), 404
    try:
        logger.info(f"Loading {name}")
        model  = keras.models.load_model(path, compile=False)
        labels = labels_for(name, model)
        with model_lock:
            current_model      = model
            current_model_name = name
            current_labels     = labels
        logger.info(f"Loaded {name} | {model.input_shape} | {len(labels)} classes")
        return jsonify({
            "success"    : True,
            "model"      : name,
            "model_type" : model_type(name),
            "input_shape": str(model.input_shape),
            "n_classes"  : len(labels),
        })
    except Exception as e:
        logger.exception("Load failed")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/status")
def api_status():
    with model_lock:
        name = current_model_name
    return jsonify({"model_loaded": name is not None, "model_name": name,
                    "model_type": model_type(name) if name else None})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Unified prediction endpoint.
    Body (JSON):
      { "type": "landmark", "landmarks": [x0,y0,z0,...] }   63 floats
      { "type": "image",    "image": "data:image/jpeg;..." } base64 crop
    """
    with model_lock:
        model  = current_model
        name   = current_model_name
        labels = current_labels

    if model is None:
        return jsonify({"error": "No model loaded"}), 400

    data = request.get_json(force=True) or {}
    kind = data.get("type", "landmark")

    try:
        if kind == "landmark":
            coords = data.get("landmarks", [])
            if len(coords) != 63:
                return jsonify({"error": f"Expected 63 values, got {len(coords)}"}), 400
            result = run_landmark(coords, model, labels)
        else:
            b64 = data.get("image", "")
            if not b64:
                return jsonify({"error": "No image"}), 400
            result = run_image(b64, model, name, labels)
        return jsonify(result)
    except Exception as e:
        logger.exception("Predict failed")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    logger.info("ASL server on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
