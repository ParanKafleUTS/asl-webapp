"""
ASL Hand Sign Recognition - Flask Backend (Vercel + Hugging Face)

Models are downloaded from Hugging Face Hub at runtime into /tmp.
MediaPipe runs in the browser (JS); backend only runs the Keras model.

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
from huggingface_hub import hf_hub_download

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tensorflow import keras

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# templates folder is one level up from api/
app = Flask(__name__, template_folder="../templates")
app.config["SECRET_KEY"] = "asl-recognition-secret-key"

# ── Hugging Face config ────────────────────────────────────────────
HF_REPO  = "ParanKafle11/asl-models"
HF_TOKEN = os.environ.get("HF_TOKEN")   # set in Vercel env vars
TMP_DIR  = "/tmp"

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


# ── Hugging Face download helper ──────────────────────────────────

def get_model_path(filename: str) -> str:
    """
    Download a file from Hugging Face to /tmp if not already cached.
    Returns the local path.
    """
    local_path = os.path.join(TMP_DIR, filename)
    if os.path.exists(local_path):
        logger.info(f"Using cached file: {local_path}")
        return local_path

    logger.info(f"Downloading {filename} from Hugging Face...")
    try:
        downloaded = hf_hub_download(
            repo_id=HF_REPO,
            filename=filename,
            local_dir=TMP_DIR,
            token=HF_TOKEN,
        )
        logger.info(f"Downloaded {filename} to {downloaded}")
        return downloaded
    except Exception as e:
        logger.error(f"Failed to download {filename}: {e}")
        raise


# ── Utility ───────────────────────────────────────────────────────

def model_type(name: str) -> str:
    n = name.lower()
    if "skeleton" in n:               return "skeleton"
    if "crop"     in n:               return "crop"
    if "landmark" in n or "mlp" in n: return "landmark"
    return "raw"


def labels_for(model_name: str, model) -> list:
    n = model.output_shape[-1]
    if "mlp" in model_name.lower() or "landmark" in model_name.lower():
        try:
            enc_path = get_model_path("landmark_label_encoder.pkl")
            with open(enc_path, "rb") as f:
                return list(pickle.load(f).classes_)
        except Exception as e:
            logger.warning(f"Could not load label encoder: {e}")
    if n == 29: return ASL_29
    if n == 28: return ASL_28
    return ASL_26


# ── Inference helpers ─────────────────────────────────────────────

def run_landmark(coords: list, model, labels: list) -> dict:
    """
    coords: flat list of 63 floats [x0,y0,z0, x1,y1,z1, ... x20,y20,z20]
    """
    inp = np.array(coords, dtype=np.float32).reshape(1, 63)
    with model_lock:
        preds = model.predict(inp, verbose=0)[0]
    return _make_result(preds, labels)


def run_image(b64_image: str, model, name: str, labels: list) -> dict:
    """
    b64_image: base64 JPEG/PNG of the hand crop sent from the browser.
    """
    buf  = base64.b64decode(b64_image.split(",", 1)[-1])
    arr  = np.frombuffer(buf, np.uint8)
    img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    H, W = model.input_shape[1], model.input_shape[2]
    img  = cv2.resize(img, (W, H))
    rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

    mtype = model_type(name)
    if mtype in ("skeleton", "crop", "fast"):
        inp = np.expand_dims(rgb, 0)
    else:
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
        {"name": m, "type": model_type(m), "exists": True}
        for m in AVAILABLE_MODELS
    ])


@app.route("/api/load_model", methods=["POST"])
def api_load_model():
    global current_model, current_model_name, current_labels

    name = (request.get_json() or {}).get("model_name", "")
    if name not in AVAILABLE_MODELS:
        return jsonify({"success": False, "error": "Unknown model"}), 400

    try:
        logger.info(f"Loading {name}")
        path   = get_model_path(name)               # download from HF if needed
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
    return jsonify({
        "model_loaded": name is not None,
        "model_name"  : name,
        "model_type"  : model_type(name) if name else None,
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
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
