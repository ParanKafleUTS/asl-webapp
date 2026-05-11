# 🤟 ASL Hand Sign Recognition — Web App

Real-time American Sign Language (A–Z) detection in the browser using your webcam,
powered by Flask + Socket.IO on the backend and MediaPipe for hand landmark extraction.

---

## Project Structure

```
asl_webapp/
├── app.py               ← Flask backend (preprocessing + inference)
├── requirements.txt     ← Python dependencies
├── README.md
└── templates/
    └── index.html       ← Frontend (webcam, UI, accumulator)
```

---

## 1. Prerequisites

| Requirement | Version |
|------------|---------|
| Python     | 3.9 – 3.11 (recommended) |
| pip        | latest  |
| Webcam     | any USB or built-in |
| GPU        | optional – CPU works fine for inference |

> **Windows note:** If you see `DLL load failed` errors with TensorFlow, install the
> [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe).

---

## 2. Installation

### Step 1 — Create a virtual environment (strongly recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 2 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If you don't have an NVIDIA GPU (or want a lighter install):

```bash
pip install tensorflow-cpu>=2.13.0
```

### Step 3 — Verify MediaPipe + OpenCV

```bash
python -c "import mediapipe, cv2, tensorflow; print('All OK')"
```

---

## 3. Model compatibility notes

### Are the `.h5` files already loadable?

If your models were saved with `model.save('name.h5')` using Keras / TensorFlow 2.x,
they load directly. If you get warnings about `compile=False`, ignore them — the app
already passes `compile=False` to `keras.models.load_model()`.

### Converting SavedModel → H5 (if needed)

```python
import tensorflow as tf
model = tf.saved_model.load("path/to/saved_model_dir")
# or
model = tf.keras.models.load_model("path/to/saved_model_dir")
model.save("output.h5")
```

### TF 2.16+ / Keras 3 compatibility

TF 2.16+ ships with Keras 3 by default. If loading fails, try:

```bash
pip install tf_keras
```

Then at the top of `app.py`, add before the keras import:
```python
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
```

---

## 4. Preprocessing pipeline (per model type)

The app auto-detects which pipeline to use from the filename:

| Filename pattern | Pipeline | What it does |
|-----------------|----------|-------------|
| `*_crop.h5`     | **crop** | MediaPipe finds hand bbox → crop → resize to model input size → normalize [0,1] |
| `*_skeleton.h5` | **skeleton** | MediaPipe 21-keypoint coords → reshaped to match model `input_shape` (flat / 2-D / 3-D) |
| `*_raw.h5`, `*_fast.h5` | **raw** | Resize full frame to model input size → normalize [0,1] |
| `landmark_mlp.h5` | **landmark** | MediaPipe coords → wrist-relative → unit-scaled → flat feature vector |

### Normalization note

By default the app normalises pixel values to **[0, 1]** (`/ 255.0`).
If your models expect ImageNet-style normalization (mean subtraction), open `app.py`
and replace the `/255.0` line in the relevant `preprocess_*` function with:

```python
# EfficientNet
from tensorflow.keras.applications.efficientnet import preprocess_input
inp = preprocess_input(crop.astype(np.float32))

# MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
inp = preprocess_input(resized.astype(np.float32))
```

---

## 5. Running the app

```bash
# Make sure your virtualenv is active
python app.py
```

Then open your browser at:

```
http://localhost:5000
```

---

## 6. Using the app

### Load a model

1. Click the **Model** dropdown in the left sidebar.
2. Select one of the seven models.
3. Click **Load Model** — the badge shows which pipeline will be used (CROP / SKELETON / RAW / LANDMARK).
4. Loading takes 5–30 seconds the first time (TF graph compilation).

### Real-time recognition

Once a model is loaded the webcam feed is processed at **10 FPS** (adjustable via the slider).

- The **large letter** overlaid on the video is the top prediction.
- The **confidence %** appears bottom-right of the video.
- The **Top Predictions** panel shows the three most likely letters with confidence bars.
- A "No hand detected" banner appears when no hand is visible.

### Accumulating text

| Action | How |
|--------|-----|
| Manually add current sign | Click **＋ Add Sign** or press **Space** |
| Add a space character | Click **Space** button |
| Delete last character | Click **⌫ Delete** or press **Backspace** |
| Auto-add (stable sign) | Toggle **Auto-add (stable 0.5 s)** — sign is added automatically when held steady |
| Copy text | Click **📋 Copy** |
| Clear all text | Click **✕ Clear** |

### Switching models

Simply select a different model from the dropdown and click **Load Model** again.
The old model is replaced in memory.

---

## 7. Class labels

The app automatically selects the label set based on the model's output size:

| Output neurons | Labels used |
|---------------|-------------|
| 26 | A – Z |
| 27 | A – Z + SPACE |
| 29 | A – Z + DEL + NOTHING + SPACE |
| other | 0, 1, 2, … (numeric) |

---

## 8. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: tensorflow` | Run `pip install tensorflow` inside the venv |
| `DLL load failed` (Windows) | Install [VC++ Redistributable](https://aka.ms/vs/17/release/vc_bedist.x64.exe) |
| Camera not opening | Allow camera permission in your browser; check no other app is using it |
| Very low confidence | Ensure good lighting; try a different model; check normalization (§4) |
| `File not found` error | Verify `MODEL_DIR` in `app.py` points to your model folder |
| Port 5000 already in use | Change `port=5000` to e.g. `port=5001` in `app.py` |
| Slow inference | Reduce FPS slider; use `mobilenetv2_*` models (faster than efficientnet) |

---

## 9. Architecture overview

```
Browser                          Flask Server (app.py)
───────                          ─────────────────────
getUserMedia() ──JPEG frame──▶  Socket.IO "frame" event
                                  │
                                  ├─ decode_frame()        (base64 → numpy BGR)
                                  ├─ get_landmarks()       (MediaPipe Hands)
                                  ├─ preprocess_*()        (per model type)
                                  ├─ model.predict()       (Keras inference)
                                  └─ emit "prediction"
                                         │
{ label, confidence, top3 } ◀───────────┘
         │
         ├─ Overlay on video canvas
         ├─ Update confidence bars
         └─ Stability timer → auto-add to text
```

---

## 10. Acknowledgements

- [MediaPipe Hands](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) — hand landmark detection
- [TensorFlow / Keras](https://www.tensorflow.org/) — model loading & inference
- [Flask-SocketIO](https://flask-socketio.readthedocs.io/) — real-time websocket communication
