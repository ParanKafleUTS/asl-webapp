import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow import keras
import os

MODEL_DIR = r"C:\Users\kafle\Documents\VS_AI\Hand_sign_to_voice"

# Load your test image (the I_test.jpg you shared)
img_path = r"C:\Users\kafle\Documents\VS_AI\I_test.jpg"   # <-- change this
img = cv2.imread(img_path)
#I_test
# Test with crop model
model = keras.models.load_model(
    os.path.join(MODEL_DIR, "mobilenetv2_crop.h5"), compile=False
)

ASL_29 = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "nothing", "space"]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)

rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
results = hands.process(rgb)

if results.multi_hand_landmarks:
    lm = results.multi_hand_landmarks[0]
    h, w = img.shape[:2]
    xs = [p.x for p in lm.landmark]; ys = [p.y for p in lm.landmark]
    x1 = max(0, int((min(xs)-0.15)*w)); x2 = min(w, int((max(xs)+0.15)*w))
    y1 = max(0, int((min(ys)-0.15)*h)); y2 = min(h, int((max(ys)+0.15)*h))
    crop = img[y1:y2, x1:x2]
    crop = cv2.resize(crop, (224, 224))
    inp = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
    preds = model.predict(np.expand_dims(inp, 0), verbose=0)[0]
    top3 = np.argsort(preds)[::-1][:3]
    print("Top 3 predictions on your test image:")
    for i in top3:
        print(f"  {ASL_29[i]}: {preds[i]*100:.1f}%")
    cv2.imshow("crop fed to model", crop)
    cv2.waitKey(0)
else:
    print("No hand detected in image!")

hands.close()