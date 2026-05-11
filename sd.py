import tensorflow as tf
import os

MODEL_DIR = r"C:\Users\kafle\Documents\VS_AI\Hand_sign_to_voice"
models = [
    "efficientnetb0_skeleton.h5",
    "mobilenetv2_skeleton.h5",
    "efficientnetb0_crop.h5",
    "mobilenetv2_crop.h5",
    "efficientnetb0_fast.h5",
    "mobilenetv2_raw.h5",
    "landmark_mlp.h5",
]

for name in models:
    path = os.path.join(MODEL_DIR, name)
    if not os.path.exists(path):
        print(f"{name}: NOT FOUND")
        continue
    try:
        m = tf.keras.models.load_model(path, compile=False)
        print(f"{name}:")
        print(f"  input_shape  : {m.input_shape}")
        print(f"  output_shape : {m.output_shape}")
        print(f"  input dtype  : {m.input.dtype}")
        # Check if preprocessing is baked in
        first_layer = m.layers[0]
        print(f"  first_layer  : {first_layer.__class__.__name__} — {first_layer.name}")
    except Exception as e:
        print(f"{name}: ERROR — {e}")
    print()