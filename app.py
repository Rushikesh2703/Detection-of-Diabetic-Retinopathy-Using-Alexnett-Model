import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# -----------------------------
# Load your saved model
# -----------------------------
MODEL_PATH = "alexnet_dr_best.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Class labels
class_names = ["Mild", "Moderate", "No_DR", "Proliferate_DR", "Severe"]

# -----------------------------
# CLAHE Preprocessing
# -----------------------------
def apply_CLAHE(image):
    img = image.astype(np.uint8)

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L_clahe = clahe.apply(L)

    lab_clahe = cv2.merge((L_clahe, A, B))
    rgb_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

    return rgb_clahe

# -----------------------------
# Prediction Function
# -----------------------------
def predict_retina(image):
    # Resize to model input size
    img = cv2.resize(image, (224, 224))

    # Apply CLAHE (same as training)
    img = apply_CLAHE(img)

    # Normalize (0–255 → 0–1)
    img = img / 255.0

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    # Predict
    preds = model.predict(img)
    class_id = np.argmax(preds)
    confidence = preds[0][class_id] * 100

    return class_names[class_id], confidence

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("👁️ Diabetic Retinopathy Detection App")
st.write("Upload a retina image to classify its DR severity using the AlexNet model.")

uploaded_file = st.file_uploader("Upload retina image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # Show uploaded image
    st.image(image_np, caption="Uploaded Image", use_column_width=True)

    # Predict
    label, confidence = predict_retina(image_np)

    st.subheader("🔍 Prediction")
    st.write(f"**Class:** {label}")
    st.write(f"**Confidence:** {confidence:.2f}%")
